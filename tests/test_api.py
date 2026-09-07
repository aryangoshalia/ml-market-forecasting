"""API tests. Offline: every endpoint here reads artefacts committed to the repository."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from market_forecast.api.app import RateLimiter, create_app
from market_forecast.api.service import ForecastService


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(ForecastService()))


class TestHealthAndUniverse:
    def test_health_reports_ready(self, client):
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        assert payload["shipped_artifacts"] is True
        assert payload["universe_size"] > 0

    def test_universe_lists_disjoint_groups(self, client):
        payload = client.get("/api/universe").json()
        assert sum(payload["groups"].values()) == len(payload["tickers"])
        assert {"dev", "val", "test"} <= set(payload["groups"])

    def test_every_ticker_has_a_group_and_sector(self, client):
        for entry in client.get("/api/universe").json()["tickers"]:
            assert entry["group"] != "unassigned"
            assert entry["sector"] != "unknown"


class TestPerformance:
    def test_returns_scores_for_every_model(self, client):
        payload = client.get("/api/performance").json()
        assert len(payload["scores"]) >= 6
        assert payload["coverage"]["folds"] > 10

    def test_baselines_are_present_and_flagged(self, client):
        scores = client.get("/api/performance").json()["scores"]
        baselines = [s for s in scores if s["is_baseline"]]
        assert len(baselines) >= 2, "a leaderboard without baselines is unreadable"

    def test_scores_are_ordered_by_auc(self, client):
        scores = client.get("/api/performance").json()["scores"]
        aucs = [s["roc_auc"] for s in scores]
        assert aucs == sorted(aucs, reverse=True)

    def test_no_model_reports_an_implausible_auc(self, client):
        """A number far above chance here would mean a leak, not a discovery."""
        for score in client.get("/api/performance").json()["scores"]:
            assert 0.4 < score["roc_auc"] < 0.6

    def test_coverage_is_reported_so_the_ui_can_label_it(self, client):
        coverage = client.get("/api/performance").json()["coverage"]
        assert coverage["first_session"] < coverage["last_session"]
        assert coverage["run_id"]
        assert coverage["source"]

    @pytest.mark.parametrize("target", ["direction", "excess_direction"])
    @pytest.mark.parametrize("horizon", [1, 5])
    def test_every_formulation_is_available(self, client, target, horizon):
        response = client.get("/api/performance", params={"target": target, "horizon": horizon})
        assert response.status_code == 200


class TestPredictions:
    def test_returns_history_for_a_universe_ticker(self, client):
        payload = client.get("/api/predictions/AAPL", params={"limit": 10}).json()
        assert payload["n"] > 1000
        assert len(payload["rows"]) == 10
        assert 0.4 < payload["accuracy"] < 0.6

    def test_rows_carry_the_outcome_and_the_fold(self, client):
        row = client.get("/api/predictions/AAPL", params={"limit": 1}).json()["rows"][0]
        assert row["predicted"] in (0, 1)
        assert row["actual"] in (0, 1)
        assert row["correct"] == (row["predicted"] == row["actual"])
        assert row["fold"] >= 0

    def test_rows_are_tagged_with_a_regime(self, client):
        rows = client.get("/api/predictions/AAPL", params={"limit": 20}).json()["rows"]
        assert all(r["regime"] is not None for r in rows)

    def test_unknown_ticker_is_a_404_not_a_crash(self, client):
        assert client.get("/api/predictions/ZZZZ").status_code == 404

    def test_limit_is_bounded(self, client):
        assert client.get("/api/predictions/AAPL", params={"limit": 99999}).status_code == 422


class TestRegimes:
    def test_timeline_is_contiguous_and_named(self, client):
        payload = client.get("/api/regimes").json()
        assert payload["inference"] == "filtered", "smoothed labels would be look-ahead"
        assert len(payload["episodes"]) > 10
        for episode in payload["episodes"]:
            assert episode["start"] <= episode["end"]
            assert episode["sessions"] > 0

    def test_state_names_come_from_statistics_not_assumptions(self, client):
        names = client.get("/api/regimes").json()["names"]
        assert names
        assert all("volatility" in n or "trend" in n or "sideways" in n for n in names.values())


class TestModelCard:
    def test_reports_provenance(self, client):
        payload = client.get("/api/model-card").json()
        card = payload["card"]
        assert card["feature_version"]
        assert card["trained_through"]
        assert card["n_features"] > 0
        assert card["packages"]

    def test_states_its_limitations(self, client):
        assert len(client.get("/api/model-card").json()["limitations"]) >= 3

    def test_carries_no_secrets(self, client):
        body = client.get("/api/model-card").text.lower()
        for term in ("api_key", "apikey", "secret", "password", "token"):
            assert term not in body


class TestInputValidation:
    @pytest.mark.parametrize(
        "ticker",
        ["AA;rm%20-rf%20/", "$(whoami)", "A" * 40, "%2e%2e%2f%2e%2e%2fetc%2fpasswd", "<script>"],
    )
    def test_malformed_tickers_are_rejected(self, client, ticker):
        response = client.get(f"/api/predictions/{ticker}")
        assert response.status_code in (400, 404), f"{ticker} reached the handler"

    def test_traversal_never_reaches_the_filesystem(self, client):
        response = client.get("/api/predictions/%2e%2e%2f%2e%2e%2fetc%2fpasswd")
        assert response.status_code in (400, 404)
        assert "passwd" not in response.text or "detail" in response.text

    @pytest.mark.parametrize("target", ["nonsense", "", "direction; DROP TABLE"])
    def test_unknown_target_is_rejected(self, client, target):
        assert client.get("/api/performance", params={"target": target}).status_code == 400

    @pytest.mark.parametrize("horizon", [0, 3, 99, -1])
    def test_unsupported_horizon_is_rejected(self, client, horizon):
        assert client.get("/api/performance", params={"horizon": horizon}).status_code == 400

    def test_errors_do_not_leak_internals(self, client):
        body = client.get("/api/predictions/ZZZZ").text
        for term in ("Traceback", "/Users/", "site-packages"):
            assert term not in body


class TestRateLimiter:
    def test_allows_up_to_the_limit(self):
        limiter = RateLimiter(limit=3, window=60.0)
        assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]

    def test_clients_are_tracked_separately(self):
        limiter = RateLimiter(limit=1, window=60.0)
        assert limiter.allow("a") is True
        assert limiter.allow("b") is True
        assert limiter.allow("a") is False

    def test_the_window_expires(self):
        limiter = RateLimiter(limit=1, window=0.01)
        assert limiter.allow("a") is True
        import time

        time.sleep(0.02)
        assert limiter.allow("a") is True


class TestCors:
    def test_local_dev_server_is_allowed(self, client):
        response = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_other_origins_get_no_cors_header(self, client):
        response = client.get("/api/health", headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in response.headers


class TestExplanationArithmetic:
    """The waterfall has to add up, and it has to say which number it adds up to."""

    @pytest.mark.network
    def test_contributions_reconstruct_the_raw_probability(self, client):
        import math

        payload = client.get("/api/explain/AAPL").json()
        reconstructed = 1.0 / (1.0 + math.exp(-payload["logodds"]))
        assert reconstructed == pytest.approx(payload["raw_probability"], abs=1e-3)

    @pytest.mark.network
    def test_raw_and_calibrated_are_reported_separately(self, client):
        payload = client.get("/api/explain/AAPL").json()
        assert "raw_probability" in payload
        assert "probability" in payload
        assert "calibration" in payload["caveat"].lower()
