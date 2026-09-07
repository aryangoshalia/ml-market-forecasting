"""Regime tests. The one that matters is that filtered inference cannot see the future."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_forecast.regime.describe import describe_states, label_states, regime_timeline
from market_forecast.regime.features import breadth_from_panel, build_regime_features
from market_forecast.regime.models import (
    GMM,
    HMM,
    KMEANS,
    assign,
    bayesian_information_criterion,
    build_regime_model,
)
from tests.factories import synthetic_prices


@pytest.fixture(scope="module")
def regime_features() -> pd.DataFrame:
    benchmark = synthetic_prices(sessions=1800, seed=21, daily_volatility=0.009)
    volatility_index = synthetic_prices(
        sessions=1800, seed=33, initial_price=18.0, daily_volatility=0.06
    )
    breadth = pd.Series(
        np.clip(np.random.default_rng(4).normal(0.6, 0.15, 1800), 0, 1), index=benchmark.index
    )
    return build_regime_features(benchmark, volatility_index, breadth)


@pytest.fixture(scope="module")
def fitted_hmm(regime_features):
    return build_regime_model(HMM, 3, seed=17).fit(regime_features.iloc[:1000])


class TestRegimeFeatures:
    def test_has_the_expected_columns(self, regime_features):
        for prefix in ("mkt_ret_", "mkt_vol_", "mkt_drawdown_", "vix_z_"):
            assert any(c.startswith(prefix) for c in regime_features.columns)
        assert "breadth_above_sma50" in regime_features.columns

    def test_no_missing_values_survive(self, regime_features):
        assert regime_features.notna().all().all()

    def test_drawdown_is_never_positive(self, regime_features):
        column = next(c for c in regime_features.columns if c.startswith("mkt_drawdown"))
        assert (regime_features[column] <= 1e-9).all()

    def test_features_do_not_read_forward(self):
        benchmark = synthetic_prices(sessions=1200, seed=5)
        corrupted = benchmark.copy()
        corrupted.iloc[800:] *= 3.0
        a = build_regime_features(benchmark)
        b = build_regime_features(corrupted)
        shared = a.index.intersection(b.index)
        shared = shared[shared <= benchmark.index[799]]
        pd.testing.assert_frame_equal(a.loc[shared], b.loc[shared])

    def test_breadth_comes_from_the_existing_feature(self):
        index = pd.MultiIndex.from_product(
            [pd.bdate_range("2020-01-01", periods=4), ["AAA", "BBB"]], names=["date", "ticker"]
        )
        frame = pd.DataFrame(
            {"px_to_sma_50": [0.1, -0.1, 0.2, 0.3, -0.1, -0.2, 0.5, 0.5]}, index=index
        )
        assert breadth_from_panel(frame).tolist() == [0.5, 1.0, 0.0, 1.0]

    def test_breadth_requires_its_column(self):
        frame = pd.DataFrame({"other": [1.0]})
        with pytest.raises(KeyError, match="px_to_sma_50"):
            breadth_from_panel(frame)


class TestFilteredInferenceIsCausal:
    """The property the whole regime component rests on."""

    @pytest.mark.parametrize("cut", [400, 800, 1200])
    def test_filtered_row_is_unchanged_by_later_data(self, fitted_hmm, regime_features, cut):
        full = fitted_hmm.filtered_proba(regime_features)
        prefix = fitted_hmm.filtered_proba(regime_features.iloc[: cut + 1])
        np.testing.assert_allclose(full[cut], prefix[cut], atol=1e-10)

    def test_smoothed_inference_does_read_the_future(self, fitted_hmm, regime_features):
        """Negative control: this is exactly why smoothed labels cannot be used."""
        full = fitted_hmm.smoothed_proba(regime_features)
        changed = 0
        for cut in (400, 800, 1200):
            prefix = fitted_hmm.smoothed_proba(regime_features.iloc[: cut + 1])
            changed += int(not np.allclose(full[cut], prefix[cut], atol=1e-6))
        assert changed > 0, "smoothed inference should depend on later data"

    def test_filtered_probabilities_are_a_distribution(self, fitted_hmm, regime_features):
        proba = fitted_hmm.filtered_proba(regime_features)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        assert (proba >= 0).all()


class TestModelProperties:
    @pytest.mark.parametrize("kind", [HMM, GMM, KMEANS])
    def test_every_model_assigns_states(self, regime_features, kind):
        model = build_regime_model(kind, 3, seed=17).fit(regime_features.iloc[:1000])
        result = assign(model, regime_features)
        assert set(result.labels.unique()) <= {0, 1, 2}
        assert len(result.labels) == len(regime_features)

    def test_hmm_states_persist_far_longer_than_a_mixture(self, regime_features):
        """The argument for the HMM: mixtures have no notion of staying put."""

        def switches(kind: str) -> float:
            model = build_regime_model(kind, 3, seed=17).fit(regime_features.iloc[:1000])
            labels = assign(model, regime_features).labels
            return float((labels.diff() != 0).mean())

        assert switches(HMM) < switches(GMM)

    def test_hmm_transition_matrix_is_stochastic(self, fitted_hmm):
        matrix = fitted_hmm.transition_matrix()
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-9)

    def test_expected_durations_are_positive(self, fitted_hmm):
        assert (fitted_hmm.expected_durations() > 0).all()

    def test_kmeans_reports_no_likelihood(self, regime_features):
        model = build_regime_model(KMEANS, 3, seed=17).fit(regime_features.iloc[:1000])
        assert np.isnan(model.log_likelihood(regime_features))
        assert np.isnan(bayesian_information_criterion(model, regime_features))

    def test_scaler_is_fitted_on_the_training_window_only(self, regime_features):
        train = regime_features.iloc[:1000]
        model = build_regime_model(HMM, 2, seed=17).fit(train)
        np.testing.assert_allclose(model.scaler.mean_, train.mean().to_numpy(), rtol=1e-9)

    def test_rejects_an_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown regime model"):
            build_regime_model("spectral", 3)

    @pytest.mark.parametrize("kind", [HMM, GMM])
    def test_fitting_is_reproducible(self, regime_features, kind):
        train = regime_features.iloc[:1000]
        first = assign(build_regime_model(kind, 3, seed=17).fit(train), regime_features).labels
        second = assign(build_regime_model(kind, 3, seed=17).fit(train), regime_features).labels
        pd.testing.assert_series_equal(first, second)


class TestDescription:
    @pytest.fixture
    def summary(self, fitted_hmm, regime_features):
        labels = assign(fitted_hmm, regime_features).labels
        return describe_states(regime_features, labels)

    def test_one_row_per_state_and_shares_sum_to_one(self, summary):
        assert len(summary) <= 3
        assert summary["share"].sum() == pytest.approx(1.0)

    def test_labels_are_derived_not_assigned(self, summary):
        names = label_states(summary)
        assert len(names) == len(summary)
        for state, name in names.items():
            vol_column = next(c for c in summary.columns if c.startswith("mkt_vol_"))
            highest = summary[vol_column].idxmax()
            if state == highest and len(summary) > 1:
                assert "high volatility" in name

    def test_timeline_covers_every_session(self, fitted_hmm, regime_features):
        labels = assign(fitted_hmm, regime_features).labels
        timeline = regime_timeline(labels)
        assert timeline["sessions"].sum() == len(labels)
        assert (timeline["end"] >= timeline["start"]).all()

    def test_timeline_handles_an_empty_series(self):
        assert regime_timeline(pd.Series(dtype="float64")).empty
