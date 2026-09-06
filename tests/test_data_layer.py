from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_forecast.data.base import (
    InvalidTickerError,
    coerce_schema,
    is_index_symbol,
    normalise_ticker,
)
from market_forecast.data.cache import ParquetCache
from market_forecast.data.loader import adjust_for_corporate_actions
from market_forecast.data.sessions import missing_sessions, trading_sessions, unexpected_sessions
from market_forecast.data.validate import validate_ohlcv


class TestTickerValidation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("aapl", "AAPL"),
            (" msft ", "MSFT"),
            ("^vix", "^VIX"),
            ("brk-b", "BRK-B"),
            ("bf.b", "BF.B"),
        ],
    )
    def test_accepts_well_formed_symbols(self, raw, expected):
        assert normalise_ticker(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "../../etc/passwd",
            "AAPL/../SPY",
            "AAPL;rm -rf /",
            "$(whoami)",
            "AAPL\x00",
            "",
            "   ",
            "A" * 20,
            "aapl aapl",
            "<script>",
        ],
    )
    def test_rejects_anything_else(self, raw):
        with pytest.raises(InvalidTickerError):
            normalise_ticker(raw)

    def test_rejects_non_strings(self):
        with pytest.raises(InvalidTickerError):
            normalise_ticker(None)

    def test_index_symbols_are_flagged(self):
        assert is_index_symbol("^VIX")
        assert not is_index_symbol("AAPL")


class TestSchemaCoercion:
    def test_sorts_deduplicates_and_orders_columns(self):
        index = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-03"])
        raw = pd.DataFrame(
            {
                "Open": [1.0, 2.0, 9.0],
                "High": [1.0, 2.0, 9.0],
                "Low": [1.0, 2.0, 9.0],
                "Close": [1.0, 2.0, 9.0],
                "Adj Close": [1.0, 2.0, 9.0],
                "Volume": [10.0, 20.0, 90.0],
            },
            index=index,
        )
        out = coerce_schema(raw, "TEST")
        assert out.index.is_monotonic_increasing
        assert not out.index.has_duplicates
        assert list(out.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
        # the later of two rows for the same session wins
        assert out.loc["2024-01-03", "close"] == 9.0

    def test_strips_timezone(self):
        index = pd.date_range("2024-01-02", periods=3, tz="America/New_York")
        raw = pd.DataFrame(
            {c: [1.0, 2.0, 3.0] for c in ("open", "high", "low", "close", "adj_close", "volume")},
            index=index,
        )
        assert coerce_schema(raw, "TEST").index.tz is None


class TestCache:
    def test_round_trip_and_incremental_merge(self, tmp_path):
        cache = ParquetCache(tmp_path, "test")
        index = pd.bdate_range("2024-01-02", periods=5, name="date")
        frame = pd.DataFrame(
            {
                c: np.arange(5, dtype=float)
                for c in ("open", "high", "low", "close", "adj_close", "volume")
            },
            index=index,
        )
        cache.write("AAPL", frame)
        assert len(cache.read("AAPL")) == 5

        fresh = frame.copy()
        fresh.index = pd.bdate_range("2024-01-05", periods=5, name="date")
        fresh["close"] = 99.0
        merged = cache.merge("AAPL", fresh)

        assert len(merged) == 8
        # overlapping sessions take the freshly fetched value
        assert merged.loc["2024-01-08", "close"] == 99.0

    def test_refuses_paths_outside_its_directory(self, tmp_path):
        cache = ParquetCache(tmp_path, "test")
        with pytest.raises(InvalidTickerError):
            cache.path_for("../escape")

    def test_index_symbols_get_a_safe_filename(self, tmp_path):
        cache = ParquetCache(tmp_path, "test")
        assert "^" not in cache.path_for("^VIX").name

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert ParquetCache(tmp_path, "test").read("AAPL").empty

    def test_corrupt_file_is_discarded_not_raised(self, tmp_path):
        cache = ParquetCache(tmp_path, "test")
        cache.path_for("AAPL").write_bytes(b"not a parquet file")
        assert cache.read("AAPL").empty


class TestSessions:
    def test_detects_a_missing_session(self):
        sessions = trading_sessions(date(2024, 3, 1), date(2024, 3, 15))
        assert len(missing_sessions(sessions.delete(3))) == 1

    def test_detects_a_row_on_an_exchange_holiday(self):
        sessions = trading_sessions(date(2024, 6, 28), date(2024, 7, 8))
        with_holiday = sessions.append(pd.DatetimeIndex([pd.Timestamp("2024-07-04")])).sort_values()
        assert pd.Timestamp("2024-07-04") in unexpected_sessions(with_holiday)

    def test_clean_index_has_no_findings(self):
        sessions = trading_sessions(date(2024, 1, 2), date(2024, 6, 28))
        assert len(missing_sessions(sessions)) == 0
        assert len(unexpected_sessions(sessions)) == 0


class TestValidation:
    def test_clean_series_passes(self, prices, data_config):
        report = validate_ohlcv(prices, "TEST", data_config.validation, check_staleness=False)
        assert report.ok, report.errors

    def test_rejects_a_short_series(self, prices, data_config):
        report = validate_ohlcv(
            prices.iloc[:50], "TEST", data_config.validation, check_staleness=False
        )
        assert not report.ok

    def test_rejects_non_positive_prices(self, prices, data_config):
        broken = prices.copy()
        broken.iloc[10, broken.columns.get_loc("close")] = -1.0
        report = validate_ohlcv(broken, "TEST", data_config.validation, check_staleness=False)
        assert any("non-positive" in e for e in report.errors)

    def test_rejects_inverted_bars(self, prices, data_config):
        broken = prices.copy()
        broken.iloc[10, broken.columns.get_loc("high")] = 0.001
        report = validate_ohlcv(broken, "TEST", data_config.validation, check_staleness=False)
        assert any("high below low" in e for e in report.errors)

    def test_rejects_an_unsorted_index(self, prices, data_config):
        report = validate_ohlcv(
            prices.iloc[::-1], "TEST", data_config.validation, check_staleness=False
        )
        assert any("sorted" in e for e in report.errors)

    def test_index_symbols_are_exempt_from_the_volume_rule(self, prices, data_config):
        no_volume = prices.copy()
        no_volume["volume"] = 0.0
        equity = validate_ohlcv(no_volume, "AAPL", data_config.validation, check_staleness=False)
        index = validate_ohlcv(no_volume, "^VIX", data_config.validation, check_staleness=False)
        assert any("zero volume" in w for w in equity.warnings)
        assert not any("zero volume" in w for w in index.warnings)


class TestCorporateActionAdjustment:
    def test_removes_the_price_discontinuity_at_a_split(self):
        index = pd.bdate_range("2024-01-02", periods=10, name="date")
        close = np.array([100.0] * 5 + [10.0] * 5)
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                # the vendor restates history, so pre-split adjusted closes are already divided
                "adj_close": np.array([10.0] * 5 + [10.0] * 5),
                "volume": np.full(10, 1e6),
            },
            index=index,
        )
        adjusted = adjust_for_corporate_actions(frame)
        returns = adjusted["close"].pct_change().abs()
        assert returns.max() < 1e-9

    def test_keeps_the_bar_internally_consistent(self, prices):
        raw = prices.copy()
        raw["adj_close"] = raw["close"] * 0.7
        adjusted = adjust_for_corporate_actions(raw)
        assert (adjusted["high"] >= adjusted[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (adjusted["low"] <= adjusted[["open", "close"]].min(axis=1) + 1e-9).all()

    def test_leaves_volume_untouched(self, prices):
        raw = prices.copy()
        raw["adj_close"] = raw["close"] * 0.7
        assert (adjust_for_corporate_actions(raw)["volume"] == raw["volume"]).all()
