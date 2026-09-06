from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_forecast.features import indicators as ind
from market_forecast.features.pipeline import FeaturePipeline
from market_forecast.features.registry import FeatureRegistry, FeatureSpec


class TestIndicators:
    def test_log_return_matches_the_definition(self, prices):
        computed = ind.log_return(prices["close"], 5)
        manual = np.log(prices["close"].iloc[100] / prices["close"].iloc[95])
        assert np.isclose(computed.iloc[100], manual)

    def test_sma_leaves_the_warm_up_undefined(self, prices):
        sma = ind.sma(prices["close"], 20)
        assert sma.iloc[:19].isna().all()
        assert np.isclose(sma.iloc[19], prices["close"].iloc[:20].mean())

    def test_rsi_is_bounded(self, prices):
        rsi = ind.rsi(prices["close"], 14).dropna()
        assert rsi.between(0.0, 100.0).all()

    def test_rsi_saturates_on_a_monotone_series(self):
        rising = pd.Series(np.arange(1, 100, dtype=float))
        assert ind.rsi(rising, 14).dropna().iloc[-1] == pytest.approx(100.0)

    def test_efficiency_ratio_is_one_for_a_straight_line(self):
        straight = pd.Series(np.arange(1, 100, dtype=float))
        assert ind.efficiency_ratio(straight, 20).dropna().iloc[-1] == pytest.approx(1.0)

    def test_efficiency_ratio_is_zero_when_price_returns_to_its_start(self):
        oscillating = pd.Series([100.0, 101.0] * 40)
        assert ind.efficiency_ratio(oscillating, 20).dropna().iloc[-1] == pytest.approx(0.0)

    def test_price_position_is_bounded(self, prices):
        position = ind.price_position(prices["close"], 252).dropna()
        assert position.between(0.0, 1.0).all()

    def test_atr_is_never_negative(self, prices):
        atr = ind.atr(prices["high"], prices["low"], prices["close"], 14).dropna()
        assert (atr >= 0.0).all()

    def test_true_range_covers_an_overnight_gap(self):
        frame = pd.DataFrame(
            {"high": [10.0, 25.0], "low": [9.0, 24.0], "close": [9.5, 24.5]},
        )
        tr = ind.true_range(frame["high"], frame["low"], frame["close"])
        # the gap from 9.5 to 24.0 dominates the 1.0 intraday range
        assert tr.iloc[1] == pytest.approx(15.5)

    def test_rolling_beta_recovers_a_known_slope(self):
        rng = np.random.default_rng(3)
        market = pd.Series(rng.normal(0.0, 0.01, 500))
        asset = 1.8 * market + rng.normal(0.0, 1e-6, 500)
        assert ind.rolling_beta(asset, market, 250).dropna().iloc[-1] == pytest.approx(
            1.8, abs=1e-3
        )

    def test_parkinson_matches_its_closed_form(self):
        # A constant log range c gives sqrt(c^2 / (4 ln 2)) * sqrt(252) exactly.
        c = 0.02
        high = pd.Series(np.full(100, 100.0 * np.exp(c)))
        low = pd.Series(np.full(100, 100.0))
        expected = np.sqrt(c**2 / (4.0 * np.log(2.0))) * np.sqrt(252.0)
        assert ind.parkinson_volatility(high, low, 20).dropna().iloc[-1] == pytest.approx(expected)

    def test_parkinson_rises_with_the_bar_range(self, prices):
        narrow = ind.parkinson_volatility(prices["high"], prices["low"], 60).dropna()
        mid = (prices["high"] + prices["low"]) / 2.0
        wider = ind.parkinson_volatility(
            mid + (prices["high"] - mid) * 3.0, mid - (mid - prices["low"]) * 3.0, 60
        ).dropna()
        assert (wider > narrow).all()

    def test_every_indicator_ignores_the_future(self, prices):
        cut = 800
        corrupted = prices.copy()
        corrupted.iloc[cut + 1 :, :] *= 5.0
        for name, call in {
            "rsi": lambda f: ind.rsi(f["close"], 14),
            "macd": lambda f: ind.macd(f["close"])[2],
            "atr": lambda f: ind.atr(f["high"], f["low"], f["close"], 14),
            "efficiency_ratio": lambda f: ind.efficiency_ratio(f["close"], 20),
            "ewma_volatility": lambda f: ind.ewma_volatility(ind.log_return(f["close"])),
            "price_position": lambda f: ind.price_position(f["close"], 252),
        }.items():
            assert np.isclose(call(prices).iloc[cut], call(corrupted).iloc[cut]), name


class TestRegistry:
    def test_rejects_duplicate_names(self):
        registry = FeatureRegistry()
        spec = FeatureSpec("a", "g", "d", "r", "f", 1)
        registry.add(spec)
        with pytest.raises(ValueError):
            registry.add(FeatureSpec("a", "g", "d2", "r2", "f2", 2))

    def test_max_lookback_is_the_longest_window(self):
        registry = FeatureRegistry()
        registry.extend(
            [FeatureSpec("a", "g", "d", "r", "f", 5), FeatureSpec("b", "g", "d", "r", "f", 252)]
        )
        assert registry.max_lookback() == 252


class TestPipeline:
    @pytest.fixture
    def matrix(self, prices, context, feature_config):
        return FeaturePipeline(feature_config).build("TEST", prices, context)

    def test_every_feature_is_documented(self, matrix):
        assert set(matrix.feature_names) == set(matrix.registry.names)
        for spec in (matrix.registry[n] for n in matrix.registry.names):
            assert spec.description and spec.rationale and spec.formula
            assert spec.lookback >= 1

    def test_warm_up_rows_are_dropped(self, matrix, prices):
        assert len(matrix.frame) == len(prices) - matrix.registry.max_lookback()

    def test_no_feature_is_entirely_missing(self, matrix):
        assert not matrix.frame.isna().all().any()

    def test_no_infinities_survive(self, matrix):
        assert np.isfinite(matrix.frame.to_numpy()[~np.isnan(matrix.frame.to_numpy())]).all()

    def test_column_order_is_deterministic(self, prices, context, feature_config):
        pipeline = FeaturePipeline(feature_config)
        first = pipeline.build("TEST", prices, context)
        second = pipeline.build("TEST", prices, context)
        assert first.feature_names == second.feature_names

    def test_market_features_require_a_context(self, prices, feature_config):
        with pytest.raises(ValueError, match="context"):
            FeaturePipeline(feature_config).build("TEST", prices, None)

    def test_bounded_features_stay_in_range(self, matrix):
        frame = matrix.frame
        assert frame["rsi_14"].dropna().between(0.0, 1.0).all()
        assert frame["price_position_252"].dropna().between(0.0, 1.0).all()
        assert frame["efficiency_ratio_20"].dropna().between(0.0, 1.0).all()
        assert frame["up_day_ratio_20"].dropna().between(0.0, 1.0).all()
        assert (frame["dist_high_252"].dropna() <= 1e-9).all()
        assert (frame["dist_low_252"].dropna() >= -1e-9).all()

    def test_a_shorter_history_yields_the_same_columns(self, prices, context, feature_config):
        pipeline = FeaturePipeline(feature_config)
        full = pipeline.build("TEST", prices, context)
        short = pipeline.build("TEST", prices.iloc[-600:], context)
        assert full.feature_names == short.feature_names
