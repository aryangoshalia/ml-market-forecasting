from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_forecast.targets import (
    NEXT_OPEN_TO_CLOSE,
    QuantileTernaryLabeller,
    VolScaledTernaryLabeller,
    binary_direction,
    build_targets,
    causal_volatility,
    excess_return,
    forward_return,
    solve_kappa,
)


class TestForwardReturn:
    @pytest.mark.parametrize("horizon", [1, 3, 5, 10])
    def test_matches_the_definition(self, prices, horizon):
        computed = forward_return(prices, horizon)
        position = 500
        manual = prices["close"].iloc[position + horizon] / prices["close"].iloc[position] - 1.0
        assert np.isclose(computed.iloc[position], manual)

    @pytest.mark.parametrize("horizon", [1, 5])
    def test_last_h_rows_have_no_label(self, prices, horizon):
        returns = forward_return(prices, horizon)
        assert returns.iloc[-horizon:].isna().all()
        assert returns.iloc[:-horizon].notna().all()

    def test_open_anchored_variant_starts_from_the_next_open(self, prices):
        computed = forward_return(prices, 5, NEXT_OPEN_TO_CLOSE)
        position = 300
        manual = prices["close"].iloc[position + 5] / prices["open"].iloc[position + 1] - 1.0
        assert np.isclose(computed.iloc[position], manual)

    def test_rejects_a_non_positive_horizon(self, prices):
        with pytest.raises(ValueError):
            forward_return(prices, 0)

    def test_rejects_an_unknown_basis(self, prices):
        with pytest.raises(ValueError):
            forward_return(prices, 1, "some_other_basis")

    def test_is_invariant_to_the_price_scale(self, prices):
        scaled = prices.copy()
        for column in ("open", "high", "low", "close", "adj_close"):
            scaled[column] *= 13.0
        pd.testing.assert_series_equal(
            forward_return(prices, 5), forward_return(scaled, 5), rtol=1e-12
        )


class TestBinaryDirection:
    def test_encodes_the_sign_of_the_forward_return(self, prices):
        returns = forward_return(prices, 5)
        labels = binary_direction(returns)
        both = pd.concat([returns, labels], axis=1).dropna()
        assert (both.iloc[:, 1] == (both.iloc[:, 0] > 0).astype(float)).all()

    def test_a_zero_return_counts_as_negative(self):
        assert binary_direction(pd.Series([0.0, 1e-12, -1e-12])).tolist() == [0.0, 1.0, 0.0]

    def test_missing_returns_stay_missing(self):
        assert binary_direction(pd.Series([np.nan, 0.1])).isna().tolist() == [True, False]


class TestExcessReturn:
    def test_subtracts_the_benchmark_over_the_same_window(self, prices, benchmark):
        stock = forward_return(prices, 5)
        market = forward_return(benchmark, 5)
        excess = excess_return(stock, market)
        assert np.isclose(excess.iloc[100], stock.iloc[100] - market.iloc[100])

    def test_a_benchmark_against_itself_is_exactly_zero(self, prices):
        stock = forward_return(prices, 5)
        assert excess_return(stock, stock).dropna().abs().max() < 1e-15


class TestVolatilityScaledBand:
    def test_kappa_hits_the_requested_neutral_share(self, prices):
        returns = forward_return(prices, 5)
        sigma = causal_volatility(prices)
        kappa = solve_kappa(returns, sigma, 5, target_fraction=1 / 3)
        # measured over the same sample solve_kappa uses: rows where both are defined
        both = pd.concat([returns.rename("r"), sigma.rename("s")], axis=1).dropna()
        share = (both["r"].abs() <= kappa * both["s"] * np.sqrt(5)).mean()
        assert share == pytest.approx(1 / 3, abs=0.005)

    def test_neutral_share_is_measured_only_where_sigma_is_defined(self, prices):
        # sigma has a warm-up; rows inside it have no band and must not be labelled
        returns, sigma = forward_return(prices, 5), causal_volatility(prices)
        labeller = VolScaledTernaryLabeller(horizon=5).fit(returns, sigma)
        labels = labeller.transform(returns, sigma)
        assert labels[sigma.isna()].isna().all()

    def test_a_wider_band_never_shrinks_the_neutral_class(self, prices):
        returns = forward_return(prices, 5).dropna()
        sigma = causal_volatility(prices).reindex(returns.index)
        shares = [
            float((returns.abs() <= k * sigma * np.sqrt(5)).mean()) for k in (0.1, 0.3, 0.6, 1.0)
        ]
        assert shares == sorted(shares)

    def test_labels_take_only_the_three_declared_values(self, prices):
        returns, sigma = forward_return(prices, 5), causal_volatility(prices)
        labeller = VolScaledTernaryLabeller(horizon=5).fit(returns, sigma)
        assert set(labeller.transform(returns, sigma).dropna().unique()) <= {-1.0, 0.0, 1.0}

    def test_the_threshold_is_known_at_prediction_time(self, prices):
        # sigma_t must not move when data after t changes, or the band itself would leak
        cut = 900
        corrupted = prices.copy()
        corrupted.iloc[cut + 1 :, :] *= 4.0
        assert np.isclose(
            causal_volatility(prices).iloc[cut], causal_volatility(corrupted).iloc[cut]
        )

    def test_must_be_fitted_before_use(self, prices):
        with pytest.raises(RuntimeError):
            VolScaledTernaryLabeller(horizon=5).transform(
                forward_return(prices, 5), causal_volatility(prices)
            )

    def test_a_band_fitted_on_train_is_applied_unchanged_to_test(self, prices):
        returns, sigma = forward_return(prices, 5), causal_volatility(prices)
        train, test = slice(None, 900), slice(900, None)
        labeller = VolScaledTernaryLabeller(horizon=5).fit(returns[train], sigma[train])
        kappa = labeller.kappa_
        labeller.transform(returns[test], sigma[test])
        assert labeller.kappa_ == kappa


class TestQuantileBand:
    def test_bounds_come_from_the_training_sample_only(self, prices):
        returns = forward_return(prices, 5)
        labeller = QuantileTernaryLabeller(horizon=5).fit(returns.iloc[:900])
        lower, upper = labeller.bounds_
        assert lower == pytest.approx(returns.iloc[:900].quantile(1 / 3))
        assert upper == pytest.approx(returns.iloc[:900].quantile(2 / 3))

    def test_labels_take_only_the_three_declared_values(self, prices):
        returns = forward_return(prices, 5)
        labeller = QuantileTernaryLabeller(horizon=5).fit(returns)
        assert set(labeller.transform(returns).dropna().unique()) <= {-1.0, 0.0, 1.0}


class TestTargetSet:
    def test_builds_every_declared_target(self, prices, benchmark, targets_config):
        targets = build_targets("TEST", prices, targets_config.horizons, benchmark_prices=benchmark)
        for horizon in targets_config.horizons:
            for name in ("fwd_ret", "direction", "direction_open", "excess_direction"):
                assert f"{name}_{horizon}d" in targets.frame.columns

    def test_excess_targets_are_absent_without_a_benchmark(self, prices, targets_config):
        targets = build_targets("TEST", prices, targets_config.horizons)
        assert not [c for c in targets.frame.columns if c.startswith("excess")]

    def test_labels_stop_h_sessions_before_the_data_ends(self, prices, benchmark, targets_config):
        targets = build_targets("TEST", prices, targets_config.horizons, benchmark_prices=benchmark)
        for horizon in targets_config.horizons:
            column = targets.frame[f"direction_{horizon}d"]
            assert column.last_valid_index() == prices.index[-horizon - 1]
