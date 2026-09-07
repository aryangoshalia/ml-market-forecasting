"""Backtest tests. The point is friction, so most of these check that costs bite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_forecast.backtest.simulate import (
    break_even_cost_bps,
    buy_and_hold,
    cost_sweep,
    simulate,
)

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def book():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=400)
    index = pd.MultiIndex.from_product([dates, TICKERS], names=["date", "ticker"])
    returns = pd.Series(rng.normal(0.0004, 0.012, len(index)), index=index, name="ret")
    # a weak but genuine edge, so the strategy has something to trade
    probabilities = 0.5 + 0.4 * np.sign(returns.to_numpy()) * rng.uniform(0, 0.2, len(index))
    predictions = pd.DataFrame({"prob": probabilities}, index=index)
    return predictions, returns


class TestSimulate:
    def test_produces_one_return_per_session(self, book):
        predictions, returns = book
        result = simulate(predictions, returns)
        assert result.n_sessions == predictions.index.get_level_values("date").nunique()

    def test_equity_starts_near_one(self, book):
        predictions, returns = book
        equity = simulate(predictions, returns).equity
        assert 0.5 < equity.iloc[0] < 1.5

    def test_costs_can_only_reduce_returns(self, book):
        predictions, returns = book
        free = simulate(predictions, returns, cost_bps=0.0)
        charged = simulate(predictions, returns, cost_bps=10.0)
        assert charged.annualised_return < free.annualised_return
        assert charged.sharpe < free.sharpe

    def test_costs_do_not_change_turnover_or_positions(self, book):
        predictions, returns = book
        free = simulate(predictions, returns, cost_bps=0.0)
        charged = simulate(predictions, returns, cost_bps=20.0)
        assert free.turnover == pytest.approx(charged.turnover)
        assert free.mean_positions == pytest.approx(charged.mean_positions)

    def test_a_higher_threshold_holds_fewer_names(self, book):
        predictions, returns = book
        loose = simulate(predictions, returns, threshold=0.45)
        tight = simulate(predictions, returns, threshold=0.60)
        assert tight.mean_positions < loose.mean_positions

    def test_a_signal_that_never_fires_produces_nothing(self, book):
        predictions, returns = book
        result = simulate(predictions, returns, threshold=1.01)
        assert result.mean_positions == 0
        assert result.annualised_return == pytest.approx(0.0, abs=1e-12)

    def test_drawdown_is_never_positive(self, book):
        predictions, returns = book
        assert simulate(predictions, returns).max_drawdown <= 0.0

    def test_raises_when_nothing_overlaps(self, book):
        predictions, _ = book
        empty = pd.Series(dtype="float64", index=predictions.index[:0])
        with pytest.raises(ValueError, match="no overlapping"):
            simulate(predictions, empty)


class TestCostSweep:
    def test_returns_fall_monotonically_with_cost(self, book):
        predictions, returns = book
        sweep = cost_sweep(predictions, returns, costs=(0.0, 5.0, 10.0, 20.0))
        assert sweep["annualised_return"].is_monotonic_decreasing

    def test_covers_every_requested_cost(self, book):
        predictions, returns = book
        sweep = cost_sweep(predictions, returns, costs=(0.0, 7.5))
        assert list(sweep.index) == [0.0, 7.5]


class TestBreakEven:
    def test_sits_where_returns_cross_zero(self, book):
        predictions, returns = book
        break_even = break_even_cost_bps(predictions, returns)
        assert break_even > 0
        below = simulate(predictions, returns, cost_bps=break_even * 0.5)
        above = simulate(predictions, returns, cost_bps=break_even * 1.5)
        assert below.annualised_return > 0 > above.annualised_return

    def test_a_worthless_signal_breaks_even_at_zero(self):
        rng = np.random.default_rng(9)
        dates = pd.bdate_range("2020-01-01", periods=300)
        index = pd.MultiIndex.from_product([dates, TICKERS], names=["date", "ticker"])
        returns = pd.Series(rng.normal(-0.001, 0.01, len(index)), index=index)
        predictions = pd.DataFrame({"prob": rng.uniform(size=len(index))}, index=index)
        assert break_even_cost_bps(predictions, returns) == 0.0


class TestBuyAndHold:
    def test_has_no_turnover(self, book):
        _, returns = book
        assert buy_and_hold(returns).turnover == 0.0

    def test_matches_the_equal_weighted_average(self, book):
        _, returns = book
        result = buy_and_hold(returns)
        expected = returns.unstack("ticker").mean(axis=1)
        pd.testing.assert_series_equal(result.daily_returns, expected.dropna(), check_names=False)
