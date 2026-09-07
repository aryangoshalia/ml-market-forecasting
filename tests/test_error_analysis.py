from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_forecast.evaluation.error_analysis import (
    confidence_deciles,
    error_breakdown,
    high_confidence_errors,
    performance_around_reversals,
    performance_by_bucket,
    performance_by_group,
    rolling_performance,
)

TICKERS = ["AAA", "BBB", "CCC"]


@pytest.fixture
def predictions():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-01", periods=300)
    index = pd.MultiIndex.from_product([dates, TICKERS], names=["date", "ticker"])
    n = len(index)
    truth = (rng.uniform(size=n) < 0.5).astype("float64")
    probability = np.clip(0.5 + 0.25 * (truth - 0.5) + rng.normal(0, 0.1, n), 0.01, 0.99)
    return pd.DataFrame({"prob": probability, "y_true": truth, "fold": 0}, index=index)


class TestErrorBreakdown:
    def test_covers_all_four_outcomes(self, predictions):
        summary = error_breakdown(predictions)
        assert set(summary.index) == {
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        }

    def test_shares_sum_to_one(self, predictions):
        assert error_breakdown(predictions)["share"].sum() == pytest.approx(1.0)

    def test_counts_sum_to_the_sample(self, predictions):
        assert error_breakdown(predictions)["count"].sum() == len(predictions)


class TestConfidenceProfile:
    def test_accuracy_rises_with_confidence_when_the_model_is_informative(self, predictions):
        deciles = confidence_deciles(predictions)
        assert deciles["accuracy"].iloc[-1] > deciles["accuracy"].iloc[0]

    def test_every_row_lands_in_a_bucket(self, predictions):
        assert confidence_deciles(predictions)["n"].sum() == len(predictions)

    def test_high_confidence_errors_are_all_wrong(self, predictions):
        wrong = high_confidence_errors(predictions, quantile=0.8)
        assert (wrong["correct"] == 0.0).all()

    def test_high_confidence_errors_are_sorted_most_confident_first(self, predictions):
        wrong = high_confidence_errors(predictions, quantile=0.8)
        assert wrong["confidence"].is_monotonic_decreasing


class TestConditionalPerformance:
    def test_buckets_partition_the_sample(self, predictions):
        conditioning = pd.Series(
            np.random.default_rng(1).normal(size=len(predictions)), index=predictions.index
        )
        table = performance_by_bucket(predictions, conditioning, bins=5)
        assert table["n"].sum() == len(predictions)
        assert len(table) == 5

    def test_bucket_means_increase(self, predictions):
        conditioning = pd.Series(
            np.arange(len(predictions), dtype="float64"), index=predictions.index
        )
        table = performance_by_bucket(predictions, conditioning, bins=4)
        assert table["mean_value"].is_monotonic_increasing

    def test_per_ticker_covers_every_ticker(self, predictions):
        table = performance_by_group(predictions, "ticker")
        assert set(table.index) == set(TICKERS)
        assert table["n"].sum() == len(predictions)

    def test_rolling_performance_is_indexed_by_date(self, predictions):
        rolling = rolling_performance(predictions, window=20)
        assert rolling.index.name == "date"
        assert rolling["n"].iloc[0] == len(TICKERS)

    def test_reversal_split_covers_every_row(self, predictions):
        trend = pd.Series(
            np.sin(np.linspace(0, 12, predictions.index.get_level_values("date").nunique())),
            index=predictions.index.get_level_values("date").unique(),
        )
        stats = performance_around_reversals(predictions, trend)
        assert stats.n_reversal_rows + stats.n_other_rows == len(predictions)
