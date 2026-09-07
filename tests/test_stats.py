from __future__ import annotations

import numpy as np
import pytest

from market_forecast.evaluation.stats import (
    benjamini_hochberg,
    block_bootstrap_ci,
    paired_fold_test,
    permutation_null_auc,
)


class TestPermutationNull:
    def test_noise_is_not_distinguishable_from_the_null(self):
        rng = np.random.default_rng(0)
        n = 4000
        target = (rng.uniform(size=n) < 0.52).astype("float64")
        result = permutation_null_auc(target, rng.uniform(size=n), draws=200, seed=1)
        assert result.p_value > 0.05
        assert result.mean == pytest.approx(0.5, abs=0.02)

    def test_a_real_edge_is_detected(self):
        rng = np.random.default_rng(1)
        n = 4000
        signal = rng.normal(size=n)
        target = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.3 * signal))).astype("float64")
        result = permutation_null_auc(target, signal, draws=200, seed=1)
        assert result.p_value < 0.01
        assert result.observed > result.quantile_95

    def test_the_null_widens_as_the_sample_shrinks(self):
        rng = np.random.default_rng(2)
        wide = permutation_null_auc(
            (rng.uniform(size=400) < 0.5).astype("float64"), rng.uniform(size=400), draws=200
        )
        narrow = permutation_null_auc(
            (rng.uniform(size=8000) < 0.5).astype("float64"), rng.uniform(size=8000), draws=200
        )
        assert wide.std > narrow.std

    def test_single_class_returns_nothing_rather_than_a_number(self):
        assert np.isnan(permutation_null_auc(np.ones(100), np.linspace(0, 1, 100)).observed)


class TestPairedFoldTest:
    def test_detects_a_consistent_improvement(self):
        scores = np.array([0.53, 0.52, 0.55, 0.51, 0.54, 0.53, 0.52, 0.56, 0.53, 0.52])
        result = paired_fold_test(scores, np.full(10, 0.50), "model", "baseline")
        assert result.wins == 10
        assert result.p_value < 0.05
        assert result.mean_difference > 0

    def test_does_not_flag_noise(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(0.5, 0.02, 20), rng.normal(0.5, 0.02, 20)
        assert paired_fold_test(a, b).p_value > 0.05

    def test_identical_scores_are_not_significant(self):
        scores = np.full(10, 0.52)
        assert paired_fold_test(scores, scores).p_value == 1.0

    def test_too_few_folds_yields_no_verdict(self):
        assert np.isnan(paired_fold_test(np.array([0.5, 0.6]), np.array([0.5, 0.5])).p_value)

    def test_ignores_missing_folds(self):
        a = np.array([0.53, np.nan, 0.55, 0.51, 0.54])
        b = np.array([0.50, 0.50, 0.50, 0.50, 0.50])
        assert paired_fold_test(a, b).n_folds == 4


class TestBlockBootstrap:
    def test_interval_contains_the_mean(self):
        rng = np.random.default_rng(4)
        values = rng.normal(0.52, 0.01, 30)
        low, high = block_bootstrap_ci(values, seed=1)
        assert low < values.mean() < high

    def test_is_wider_than_an_independent_interval(self):
        """Blocks preserve the correlation between adjacent folds, so the interval is honest."""
        rng = np.random.default_rng(5)
        noise = rng.normal(0, 0.01, 60)
        correlated = np.convolve(noise, np.ones(5) / 5, mode="same") + 0.52
        blocked = block_bootstrap_ci(correlated, block_size=8, seed=2)
        independent = block_bootstrap_ci(correlated, block_size=1, seed=2)
        assert (blocked[1] - blocked[0]) > (independent[1] - independent[0])

    def test_too_few_values_yields_no_interval(self):
        assert np.isnan(block_bootstrap_ci(np.array([0.5]))[0])


class TestMultipleComparisons:
    def test_adjusted_values_are_never_smaller(self):
        raw = np.array([0.001, 0.02, 0.04, 0.3, 0.8])
        assert (benjamini_hochberg(raw) >= raw - 1e-12).all()

    def test_ordering_is_preserved(self):
        raw = np.array([0.001, 0.02, 0.04, 0.3, 0.8])
        assert list(np.argsort(benjamini_hochberg(raw))) == list(np.argsort(raw))

    def test_stays_within_range(self):
        rng = np.random.default_rng(6)
        adjusted = benjamini_hochberg(rng.uniform(size=50))
        assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))

    def test_passes_through_missing_values(self):
        adjusted = benjamini_hochberg(np.array([0.01, np.nan, 0.5]))
        assert np.isnan(adjusted[1])
        assert np.isfinite(adjusted[[0, 2]]).all()


class TestPooledPermutationNull:
    """A predictor that only tracks its own fold's base rate must not look skilful."""

    @pytest.fixture
    def pooled(self):
        rng = np.random.default_rng(11)
        folds, dates, y, base = [], [], [], []
        for fold in range(20):
            rate = 0.45 + 0.01 * fold  # base rate drifts across folds
            for day in range(60):
                labels = (rng.uniform(size=10) < rate).astype("float64")
                y.extend(labels)
                folds.extend([fold] * 10)
                dates.extend([fold * 60 + day] * 10)
                base.extend([rate] * 10)
        return (np.array(y), np.array(base), np.array(dates), np.array(folds))

    def test_naive_null_is_fooled_by_base_rate_drift(self, pooled):
        from market_forecast.evaluation.stats import permutation_null_auc

        y, base, _, _ = pooled
        result = permutation_null_auc(y, base, draws=200, seed=1)
        # the constant-per-fold predictor scores above 0.5 purely from drift
        assert result.observed > 0.52
        assert result.p_value < 0.05

    def test_pooled_null_is_not_fooled(self, pooled):
        from market_forecast.evaluation.stats import permutation_null_pooled

        y, base, dates, folds = pooled
        result = permutation_null_pooled(y, base, dates, folds, draws=200, seed=1)
        assert result.p_value > 0.05, "base-rate drift alone should not read as skill"

    def test_pooled_null_still_detects_real_signal(self, pooled):
        from market_forecast.evaluation.stats import permutation_null_pooled

        y, base, dates, folds = pooled
        rng = np.random.default_rng(3)
        informative = base + 0.20 * (y - 0.5) + rng.normal(0, 0.02, len(y))
        result = permutation_null_pooled(y, informative, dates, folds, draws=200, seed=1)
        assert result.p_value < 0.01
        assert result.observed > result.quantile_95
