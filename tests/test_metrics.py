from __future__ import annotations

import numpy as np
import pytest

from market_forecast.evaluation.metrics import (
    ClassificationMetrics,
    evaluate,
    expected_calibration_error,
    reliability_curve,
)


@pytest.fixture
def calibrated():
    rng = np.random.default_rng(11)
    n = 20000
    probabilities = rng.beta(2, 2, n)
    target = (rng.uniform(size=n) < probabilities).astype("float64")
    return target, probabilities


class TestEvaluate:
    def test_reports_the_base_rate(self, calibrated):
        target, probabilities = calibrated
        assert evaluate(target, probabilities).base_rate == pytest.approx(target.mean())

    def test_perfect_predictions_score_perfectly(self):
        target = np.array([0.0, 0.0, 1.0, 1.0])
        metrics = evaluate(target, np.array([0.01, 0.02, 0.98, 0.99]))
        assert metrics.accuracy == 1.0
        assert metrics.roc_auc == 1.0
        assert metrics.f1 == 1.0

    def test_inverted_predictions_score_zero_auc(self):
        target = np.array([0.0, 0.0, 1.0, 1.0])
        assert evaluate(target, np.array([0.99, 0.98, 0.02, 0.01])).roc_auc == 0.0

    def test_random_predictions_score_half_auc(self, calibrated):
        target, _ = calibrated
        rng = np.random.default_rng(2)
        assert evaluate(target, rng.uniform(size=len(target))).roc_auc == pytest.approx(
            0.5, abs=0.02
        )

    def test_accuracy_over_base_rate_uses_the_larger_class(self):
        target = np.concatenate([np.ones(80), np.zeros(20)])
        metrics = evaluate(target, np.full(100, 0.9))
        assert metrics.accuracy == pytest.approx(0.8)
        assert metrics.accuracy_over_base_rate == pytest.approx(0.0)

    def test_confusion_counts_sum_to_the_sample(self, calibrated):
        target, probabilities = calibrated
        metrics = evaluate(target, probabilities)
        total = (
            metrics.true_negatives
            + metrics.false_positives
            + metrics.false_negatives
            + metrics.true_positives
        )
        assert total == metrics.n

    def test_threshold_changes_the_positive_rate(self, calibrated):
        target, probabilities = calibrated
        low = evaluate(target, probabilities, threshold=0.3)
        high = evaluate(target, probabilities, threshold=0.7)
        assert low.positive_rate > high.positive_rate

    def test_threshold_does_not_change_ranking_metrics(self, calibrated):
        target, probabilities = calibrated
        low = evaluate(target, probabilities, threshold=0.3)
        high = evaluate(target, probabilities, threshold=0.7)
        assert low.roc_auc == pytest.approx(high.roc_auc)
        assert low.brier == pytest.approx(high.brier)

    def test_drops_non_finite_rows(self):
        target = np.array([0.0, 1.0, np.nan, 1.0])
        probabilities = np.array([0.2, 0.8, 0.5, np.nan])
        assert evaluate(target, probabilities).n == 2

    def test_raises_when_nothing_is_left(self):
        with pytest.raises(ValueError, match="no finite"):
            evaluate(np.array([np.nan]), np.array([np.nan]))

    def test_single_class_yields_no_auc(self):
        metrics = evaluate(np.ones(50), np.linspace(0.1, 0.9, 50))
        assert np.isnan(metrics.roc_auc)
        assert np.isfinite(metrics.brier)

    def test_serialises_to_a_flat_dictionary(self, calibrated):
        target, probabilities = calibrated
        payload = evaluate(target, probabilities).to_dict()
        assert isinstance(payload, dict)
        assert set(ClassificationMetrics.__dataclass_fields__) <= set(payload)


class TestCalibrationError:
    def test_is_near_zero_for_calibrated_probabilities(self, calibrated):
        target, probabilities = calibrated
        ece, _ = expected_calibration_error(target, probabilities)
        assert ece < 0.02

    def test_is_large_for_overconfident_probabilities(self, calibrated):
        target, probabilities = calibrated
        stretched = np.clip((probabilities - 0.5) * 2.5 + 0.5, 0.001, 0.999)
        ece, _ = expected_calibration_error(target, stretched)
        assert ece > 0.10

    def test_max_error_is_at_least_the_mean_error(self, calibrated):
        target, probabilities = calibrated
        ece, mce = expected_calibration_error(target, probabilities)
        assert mce >= ece

    def test_a_constant_prediction_measures_its_own_gap(self):
        target = np.concatenate([np.ones(70), np.zeros(30)])
        ece, _ = expected_calibration_error(target, np.full(100, 0.5))
        assert ece == pytest.approx(0.2, abs=1e-9)


class TestReliabilityCurve:
    def test_tracks_the_diagonal_when_calibrated(self, calibrated):
        target, probabilities = calibrated
        curve = reliability_curve(target, probabilities, bins=10)
        gaps = np.abs(np.array(curve["predicted"]) - np.array(curve["observed"]))
        assert gaps.max() < 0.05

    def test_counts_cover_every_observation(self, calibrated):
        target, probabilities = calibrated
        curve = reliability_curve(target, probabilities, bins=10)
        assert sum(curve["count"]) == len(target)

    def test_handles_a_constant_prediction(self):
        curve = reliability_curve(np.ones(10), np.full(10, 0.5))
        assert curve["predicted"] == [] or len(curve["predicted"]) == 1
