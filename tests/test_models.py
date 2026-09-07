from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_forecast.models.base import ModelSpec, SklearnModel
from market_forecast.models.baselines import AlwaysLong, MajorityClass, Persistence
from market_forecast.models.calibration import (
    ISOTONIC,
    NONE,
    PLATT,
    ProbabilityCalibrator,
    select_threshold,
)
from market_forecast.models.registry import build_model, default_specs

SEED = 17


@pytest.fixture
def dataset():
    rng = np.random.default_rng(SEED)
    n = 3000
    features = pd.DataFrame(rng.normal(size=(n, 10)), columns=[f"f{i}" for i in range(10)])
    features["ret_1d"] = rng.normal(size=n)
    logit = 0.6 * features["f0"] - 0.4 * features["f1"]
    target = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype("float64"))
    return features, target


class TestEveryModelBuilds:
    @pytest.mark.parametrize("spec", default_specs(), ids=lambda s: s.name)
    def test_fits_and_predicts_probabilities(self, spec, dataset):
        features, target = dataset
        model = build_model(spec, seed=SEED).fit(features.iloc[:2000], target.iloc[:2000])
        probabilities = model.predict_proba(features.iloc[2000:])

        assert probabilities.shape == (1000,)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        assert np.isfinite(probabilities).all()

    @pytest.mark.parametrize("spec", default_specs(), ids=lambda s: s.name)
    def test_is_reproducible(self, spec, dataset):
        features, target = dataset
        first = build_model(spec, seed=SEED).fit(features, target).predict_proba(features)
        second = build_model(spec, seed=SEED).fit(features, target).predict_proba(features)
        np.testing.assert_allclose(first, second)

    def test_predicting_before_fitting_raises(self, dataset):
        features, _ = dataset
        model = build_model(default_specs()[3], seed=SEED)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(features)

    def test_rejects_an_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown model kind"):
            build_model(ModelSpec("mystery", "mystery"), seed=SEED)

    def test_feature_order_does_not_change_predictions(self, dataset):
        features, target = dataset
        model = build_model(default_specs()[3], seed=SEED).fit(features, target)
        shuffled = features[list(reversed(features.columns))]
        np.testing.assert_allclose(model.predict_proba(features), model.predict_proba(shuffled))


class TestPreprocessingIsFittedInsideTheModel:
    def test_scaler_uses_training_statistics_only(self, dataset):
        features, target = dataset
        spec = ModelSpec("logistic", "logistic", needs_scaling=True)
        model = build_model(spec, seed=SEED).fit(features.iloc[:2000], target.iloc[:2000])
        scaler = model.pipeline.named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, features.iloc[:2000].mean().to_numpy(), rtol=1e-9)

    def test_predictions_do_not_change_when_later_rows_change(self, dataset):
        """Rescoring one row must not depend on what else is in the batch."""
        features, target = dataset
        model = build_model(default_specs()[3], seed=SEED).fit(
            features.iloc[:2000], target.iloc[:2000]
        )
        batch = model.predict_proba(features.iloc[2000:])
        single = model.predict_proba(features.iloc[[2000]])
        assert np.isclose(batch[0], single[0])

    def test_missing_values_are_handled(self, dataset):
        features, target = dataset
        holey = features.copy()
        holey.iloc[::7, 0] = np.nan
        for spec in default_specs():
            model = build_model(spec, seed=SEED).fit(holey.iloc[:2000], target.iloc[:2000])
            assert np.isfinite(model.predict_proba(holey.iloc[2000:])).all(), spec.name


class TestBaselines:
    def test_majority_predicts_the_training_base_rate(self, dataset):
        features, target = dataset
        model = MajorityClass().fit(features, target)
        probabilities = model.predict_proba(features)[:, 1]
        assert np.allclose(probabilities, target.mean())

    def test_majority_ignores_the_features_entirely(self, dataset):
        features, target = dataset
        model = MajorityClass().fit(features, target)
        assert len(np.unique(model.predict_proba(features)[:, 1])) == 1

    def test_always_long_predicts_up(self, dataset):
        features, target = dataset
        probabilities = AlwaysLong().fit(features, target).predict_proba(features)[:, 1]
        assert (probabilities > 0.99).all()

    def test_persistence_conditions_on_the_last_return(self, dataset):
        features, target = dataset
        model = Persistence().fit(features, target)
        probabilities = model.predict_proba(features)[:, 1]
        assert len(np.unique(probabilities)) == 2
        rising = features["ret_1d"] > 0
        assert np.isclose(probabilities[rising.to_numpy()][0], target[rising].mean())

    def test_persistence_needs_its_signal_column(self, dataset):
        features, target = dataset
        with pytest.raises(KeyError, match="ret_1d"):
            Persistence().fit(features.drop(columns=["ret_1d"]), target)

    def test_baselines_score_no_better_than_chance_on_noise(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(0)
        features = pd.DataFrame({"ret_1d": rng.normal(size=4000)})
        target = pd.Series(rng.integers(0, 2, 4000).astype("float64"))
        for estimator in (MajorityClass(), AlwaysLong()):
            spec = ModelSpec("b", "b", needs_imputation=False)
            model = SklearnModel(spec, estimator).fit(features, target)
            assert roc_auc_score(target, model.predict_proba(features)) == pytest.approx(0.5)


class TestCalibration:
    @pytest.fixture
    def overconfident(self):
        rng = np.random.default_rng(3)
        n = 8000
        true_probability = rng.beta(2, 2, n)
        target = (rng.uniform(size=n) < true_probability).astype("float64")
        stretched = np.clip((true_probability - 0.5) * 2.2 + 0.5, 1e-3, 1 - 1e-3)
        return target, stretched

    @pytest.mark.parametrize("method", [ISOTONIC, PLATT])
    def test_calibration_reduces_the_calibration_error(self, overconfident, method):
        from market_forecast.evaluation.metrics import expected_calibration_error

        target, probabilities = overconfident
        calibrator = ProbabilityCalibrator(method).fit(probabilities[:4000], target[:4000])
        adjusted = calibrator.transform(probabilities[4000:])

        before, _ = expected_calibration_error(target[4000:], probabilities[4000:])
        after, _ = expected_calibration_error(target[4000:], adjusted)
        assert after < before

    @pytest.mark.parametrize("method", [ISOTONIC, PLATT])
    def test_output_stays_a_probability(self, overconfident, method):
        target, probabilities = overconfident
        calibrator = ProbabilityCalibrator(method).fit(probabilities, target)
        adjusted = calibrator.transform(probabilities)
        assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))

    def test_calibration_preserves_ranking(self, overconfident):
        """Both methods are monotone, so AUC must not move."""
        from sklearn.metrics import roc_auc_score

        target, probabilities = overconfident
        calibrator = ProbabilityCalibrator(PLATT).fit(probabilities, target)
        adjusted = calibrator.transform(probabilities)
        assert roc_auc_score(target, adjusted) == pytest.approx(
            roc_auc_score(target, probabilities), abs=1e-6
        )

    def test_none_is_a_pass_through(self, overconfident):
        target, probabilities = overconfident
        calibrator = ProbabilityCalibrator(NONE).fit(probabilities, target)
        np.testing.assert_allclose(calibrator.transform(probabilities), probabilities)

    def test_must_be_fitted_before_use(self):
        with pytest.raises(RuntimeError, match="fitted"):
            ProbabilityCalibrator(ISOTONIC).transform(np.array([0.5]))

    def test_rejects_an_unknown_method(self, overconfident):
        target, probabilities = overconfident
        with pytest.raises(ValueError, match="unknown calibration"):
            ProbabilityCalibrator("bayesian").fit(probabilities, target)

    def test_survives_a_single_class_window(self):
        probabilities = np.linspace(0.1, 0.9, 100)
        calibrator = ProbabilityCalibrator(ISOTONIC).fit(probabilities, np.ones(100))
        assert np.isfinite(calibrator.transform(probabilities)).all()


class TestThresholdSelection:
    def test_finds_a_better_threshold_than_the_default_when_classes_are_skewed(self):
        rng = np.random.default_rng(5)
        n = 5000
        target = (rng.uniform(size=n) < 0.2).astype("float64")
        probabilities = np.clip(0.2 + 0.35 * (target - 0.5) + rng.normal(0, 0.12, n), 0.01, 0.99)

        from sklearn.metrics import f1_score

        choice = select_threshold(target, probabilities, "f1")
        at_half = f1_score(target, (probabilities >= 0.5).astype(float), zero_division=0)
        assert choice.score >= at_half

    def test_threshold_stays_in_range(self):
        rng = np.random.default_rng(6)
        target = rng.integers(0, 2, 1000).astype("float64")
        probabilities = rng.uniform(size=1000)
        assert 0.0 < select_threshold(target, probabilities).threshold < 1.0

    def test_handles_a_single_class(self):
        choice = select_threshold(np.ones(100), np.linspace(0.1, 0.9, 100))
        assert choice.threshold == 0.5

    def test_rejects_an_unknown_metric(self):
        rng = np.random.default_rng(7)
        with pytest.raises(ValueError, match="threshold metric"):
            select_threshold(rng.integers(0, 2, 100).astype(float), rng.uniform(size=100), "kappa")

    def test_isotonic_never_returns_a_degenerate_probability(self):
        """A calibrated 0 that resolves to 1 has unbounded loss, so the ends are barred."""
        rng = np.random.default_rng(9)
        n = 4000
        probabilities = rng.uniform(0.2, 0.8, n)
        # a perfectly separable tail is what drives isotonic to a pure bin
        target = (probabilities > 0.5).astype("float64")
        calibrator = ProbabilityCalibrator(ISOTONIC).fit(probabilities, target)
        adjusted = calibrator.transform(probabilities)
        assert adjusted.min() > 0.0
        assert adjusted.max() < 1.0
        assert calibrator.floor_ > 0.0

    def test_the_bound_tightens_with_more_calibration_data(self):
        rng = np.random.default_rng(10)
        small = rng.uniform(size=200)
        large = rng.uniform(size=20000)
        a = ProbabilityCalibrator(ISOTONIC).fit(small, (small > 0.5).astype("float64"))
        b = ProbabilityCalibrator(ISOTONIC).fit(large, (large > 0.5).astype("float64"))
        assert a.floor_ > b.floor_

    def test_clipping_does_not_change_ranking(self):
        rng = np.random.default_rng(11)
        n = 3000
        probabilities = rng.uniform(size=n)
        target = (rng.uniform(size=n) < probabilities).astype("float64")
        calibrator = ProbabilityCalibrator(ISOTONIC).fit(probabilities, target)
        adjusted = calibrator.transform(probabilities)
        from sklearn.metrics import roc_auc_score

        assert roc_auc_score(target, adjusted) == pytest.approx(
            roc_auc_score(target, calibrator._model.predict(probabilities)), abs=1e-9
        )

    def test_f1_degenerates_on_a_weak_classifier(self):
        """Negative control for the default: F1 collapses to always predicting positive."""
        rng = np.random.default_rng(21)
        n = 20000
        target = (rng.uniform(size=n) < 0.5).astype("float64")
        # ranking barely better than chance, which is the regime this project operates in
        probabilities = np.clip(0.5 + 0.01 * (target - 0.5) + rng.normal(0, 0.05, n), 0.01, 0.99)

        by_f1 = select_threshold(target, probabilities, "f1")
        by_balanced = select_threshold(target, probabilities, "balanced_accuracy")

        assert (probabilities >= by_f1.threshold).mean() > 0.95
        assert 0.2 < (probabilities >= by_balanced.threshold).mean() < 0.8

    def test_the_default_metric_is_balanced_accuracy(self):
        rng = np.random.default_rng(22)
        target = (rng.uniform(size=5000) < 0.5).astype("float64")
        probabilities = rng.uniform(0.3, 0.7, 5000)
        assert select_threshold(target, probabilities).metric == "balanced_accuracy"

    def test_a_constant_predictor_gets_the_neutral_threshold(self):
        """Baselines tie at every threshold, so searching would return an arbitrary edge."""
        rng = np.random.default_rng(23)
        target = (rng.uniform(size=2000) < 0.52).astype("float64")
        for constant in (0.52, 1.0 - 1e-6, 0.0):
            choice = select_threshold(target, np.full(2000, constant))
            assert choice.threshold == 0.5

    def test_a_varying_predictor_still_gets_a_searched_threshold(self):
        rng = np.random.default_rng(24)
        n = 5000
        target = (rng.uniform(size=n) < 0.5).astype("float64")
        probabilities = np.clip(0.5 + 0.05 * (target - 0.5) + rng.normal(0, 0.05, n), 0.01, 0.99)
        choice = select_threshold(target, probabilities)
        assert 0.2 < choice.threshold < 0.8
