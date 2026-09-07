"""Probability calibration and decision-threshold selection.

Both are fitted on the inner validation window and applied unchanged to the test window.
Fitting either on the test window would be choosing the answer after seeing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from market_forecast.logging import get_logger

ISOTONIC = "isotonic"
PLATT = "platt"
NONE = "none"
CALIBRATION_METHODS = (NONE, ISOTONIC, PLATT)

logger = get_logger(__name__)

_EPSILON = 1e-6


@dataclass
class ProbabilityCalibrator:
    """Isotonic regression maps a pure bin to exactly 0 or 1, which claims certainty no
    finite sample supports and makes log loss unbounded. Outputs are therefore bounded
    away from the ends by Laplace's rule of succession on the calibration sample."""

    method: str = ISOTONIC
    fitted_: bool = False
    floor_: float = _EPSILON

    def fit(self, probabilities: np.ndarray, target: np.ndarray) -> ProbabilityCalibrator:
        if self.method not in CALIBRATION_METHODS:
            raise ValueError(f"unknown calibration method: {self.method!r}")

        clean = np.isfinite(probabilities) & np.isfinite(target)
        p, y = probabilities[clean], target[clean]

        self.floor_ = min(1.0 / (len(y) + 2.0), 0.01) if len(y) else _EPSILON

        if self.method == NONE or len(np.unique(y)) < 2:
            self.method = NONE if self.method == NONE else self.method
            self._model = None
            self.fitted_ = True
            return self

        if self.method == ISOTONIC:
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(p, y)
        else:
            self._model = LogisticRegression(C=1e6, solver="lbfgs")
            self._model.fit(_logit(p).reshape(-1, 1), y)

        self.fitted_ = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("calibrator must be fitted before use")
        if self._model is None:
            return probabilities
        if self.method == ISOTONIC:
            adjusted = self._model.predict(probabilities)
        else:
            adjusted = self._model.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]
        return np.clip(adjusted, self.floor_, 1.0 - self.floor_)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class ThresholdChoice:
    threshold: float
    metric: str
    score: float


def select_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "balanced_accuracy",
    grid: np.ndarray | None = None,
) -> ThresholdChoice:
    """Pick the decision threshold that maximises ``metric`` on the supplied window.

    The default is balanced accuracy rather than F1. F1 rewards recall, so against a
    classifier whose ranking is barely better than chance its maximum sits at the lowest
    threshold on the grid: predict positive for everything, take recall of one, and
    accept precision near the base rate. That is the metric behaving as defined, and it
    is the wrong metric for choosing a decision boundary at this signal level.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    clean = np.isfinite(probabilities) & np.isfinite(target)
    p, y = probabilities[clean], target[clean]
    candidates = grid if grid is not None else np.linspace(0.05, 0.95, 91)

    if len(np.unique(y)) < 2:
        return ThresholdChoice(threshold=0.5, metric=metric, score=float("nan"))

    # A constant predictor scores identically at every threshold below its own value, so
    # searching returns an arbitrary tie rather than a choice. The baselines are exactly
    # this, and warning about them would bury the cases that matter.
    if len(np.unique(p)) < 2:
        return ThresholdChoice(threshold=0.5, metric=metric, score=float("nan"))

    scorers = {
        "f1": lambda yy, pred: f1_score(yy, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
        "accuracy": lambda yy, pred: float((yy == pred).mean()),
    }
    if metric not in scorers:
        raise ValueError(f"unknown threshold metric: {metric!r}")
    score_fn = scorers[metric]

    best, best_score = 0.5, -np.inf
    for candidate in candidates:
        score = float(score_fn(y, (p >= candidate).astype(float)))
        if score > best_score:
            best, best_score = float(candidate), score

    if best <= candidates[0] or best >= candidates[-1]:
        logger.warning(
            "threshold for %s settled on the edge of the search grid at %.2f, "
            "which usually means the metric is degenerate for this classifier",
            metric,
            best,
        )
    return ThresholdChoice(threshold=best, metric=metric, score=best_score)
