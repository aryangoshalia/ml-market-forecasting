"""Classification and probability-quality metrics, always reported against the base rate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5


@dataclass
class ClassificationMetrics:
    n: int
    base_rate: float
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    ece: float
    mce: float
    positive_rate: float
    accuracy_over_base_rate: float
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"n={self.n} base={self.base_rate:.4f} acc={self.accuracy:.4f} "
            f"(+{self.accuracy_over_base_rate:+.4f}) auc={self.roc_auc:.4f} "
            f"pr_auc={self.pr_auc:.4f} brier={self.brier:.4f} ece={self.ece:.4f}"
        )


def expected_calibration_error(
    target: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> tuple[float, float]:
    """Equal-frequency binned calibration error.

    Quantile bins rather than equal-width, because predicted probabilities here cluster
    tightly around the base rate and equal-width bins would leave most of them empty.
    """
    if len(probabilities) == 0:
        return float("nan"), float("nan")

    edges = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return float(abs(probabilities.mean() - target.mean())), float(
            abs(probabilities.mean() - target.mean())
        )

    assignments = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, len(edges) - 2)
    total, weighted, worst = len(probabilities), 0.0, 0.0
    for index in range(len(edges) - 1):
        selected = assignments == index
        count = int(selected.sum())
        if count == 0:
            continue
        gap = abs(float(probabilities[selected].mean()) - float(target[selected].mean()))
        weighted += (count / total) * gap
        worst = max(worst, gap)
    return float(weighted), float(worst)


def evaluate(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    bins: int = 10,
) -> ClassificationMetrics:
    y = np.asarray(target, dtype="float64")
    p = np.asarray(probabilities, dtype="float64")

    clean = np.isfinite(y) & np.isfinite(p)
    y, p = y[clean], p[clean]
    if len(y) == 0:
        raise ValueError("no finite observations to evaluate")

    predictions = (p >= threshold).astype("float64")
    base_rate = float(y.mean())
    both_classes = len(np.unique(y)) > 1

    accuracy = float(accuracy_score(y, predictions))
    ece, mce = expected_calibration_error(y, p, bins=bins)

    matrix = confusion_matrix(y, predictions, labels=[0.0, 1.0])
    true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()

    return ClassificationMetrics(
        n=int(len(y)),
        base_rate=base_rate,
        threshold=float(threshold),
        accuracy=accuracy,
        precision=float(precision_score(y, predictions, zero_division=0)),
        recall=float(recall_score(y, predictions, zero_division=0)),
        f1=float(f1_score(y, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)) if both_classes else float("nan"),
        pr_auc=float(average_precision_score(y, p)) if both_classes else float("nan"),
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0.0, 1.0])),
        ece=ece,
        mce=mce,
        positive_rate=float(predictions.mean()),
        accuracy_over_base_rate=accuracy - max(base_rate, 1.0 - base_rate),
        true_negatives=int(true_negatives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_positives=int(true_positives),
    )


def reliability_curve(
    target: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> dict[str, list[float]]:
    """Points for a reliability diagram, using equal-frequency bins."""
    edges = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return {"predicted": [], "observed": [], "count": []}

    assignments = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, len(edges) - 2)
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[float] = []
    for index in range(len(edges) - 1):
        selected = assignments == index
        if not selected.any():
            continue
        predicted.append(float(probabilities[selected].mean()))
        observed.append(float(target[selected].mean()))
        counts.append(float(selected.sum()))
    return {"predicted": predicted, "observed": observed, "count": counts}
