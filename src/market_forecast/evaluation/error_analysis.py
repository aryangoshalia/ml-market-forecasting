"""Where the model fails, rather than how well it does on average.

Aggregate metrics hide the two things worth knowing: whether errors concentrate in
particular conditions, and whether the model is wrong precisely when it is most
confident. Both are asked directly here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_THRESHOLD = 0.5


def _prepare(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = predictions.copy()
    frame["predicted"] = (frame["prob"] >= threshold).astype("float64")
    frame["correct"] = (frame["predicted"] == frame["y_true"]).astype("float64")
    frame["confidence"] = (frame["prob"] - 0.5).abs()
    frame["error_type"] = np.where(
        frame["correct"] == 1.0,
        np.where(frame["y_true"] == 1.0, "true_positive", "true_negative"),
        np.where(frame["predicted"] == 1.0, "false_positive", "false_negative"),
    )
    return frame


def error_breakdown(
    predictions: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    frame = _prepare(predictions, threshold)
    summary = frame.groupby("error_type").agg(
        count=("correct", "size"),
        mean_probability=("prob", "mean"),
        mean_confidence=("confidence", "mean"),
    )
    summary["share"] = summary["count"] / summary["count"].sum()
    return summary.sort_index()


def confidence_deciles(
    predictions: pd.DataFrame, bins: int = 10, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Accuracy by how confident the model was.

    A model that knows what it does not know shows accuracy rising with confidence. A
    flat or inverted profile means the probabilities carry no reliable ordering.
    """
    frame = _prepare(predictions, threshold)
    edges = np.unique(np.quantile(frame["confidence"], np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return pd.DataFrame()
    frame["bucket"] = pd.cut(frame["confidence"], edges, include_lowest=True, labels=False)
    return frame.groupby("bucket").agg(
        n=("correct", "size"),
        mean_confidence=("confidence", "mean"),
        accuracy=("correct", "mean"),
        base_rate=("y_true", "mean"),
    )


def high_confidence_errors(
    predictions: pd.DataFrame, quantile: float = 0.9, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """The rows a user would have trusted most and the model got wrong."""
    frame = _prepare(predictions, threshold)
    cutoff = frame["confidence"].quantile(quantile)
    confident = frame[frame["confidence"] >= cutoff]
    wrong = confident[confident["correct"] == 0.0]
    return wrong.sort_values("confidence", ascending=False)


def performance_by_bucket(
    predictions: pd.DataFrame,
    conditioning: pd.Series,
    bins: int = 5,
    labels: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Accuracy and AUC within quantile buckets of some conditioning variable."""
    frame = _prepare(predictions, threshold)
    aligned = conditioning.reindex(frame.index)
    usable = aligned.notna()
    frame, aligned = frame[usable], aligned[usable]

    try:
        frame["bucket"] = pd.qcut(aligned, bins, labels=labels, duplicates="drop")
    except ValueError:
        frame["bucket"] = pd.cut(aligned, bins, labels=labels)

    rows = []
    for bucket, block in frame.groupby("bucket", observed=True):
        rows.append(
            {
                "bucket": str(bucket),
                "n": int(len(block)),
                "accuracy": float(block["correct"].mean()),
                "base_rate": float(block["y_true"].mean()),
                "roc_auc": (
                    float(roc_auc_score(block["y_true"], block["prob"]))
                    if block["y_true"].nunique() > 1
                    else np.nan
                ),
                "mean_value": float(aligned[block.index].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("bucket")


def performance_by_group(
    predictions: pd.DataFrame, level: str = "ticker", threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    frame = _prepare(predictions, threshold)
    rows = []
    for name, block in frame.groupby(level=level):
        rows.append(
            {
                level: name,
                "n": int(len(block)),
                "accuracy": float(block["correct"].mean()),
                "base_rate": float(block["y_true"].mean()),
                "roc_auc": (
                    float(roc_auc_score(block["y_true"], block["prob"]))
                    if block["y_true"].nunique() > 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index(level).sort_values("roc_auc", ascending=False)


def rolling_performance(
    predictions: pd.DataFrame, window: int = 60, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Accuracy over time, so a model that decays is visible rather than averaged away."""
    frame = _prepare(predictions, threshold)
    daily = frame.groupby(level="date").agg(accuracy=("correct", "mean"), n=("correct", "size"))
    daily["rolling_accuracy"] = daily["accuracy"].rolling(window, min_periods=window // 2).mean()
    return daily


@dataclass
class ReversalStats:
    around_reversals: float
    elsewhere: float
    n_reversal_rows: int
    n_other_rows: int

    @property
    def gap(self) -> float:
        return self.around_reversals - self.elsewhere


def performance_around_reversals(
    predictions: pd.DataFrame,
    trend: pd.Series,
    window: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
) -> ReversalStats:
    """Accuracy on sessions near a change in the sign of a trend measure.

    Turning points are where a momentum-driven model should struggle most, so this
    isolates them instead of letting them average into a long trending sample.
    """
    frame = _prepare(predictions, threshold)
    sign = np.sign(trend.dropna())
    flips = sign.index[sign.diff().fillna(0.0) != 0.0]

    dates = frame.index.get_level_values("date")
    near = np.zeros(len(frame), dtype=bool)
    for flip in flips:
        low = flip - pd.Timedelta(days=window * 2)
        high = flip + pd.Timedelta(days=window * 2)
        near |= (dates >= low) & (dates <= high)

    return ReversalStats(
        around_reversals=float(frame.loc[near, "correct"].mean()) if near.any() else np.nan,
        elsewhere=float(frame.loc[~near, "correct"].mean()) if (~near).any() else np.nan,
        n_reversal_rows=int(near.sum()),
        n_other_rows=int((~near).sum()),
    )
