"""SHAP explanations for the tree models.

Contributions are reported in log-odds, the space the trees actually add in, so the base
value plus the contributions reconstructs the prediction exactly. Reporting them as
probability shares would not sum correctly and would hide how much of a move happens in
the flat part of the logistic curve.

An attribution says what this model did with these correlated inputs. It is not evidence
that a feature causes future returns, and where features are collinear the credit
assigned among them is arbitrary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_forecast.logging import get_logger
from market_forecast.models.base import SklearnModel

logger = get_logger(__name__)

TREE_KINDS = ("xgboost", "lightgbm", "random_forest")


@dataclass
class LocalExplanation:
    base_value: float
    contributions: pd.Series
    probability: float
    feature_values: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))

    @property
    def logodds(self) -> float:
        return float(self.base_value + self.contributions.sum())

    def top(self, n: int = 5, positive: bool = True) -> pd.Series:
        ordered = self.contributions.sort_values(ascending=not positive)
        selected = ordered[ordered > 0] if positive else ordered[ordered < 0]
        return selected.head(n)

    def as_records(self, n: int = 5) -> list[dict[str, float | str]]:
        records: list[dict[str, float | str]] = []
        for direction, positive in (("positive", True), ("negative", False)):
            for name, value in self.top(n, positive).items():
                records.append(
                    {
                        "feature": str(name),
                        "direction": direction,
                        "contribution": float(value),
                        "value": float(self.feature_values.get(name, np.nan)),
                    }
                )
        return records


class ShapExplainer:
    """Wraps shap.TreeExplainer around a fitted pipeline model."""

    def __init__(self, model: SklearnModel) -> None:
        import shap

        if model.pipeline is None:
            raise RuntimeError("model must be fitted before it can be explained")
        self.model = model
        self.feature_names = list(model.feature_names_)
        estimator = model.fitted_estimator()
        self._explainer = shap.TreeExplainer(estimator)
        self._is_sklearn_forest = type(estimator).__name__ == "RandomForestClassifier"

    def _values(self, features: pd.DataFrame) -> tuple[np.ndarray, float]:
        transformed = self.model.transform_features(features)
        raw = self._explainer.shap_values(transformed, check_additivity=False)
        expected = self._explainer.expected_value

        values = np.asarray(raw)
        if values.ndim == 3:
            # (rows, features, classes) for multiclass output; keep the positive class
            values = values[:, :, -1]
        if isinstance(expected, (list, np.ndarray)):
            expected = float(np.asarray(expected).ravel()[-1])
        return values, float(expected)

    def explain_local(self, features: pd.DataFrame) -> list[LocalExplanation]:
        values, base = self._values(features)
        probabilities = self.model.predict_proba(features)
        out = []
        for position in range(len(features)):
            out.append(
                LocalExplanation(
                    base_value=base,
                    contributions=pd.Series(values[position], index=self.feature_names),
                    probability=float(probabilities[position]),
                    feature_values=features.iloc[position][self.feature_names],
                )
            )
        return out

    def explain_global(self, features: pd.DataFrame) -> pd.DataFrame:
        values, _ = self._values(features)
        magnitude = pd.Series(np.abs(values).mean(axis=0), index=self.feature_names)
        signed = pd.Series(values.mean(axis=0), index=self.feature_names)
        frame = pd.DataFrame({"mean_abs_shap": magnitude, "mean_shap": signed})
        frame["share"] = frame["mean_abs_shap"] / frame["mean_abs_shap"].sum()
        return frame.sort_values("mean_abs_shap", ascending=False)

    def is_probability_space(self) -> bool:
        """sklearn forests attribute in probability space; boosted trees in log-odds."""
        return self._is_sklearn_forest


def permutation_importance(
    model: SklearnModel,
    features: pd.DataFrame,
    target: pd.Series,
    repeats: int = 5,
    seed: int = 17,
) -> pd.DataFrame:
    """Drop in AUC when one column is shuffled. An independent read on what matters."""
    from sklearn.metrics import roc_auc_score

    truth = target.to_numpy(dtype="float64")
    if len(np.unique(truth)) < 2:
        raise ValueError("permutation importance needs both classes present")

    baseline = roc_auc_score(truth, model.predict_proba(features))
    rng = np.random.default_rng(seed)

    rows = []
    for name in model.feature_names_:
        drops = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled[name] = rng.permutation(shuffled[name].to_numpy())
            drops.append(baseline - roc_auc_score(truth, model.predict_proba(shuffled)))
        rows.append(
            {"feature": name, "auc_drop": float(np.mean(drops)), "std": float(np.std(drops))}
        )
    return pd.DataFrame(rows).set_index("feature").sort_values("auc_drop", ascending=False)


def importance_stability(per_fold: dict[int, pd.Series], top_n: int = 10) -> pd.DataFrame:
    """How often each feature appears in the top n across folds.

    A ranking that reshuffles every six months is itself a finding: it says the
    relationship the model is using does not hold still.
    """
    counts: dict[str, int] = {}
    ranks: dict[str, list[float]] = {}
    for series in per_fold.values():
        ordered = series.sort_values(ascending=False)
        for rank, name in enumerate(ordered.index, start=1):
            ranks.setdefault(str(name), []).append(float(rank))
            if rank <= top_n:
                counts[str(name)] = counts.get(str(name), 0) + 1

    rows = [
        {
            "feature": name,
            "top_n_appearances": counts.get(name, 0),
            "share_of_folds": counts.get(name, 0) / max(len(per_fold), 1),
            "mean_rank": float(np.mean(values)),
            "rank_std": float(np.std(values)),
        }
        for name, values in ranks.items()
    ]
    return (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values(["share_of_folds", "mean_rank"], ascending=[False, True])
    )
