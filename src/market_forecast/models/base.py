"""Uniform model interface. Preprocessing lives inside the estimator so it can only ever
be fitted on the window the model is fitted on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

POSITIVE_CLASS = 1.0


@runtime_checkable
class Model(Protocol):
    name: str

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Model: ...

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ModelSpec:
    """Everything needed to build and identify one model."""

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    needs_scaling: bool = False
    needs_imputation: bool = True
    handles_nan: bool = False
    is_baseline: bool = False

    def signature(self) -> str:
        if not self.params:
            return self.name
        rendered = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({rendered})"


class SklearnModel:
    """Wraps a scikit-learn compatible classifier with its own preprocessing."""

    def __init__(self, spec: ModelSpec, estimator: BaseEstimator) -> None:
        self.spec = spec
        self.name = spec.name
        self.estimator = estimator
        self.pipeline: Pipeline | None = None
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray | None = None

    def _build_pipeline(self) -> Pipeline:
        steps: list[tuple[str, Any]] = []
        if self.spec.needs_imputation and not self.spec.handles_nan:
            steps.append(("impute", SimpleImputer(strategy="median")))
        if self.spec.needs_scaling:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", self.estimator))
        return Pipeline(steps)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> SklearnModel:
        self.feature_names_ = list(features.columns)
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(features, target)
        self.classes_ = getattr(self.pipeline.named_steps["model"], "classes_", None)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        aligned = features[self.feature_names_]
        probabilities = self.pipeline.predict_proba(aligned)
        return _positive_column(probabilities, self.classes_)

    def fitted_estimator(self) -> Any:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} must be fitted before inspection")
        return self.pipeline.named_steps["model"]

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Features as the final estimator sees them, needed for tree explanations."""
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} must be fitted before inspection")
        aligned = features[self.feature_names_]
        for name, step in self.pipeline.steps[:-1]:
            aligned = pd.DataFrame(
                step.transform(aligned), index=aligned.index, columns=self.feature_names_
            )
            del name
        return aligned


def _positive_column(probabilities: np.ndarray, classes: np.ndarray | None) -> np.ndarray:
    if probabilities.ndim == 1:
        return probabilities
    if classes is None:
        return probabilities[:, -1]
    matches = np.flatnonzero(np.asarray(classes) == POSITIVE_CLASS)
    column = int(matches[0]) if len(matches) else probabilities.shape[1] - 1
    return probabilities[:, column]


class ConstantProbability(BaseEstimator, ClassifierMixin):
    """Predicts the training base rate for every row."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ConstantProbability:
        values = np.asarray(target, dtype="float64")
        self.classes_ = np.array([0.0, 1.0])
        self.rate_ = float(np.nanmean(values)) if len(values) else 0.5
        del features
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        rate = np.full(len(features), self.rate_)
        return np.column_stack([1.0 - rate, rate])
