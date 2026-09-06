"""Baselines. Every reported metric is meaningless without these next to it."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityClass(BaseEstimator, ClassifierMixin):
    """Predicts the training base rate. The floor an accuracy figure must clear."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> MajorityClass:
        values = np.asarray(target, dtype="float64")
        self.classes_ = np.array([0.0, 1.0])
        self.rate_ = float(np.nanmean(values)) if len(values) else 0.5
        del features
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        rate = np.full(len(features), self.rate_)
        return np.column_stack([1.0 - rate, rate])


class AlwaysLong(BaseEstimator, ClassifierMixin):
    """Always predicts up. Isolates how much of any result is unconditional equity drift."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> AlwaysLong:
        self.classes_ = np.array([0.0, 1.0])
        del features, target
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        rate = np.full(len(features), 1.0 - 1e-6)
        return np.column_stack([1.0 - rate, rate])


class Persistence(BaseEstimator, ClassifierMixin):
    """Predicts that the last move repeats.

    Tests whether any apparent skill is just first-order autocorrelation. The probability
    is the training base rate conditional on the sign of the most recent return, so the
    baseline is calibrated rather than a bare 0/1 guess.
    """

    def __init__(self, signal_column: str = "ret_1d") -> None:
        self.signal_column = signal_column

    def _signal(self, features: pd.DataFrame) -> np.ndarray:
        if self.signal_column not in features.columns:
            raise KeyError(f"persistence baseline needs the {self.signal_column!r} feature")
        return np.asarray(features[self.signal_column], dtype="float64")

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Persistence:
        signal = self._signal(features)
        values = np.asarray(target, dtype="float64")
        self.classes_ = np.array([0.0, 1.0])

        rising = signal > 0.0
        self.rate_up_ = _safe_mean(values[rising], fallback=0.5)
        self.rate_down_ = _safe_mean(values[~rising], fallback=0.5)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        signal = self._signal(features)
        rate = np.where(signal > 0.0, self.rate_up_, self.rate_down_)
        rate = np.where(np.isnan(signal), 0.5, rate)
        return np.column_stack([1.0 - rate, rate])


def _safe_mean(values: np.ndarray, fallback: float) -> float:
    usable = values[~np.isnan(values)]
    return float(usable.mean()) if len(usable) else fallback
