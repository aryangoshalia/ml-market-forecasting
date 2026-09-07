"""Regime models behind one interface: hidden Markov, Gaussian mixture and k-means.

The distinction that matters is filtered versus smoothed inference. Smoothed state
probabilities condition on the whole sample, including sessions after t, so using them
to describe the regime at t is look-ahead. Only the filtered posterior P(s_t | x_1..t)
is available to someone standing at t, and that is what the forecaster and the interface
consume. Both are exposed here so the size of that difference can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from market_forecast.logging import get_logger

logger = get_logger(__name__)

HMM = "hmm"
GMM = "gmm"
KMEANS = "kmeans"
REGIME_KINDS = (HMM, GMM, KMEANS)


@runtime_checkable
class RegimeModel(Protocol):
    kind: str
    n_states: int

    def fit(self, features: pd.DataFrame) -> RegimeModel: ...

    def filtered_proba(self, features: pd.DataFrame) -> np.ndarray: ...

    def smoothed_proba(self, features: pd.DataFrame) -> np.ndarray: ...

    def log_likelihood(self, features: pd.DataFrame) -> float: ...

    def n_parameters(self) -> int: ...


class _Scaled:
    """Shared plumbing: the scaler is fitted on the training window and reused unchanged."""

    def __init__(self, n_states: int, seed: int) -> None:
        self.n_states = n_states
        self.seed = seed
        self.scaler = StandardScaler()
        self.columns_: list[str] = []

    def _fit_scale(self, features: pd.DataFrame) -> np.ndarray:
        self.columns_ = list(features.columns)
        return self.scaler.fit_transform(features.to_numpy(dtype="float64"))

    def _scale(self, features: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(features[self.columns_].to_numpy(dtype="float64"))


class HmmRegime(_Scaled):
    """Gaussian hidden Markov model. The only option that represents persistence."""

    kind = HMM

    def __init__(
        self, n_states: int = 3, seed: int = 17, covariance_type: str = "full", n_iter: int = 200
    ) -> None:
        super().__init__(n_states, seed)
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.model: Any = None

    def fit(self, features: pd.DataFrame) -> HmmRegime:
        from hmmlearn.hmm import GaussianHMM

        scaled = self._fit_scale(features)
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.seed,
            tol=1e-4,
        )
        self.model.fit(scaled)
        return self

    def _log_emissions(self, features: pd.DataFrame) -> np.ndarray:
        return self.model._compute_log_likelihood(self._scale(features))

    def filtered_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Forward recursion only, so row t uses observations up to and including t."""
        log_b = self._log_emissions(features)
        log_start = np.log(np.maximum(self.model.startprob_, 1e-300))
        log_transition = np.log(np.maximum(self.model.transmat_, 1e-300))

        n_rows = log_b.shape[0]
        filtered = np.empty((n_rows, self.n_states))
        log_alpha = log_start + log_b[0]
        filtered[0] = _normalise(log_alpha)

        for t in range(1, n_rows):
            log_alpha = logsumexp(log_alpha[:, None] + log_transition, axis=0) + log_b[t]
            filtered[t] = _normalise(log_alpha)
        return filtered

    def smoothed_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._scale(features))

    def log_likelihood(self, features: pd.DataFrame) -> float:
        return float(self.model.score(self._scale(features)))

    def n_parameters(self) -> int:
        d = len(self.columns_)
        transitions = self.n_states * (self.n_states - 1) + (self.n_states - 1)
        means = self.n_states * d
        covariances = self.n_states * d * (d + 1) // 2
        return transitions + means + covariances

    def transition_matrix(self) -> np.ndarray:
        return np.asarray(self.model.transmat_)

    def expected_durations(self) -> np.ndarray:
        """Mean sessions spent in each state before leaving it."""
        stay = np.diag(self.transition_matrix())
        return 1.0 / np.maximum(1.0 - stay, 1e-9)


class GmmRegime(_Scaled):
    """Gaussian mixture. Soft assignment with full covariance, but sessions are independent."""

    kind = GMM

    def __init__(self, n_states: int = 3, seed: int = 17, covariance_type: str = "full") -> None:
        super().__init__(n_states, seed)
        self.covariance_type = covariance_type
        self.model: Any = None

    def fit(self, features: pd.DataFrame) -> GmmRegime:
        scaled = self._fit_scale(features)
        self.model = GaussianMixture(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            random_state=self.seed,
            n_init=5,
        )
        self.model.fit(scaled)
        return self

    def filtered_proba(self, features: pd.DataFrame) -> np.ndarray:
        # Memoryless, so the posterior at t already depends only on row t.
        return self.model.predict_proba(self._scale(features))

    def smoothed_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.filtered_proba(features)

    def log_likelihood(self, features: pd.DataFrame) -> float:
        scaled = self._scale(features)
        return float(self.model.score(scaled) * len(scaled))

    def n_parameters(self) -> int:
        d = len(self.columns_)
        return (self.n_states - 1) + self.n_states * d + self.n_states * d * (d + 1) // 2


class KMeansRegime(_Scaled):
    """Hard assignment, spherical clusters. Included as the floor, not as a candidate."""

    kind = KMEANS

    def __init__(self, n_states: int = 3, seed: int = 17) -> None:
        super().__init__(n_states, seed)
        self.model: Any = None

    def fit(self, features: pd.DataFrame) -> KMeansRegime:
        scaled = self._fit_scale(features)
        self.model = KMeans(n_clusters=self.n_states, random_state=self.seed, n_init=10)
        self.model.fit(scaled)
        return self

    def filtered_proba(self, features: pd.DataFrame) -> np.ndarray:
        labels = self.model.predict(self._scale(features))
        out = np.zeros((len(labels), self.n_states))
        out[np.arange(len(labels)), labels] = 1.0
        return out

    def smoothed_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.filtered_proba(features)

    def log_likelihood(self, features: pd.DataFrame) -> float:
        """Undefined for k-means. Returning NaN keeps it out of any BIC comparison."""
        del features
        return float("nan")

    def inertia(self, features: pd.DataFrame) -> float:
        return float(-self.model.score(self._scale(features)))

    def n_parameters(self) -> int:
        return self.n_states * len(self.columns_)


def _normalise(log_alpha: np.ndarray) -> np.ndarray:
    return np.exp(log_alpha - logsumexp(log_alpha))


def build_regime_model(kind: str, n_states: int, seed: int = 17) -> RegimeModel:
    if kind == HMM:
        return HmmRegime(n_states=n_states, seed=seed)
    if kind == GMM:
        return GmmRegime(n_states=n_states, seed=seed)
    if kind == KMEANS:
        return KMeansRegime(n_states=n_states, seed=seed)
    raise ValueError(f"unknown regime model: {kind!r}; expected one of {REGIME_KINDS}")


def bayesian_information_criterion(model: RegimeModel, features: pd.DataFrame) -> float:
    n = len(features)
    return float(model.n_parameters() * np.log(n) - 2.0 * model.log_likelihood(features))


@dataclass
class RegimeAssignment:
    labels: pd.Series
    probabilities: pd.DataFrame
    kind: str
    n_states: int
    metadata: dict[str, Any] = field(default_factory=dict)


def assign(
    model: RegimeModel, features: pd.DataFrame, use_filtered: bool = True
) -> RegimeAssignment:
    proba = model.filtered_proba(features) if use_filtered else model.smoothed_proba(features)
    frame = pd.DataFrame(
        proba, index=features.index, columns=[f"state_{i}" for i in range(model.n_states)]
    )
    return RegimeAssignment(
        labels=pd.Series(proba.argmax(axis=1), index=features.index, name="regime"),
        probabilities=frame,
        kind=model.kind,
        n_states=model.n_states,
        metadata={"inference": "filtered" if use_filtered else "smoothed"},
    )
