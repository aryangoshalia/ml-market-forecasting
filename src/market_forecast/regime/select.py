"""Choosing the regime model and the number of states.

Selection is not by likelihood alone. A mixture with many components will always fit
better in sample while producing states that flicker daily, which is useless as market
context. Candidates are therefore judged on out-of-sample likelihood, on whether the
implied regime durations are plausible, and on how stable the assignment is when the
model is refitted on a different window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from market_forecast.logging import get_logger
from market_forecast.regime.models import (
    HMM,
    REGIME_KINDS,
    HmmRegime,
    assign,
    bayesian_information_criterion,
    build_regime_model,
)

logger = get_logger(__name__)

MIN_PLAUSIBLE_DURATION = 10.0

# k-means defines no likelihood, so BIC and held-out likelihood are not comparable for it.
LIKELIHOOD_FREE = ("kmeans",)


@dataclass
class Candidate:
    kind: str
    n_states: int
    bic: float
    train_log_likelihood: float
    holdout_log_likelihood: float
    min_duration: float
    median_duration: float
    smallest_state_share: float
    stability: float
    switches_per_year: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def plausible(self) -> bool:
        return self.min_duration >= MIN_PLAUSIBLE_DURATION and self.smallest_state_share >= 0.05


def _durations_from_labels(labels: pd.Series) -> np.ndarray:
    values = labels.to_numpy()
    if len(values) == 0:
        return np.array([1.0])
    boundaries = np.flatnonzero(np.diff(values) != 0)
    lengths = np.diff(np.concatenate([[-1], boundaries, [len(values) - 1]]))
    return lengths.astype("float64")


def evaluate_candidate(
    features: pd.DataFrame,
    kind: str,
    n_states: int,
    holdout_fraction: float = 0.25,
    seed: int = 17,
) -> Candidate:
    split = int(len(features) * (1.0 - holdout_fraction))
    train, holdout = features.iloc[:split], features.iloc[split:]

    model = build_regime_model(kind, n_states, seed=seed).fit(train)
    labels = assign(model, features, use_filtered=True).labels

    durations = _durations_from_labels(labels)
    shares = labels.value_counts(normalize=True)
    switches = float((labels.diff() != 0).sum())
    years = max(len(labels) / 252.0, 1e-9)

    # Refit on a shifted window and measure how often the assignment survives.
    shifted = features.iloc[int(len(features) * 0.1) : split]
    rival = build_regime_model(kind, n_states, seed=seed + 1).fit(shifted)
    rival_labels = assign(rival, features, use_filtered=True).labels
    stability = _matched_agreement(labels, rival_labels, n_states)

    if kind == HMM and isinstance(model, HmmRegime):
        implied = float(np.min(model.expected_durations()))
    else:
        implied = float(np.min(durations)) if len(durations) else 0.0

    return Candidate(
        kind=kind,
        n_states=n_states,
        bic=bayesian_information_criterion(model, train),
        train_log_likelihood=model.log_likelihood(train),
        holdout_log_likelihood=model.log_likelihood(holdout),
        min_duration=implied,
        median_duration=float(np.median(durations)) if len(durations) else 0.0,
        smallest_state_share=float(shares.min()) if len(shares) else 0.0,
        stability=stability,
        switches_per_year=switches / years,
    )


def _matched_agreement(left: pd.Series, right: pd.Series, n_states: int) -> float:
    """Agreement after matching state labels, which are arbitrary integers."""
    from itertools import permutations

    a, b = left.to_numpy(), right.to_numpy()
    best = 0.0
    for order in permutations(range(n_states)):
        mapped = np.array(order)[b]
        best = max(best, float((a == mapped).mean()))
    return best


def compare(
    features: pd.DataFrame,
    kinds: tuple[str, ...] = REGIME_KINDS,
    state_range: tuple[int, ...] = (2, 3, 4, 5),
    seed: int = 17,
) -> pd.DataFrame:
    rows = []
    for kind in kinds:
        for n_states in state_range:
            try:
                candidate = evaluate_candidate(features, kind, n_states, seed=seed)
            except Exception as exc:
                logger.warning("%s with %d states failed: %s", kind, n_states, exc)
                continue
            rows.append(
                {
                    "kind": candidate.kind,
                    "n_states": candidate.n_states,
                    "bic": candidate.bic,
                    "holdout_ll": candidate.holdout_log_likelihood,
                    "min_duration": candidate.min_duration,
                    "median_duration": candidate.median_duration,
                    "smallest_share": candidate.smallest_state_share,
                    "stability": candidate.stability,
                    "switches_per_year": candidate.switches_per_year,
                    "plausible": candidate.plausible,
                }
            )
    return pd.DataFrame(rows)
