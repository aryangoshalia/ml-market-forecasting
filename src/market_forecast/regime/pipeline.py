"""Per-fold regime assignment.

The regime model is refitted inside each fold's training window and applied to that
fold's test window by forward filtering, so the regime attached to a session was
knowable at that session. Fitting once on the whole sample would put the shape of the
2020 crash into the description of 2013.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_forecast.logging import get_logger
from market_forecast.regime.models import HMM, RegimeModel, build_regime_model
from market_forecast.validation.splits import Fold

logger = get_logger(__name__)


@dataclass
class FoldRegimes:
    labels: pd.Series
    probabilities: pd.DataFrame
    state_summaries: dict[int, pd.DataFrame]
    kind: str
    n_states: int


def assign_by_fold(
    features: pd.DataFrame,
    folds: list[Fold],
    kind: str = HMM,
    n_states: int = 3,
    seed: int = 17,
) -> FoldRegimes:
    """Regime label for every test session, from a model that never saw that session."""
    label_blocks: list[pd.Series] = []
    probability_blocks: list[pd.DataFrame] = []
    summaries: dict[int, pd.DataFrame] = {}

    for fold in folds:
        dates = pd.Series(features.index, index=features.index)
        train_mask = (dates >= fold.dates[fold.fit[0]]) & (dates <= fold.dates[fold.inner[1]])
        train = features[train_mask.to_numpy()]
        test_start, test_end = fold.test_dates
        if len(train) < 250:
            logger.warning("fold %d has only %d regime rows, skipping", fold.index, len(train))
            continue

        model = build_regime_model(kind, n_states, seed=seed).fit(train)

        # Filter from the start of the series so the posterior at t reflects all history
        # available at t, then keep only this fold's test sessions.
        through_test = features[(dates <= test_end).to_numpy()]
        filtered = model.filtered_proba(through_test)
        keep = (through_test.index >= test_start) & (through_test.index <= test_end)

        block = pd.DataFrame(
            filtered[keep],
            index=through_test.index[keep],
            columns=[f"state_{i}" for i in range(n_states)],
        )
        probability_blocks.append(block)
        label_blocks.append(pd.Series(filtered[keep].argmax(axis=1), index=block.index))
        summaries[fold.index] = _state_means(model, train, n_states)

    if not label_blocks:
        raise RuntimeError("no folds produced regime assignments")

    labels = pd.concat(label_blocks).sort_index().rename("regime")
    probabilities = pd.concat(probability_blocks).sort_index()
    return FoldRegimes(
        labels=labels[~labels.index.duplicated(keep="first")],
        probabilities=probabilities[~probabilities.index.duplicated(keep="first")],
        state_summaries=summaries,
        kind=kind,
        n_states=n_states,
    )


def _state_means(model: RegimeModel, train: pd.DataFrame, n_states: int) -> pd.DataFrame:
    assignment = model.filtered_proba(train).argmax(axis=1)
    rows = []
    for state in range(n_states):
        selected = train[assignment == state]
        row: dict[str, float] = {"state": float(state), "sessions": float(len(selected))}
        row.update(
            {c: float(selected[c].mean()) if len(selected) else np.nan for c in train.columns}
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def canonical_state_order(summary: pd.DataFrame, by: str = "mkt_vol_20") -> dict[int, int]:
    """Map estimator state numbers onto a stable order, lowest value first.

    Fold-to-fold the estimator numbers states arbitrarily, so comparing regime 1 in 2013
    against regime 1 in 2020 is meaningless without this.
    """
    if by not in summary.columns:
        raise KeyError(f"cannot order states by missing column {by!r}")
    order = summary[by].sort_values().index.tolist()
    return {int(state): rank for rank, state in enumerate(order)}
