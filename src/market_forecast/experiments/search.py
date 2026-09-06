"""Random hyperparameter search, confined to the period before the first test window.

The chosen settings are frozen and applied to every fold, so the search must not see any
session that a fold is later scored on. Iterating over the first few walk-forward folds
would violate that: fold one's inner window sits after fold zero's test window, so
settings picked there would be applied backwards in time. Instead the search builds its
own nested validation windows entirely inside the initial training period.

Searching independently within every fold would also be defensible, and roughly
twenty-five times more expensive for a decision that barely moves at this signal level.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from market_forecast.config import AppConfig
from market_forecast.dataset import Panel
from market_forecast.logging import get_logger
from market_forecast.models.base import ModelSpec
from market_forecast.models.registry import build_model
from market_forecast.validation.splits import WalkForwardSplitter

logger = get_logger(__name__)

SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "logistic": {"C": [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "elastic_net_logistic": {
        "C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "random_forest": {
        "n_estimators": [300, 500],
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [20, 50, 100, 200],
        "max_features": ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "n_estimators": [200, 400, 700],
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0, 20.0],
        "min_child_weight": [1, 10, 50],
    },
    "lightgbm": {
        "n_estimators": [200, 400, 700],
        "num_leaves": [7, 15, 31],
        "learning_rate": [0.01, 0.03, 0.05],
        "colsample_bytree": [0.5, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0, 20.0],
        "min_child_samples": [50, 100, 300],
    },
}


@dataclass
class SearchTrial:
    model: str
    params: dict[str, Any]
    mean_auc: float
    std_auc: float
    fold_scores: list[float] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class SearchResult:
    model: str
    best: SearchTrial
    trials: list[SearchTrial]
    folds_used: int
    cutoff: str

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"model": t.model, "mean_auc": t.mean_auc, "std_auc": t.std_auc, **t.params}
                for t in sorted(self.trials, key=lambda t: -t.mean_auc)
            ]
        )


def sample_params(space: dict[str, list[Any]], rng: np.random.Generator) -> dict[str, Any]:
    return {name: values[int(rng.integers(len(values)))] for name, values in space.items()}


def search_model(
    panel: Panel,
    config: AppConfig,
    spec: ModelSpec,
    target: str,
    horizon: int,
    groups: list[str],
    n_trials: int = 25,
    n_folds: int = 3,
    seed: int = 17,
) -> SearchResult:
    space = SEARCH_SPACES.get(spec.kind)
    if not space:
        raise ValueError(f"no search space defined for {spec.kind!r}")

    rows = panel.frame[panel.frame["group"].isin(groups)]
    label = f"{target}_{horizon}d"
    rows = rows[rows[label].notna()]

    dates = pd.Series(rows.index.get_level_values("date"), index=rows.index)
    sessions = pd.DatetimeIndex(dates.unique()).sort_values()

    windows, boundaries = _pre_test_windows(rows, dates, sessions, config, horizon, n_folds)
    cutoff = str(boundaries[-1].date())

    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    trials: list[SearchTrial] = []

    for _ in range(n_trials):
        params = sample_params(space, rng)
        key = str(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)

        started = time.perf_counter()
        scores: list[float] = []
        for fit_rows, inner_rows in windows:
            candidate = ModelSpec(
                name=spec.name,
                kind=spec.kind,
                params={**spec.params, **params},
                needs_scaling=spec.needs_scaling,
                needs_imputation=spec.needs_imputation,
                handles_nan=spec.handles_nan,
            )
            model = build_model(candidate, seed=seed)
            model.fit(fit_rows[panel.feature_names], fit_rows[label])
            probabilities = model.predict_proba(inner_rows[panel.feature_names])
            truth = inner_rows[label].to_numpy(dtype="float64")
            scores.append(
                float(roc_auc_score(truth, probabilities)) if len(np.unique(truth)) > 1 else np.nan
            )

        trials.append(
            SearchTrial(
                model=spec.name,
                params=params,
                mean_auc=float(np.nanmean(scores)),
                std_auc=float(np.nanstd(scores, ddof=1)) if len(scores) > 1 else 0.0,
                fold_scores=scores,
                seconds=time.perf_counter() - started,
            )
        )
        logger.info(
            "%s trial %2d/%d  inner auc %.4f  %s",
            spec.name,
            len(trials),
            n_trials,
            trials[-1].mean_auc,
            params,
        )

    best = max(trials, key=lambda t: t.mean_auc)
    return SearchResult(
        model=spec.name,
        best=best,
        trials=trials,
        folds_used=len(windows),
        cutoff=cutoff,
    )


def _pre_test_windows(
    rows: pd.DataFrame,
    dates: pd.Series,
    sessions: pd.DatetimeIndex,
    config: AppConfig,
    horizon: int,
    n_folds: int,
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame]], list[pd.Timestamp]]:
    """Nested fit and validation windows lying entirely before the first test session."""
    walkforward = config.walkforward
    splitter = WalkForwardSplitter(walkforward, horizon)
    first_test = splitter.split(sessions)[0].test[0]

    validation_length = walkforward.inner_validation_sessions
    gap = splitter.gap

    windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    boundaries: list[pd.Timestamp] = []

    for step in range(n_folds):
        validation_end = first_test - 1 - gap - step * validation_length
        validation_start = validation_end - validation_length + 1
        fit_end = validation_start - gap - 1
        if validation_start <= 0 or fit_end <= 0:
            break

        fit_mask = dates <= sessions[fit_end]
        validation_mask = (dates >= sessions[validation_start]) & (
            dates <= sessions[validation_end]
        )
        windows.append((rows[fit_mask.to_numpy()], rows[validation_mask.to_numpy()]))
        boundaries.append(sessions[validation_end])

    if not windows:
        raise ValueError("not enough history before the first test window to tune on")
    return windows, boundaries
