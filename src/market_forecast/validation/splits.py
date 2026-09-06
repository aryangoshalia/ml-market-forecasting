"""Walk-forward splitting. Fit, inner and test windows separated by a purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_forecast.config import WalkForwardConfig
from market_forecast.logging import get_logger

logger = get_logger(__name__)

ANCHORED = "anchored"
ROLLING = "rolling"


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, expressed both as positions and as dates."""

    index: int
    fit: tuple[int, int]
    inner: tuple[int, int]
    test: tuple[int, int]
    gap: int
    horizon: int
    dates: pd.DatetimeIndex

    def _span(self, bounds: tuple[int, int]) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.dates[bounds[0]], self.dates[bounds[1]]

    @property
    def fit_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self._span(self.fit)

    @property
    def inner_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self._span(self.inner)

    @property
    def test_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self._span(self.test)

    @property
    def train_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.dates[self.fit[0]], self.dates[self.inner[1]]

    @property
    def n_fit(self) -> int:
        return self.fit[1] - self.fit[0] + 1

    @property
    def n_inner(self) -> int:
        return self.inner[1] - self.inner[0] + 1

    @property
    def n_test(self) -> int:
        return self.test[1] - self.test[0] + 1

    def mask(self, dates: pd.Series, part: str) -> pd.Series:
        """Boolean mask over a date column, selecting one part of this fold."""
        bounds = {"fit": self.fit, "inner": self.inner, "test": self.test}[part]
        start, end = self.dates[bounds[0]], self.dates[bounds[1]]
        return (dates >= start) & (dates <= end)

    def describe(self) -> str:
        fit_start, fit_end = self.fit_dates
        inner_start, inner_end = self.inner_dates
        test_start, test_end = self.test_dates
        return (
            f"fold {self.index:2d}  "
            f"fit {fit_start:%Y-%m-%d}..{fit_end:%Y-%m-%d} ({self.n_fit})  "
            f"inner {inner_start:%Y-%m-%d}..{inner_end:%Y-%m-%d} ({self.n_inner})  "
            f"test {test_start:%Y-%m-%d}..{test_end:%Y-%m-%d} ({self.n_test})"
        )


class WalkForwardSplitter:
    def __init__(self, config: WalkForwardConfig, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least one session")
        if config.scheme not in (ANCHORED, ROLLING):
            raise ValueError(f"unknown scheme: {config.scheme!r}")
        self.config = config
        self.horizon = horizon
        self.purge = config.purge_for(horizon)
        self.gap = self.purge + config.embargo_sessions

    def split(self, dates: pd.DatetimeIndex) -> list[Fold]:
        sessions = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(dates))).sort_values()
        n = len(sessions)
        config = self.config

        folds: list[Fold] = []
        index = 0
        while True:
            test_start = config.initial_train_sessions + index * config.step_sessions
            test_end = test_start + config.test_sessions - 1
            if test_end >= n:
                break

            inner_end = test_start - self.gap - 1
            inner_start = inner_end - config.inner_validation_sessions + 1
            fit_end = inner_start - self.gap - 1
            fit_start = (
                0
                if config.scheme == ANCHORED
                else max(0, fit_end - config.rolling_train_sessions + 1)
            )

            if inner_start <= 0 or fit_end <= fit_start:
                index += 1
                continue

            folds.append(
                Fold(
                    index=len(folds),
                    fit=(fit_start, fit_end),
                    inner=(inner_start, inner_end),
                    test=(test_start, test_end),
                    gap=self.gap,
                    horizon=self.horizon,
                    dates=sessions,
                )
            )
            index += 1

        if not folds:
            raise ValueError(
                f"no folds fit in {n} sessions with initial_train="
                f"{config.initial_train_sessions}, test={config.test_sessions}, gap={self.gap}"
            )
        return folds

    def describe(self, dates: pd.DatetimeIndex) -> str:
        folds = self.split(dates)
        first, last = folds[0], folds[-1]
        return (
            f"{len(folds)} folds, horizon {self.horizon}, scheme {self.config.scheme}, "
            f"purge {self.purge} + embargo {self.config.embargo_sessions} = {self.gap} sessions; "
            f"test coverage {first.test_dates[0]:%Y-%m-%d}..{last.test_dates[1]:%Y-%m-%d}"
        )


def assert_no_overlap(fold: Fold) -> None:
    """Fail loudly if a fold's windows touch. Cheap insurance against an off-by-one."""
    if fold.fit[1] >= fold.inner[0]:
        raise ValueError(f"fold {fold.index}: fit window runs into the inner window")
    if fold.inner[1] >= fold.test[0]:
        raise ValueError(f"fold {fold.index}: inner window runs into the test window")
    if fold.inner[0] - fold.fit[1] <= fold.horizon:
        raise ValueError(f"fold {fold.index}: gap before inner is not larger than the horizon")
    if fold.test[0] - fold.inner[1] <= fold.horizon:
        raise ValueError(f"fold {fold.index}: gap before test is not larger than the horizon")


def label_reach(dates: pd.DatetimeIndex, position: int, horizon: int) -> pd.Timestamp:
    """The last session a label at ``position`` is defined by."""
    return dates[min(position + horizon, len(dates) - 1)]


def evaluated_sessions(folds: list[Fold]) -> pd.DatetimeIndex:
    """Every session that appears in some test window, used for out-of-sample series."""
    parts = [fold.dates[fold.test[0] : fold.test[1] + 1] for fold in folds]
    return pd.DatetimeIndex(np.concatenate([p.to_numpy() for p in parts])).sort_values()
