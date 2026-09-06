"""Structural and plausibility checks on a downloaded price series."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_forecast.config import DataValidationConfig
from market_forecast.data.base import OHLCV_COLUMNS, is_index_symbol
from market_forecast.data.sessions import (
    last_closed_session,
    missing_sessions,
    unexpected_sessions,
)


@dataclass
class ValidationReport:
    ticker: str
    n_rows: int = 0
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None
    missing_sessions: int = 0
    unexpected_sessions: int = 0
    sessions_stale: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        span = (
            f"{self.first_date:%Y-%m-%d}..{self.last_date:%Y-%m-%d}"
            if self.first_date is not None
            else "empty"
        )
        state = "ok" if self.ok else f"{len(self.errors)} errors"
        return f"{self.ticker}: {self.n_rows} rows {span} [{state}, {len(self.warnings)} warnings]"


def validate_ohlcv(
    frame: pd.DataFrame,
    ticker: str,
    config: DataValidationConfig,
    calendar: str = "XNYS",
    check_staleness: bool = True,
) -> ValidationReport:
    report = ValidationReport(ticker=ticker, n_rows=len(frame))

    missing_columns = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing_columns:
        report.errors.append(f"missing columns: {missing_columns}")
        return report

    if frame.empty:
        report.errors.append("no rows returned")
        return report

    report.first_date = frame.index[0]
    report.last_date = frame.index[-1]

    if not isinstance(frame.index, pd.DatetimeIndex):
        report.errors.append("index is not a DatetimeIndex")
        return report
    if not frame.index.is_monotonic_increasing:
        report.errors.append("index is not sorted chronologically")
    if frame.index.has_duplicates:
        report.errors.append(f"{int(frame.index.duplicated().sum())} duplicate sessions")
    if len(frame) < config.min_rows:
        report.errors.append(f"only {len(frame)} rows, need at least {config.min_rows}")

    prices = frame[["open", "high", "low", "close", "adj_close"]]
    non_positive = int((prices <= 0).to_numpy().sum())
    if non_positive:
        report.errors.append(f"{non_positive} non-positive price values")

    fully_null = int(prices.isna().all(axis=1).sum())
    if fully_null:
        report.errors.append(f"{fully_null} sessions with no price data")

    partial_null = int(prices.isna().any(axis=1).sum()) - fully_null
    if partial_null:
        report.warnings.append(f"{partial_null} sessions with partially missing prices")

    inverted = int((frame["high"] < frame["low"]).sum())
    if inverted:
        report.errors.append(f"{inverted} sessions with high below low")

    outside = int(
        (
            (frame["high"] < frame[["open", "close"]].max(axis=1) - 1e-6)
            | (frame["low"] > frame[["open", "close"]].min(axis=1) + 1e-6)
        ).sum()
    )
    if outside:
        report.warnings.append(
            f"{outside} sessions where open/close sit outside the high-low range"
        )

    gaps = missing_sessions(frame.index, calendar)
    report.missing_sessions = len(gaps)
    if len(frame) and report.missing_sessions / len(frame) > config.max_missing_session_ratio:
        report.errors.append(
            f"{report.missing_sessions} exchange sessions absent "
            f"({report.missing_sessions / len(frame):.2%} of rows)"
        )
    elif report.missing_sessions:
        report.warnings.append(f"{report.missing_sessions} exchange sessions absent")

    stray = unexpected_sessions(frame.index, calendar)
    report.unexpected_sessions = len(stray)
    if report.unexpected_sessions:
        shown = ", ".join(d.strftime("%Y-%m-%d") for d in stray[:3])
        report.warnings.append(
            f"{report.unexpected_sessions} rows dated on exchange holidays ({shown})"
        )

    is_index = is_index_symbol(ticker)

    returns = frame["adj_close"].pct_change()
    extreme = returns.abs() > config.max_abs_daily_return
    if int(extreme.sum()) and not is_index:
        worst = returns[extreme].abs().max()
        report.warnings.append(
            f"{int(extreme.sum())} sessions move more than "
            f"{config.max_abs_daily_return:.0%} (max {worst:.1%}); check for unadjusted actions"
        )

    # Index symbols carry no share volume, so the zero-volume rule does not apply.
    volume = frame["volume"]
    zero_ratio = float((volume.fillna(0) <= 0).mean())
    if zero_ratio > config.allow_zero_volume_ratio and not is_index:
        report.warnings.append(f"{zero_ratio:.2%} of sessions have zero volume")

    if not np.isfinite(frame[list(OHLCV_COLUMNS)].to_numpy(dtype="float64")).any():
        report.errors.append("no finite values in the series")

    if check_staleness:
        latest = last_closed_session(calendar)
        if report.last_date is not None and report.last_date < latest:
            gap = missing_sessions(
                pd.DatetimeIndex([report.last_date, latest]).sort_values(), calendar
            )
            report.sessions_stale = len(gap) + 1
            report.warnings.append(
                f"last observation {report.last_date:%Y-%m-%d} is "
                f"{report.sessions_stale} sessions behind {latest:%Y-%m-%d}"
            )

    return report
