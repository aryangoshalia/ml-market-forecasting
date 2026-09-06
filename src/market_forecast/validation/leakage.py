"""Audits that future information cannot reach a feature row, and that a label reads
exactly its own horizon. Each audit is also required to fail on a planted bug."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_forecast.features.market import MarketContext
from market_forecast.logging import get_logger

logger = get_logger(__name__)

BuildFn = Callable[[pd.DataFrame, MarketContext | None], pd.DataFrame]

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


@dataclass
class AuditFinding:
    check: str
    detail: str
    feature: str | None = None
    position: int | None = None
    severity: str = "error"


@dataclass
class AuditReport:
    name: str
    checks_run: int = 0
    features_checked: int = 0
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    @property
    def errors(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == "error"]

    def summary(self) -> str:
        state = "pass" if self.ok else f"FAIL ({len(self.errors)} findings)"
        return (
            f"{self.name}: {state}, {self.checks_run} checks over {self.features_checked} features"
        )

    def merge(self, other: AuditReport) -> AuditReport:
        return AuditReport(
            name=f"{self.name}+{other.name}",
            checks_run=self.checks_run + other.checks_run,
            features_checked=max(self.features_checked, other.features_checked),
            findings=self.findings + other.findings,
        )


def corrupt_after(
    prices: pd.DataFrame, position: int, seed: int = 0, drift: float = 0.05
) -> pd.DataFrame:
    """Replace every row after ``position`` with a different, still plausible price path."""
    rng = np.random.default_rng(seed)
    out = prices.copy()
    tail = out.index[position + 1 :]
    if len(tail) == 0:
        return out

    anchor = float(out["close"].iloc[position])
    shocks = rng.normal(drift, 0.05, size=len(tail))
    path = anchor * np.exp(np.cumsum(shocks))

    out.loc[tail, "close"] = path
    out.loc[tail, "open"] = path * (1.0 + rng.normal(0.0, 0.01, len(tail)))
    out.loc[tail, "high"] = np.maximum(out.loc[tail, "open"], path) * 1.01
    out.loc[tail, "low"] = np.minimum(out.loc[tail, "open"], path) * 0.99
    out.loc[tail, "adj_close"] = path
    out.loc[tail, "volume"] = np.abs(rng.normal(5e7, 1e7, len(tail)))
    if "adj_factor" in out.columns:
        out.loc[tail, "adj_factor"] = 1.0
    return out


def _corrupt_context(
    context: MarketContext | None, cutoff: pd.Timestamp, seed: int
) -> MarketContext | None:
    if context is None:
        return None

    def corrupt(frame: pd.DataFrame, offset: int) -> pd.DataFrame:
        if frame.empty:
            return frame
        positions = np.flatnonzero(frame.index <= cutoff)
        if len(positions) == 0:
            return frame
        return corrupt_after(frame, int(positions[-1]), seed=seed + offset)

    def corrupt_optional(frame: pd.DataFrame | None, offset: int) -> pd.DataFrame | None:
        return None if frame is None else corrupt(frame, offset)

    return MarketContext(
        benchmark_symbol=context.benchmark_symbol,
        benchmark=corrupt(context.benchmark, 1),
        volatility_index=corrupt_optional(context.volatility_index, 2),
        volatility_index_symbol=context.volatility_index_symbol,
        sector=corrupt_optional(context.sector, 3),
        sector_symbol=context.sector_symbol,
    )


def _compare_row(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    session: pd.Timestamp,
    check: str,
    position: int,
    rtol: float,
    atol: float,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if session not in reference.index or session not in candidate.index:
        return [
            AuditFinding(
                check=check,
                detail=f"session {session:%Y-%m-%d} absent from one of the two builds",
                position=position,
            )
        ]

    left = reference.loc[session]
    right = candidate.loc[session]
    for feature in reference.columns:
        a, b = float(np.asarray(left[feature]).item()), float(np.asarray(right[feature]).item())
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) != np.isnan(b) or not np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            findings.append(
                AuditFinding(
                    check=check,
                    feature=feature,
                    position=position,
                    detail=(
                        f"{feature} at {session:%Y-%m-%d} changed from {a!r} to {b!r} "
                        "when only future data was altered"
                    ),
                )
            )
    return findings


def audit_perturbation(
    build: BuildFn,
    prices: pd.DataFrame,
    context: MarketContext | None = None,
    positions: list[int] | None = None,
    seed: int = 0,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> AuditReport:
    reference = build(prices, context)
    report = AuditReport(name="perturbation", features_checked=len(reference.columns))
    for position in positions or _default_positions(prices):
        session = prices.index[position]
        if session not in reference.index:
            continue
        corrupted = build(
            corrupt_after(prices, position, seed=seed),
            _corrupt_context(context, session, seed),
        )
        report.checks_run += 1
        report.findings.extend(
            _compare_row(reference, corrupted, session, "perturbation", position, rtol, atol)
        )
    return report


def audit_truncation(
    build: BuildFn,
    prices: pd.DataFrame,
    context: MarketContext | None = None,
    positions: list[int] | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> AuditReport:
    reference = build(prices, context)
    report = AuditReport(name="truncation", features_checked=len(reference.columns))
    for position in positions or _default_positions(prices):
        session = prices.index[position]
        if session not in reference.index:
            continue
        truncated_context = _truncate_context(context, session)
        truncated = build(prices.iloc[: position + 1], truncated_context)
        report.checks_run += 1
        report.findings.extend(
            _compare_row(reference, truncated, session, "truncation", position, rtol, atol)
        )
    return report


def _truncate_context(context: MarketContext | None, cutoff: pd.Timestamp) -> MarketContext | None:
    if context is None:
        return None
    return MarketContext(
        benchmark_symbol=context.benchmark_symbol,
        benchmark=context.benchmark.loc[:cutoff],
        volatility_index=(
            context.volatility_index.loc[:cutoff] if context.volatility_index is not None else None
        ),
        volatility_index_symbol=context.volatility_index_symbol,
        sector=context.sector.loc[:cutoff] if context.sector is not None else None,
        sector_symbol=context.sector_symbol,
    )


def audit_scale_invariance(
    build: BuildFn,
    prices: pd.DataFrame,
    context: MarketContext | None = None,
    price_factor: float = 7.0,
    volume_factor: float = 3.0,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> AuditReport:
    """Rescaling prices and volumes must leave every feature untouched.

    This is the property that lets one model be trained across tickers whose prices
    differ by orders of magnitude. A feature that fails here is a level, not a signal.
    """
    reference = build(prices, context)
    report = AuditReport(name="scale", features_checked=len(reference.columns), checks_run=1)

    scaled = prices.copy()
    for column in ("open", "high", "low", "close", "adj_close"):
        scaled[column] = scaled[column] * price_factor
    scaled["volume"] = scaled["volume"] * volume_factor

    scaled_context = None
    if context is not None:

        def rescale(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            out = frame.copy()
            for column in ("open", "high", "low", "close", "adj_close"):
                out[column] = out[column] * price_factor
            out["volume"] = out["volume"] * volume_factor
            return out

        scaled_context = MarketContext(
            benchmark_symbol=context.benchmark_symbol,
            benchmark=rescale(context.benchmark),
            volatility_index=context.volatility_index,
            volatility_index_symbol=context.volatility_index_symbol,
            sector=None if context.sector is None else rescale(context.sector),
            sector_symbol=context.sector_symbol,
        )

    candidate = build(scaled, scaled_context)
    shared = reference.index.intersection(candidate.index)
    for feature in reference.columns:
        left = reference.loc[shared, feature].to_numpy()
        right = candidate.loc[shared, feature].to_numpy()
        if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
            worst = np.nanmax(np.abs(left - right))
            report.findings.append(
                AuditFinding(
                    check="scale",
                    feature=feature,
                    detail=(
                        f"{feature} changed by up to {worst:.3e} when prices were scaled by "
                        f"{price_factor} and volume by {volume_factor}"
                    ),
                )
            )
    return report


def audit_target_causality(
    prices: pd.DataFrame,
    horizon: int,
    build_target: Callable[[pd.DataFrame], pd.Series],
    positions: list[int] | None = None,
    seed: int = 0,
) -> AuditReport:
    """A label at t must depend on (t, t+h] and on nothing beyond it."""
    reference = build_target(prices)
    report = AuditReport(name=f"target_h{horizon}", features_checked=1)

    for position in positions or _default_positions(prices, margin=horizon + 5):
        session = prices.index[position]
        report.checks_run += 1

        beyond = build_target(corrupt_after(prices, position + horizon, seed=seed))
        before, after = reference.loc[session], beyond.loc[session]
        if not np.isclose(before, after, equal_nan=True):
            report.findings.append(
                AuditFinding(
                    check="target_horizon_upper_bound",
                    position=position,
                    detail=(
                        f"label at {session:%Y-%m-%d} changed from {before} to {after} when data "
                        f"beyond t+{horizon} was altered, so the label reaches too far forward"
                    ),
                )
            )

        # A binary label only takes two values, so a single corruption can match the
        # original by chance. Drive the horizon window up and down instead: a label that
        # genuinely reads (t, t+h] must disagree between the two.
        rising = build_target(corrupt_after(prices, position, seed=seed + 1, drift=0.05))
        falling = build_target(corrupt_after(prices, position, seed=seed + 1, drift=-0.05))
        if np.isclose(rising.loc[session], falling.loc[session], equal_nan=True):
            report.findings.append(
                AuditFinding(
                    check="target_responds_to_horizon",
                    position=position,
                    detail=(
                        f"label at {session:%Y-%m-%d} was identical under a rising and a falling "
                        f"path over (t, t+{horizon}], so it does not depend on its own horizon"
                    ),
                )
            )
    return report


def _default_positions(prices: pd.DataFrame, margin: int = 30, count: int = 6) -> list[int]:
    usable = len(prices) - margin
    if usable <= 300:
        return [max(0, usable - 1)]
    return [int(p) for p in np.linspace(300, usable - 1, count).astype(int)]
