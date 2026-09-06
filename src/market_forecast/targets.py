"""Target construction. The mathematical definition of each target is in docs/REPORT.md."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_forecast.features.indicators import ewma_volatility, log_return
from market_forecast.logging import get_logger

logger = get_logger(__name__)

CLOSE_TO_CLOSE = "close_to_close"
NEXT_OPEN_TO_CLOSE = "next_open_to_close"

DOWN, NEUTRAL, UP = -1, 0, 1


def forward_return(prices: pd.DataFrame, horizon: int, basis: str = CLOSE_TO_CLOSE) -> pd.Series:
    """Simple forward return over ``horizon`` sessions, NaN on the final h rows."""
    if horizon < 1:
        raise ValueError("horizon must be at least one session")
    close = prices["close"]
    if basis == CLOSE_TO_CLOSE:
        anchor = close
        target = close.shift(-horizon)
    elif basis == NEXT_OPEN_TO_CLOSE:
        anchor = prices["open"].shift(-1)
        target = close.shift(-horizon)
    else:
        raise ValueError(f"unknown return basis: {basis!r}")
    return (target / anchor - 1.0).rename(f"fwd_ret_{horizon}d")


def binary_direction(returns: pd.Series) -> pd.Series:
    return returns.gt(0.0).astype("float64").where(returns.notna())


def excess_return(returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    aligned = benchmark_returns.reindex(returns.index)
    return returns - aligned


def causal_volatility(prices: pd.DataFrame, lam: float = 0.94, min_periods: int = 60) -> pd.Series:
    return ewma_volatility(log_return(prices["close"], 1), lam=lam, min_periods=min_periods).rename(
        "ewma_sigma"
    )


def _neutral_fraction(returns: np.ndarray, band: np.ndarray, kappa: float) -> float:
    return float(np.mean(np.abs(returns) <= kappa * band))


def solve_kappa(
    returns: pd.Series,
    sigma: pd.Series,
    horizon: int,
    target_fraction: float,
    search: tuple[float, float] = (0.05, 1.5),
    tolerance: float = 1e-4,
    max_iterations: int = 80,
) -> float:
    """Bisect for the kappa that puts ``target_fraction`` of training labels in the neutral band.

    The neutral share is monotone non-decreasing in kappa, so bisection is exact up to
    the granularity of the sample.
    """
    frame = pd.concat([returns.rename("r"), sigma.rename("s")], axis=1).dropna()
    if frame.empty:
        raise ValueError("cannot solve kappa on an empty training sample")

    values = frame["r"].to_numpy()
    band = frame["s"].to_numpy() * np.sqrt(horizon)

    low, high = search
    if _neutral_fraction(values, band, high) < target_fraction:
        logger.warning(
            "kappa search capped at %.2f, neutral share only %.3f of the requested %.3f",
            high,
            _neutral_fraction(values, band, high),
            target_fraction,
        )
        return high
    if _neutral_fraction(values, band, low) > target_fraction:
        return low

    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        share = _neutral_fraction(values, band, middle)
        if abs(share - target_fraction) < tolerance:
            return middle
        if share < target_fraction:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass
class VolScaledTernaryLabeller:
    """Fitted on a training window, applied unchanged to the evaluation window."""

    horizon: int
    target_neutral_fraction: float = 1.0 / 3.0
    search: tuple[float, float] = (0.05, 1.5)
    kappa_: float | None = None

    def fit(self, returns: pd.Series, sigma: pd.Series) -> VolScaledTernaryLabeller:
        self.kappa_ = solve_kappa(
            returns, sigma, self.horizon, self.target_neutral_fraction, self.search
        )
        return self

    def threshold(self, sigma: pd.Series) -> pd.Series:
        if self.kappa_ is None:
            raise RuntimeError("labeller must be fitted before use")
        return (self.kappa_ * sigma * np.sqrt(self.horizon)).rename("tau")

    def transform(self, returns: pd.Series, sigma: pd.Series) -> pd.Series:
        tau = self.threshold(sigma).reindex(returns.index)
        labels = pd.Series(np.full(len(returns), np.nan), index=returns.index, dtype="float64")
        usable = returns.notna() & tau.notna()
        labels[usable & (returns > tau)] = float(UP)
        labels[usable & (returns < -tau)] = float(DOWN)
        labels[usable & (returns.abs() <= tau)] = float(NEUTRAL)
        return labels.rename(f"ternary_{self.horizon}d")


@dataclass
class QuantileTernaryLabeller:
    """Ablation for the volatility-scaled band: unconditional tertiles of training returns."""

    horizon: int
    lower_quantile: float = 1.0 / 3.0
    upper_quantile: float = 2.0 / 3.0
    bounds_: tuple[float, float] | None = None

    def fit(self, returns: pd.Series, sigma: pd.Series | None = None) -> QuantileTernaryLabeller:  # noqa: ARG002
        clean = returns.dropna()
        if clean.empty:
            raise ValueError("cannot fit quantile bounds on an empty training sample")
        self.bounds_ = (
            float(clean.quantile(self.lower_quantile)),
            float(clean.quantile(self.upper_quantile)),
        )
        return self

    def transform(self, returns: pd.Series, sigma: pd.Series | None = None) -> pd.Series:  # noqa: ARG002
        if self.bounds_ is None:
            raise RuntimeError("labeller must be fitted before use")
        lower, upper = self.bounds_
        labels = pd.Series(np.full(len(returns), np.nan), index=returns.index, dtype="float64")
        usable = returns.notna()
        labels[usable & (returns > upper)] = float(UP)
        labels[usable & (returns < lower)] = float(DOWN)
        labels[usable & (returns >= lower) & (returns <= upper)] = float(NEUTRAL)
        return labels.rename(f"ternary_quantile_{self.horizon}d")


@dataclass
class TargetSet:
    ticker: str
    frame: pd.DataFrame
    horizons: list[int]
    columns: dict[str, list[str]] = field(default_factory=dict)

    def label(self, name: str, horizon: int) -> pd.Series:
        return self.frame[f"{name}_{horizon}d"]

    @property
    def names(self) -> list[str]:
        return list(self.frame.columns)


def build_targets(
    ticker: str,
    prices: pd.DataFrame,
    horizons: list[int],
    benchmark_prices: pd.DataFrame | None = None,
    ewma_lambda: float = 0.94,
    ewma_min_periods: int = 60,
) -> TargetSet:
    """Build every label that needs no fitted parameter, plus the inputs for those that do.

    The ternary labels are deliberately absent here: their threshold is fitted inside a
    training window, so they are produced during walk-forward evaluation rather than
    baked into a static table.
    """
    columns: dict[str, pd.Series] = {}
    grouped: dict[str, list[str]] = {}

    sigma = causal_volatility(prices, lam=ewma_lambda, min_periods=ewma_min_periods)
    columns["ewma_sigma"] = sigma

    for horizon in horizons:
        close_return = forward_return(prices, horizon, CLOSE_TO_CLOSE)
        open_return = forward_return(prices, horizon, NEXT_OPEN_TO_CLOSE)

        columns[f"fwd_ret_{horizon}d"] = close_return
        columns[f"fwd_ret_open_{horizon}d"] = open_return
        columns[f"direction_{horizon}d"] = binary_direction(close_return)
        columns[f"direction_open_{horizon}d"] = binary_direction(open_return)
        grouped.setdefault("direction", []).append(f"direction_{horizon}d")
        grouped.setdefault("direction_open_anchored", []).append(f"direction_open_{horizon}d")

        if benchmark_prices is not None and not benchmark_prices.empty:
            benchmark_return = forward_return(benchmark_prices, horizon, CLOSE_TO_CLOSE)
            excess = excess_return(close_return, benchmark_return)
            columns[f"fwd_excess_ret_{horizon}d"] = excess
            columns[f"excess_direction_{horizon}d"] = binary_direction(excess)
            grouped.setdefault("excess_direction", []).append(f"excess_direction_{horizon}d")

    frame = pd.DataFrame(columns, index=prices.index)
    return TargetSet(ticker=ticker, frame=frame, horizons=list(horizons), columns=grouped)


def valid_label_mask(frame: pd.DataFrame, horizon: int, target: str = "direction") -> pd.Series:
    return frame[f"{target}_{horizon}d"].notna()
