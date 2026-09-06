"""Market-context features: benchmark, volatility index and sector behaviour.

Context is aligned onto the ticker's own sessions forward only, never backward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_forecast.features.indicators import (
    log_return,
    realised_volatility,
    rolling_beta,
    rolling_zscore,
    sma,
)
from market_forecast.features.registry import BuildResult, FeatureSpec

GROUP = "market"


@dataclass
class MarketContext:
    benchmark_symbol: str
    benchmark: pd.DataFrame
    volatility_index: pd.DataFrame | None = None
    volatility_index_symbol: str | None = None
    sector: pd.DataFrame | None = None
    sector_symbol: str | None = None


def align(series: pd.Series, index: pd.Index, limit: int) -> pd.Series:
    return series.reindex(index).ffill(limit=limit)


def staleness(series: pd.Series, index: pd.Index) -> int:
    return int(series.reindex(index).isna().sum())


def build(
    frame: pd.DataFrame,
    params: dict[str, Any],
    context: MarketContext,
    forward_fill_limit: int = 3,
) -> BuildResult:
    index = frame.index
    close = frame["close"]
    columns: dict[str, pd.Series] = {}
    specs: list[FeatureSpec] = []

    benchmark_close = align(context.benchmark["close"], index, forward_fill_limit)
    benchmark_returns = log_return(benchmark_close, 1)

    for horizon in params.get("benchmark_return_horizons", [1, 5, 20]):
        name = f"bench_ret_{horizon}d"
        columns[name] = log_return(benchmark_close, horizon)
        specs.append(
            FeatureSpec(
                name=name,
                group=GROUP,
                description=f"{horizon}-session benchmark ({context.benchmark_symbol}) log return",
                formula=f"log(B_t / B_(t-{horizon}))",
                rationale=(
                    "Most of a single stock's daily variance is the market factor, so the "
                    "market's own recent path is directly relevant"
                ),
                lookback=horizon + 1,
            )
        )

    vol_window = int(params.get("benchmark_vol_window", 20))
    columns[f"bench_vol_{vol_window}"] = realised_volatility(benchmark_returns, vol_window)
    specs.append(
        FeatureSpec(
            name=f"bench_vol_{vol_window}",
            group=GROUP,
            description=f"Annualised benchmark volatility over {vol_window} sessions",
            formula=f"std_{vol_window}(benchmark log returns) * sqrt(252)",
            rationale=(
                "Systematic risk level, which conditions how much idiosyncratic signal survives"
            ),
            lookback=vol_window + 1,
        )
    )

    sma_window = int(params.get("benchmark_sma_window", 50))
    columns[f"bench_px_to_sma_{sma_window}"] = (
        benchmark_close / sma(benchmark_close, sma_window) - 1.0
    )
    specs.append(
        FeatureSpec(
            name=f"bench_px_to_sma_{sma_window}",
            group=GROUP,
            description=f"Benchmark relative to its {sma_window}-session moving average",
            formula=f"B_t / SMA_{sma_window}(B)_t - 1",
            rationale=(
                "Broad-market trend state, a common conditioning variable for stock-level signals"
            ),
            lookback=sma_window,
        )
    )

    stock_returns = log_return(close, 1)
    beta_window = int(params.get("beta_window", 60))
    beta = rolling_beta(stock_returns, benchmark_returns, beta_window)
    columns[f"beta_{beta_window}"] = beta
    specs.append(
        FeatureSpec(
            name=f"beta_{beta_window}",
            group=GROUP,
            description=f"Rolling market beta over {beta_window} sessions",
            formula=f"cov_{beta_window}(r, r_m) / var_{beta_window}(r_m)",
            rationale=(
                "How much of the stock's movement is market driven; high-beta names respond "
                "differently to the same market signal than defensive ones"
            ),
            lookback=beta_window + 1,
        )
    )

    corr_window = int(params.get("correlation_window", 60))
    columns[f"corr_bench_{corr_window}"] = stock_returns.rolling(
        corr_window, min_periods=corr_window
    ).corr(benchmark_returns)
    specs.append(
        FeatureSpec(
            name=f"corr_bench_{corr_window}",
            group=GROUP,
            description=f"Correlation with the benchmark over {corr_window} sessions",
            formula=f"corr_{corr_window}(r, r_m)",
            rationale=(
                "Unlike beta this is scale free, and it separates a stock trading on its own "
                "news from one moving with the index"
            ),
            lookback=corr_window + 1,
        )
    )

    residual_horizon = int(params.get("residual_return_horizon", 5))
    columns[f"resid_ret_{residual_horizon}d"] = log_return(
        close, residual_horizon
    ) - beta * log_return(benchmark_close, residual_horizon)
    specs.append(
        FeatureSpec(
            name=f"resid_ret_{residual_horizon}d",
            group=GROUP,
            description=f"{residual_horizon}-session return net of the beta-scaled market return",
            formula=f"r_t({residual_horizon}) - beta_t * r_m,t({residual_horizon})",
            rationale=(
                "Isolates stock-specific movement; residual momentum is a cleaner signal than "
                "total momentum because it removes the shared market factor"
            ),
            lookback=max(beta_window, residual_horizon) + 1,
            leakage_note=(
                "Beta is estimated on the trailing window ending at t, so no future returns "
                "enter the residual."
            ),
        )
    )

    if context.volatility_index is not None and not context.volatility_index.empty:
        vix_close = align(context.volatility_index["close"], index, forward_fill_limit)
        columns["vix_level"] = vix_close / 100.0
        zscore_window = int(params.get("vix_zscore_window", 60))
        columns[f"vix_z_{zscore_window}"] = rolling_zscore(vix_close, zscore_window)
        change_horizon = int(params.get("vix_change_horizon", 5))
        columns[f"vix_chg_{change_horizon}d"] = pd.Series(
            np.log(vix_close.to_numpy() / vix_close.shift(change_horizon).to_numpy()),
            index=index,
        )
        specs.extend(
            [
                FeatureSpec(
                    name="vix_level",
                    group=GROUP,
                    description="Volatility index level expressed as a decimal",
                    formula="VIX_t / 100",
                    rationale=(
                        "The market's priced expectation of future volatility, which is forward "
                        "looking in a way realised volatility is not"
                    ),
                    lookback=1,
                ),
                FeatureSpec(
                    name=f"vix_z_{zscore_window}",
                    group=GROUP,
                    description=f"Volatility index z-score over {zscore_window} sessions",
                    formula=f"(VIX_t - mean_{zscore_window}) / std_{zscore_window}",
                    rationale=(
                        "The level alone drifts across decades; the z-score asks whether fear is "
                        "elevated relative to the recent regime"
                    ),
                    lookback=zscore_window,
                ),
                FeatureSpec(
                    name=f"vix_chg_{change_horizon}d",
                    group=GROUP,
                    description=f"{change_horizon}-session log change in the volatility index",
                    formula=f"log(VIX_t / VIX_(t-{change_horizon}))",
                    rationale="Direction of repricing in risk, which often leads price weakness",
                    lookback=change_horizon + 1,
                ),
            ]
        )

    if context.sector is not None and not context.sector.empty:
        sector_close = align(context.sector["close"], index, forward_fill_limit)
        horizon = int(params.get("sector_return_horizon", 5))
        sector_return = log_return(sector_close, horizon)
        columns[f"sector_ret_{horizon}d"] = sector_return
        columns[f"rel_sector_ret_{horizon}d"] = log_return(close, horizon) - sector_return
        specs.extend(
            [
                FeatureSpec(
                    name=f"sector_ret_{horizon}d",
                    group=GROUP,
                    description=f"{horizon}-session return of the sector ETF proxy",
                    formula=f"log(S_t / S_(t-{horizon}))",
                    rationale=(
                        "Sector rotation explains a large share of the return left after "
                        "the market factor"
                    ),
                    lookback=horizon + 1,
                    leakage_note=(
                        "Sector membership is taken from current classification data, a mild "
                        "look-ahead for firms that have since been reclassified."
                    ),
                ),
                FeatureSpec(
                    name=f"rel_sector_ret_{horizon}d",
                    group=GROUP,
                    description=f"{horizon}-session return relative to the sector ETF",
                    formula=f"r_t({horizon}) - r_sector,t({horizon})",
                    rationale=(
                        "Relative strength within a sector separates a stock leading its peers "
                        "from one simply carried by them"
                    ),
                    lookback=horizon + 1,
                    leakage_note=(
                        "Sector membership is taken from current classification data, a mild "
                        "look-ahead for firms that have since been reclassified."
                    ),
                ),
            ]
        )

    return BuildResult(frame=pd.DataFrame(columns, index=index), specs=specs)
