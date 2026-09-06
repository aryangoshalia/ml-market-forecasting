"""Volatility features: realised volatility, its term structure and range-based measures."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_forecast.features.indicators import (
    atr,
    downside_volatility,
    log_return,
    parkinson_volatility,
    realised_volatility,
)
from market_forecast.features.registry import BuildResult, FeatureSpec

GROUP = "volatility"


def build(frame: pd.DataFrame, params: dict[str, Any]) -> BuildResult:
    close, high, low = frame["close"], frame["high"], frame["low"]
    returns = log_return(close, 1)

    columns: dict[str, pd.Series] = {}
    specs: list[FeatureSpec] = []

    volatilities: dict[int, pd.Series] = {}
    for window in params.get("windows", [5, 20, 60]):
        volatilities[window] = realised_volatility(returns, window)
        name = f"vol_{window}"
        columns[name] = volatilities[window]
        specs.append(
            FeatureSpec(
                name=name,
                group=GROUP,
                description=f"Annualised realised volatility over {window} sessions",
                formula=f"std_{window}(log returns) * sqrt(252)",
                rationale=(
                    "Volatility is strongly autocorrelated and conditions the size of any "
                    "future move, so it calibrates how much weight a directional signal deserves"
                ),
                lookback=window + 1,
            )
        )

    for short, long in params.get("ratio_pairs", [[5, 20], [20, 60]]):
        short_vol = volatilities.get(short, realised_volatility(returns, short))
        long_vol = volatilities.get(long, realised_volatility(returns, long))
        name = f"vol_ratio_{short}_{long}"
        columns[name] = short_vol / long_vol.replace(0.0, pd.NA).astype("float64")
        specs.append(
            FeatureSpec(
                name=name,
                group=GROUP,
                description=f"{short}-session volatility relative to {long}-session volatility",
                formula=f"vol_{short} / vol_{long}",
                rationale=(
                    "The volatility term structure: values above one mean volatility is "
                    "expanding, which historically coincides with weaker forward returns"
                ),
                lookback=long + 1,
            )
        )

    atr_window = int(params.get("atr_window", 14))
    columns[f"atr_{atr_window}_norm"] = atr(high, low, close, atr_window) / close
    specs.append(
        FeatureSpec(
            name=f"atr_{atr_window}_norm",
            group=GROUP,
            description=f"Average true range over {atr_window} sessions, divided by price",
            formula="Wilder ATR / C_t",
            rationale=(
                "Uses the full bar including overnight gaps, so it captures risk that a "
                "close-to-close estimate misses"
            ),
            lookback=atr_window * 3,
        )
    )

    parkinson_window = int(params.get("parkinson_window", 20))
    columns[f"parkinson_vol_{parkinson_window}"] = parkinson_volatility(high, low, parkinson_window)
    specs.append(
        FeatureSpec(
            name=f"parkinson_vol_{parkinson_window}",
            group=GROUP,
            description=f"Parkinson high-low volatility estimator over {parkinson_window} sessions",
            formula=f"sqrt(mean_{parkinson_window}(log(H/L)^2) / (4 log 2)) * sqrt(252)",
            rationale=(
                "Roughly five times more efficient than the close-to-close estimator at "
                "the same window length, so it reacts to changes in risk sooner"
            ),
            lookback=parkinson_window,
        )
    )

    downside_window = int(params.get("downside_window", 20))
    total = realised_volatility(returns, downside_window).replace(0.0, pd.NA).astype("float64")
    columns[f"downside_vol_ratio_{downside_window}"] = (
        downside_volatility(returns, downside_window) / total
    )
    specs.append(
        FeatureSpec(
            name=f"downside_vol_ratio_{downside_window}",
            group=GROUP,
            description=f"Share of {downside_window}-session volatility contributed by down moves",
            formula=f"downside_vol_{downside_window} / vol_{downside_window}",
            rationale=(
                "Separates volatility driven by selling from volatility driven by rallies, "
                "which a symmetric estimator cannot distinguish"
            ),
            lookback=downside_window + 1,
        )
    )

    move_window = int(params.get("standardised_move_window", 20))
    daily_vol = realised_volatility(returns, move_window, annualise=False)
    columns["standardised_move"] = returns / daily_vol.replace(0.0, pd.NA).astype("float64")
    specs.append(
        FeatureSpec(
            name="standardised_move",
            group=GROUP,
            description=f"Today's return in units of its own {move_window}-session volatility",
            formula=f"r_t / std_{move_window}(r)",
            rationale=(
                "A one percent move means something different in calm and stressed markets; "
                "this expresses the latest move on a comparable scale"
            ),
            lookback=move_window + 1,
        )
    )

    return BuildResult(frame=pd.DataFrame(columns, index=frame.index), specs=specs)
