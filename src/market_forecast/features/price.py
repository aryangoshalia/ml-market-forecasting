"""Return features: multi-horizon log returns and short lags of the daily return."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_forecast.features.indicators import log_return
from market_forecast.features.registry import BuildResult, FeatureSpec

GROUP = "returns"


def build(frame: pd.DataFrame, params: dict[str, Any]) -> BuildResult:
    close = frame["close"]
    horizons: list[int] = params.get("horizons", [1, 3, 5, 10, 20])
    lags: list[int] = params.get("lags", [1, 2, 3, 4, 5])

    columns: dict[str, pd.Series] = {}
    specs: list[FeatureSpec] = []

    for horizon in horizons:
        name = f"ret_{horizon}d"
        columns[name] = log_return(close, horizon)
        specs.append(
            FeatureSpec(
                name=name,
                group=GROUP,
                description=f"{horizon}-session log return",
                formula=f"log(C_t / C_(t-{horizon}))",
                rationale=(
                    "Short horizons carry weak reversal, longer horizons carry weak "
                    "continuation; together they describe the recent return path"
                ),
                lookback=horizon + 1,
            )
        )

    daily = columns["ret_1d"] if "ret_1d" in columns else log_return(close, 1)
    for lag in lags:
        name = f"ret_1d_lag{lag}"
        columns[name] = daily.shift(lag)
        specs.append(
            FeatureSpec(
                name=name,
                group=GROUP,
                description=f"Daily log return lagged {lag} sessions",
                formula=f"log(C_(t-{lag}) / C_(t-{lag}-1))",
                rationale=(
                    "Lets the model learn sign patterns in the recent return sequence "
                    "rather than only its aggregate"
                ),
                lookback=lag + 2,
            )
        )

    return BuildResult(frame=pd.DataFrame(columns, index=frame.index), specs=specs)
