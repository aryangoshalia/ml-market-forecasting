"""Trend-quality features: how directional recent movement has been, and where price sits."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_forecast.features.indicators import efficiency_ratio, log_return
from market_forecast.features.registry import BuildResult, FeatureSpec

GROUP = "trend"


def build(frame: pd.DataFrame, params: dict[str, Any]) -> BuildResult:
    close = frame["close"]
    columns: dict[str, pd.Series] = {}
    specs: list[FeatureSpec] = []

    er_window = int(params.get("efficiency_ratio_window", 20))
    columns[f"efficiency_ratio_{er_window}"] = efficiency_ratio(close, er_window)
    specs.append(
        FeatureSpec(
            name=f"efficiency_ratio_{er_window}",
            group=GROUP,
            description=f"Net move divided by total path length over {er_window} sessions",
            formula=f"|C_t - C_(t-{er_window})| / sum(|C_i - C_(i-1)|)",
            rationale=(
                "Bounded in [0, 1] and separates a clean trend from a choppy market that "
                "travelled the same net distance; conditions whether momentum should be trusted"
            ),
            lookback=er_window + 1,
        )
    )

    extreme_window = int(params.get("extreme_window", 252))
    highest = close.rolling(extreme_window, min_periods=extreme_window).max()
    lowest = close.rolling(extreme_window, min_periods=extreme_window).min()
    columns[f"dist_high_{extreme_window}"] = close / highest - 1.0
    columns[f"dist_low_{extreme_window}"] = close / lowest - 1.0
    specs.extend(
        [
            FeatureSpec(
                name=f"dist_high_{extreme_window}",
                group=GROUP,
                description=f"Drawdown from the {extreme_window}-session high",
                formula=f"C_t / max_{extreme_window}(C)_t - 1",
                rationale=(
                    "Never positive; how deep the current drawdown is separates a pullback "
                    "from a broken trend"
                ),
                lookback=extreme_window,
            ),
            FeatureSpec(
                name=f"dist_low_{extreme_window}",
                group=GROUP,
                description=f"Advance above the {extreme_window}-session low",
                formula=f"C_t / min_{extreme_window}(C)_t - 1",
                rationale="Never negative; measures how far a recovery has already run",
                lookback=extreme_window,
            ),
        ]
    )

    up_window = int(params.get("up_day_window", 20))
    returns = log_return(close, 1)
    columns[f"up_day_ratio_{up_window}"] = (
        (returns > 0.0).rolling(up_window, min_periods=up_window).mean()
    )
    specs.append(
        FeatureSpec(
            name=f"up_day_ratio_{up_window}",
            group=GROUP,
            description=f"Fraction of the last {up_window} sessions that closed higher",
            formula=f"mean_{up_window}(1[r_t > 0])",
            rationale=(
                "Counts direction without weighting by size, so a single large move cannot "
                "dominate the way it does in a mean return"
            ),
            lookback=up_window + 1,
        )
    )

    return BuildResult(frame=pd.DataFrame(columns, index=frame.index), specs=specs)
