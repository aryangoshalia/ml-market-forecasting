"""Volume features, all expressed relative to the ticker's own recent activity."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from market_forecast.features.indicators import log_return, rolling_zscore, sma
from market_forecast.features.registry import BuildResult, FeatureSpec

GROUP = "volume"


def _log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(np.log(numerator.to_numpy() / denominator.to_numpy()), index=numerator.index)


def build(frame: pd.DataFrame, params: dict[str, Any]) -> BuildResult:
    volume = frame["volume"].astype("float64")
    close = frame["close"]
    positive_volume = volume.where(volume > 0.0)

    columns: dict[str, pd.Series] = {}
    specs: list[FeatureSpec] = []

    relative_window = int(params.get("relative_window", 20))
    average_volume = sma(positive_volume, relative_window)
    columns[f"rel_volume_{relative_window}"] = _log_ratio(
        positive_volume, average_volume.replace(0.0, np.nan)
    )
    specs.append(
        FeatureSpec(
            name=f"rel_volume_{relative_window}",
            group=GROUP,
            description=f"Log of volume against its {relative_window}-session average",
            formula=f"log(V_t / SMA_{relative_window}(V)_t)",
            rationale=(
                "Participation relative to the ticker's own norm; share counts differ by "
                "orders of magnitude across tickers so the raw level is unusable"
            ),
            lookback=relative_window,
        )
    )

    zscore_window = int(params.get("zscore_window", 20))
    columns[f"volume_z_{zscore_window}"] = rolling_zscore(positive_volume, zscore_window)
    specs.append(
        FeatureSpec(
            name=f"volume_z_{zscore_window}",
            group=GROUP,
            description=f"Volume z-score over {zscore_window} sessions",
            formula=f"(V_t - mean_{zscore_window}(V)) / std_{zscore_window}(V)",
            rationale=(
                "Scales the surprise in volume by how variable volume normally is, which "
                "a plain ratio ignores"
            ),
            lookback=zscore_window,
        )
    )

    dollar_window = int(params.get("dollar_volume_window", 20))
    dollar_volume = (close * positive_volume).where(lambda s: s > 0.0)
    columns[f"dollar_volume_ratio_{dollar_window}"] = _log_ratio(
        dollar_volume, sma(dollar_volume, dollar_window).replace(0.0, np.nan)
    )
    specs.append(
        FeatureSpec(
            name=f"dollar_volume_ratio_{dollar_window}",
            group=GROUP,
            description=f"Log traded value against its {dollar_window}-session average",
            formula=f"log(C_t V_t / SMA_{dollar_window}(C V)_t)",
            rationale=(
                "Traded value tracks the capital committed to a move better than share "
                "count, which drifts with the price level"
            ),
            lookback=dollar_window,
        )
    )

    corr_window = int(params.get("price_volume_corr_window", 20))
    returns = log_return(close, 1)
    volume_change = _log_ratio(positive_volume, positive_volume.shift(1))
    columns[f"price_volume_corr_{corr_window}"] = returns.rolling(
        corr_window, min_periods=corr_window
    ).corr(volume_change)
    specs.append(
        FeatureSpec(
            name=f"price_volume_corr_{corr_window}",
            group=GROUP,
            description=f"Correlation of returns and volume changes over {corr_window} sessions",
            formula=f"corr_{corr_window}(r_t, log(V_t / V_(t-1)))",
            rationale=(
                "Distinguishes advances on rising volume from advances on fading volume, "
                "a conviction signal neither series carries alone"
            ),
            lookback=corr_window + 1,
        )
    )

    return BuildResult(frame=pd.DataFrame(columns, index=frame.index), specs=specs)
