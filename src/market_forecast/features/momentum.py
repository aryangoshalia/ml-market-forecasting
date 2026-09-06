"""Momentum features: moving-average distances, oscillators and skip-month momentum."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from market_forecast.features.indicators import macd, price_position, rsi, sma
from market_forecast.features.registry import BuildResult, FeatureSpec

GROUP = "momentum"


def build(frame: pd.DataFrame, params: dict[str, Any]) -> BuildResult:
    close = frame["close"]
    columns: dict[str, pd.Series] = {}
    specs: list[FeatureSpec] = []

    averages: dict[int, pd.Series] = {}
    for window in params.get("sma_windows", [10, 20, 50, 200]):
        averages[window] = sma(close, window)
        name = f"px_to_sma_{window}"
        columns[name] = close / averages[window] - 1.0
        specs.append(
            FeatureSpec(
                name=name,
                group=GROUP,
                description=f"Price relative to its {window}-session simple moving average",
                formula=f"C_t / SMA_{window}(C)_t - 1",
                rationale=(
                    "Distance from a trailing average is the scale-free form of trend "
                    "position; the raw average level is not comparable across tickers"
                ),
                lookback=window,
            )
        )

    for short, long in params.get("sma_ratio_pairs", [[20, 50], [50, 200]]):
        short_avg = averages.get(short, sma(close, short))
        long_avg = averages.get(long, sma(close, long))
        name = f"sma_ratio_{short}_{long}"
        columns[name] = short_avg / long_avg - 1.0
        specs.append(
            FeatureSpec(
                name=name,
                group=GROUP,
                description=f"{short}-session average relative to the {long}-session average",
                formula=f"SMA_{short}(C)_t / SMA_{long}(C)_t - 1",
                rationale=(
                    "Continuous form of the moving-average crossover: sign gives the "
                    "trend direction, magnitude gives its separation"
                ),
                lookback=long,
            )
        )

    rsi_window = int(params.get("rsi_window", 14))
    columns[f"rsi_{rsi_window}"] = rsi(close, rsi_window) / 100.0
    specs.append(
        FeatureSpec(
            name=f"rsi_{rsi_window}",
            group=GROUP,
            description=(
                f"Wilder relative strength index over {rsi_window} sessions, scaled to [0, 1]"
            ),
            formula="100 - 100 / (1 + avg_gain / avg_loss), divided by 100",
            rationale=(
                "Bounded measure of how one-sided recent moves have been; extremes are "
                "associated with short-term mean reversion"
            ),
            lookback=rsi_window * 3,
        )
    )

    macd_params = params.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    line, signal_line, histogram = macd(
        close, macd_params["fast"], macd_params["slow"], macd_params["signal"]
    )
    columns["macd_norm"] = line / close
    columns["macd_hist_norm"] = histogram / close
    specs.extend(
        [
            FeatureSpec(
                name="macd_norm",
                group=GROUP,
                description="MACD line divided by price",
                formula=f"(EMA_{macd_params['fast']}(C) - EMA_{macd_params['slow']}(C)) / C_t",
                rationale=(
                    "Difference of two exponential averages responds faster than a simple "
                    "crossover; dividing by price makes it comparable across tickers"
                ),
                lookback=macd_params["slow"] * 3,
            ),
            FeatureSpec(
                name="macd_hist_norm",
                group=GROUP,
                description="MACD histogram divided by price",
                formula="(MACD_t - signal_t) / C_t",
                rationale="Rate of change of the MACD line, an early indication of a turning trend",
                lookback=macd_params["slow"] * 3 + macd_params["signal"],
            ),
        ]
    )

    skip = params.get("skip_month_momentum", {"lookback": 252, "skip": 21})
    lookback, skip_days = int(skip["lookback"]), int(skip["skip"])
    columns["mom_12_1"] = pd.Series(
        np.log(close.shift(skip_days).to_numpy() / close.shift(lookback).to_numpy()),
        index=close.index,
    )
    specs.append(
        FeatureSpec(
            name="mom_12_1",
            group=GROUP,
            description=(
                f"Return from {lookback} to {skip_days} sessions ago, "
                "skipping the most recent month"
            ),
            formula=f"log(C_(t-{skip_days}) / C_(t-{lookback}))",
            rationale=(
                "The classic cross-sectional momentum construction; the recent month is "
                "skipped because it carries short-term reversal that offsets the momentum effect"
            ),
            lookback=lookback + 1,
        )
    )

    position_window = int(params.get("price_position_window", 252))
    columns[f"price_position_{position_window}"] = price_position(close, position_window)
    specs.append(
        FeatureSpec(
            name=f"price_position_{position_window}",
            group=GROUP,
            description=f"Where price sits within its {position_window}-session range",
            formula=(
                f"(C_t - min_{position_window}(C)) / "
                f"(max_{position_window}(C) - min_{position_window}(C))"
            ),
            rationale=(
                "Proximity to a 52-week extreme is a documented anchor for investor "
                "behaviour and is naturally bounded"
            ),
            lookback=position_window,
        )
    )

    return BuildResult(frame=pd.DataFrame(columns, index=frame.index), specs=specs)
