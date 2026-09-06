"""Calendar features. Weak but documented seasonal effects, encoded without implying an ordering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from market_forecast.features.registry import BuildResult, FeatureSpec

GROUP = "calendar"

_WEEKDAYS = {1: "tue", 2: "wed", 3: "thu", 4: "fri"}


def build(frame: pd.DataFrame, params: dict[str, Any]) -> BuildResult:
    index = pd.DatetimeIndex(frame.index)
    columns: dict[str, pd.Series] = {}
    specs: list[FeatureSpec] = []

    if params.get("day_of_week", True):
        weekday = pd.Series(index.dayofweek, index=index)
        for code, label in _WEEKDAYS.items():
            name = f"dow_{label}"
            columns[name] = (weekday == code).astype("float64")
            specs.append(
                FeatureSpec(
                    name=name,
                    group=GROUP,
                    description=f"Indicator for {label}",
                    formula=f"1[weekday(t) == {code}]",
                    rationale=(
                        "Day-of-week effects are small but documented; Monday is the omitted "
                        "baseline so the dummies are not collinear with an intercept"
                    ),
                    lookback=1,
                    leakage_note="Derived from the timestamp of row t itself.",
                )
            )

    if params.get("month", True):
        month = pd.Series(index.month, index=index).astype("float64")
        radians = 2.0 * np.pi * month.to_numpy() / 12.0
        columns["month_sin"] = pd.Series(np.sin(radians), index=index)
        columns["month_cos"] = pd.Series(np.cos(radians), index=index)
        specs.extend(
            [
                FeatureSpec(
                    name="month_sin",
                    group=GROUP,
                    description="Sine component of the month-of-year cycle",
                    formula="sin(2 pi month / 12)",
                    rationale=(
                        "Encodes seasonality as a cycle so December and January are adjacent, "
                        "which an integer month code gets wrong"
                    ),
                    lookback=1,
                    leakage_note="Derived from the timestamp of row t itself.",
                ),
                FeatureSpec(
                    name="month_cos",
                    group=GROUP,
                    description="Cosine component of the month-of-year cycle",
                    formula="cos(2 pi month / 12)",
                    rationale="Second component needed to make the cyclic encoding one-to-one",
                    lookback=1,
                    leakage_note="Derived from the timestamp of row t itself.",
                ),
            ]
        )

    if params.get("days_to_month_end", True):
        month_end = pd.DatetimeIndex(index.to_period("M").to_timestamp("M"))
        remaining = (month_end - index).days.to_numpy(dtype="float64")
        columns["days_to_month_end"] = pd.Series(remaining / 31.0, index=index)
        specs.append(
            FeatureSpec(
                name="days_to_month_end",
                group=GROUP,
                description="Calendar days until month end, scaled to roughly [0, 1]",
                formula="(month_end(t) - t) / 31",
                rationale=(
                    "The turn of the month coincides with recurring flows from pension and "
                    "index rebalancing, a documented seasonal pattern"
                ),
                lookback=1,
                leakage_note=(
                    "Uses only the calendar position of row t; the month-end date is known "
                    "in advance and is not an observation."
                ),
            )
        )

    return BuildResult(frame=pd.DataFrame(columns, index=index), specs=specs)
