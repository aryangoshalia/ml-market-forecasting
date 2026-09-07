"""Describing regimes from their fitted statistics.

States are numbered by the estimator, and the numbering carries no meaning. A label is
produced only after looking at what each state actually did, so no cluster is called
bullish or bearish before the numbers say it behaved that way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def describe_states(
    features: pd.DataFrame, labels: pd.Series, benchmark_returns: pd.Series | None = None
) -> pd.DataFrame:
    aligned = labels.reindex(features.index).dropna()
    rows = []
    for state in sorted(aligned.unique()):
        selected = features.loc[aligned[aligned == state].index]
        row: dict[str, float | int | str] = {
            "state": int(state),
            "sessions": int(len(selected)),
            "share": float(len(selected) / len(aligned)),
        }
        row.update({column: float(selected[column].mean()) for column in features.columns})
        if benchmark_returns is not None:
            realised = benchmark_returns.reindex(selected.index).dropna()
            if len(realised):
                row["mean_daily_return"] = float(realised.mean())
                row["annualised_return"] = float(realised.mean() * TRADING_DAYS)
                row["annualised_vol"] = float(realised.std(ddof=1) * np.sqrt(TRADING_DAYS))
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def label_states(summary: pd.DataFrame) -> dict[int, str]:
    """Generate a readable name for each state from its own statistics."""
    labels: dict[int, str] = {}
    vol_column = next((c for c in summary.columns if c.startswith("mkt_vol_")), None)
    ret_column = next((c for c in summary.columns if c.startswith("mkt_ret_")), None)

    vol_rank = summary[vol_column].rank(pct=True) if vol_column else None
    for state in summary.index:
        parts: list[str] = []
        if vol_rank is not None:
            share = float(vol_rank.loc[state])
            parts.append(
                "high volatility"
                if share > 0.66
                else "low volatility"
                if share < 0.34
                else "moderate volatility"
            )
        if ret_column is not None:
            mean_return = float(summary.loc[state, ret_column])
            parts.append(
                "negative trend"
                if mean_return < -0.005
                else "positive trend"
                if mean_return > 0.005
                else "sideways"
            )
        labels[int(state)] = " / ".join(parts) if parts else f"state {state}"
    return labels


def regime_timeline(labels: pd.Series, names: dict[int, str] | None = None) -> pd.DataFrame:
    """Contiguous runs of one regime, for plotting and for the interface."""
    values = labels.dropna()
    if values.empty:
        return pd.DataFrame(columns=["state", "name", "start", "end", "sessions"])

    changed = values.ne(values.shift()).cumsum()
    rows = []
    for _, block in values.groupby(changed):
        state = int(block.iloc[0])
        rows.append(
            {
                "state": state,
                "name": (names or {}).get(state, f"state {state}"),
                "start": block.index[0],
                "end": block.index[-1],
                "sessions": len(block),
            }
        )
    return pd.DataFrame(rows)
