"""Synthetic price generation for offline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_SESSIONS = 1600


def synthetic_prices(
    sessions: int = TRADING_SESSIONS,
    seed: int = 7,
    start: str = "2015-01-02",
    initial_price: float = 100.0,
    daily_volatility: float = 0.013,
    drift: float = 0.0003,
) -> pd.DataFrame:
    """A geometric random walk with plausible intraday ranges and volume."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=sessions, name="date")

    returns = rng.normal(drift, daily_volatility, sessions)
    close = initial_price * np.exp(np.cumsum(returns))
    gaps = rng.normal(0.0, daily_volatility / 3.0, sessions)
    open_ = close * np.exp(-returns + gaps)
    spread = np.abs(rng.normal(0.0, daily_volatility, sessions))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = np.abs(rng.lognormal(16.0, 0.4, sessions))

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
            "adj_factor": np.ones(sessions),
        },
        index=index,
    )
