"""Market-level features describing the environment all tickers share.

Regimes are fitted on the market, not on individual stocks, so one regime series
conditions every ticker and stays comparable across them. Every column is causal at t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_forecast.features.indicators import (
    log_return,
    realised_volatility,
    rolling_zscore,
)
from market_forecast.logging import get_logger

logger = get_logger(__name__)

REGIME_FEATURES = (
    "mkt_ret_20",
    "mkt_ret_60",
    "mkt_vol_20",
    "mkt_vol_ratio_20_60",
    "vix_z_60",
    "breadth_above_sma50",
    "mkt_drawdown_252",
)


@dataclass
class RegimeFeatureConfig:
    return_windows: tuple[int, int] = (20, 60)
    vol_windows: tuple[int, int] = (20, 60)
    vix_zscore_window: int = 60
    drawdown_window: int = 252


def build_regime_features(
    benchmark: pd.DataFrame,
    volatility_index: pd.DataFrame | None = None,
    breadth: pd.Series | None = None,
    config: RegimeFeatureConfig | None = None,
) -> pd.DataFrame:
    """One row per session describing the market environment observable at that close."""
    settings = config or RegimeFeatureConfig()
    close = benchmark["close"]
    returns = log_return(close, 1)
    short_return, long_return = settings.return_windows
    short_vol, long_vol = settings.vol_windows

    columns: dict[str, pd.Series] = {
        f"mkt_ret_{short_return}": log_return(close, short_return),
        f"mkt_ret_{long_return}": log_return(close, long_return),
        f"mkt_vol_{short_vol}": realised_volatility(returns, short_vol),
    }
    slow = realised_volatility(returns, long_vol)
    columns[f"mkt_vol_ratio_{short_vol}_{long_vol}"] = columns[
        f"mkt_vol_{short_vol}"
    ] / slow.replace(0.0, np.nan)

    peak = close.rolling(settings.drawdown_window, min_periods=settings.drawdown_window).max()
    columns[f"mkt_drawdown_{settings.drawdown_window}"] = close / peak - 1.0

    if volatility_index is not None and not volatility_index.empty:
        vix = volatility_index["close"].reindex(close.index).ffill(limit=3)
        columns[f"vix_z_{settings.vix_zscore_window}"] = rolling_zscore(
            vix, settings.vix_zscore_window
        )

    if breadth is not None:
        columns["breadth_above_sma50"] = breadth.reindex(close.index).ffill(limit=3)

    frame = pd.DataFrame(columns, index=close.index).replace([np.inf, -np.inf], np.nan)
    return frame.dropna()


def breadth_from_panel(panel_frame: pd.DataFrame, column: str = "px_to_sma_50") -> pd.Series:
    """Fraction of the universe trading above its own 50-session average.

    Computed from the per-ticker feature that already exists, so it inherits that
    feature's causality rather than introducing a second definition.
    """
    if column not in panel_frame.columns:
        raise KeyError(f"panel has no {column!r} column")
    above = (panel_frame[column] > 0.0).groupby(level="date").mean()
    return above.rename("breadth_above_sma50")
