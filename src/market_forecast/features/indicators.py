"""Technical indicator primitives. Every function is causal and returns NaN during warm-up."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    ratio = series.to_numpy() / series.shift(periods).to_numpy()
    return pd.Series(np.log(ratio), index=series.index, name=series.name)


def simple_return(series: pd.Series, periods: int = 1) -> pd.Series:
    return series / series.shift(periods) - 1.0


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_smooth(gain, window)
    avg_loss = wilder_smooth(loss, window)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    return pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), window)


def realised_volatility(returns: pd.Series, window: int, annualise: bool = True) -> pd.Series:
    vol = returns.rolling(window, min_periods=window).std(ddof=1)
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    squared = pd.Series(np.log(high.to_numpy() / low.to_numpy()) ** 2, index=high.index)
    mean_squared = squared.rolling(window, min_periods=window).mean()
    return np.sqrt(mean_squared / (4.0 * np.log(2.0))) * np.sqrt(TRADING_DAYS)


def downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    negative = returns.where(returns < 0.0, 0.0)
    return negative.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return net / path.replace(0.0, np.nan)


def price_position(close: pd.Series, window: int) -> pd.Series:
    lowest = close.rolling(window, min_periods=window).min()
    highest = close.rolling(window, min_periods=window).max()
    span = (highest - lowest).replace(0.0, np.nan)
    return (close - lowest) / span


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=1).replace(0.0, np.nan)
    return (series - mean) / std


def rolling_beta(asset: pd.Series, market: pd.Series, window: int) -> pd.Series:
    covariance = asset.rolling(window, min_periods=window).cov(market)
    variance = market.rolling(window, min_periods=window).var(ddof=1).replace(0.0, np.nan)
    return covariance / variance


def ewma_volatility(returns: pd.Series, lam: float = 0.94, min_periods: int = 60) -> pd.Series:
    """RiskMetrics EWMA volatility of daily returns, causal at t."""
    variance = returns.pow(2).ewm(alpha=1.0 - lam, adjust=False, min_periods=min_periods).mean()
    return np.sqrt(variance)
