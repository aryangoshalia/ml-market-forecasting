"""Yahoo Finance provider. Returns unadjusted OHLCV plus the adjusted close."""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from market_forecast.data.base import (
    AssetMetadata,
    ProviderError,
    coerce_schema,
    empty_ohlcv,
    normalise_ticker,
)
from market_forecast.logging import get_logger

logger = get_logger(__name__)


class YFinanceProvider:
    name = "yfinance"

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def get_ohlcv(self, ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
        symbol = normalise_ticker(ticker)
        # yfinance treats `end` as exclusive; extend it so the final session is included.
        end_exclusive = (end + timedelta(days=1)) if end else None

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = yf.download(
                    symbol,
                    start=start.isoformat(),
                    end=end_exclusive.isoformat() if end_exclusive else None,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=False,
                    timeout=self.timeout,
                )
            except Exception as exc:
                last_error = exc
                logger.warning("%s: download attempt %d failed (%s)", symbol, attempt, exc)
                time.sleep(self.backoff_seconds * attempt)
                continue

            if raw is None or raw.empty:
                last_error = ProviderError(f"{symbol}: empty response")
                logger.warning("%s: empty response on attempt %d", symbol, attempt)
                time.sleep(self.backoff_seconds * attempt)
                continue

            return coerce_schema(_flatten_columns(raw, symbol), symbol)

        raise ProviderError(f"{symbol}: no data after {self.max_retries} attempts") from last_error

    def get_metadata(self, ticker: str) -> AssetMetadata:
        symbol = normalise_ticker(ticker)
        try:
            info = yf.Ticker(symbol).get_info()
        except Exception as exc:
            logger.debug("%s: metadata lookup failed (%s)", symbol, exc)
            return AssetMetadata(ticker=symbol)
        return AssetMetadata(
            ticker=symbol,
            name=info.get("longName") or info.get("shortName"),
            exchange=info.get("fullExchangeName") or info.get("exchange"),
            sector=info.get("sector"),
            currency=info.get("currency"),
        )


def _flatten_columns(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinance returns MultiIndex columns keyed by ticker even for single downloads."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [
            lvl for lvl in range(frame.columns.nlevels) if symbol in frame.columns.levels[lvl]
        ]
        if levels:
            extracted = frame.xs(symbol, axis=1, level=levels[0])
            frame = extracted.to_frame() if isinstance(extracted, pd.Series) else extracted
        else:
            frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        return empty_ohlcv()
    return frame
