"""Alpha Vantage provider. Optional; the key is read from ALPHAVANTAGE_API_KEY only."""

from __future__ import annotations

import os
from datetime import date

import httpx
import pandas as pd

from market_forecast.data.base import (
    AssetMetadata,
    ProviderError,
    coerce_schema,
    normalise_ticker,
)
from market_forecast.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"
_ENV_VAR = "ALPHAVANTAGE_API_KEY"


class AlphaVantageProvider:
    name = "alphavantage"

    def __init__(self, timeout: float = 30.0) -> None:
        key = os.environ.get(_ENV_VAR, "").strip()
        if not key:
            raise ProviderError(f"{_ENV_VAR} is not set")
        self._key = key
        self.timeout = timeout

    def get_ohlcv(self, ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
        symbol = normalise_ticker(ticker)
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "json",
            "apikey": self._key,
        }
        try:
            response = httpx.get(_BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise ProviderError(f"{symbol}: alphavantage request failed") from exc

        series = payload.get("Time Series (Daily)")
        if not series:
            note = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
            raise ProviderError(f"{symbol}: alphavantage returned no series ({note})")

        frame = pd.DataFrame.from_dict(series, orient="index").rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. adjusted close": "adj_close",
                "6. volume": "volume",
            }
        )
        frame.index = pd.to_datetime(frame.index)
        frame = coerce_schema(frame, symbol)
        upper = pd.Timestamp(end) if end else frame.index[-1]
        return frame.loc[pd.Timestamp(start) : upper]

    def get_metadata(self, ticker: str) -> AssetMetadata:
        return AssetMetadata(ticker=normalise_ticker(ticker))
