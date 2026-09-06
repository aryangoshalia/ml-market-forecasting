"""Stooq provider. Serves prices already adjusted, so close and adj_close are identical.

The CSV endpoint is now behind a browser-verification challenge for some clients; this
provider detects that and raises rather than trying to defeat it."""

from __future__ import annotations

import io
from datetime import date

import httpx
import pandas as pd

from market_forecast.data.base import (
    AssetMetadata,
    ProviderError,
    coerce_schema,
    is_index_symbol,
    normalise_ticker,
)
from market_forecast.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://stooq.com/q/d/l/"
_INDEX_ALIASES = {"^VIX": "^VIX", "^GSPC": "^SPX"}


class StooqProvider:
    name = "stooq"

    def __init__(self, timeout: float = 30.0, max_retries: int = 2) -> None:
        self.timeout = timeout
        self.max_retries = max_retries

    def _symbol_for_stooq(self, ticker: str) -> str:
        if is_index_symbol(ticker):
            return _INDEX_ALIASES.get(ticker, ticker).lower()
        return f"{ticker.replace('-', '.').lower()}.us"

    def get_ohlcv(self, ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
        symbol = normalise_ticker(ticker)
        params = {
            "s": self._symbol_for_stooq(symbol),
            "d1": start.strftime("%Y%m%d"),
            "d2": (end or date.today()).strftime("%Y%m%d"),
            "i": "d",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.get(_BASE_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
            except Exception as exc:
                last_error = exc
                logger.warning("%s: stooq attempt %d failed (%s)", symbol, attempt, exc)
                continue

            text = response.text
            if _is_bot_challenge(text):
                raise ProviderError(
                    f"{symbol}: stooq served a browser-verification challenge; "
                    "use a different provider"
                )
            if not text.startswith("Date"):
                last_error = ProviderError(f"{symbol}: stooq returned {text[:60]!r}")
                continue

            frame = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
            if frame.empty:
                last_error = ProviderError(f"{symbol}: stooq returned no rows")
                continue
            frame["Adj Close"] = frame["Close"]
            return coerce_schema(frame, symbol)

        raise ProviderError(
            f"{symbol}: stooq failed after {self.max_retries} attempts"
        ) from last_error

    def get_metadata(self, ticker: str) -> AssetMetadata:
        return AssetMetadata(ticker=normalise_ticker(ticker))


def _is_bot_challenge(text: str) -> bool:
    head = text[:400].lower()
    return "<!doctype html" in head or "requires javascript" in head
