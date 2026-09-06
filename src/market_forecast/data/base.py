"""Canonical market-data schema and the provider interface everything downstream depends on."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")
DATE_INDEX_NAME = "date"

# Equity symbols, plus a leading caret for index symbols such as ^VIX. The pattern is
# deliberately narrow: the ticker reaches the filesystem as a cache filename.
_TICKER_PATTERN = re.compile(r"^\^?[A-Z][A-Z0-9]{0,9}(?:[.\-][A-Z0-9]{1,4})?$")
_MAX_TICKER_LENGTH = 16


class DataError(Exception):
    """Base class for data-layer failures."""


class InvalidTickerError(DataError):
    """The supplied symbol is not a well-formed ticker."""


class ProviderError(DataError):
    """A data provider failed to return usable data."""


class DataValidationError(DataError):
    """A downloaded series failed validation."""


def normalise_ticker(raw: str) -> str:
    """Return the canonical uppercase symbol, or raise if it is not well formed."""
    if not isinstance(raw, str):
        raise InvalidTickerError("ticker must be a string")
    candidate = raw.strip().upper()
    if not candidate:
        raise InvalidTickerError("ticker must not be empty")
    if len(candidate) > _MAX_TICKER_LENGTH:
        raise InvalidTickerError(f"ticker too long: {len(candidate)} characters")
    if not _TICKER_PATTERN.match(candidate):
        raise InvalidTickerError(f"malformed ticker: {raw!r}")
    return candidate


def is_index_symbol(ticker: str) -> bool:
    return ticker.startswith("^")


@dataclass(frozen=True)
class AssetMetadata:
    ticker: str
    name: str | None = None
    exchange: str | None = None
    sector: str | None = None
    currency: str | None = None


@runtime_checkable
class MarketDataProvider(Protocol):
    """Everything downstream of this interface is provider agnostic."""

    name: str

    def get_ohlcv(self, ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
        """Daily bars indexed by a tz-naive DatetimeIndex named ``date``."""

    def get_metadata(self, ticker: str) -> AssetMetadata: ...


def empty_ohlcv() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(OHLCV_COLUMNS), dtype="float64")
    frame.index = pd.DatetimeIndex([], name=DATE_INDEX_NAME)
    return frame


def coerce_schema(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Force any provider output into the canonical schema.

    Sorts chronologically, drops duplicate sessions keeping the last observation,
    normalises timestamps to tz-naive midnight, and orders the columns.
    """
    if frame is None or len(frame) == 0:
        return empty_ohlcv()

    out = frame.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    renames = {"adj._close": "adj_close", "adjclose": "adj_close", "adjusted_close": "adj_close"}
    out = out.rename(columns=renames)

    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ProviderError(f"{ticker}: provider output missing columns {missing}")

    index = pd.to_datetime(out.index, errors="coerce", utc=False)
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        index = index.tz_convert(None)
    out.index = pd.DatetimeIndex(index).normalize()
    out.index.name = DATE_INDEX_NAME

    out = out.loc[out.index.notna(), list(OHLCV_COLUMNS)]
    out = out.astype("float64")
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()
