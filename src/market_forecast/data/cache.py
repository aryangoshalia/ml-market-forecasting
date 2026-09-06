"""On-disk parquet cache for daily bars, with incremental refresh."""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from market_forecast.data.base import (
    DATE_INDEX_NAME,
    OHLCV_COLUMNS,
    DataError,
    coerce_schema,
    empty_ohlcv,
    normalise_ticker,
)
from market_forecast.data.sessions import last_closed_session
from market_forecast.logging import get_logger

logger = get_logger(__name__)


def _safe_filename(ticker: str) -> str:
    return f"{ticker.replace('^', 'INDEX_')}.parquet"


class ParquetCache:
    def __init__(self, root: Path, provider: str, ttl_hours: float = 12.0) -> None:
        self.root = Path(root).resolve()
        self.provider = provider
        self.ttl_seconds = ttl_hours * 3600.0
        self.directory = (self.root / provider).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, ticker: str) -> Path:
        symbol = normalise_ticker(ticker)
        candidate = (self.directory / _safe_filename(symbol)).resolve()
        if candidate.parent != self.directory:
            raise DataError(f"refusing to resolve cache path outside {self.directory}")
        return candidate

    def read(self, ticker: str) -> pd.DataFrame:
        path = self.path_for(ticker)
        if not path.exists():
            return empty_ohlcv()
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("unreadable cache file %s (%s), discarding", path.name, exc)
            path.unlink(missing_ok=True)
            return empty_ohlcv()
        return coerce_schema(frame, ticker)

    def write(self, ticker: str, frame: pd.DataFrame) -> None:
        path = self.path_for(ticker)
        payload = coerce_schema(frame, ticker)
        tmp = path.with_suffix(".parquet.tmp")
        payload.to_parquet(tmp, engine="pyarrow", index=True)
        tmp.replace(path)

    def merge(self, ticker: str, fresh: pd.DataFrame) -> pd.DataFrame:
        """Overlay newly fetched rows on the cache, preferring the fresh values."""
        existing = self.read(ticker)
        incoming = coerce_schema(fresh, ticker)
        if incoming.empty:
            return existing
        if existing.empty:
            combined = incoming
        else:
            combined = pd.concat([existing, incoming])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        self.write(ticker, combined)
        return combined

    def age_seconds(self, ticker: str) -> float | None:
        path = self.path_for(ticker)
        if not path.exists():
            return None
        return time.time() - path.stat().st_mtime

    def is_fresh(self, ticker: str, calendar: str = "XNYS", now: datetime | None = None) -> bool:
        """Fresh means written within the TTL and already covering the last closed session."""
        age = self.age_seconds(ticker)
        if age is None or age > self.ttl_seconds:
            return False
        cached = self.read(ticker)
        if cached.empty:
            return False
        return cached.index[-1] >= last_closed_session(calendar, now)

    def coverage(self, ticker: str) -> tuple[date, date] | None:
        cached = self.read(ticker)
        if cached.empty:
            return None
        return cached.index[0].date(), cached.index[-1].date()

    def clear(self, ticker: str | None = None) -> None:
        if ticker is None:
            for path in self.directory.glob("*.parquet"):
                path.unlink(missing_ok=True)
            return
        self.path_for(ticker).unlink(missing_ok=True)


def build_frame(rows: dict[str, list[float]], index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(rows, index=index, columns=list(OHLCV_COLUMNS))
    frame.index.name = DATE_INDEX_NAME
    return frame
