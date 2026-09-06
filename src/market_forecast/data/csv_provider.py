"""Local CSV provider. Offline source for tests, fixtures and reproducible demos."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from market_forecast.data.base import (
    AssetMetadata,
    ProviderError,
    coerce_schema,
    normalise_ticker,
)


class CsvProvider:
    name = "csv"

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()

    def _path_for(self, ticker: str) -> Path:
        filename = f"{ticker.replace('^', 'INDEX_')}.csv"
        candidate = (self.directory / filename).resolve()
        if candidate.parent != self.directory:
            raise ProviderError(f"refusing to read outside {self.directory}")
        return candidate

    def get_ohlcv(self, ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
        symbol = normalise_ticker(ticker)
        path = self._path_for(symbol)
        if not path.exists():
            raise ProviderError(f"{symbol}: no CSV at {path}")
        frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
        frame = coerce_schema(frame, symbol)
        upper = pd.Timestamp(end) if end else frame.index[-1]
        return frame.loc[pd.Timestamp(start) : upper]

    def get_metadata(self, ticker: str) -> AssetMetadata:
        return AssetMetadata(ticker=normalise_ticker(ticker))
