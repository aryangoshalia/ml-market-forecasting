"""Provider construction by name."""

from __future__ import annotations

from pathlib import Path

from market_forecast.config import DataConfig, project_root
from market_forecast.data.base import MarketDataProvider
from market_forecast.data.csv_provider import CsvProvider
from market_forecast.data.stooq_provider import StooqProvider
from market_forecast.data.yfinance_provider import YFinanceProvider

PROVIDER_NAMES = ("yfinance", "stooq", "csv", "alphavantage")


def build_provider(
    name: str, config: DataConfig, csv_directory: Path | None = None
) -> MarketDataProvider:
    key = name.lower()
    if key == "yfinance":
        return YFinanceProvider(
            timeout=config.request_timeout_seconds,
            max_retries=config.max_retries,
            backoff_seconds=config.retry_backoff_seconds,
        )
    if key == "stooq":
        return StooqProvider(timeout=config.request_timeout_seconds, max_retries=config.max_retries)
    if key == "csv":
        return CsvProvider(csv_directory or project_root() / "data" / "csv")
    if key == "alphavantage":
        from market_forecast.data.alphavantage_provider import AlphaVantageProvider

        return AlphaVantageProvider(timeout=config.request_timeout_seconds)
    raise ValueError(f"unknown provider: {name!r}; expected one of {PROVIDER_NAMES}")
