"""Fetch, cache, validate and corporate-action-adjust daily bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from market_forecast.config import DataConfig
from market_forecast.data.base import (
    DataValidationError,
    MarketDataProvider,
    ProviderError,
    normalise_ticker,
)
from market_forecast.data.cache import ParquetCache
from market_forecast.data.providers import build_provider
from market_forecast.data.sessions import unexpected_sessions
from market_forecast.data.validate import ValidationReport, validate_ohlcv
from market_forecast.logging import get_logger

logger = get_logger(__name__)

# Refetch a short overlap so vendor restatements of recent bars are picked up.
_REFETCH_OVERLAP_DAYS = 10

ADJUSTED_COLUMNS = ("open", "high", "low", "close")


@dataclass
class LoadResult:
    ticker: str
    frame: pd.DataFrame
    report: ValidationReport
    provider: str
    from_cache: bool

    @property
    def last_session(self) -> pd.Timestamp | None:
        return self.frame.index[-1] if len(self.frame) else None


def drop_non_sessions(frame: pd.DataFrame, ticker: str, calendar: str = "XNYS") -> pd.DataFrame:
    """Remove rows dated on days the exchange was closed.

    Vendors sometimes publish a live quote on a holiday. Index symbols are the usual
    offender because they are computed rather than traded. The exchange calendar decides
    what counts as a session, so those rows are dropped rather than carried forward as a
    phantom bar that would shift the latest-session date and every trailing window.
    """
    if frame.empty:
        return frame
    stray = unexpected_sessions(pd.DatetimeIndex(frame.index), calendar)
    if len(stray) == 0:
        return frame
    logger.warning(
        "%s: dropping %d rows dated outside exchange sessions (%s)",
        ticker,
        len(stray),
        ", ".join(d.strftime("%Y-%m-%d") for d in stray[:3]),
    )
    return frame.drop(index=stray)


def adjust_for_corporate_actions(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale open/high/low by the close-to-adjusted-close ratio.

    Yahoo delivers OHLC already adjusted for splits but not dividends, so this factor
    is the dividend adjustment. Applying it to the whole bar keeps high, low and close
    on one consistent scale, which range-based estimators such as ATR and Parkinson
    volatility require. Volume is left untouched because it is already split adjusted
    and dividends do not change share counts.
    """
    out = frame.copy()
    factor = (out["adj_close"] / out["close"]).replace([float("inf"), -float("inf")], pd.NA)
    factor = factor.astype("float64").ffill().fillna(1.0)
    for column in ADJUSTED_COLUMNS:
        out[column] = out[column] * factor
    out["adj_factor"] = factor
    return out


class MarketDataLoader:
    def __init__(
        self,
        config: DataConfig,
        provider: MarketDataProvider | None = None,
        cache: ParquetCache | None = None,
    ) -> None:
        self.config = config
        self.provider = provider or build_provider(config.provider, config)
        self.cache = cache or ParquetCache(
            root=config.cache_path,
            provider=self.provider.name,
            ttl_hours=config.cache_ttl_hours,
        )

    def load(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
        force_refresh: bool = False,
        validate: bool = True,
        raise_on_error: bool = True,
    ) -> LoadResult:
        symbol = normalise_ticker(ticker)
        window_start = start or self.config.start_date
        window_end = end or self.config.end_date

        from_cache = True
        if force_refresh or not self.cache.is_fresh(symbol, self.config.exchange_calendar):
            from_cache = False
            self._refresh(symbol, window_start, window_end, force_refresh)

        frame = self.cache.read(symbol)
        frame = drop_non_sessions(frame, symbol, self.config.exchange_calendar)
        if window_start:
            frame = frame.loc[pd.Timestamp(window_start) :]
        if window_end:
            frame = frame.loc[: pd.Timestamp(window_end)]

        report = (
            validate_ohlcv(
                frame,
                symbol,
                self.config.validation,
                calendar=self.config.exchange_calendar,
                check_staleness=window_end is None,
            )
            if validate
            else ValidationReport(ticker=symbol, n_rows=len(frame))
        )

        if validate and not report.ok:
            logger.error("%s failed validation: %s", symbol, "; ".join(report.errors))
            if raise_on_error:
                raise DataValidationError(f"{symbol}: {'; '.join(report.errors)}")
        for warning in report.warnings:
            logger.warning("%s: %s", symbol, warning)

        return LoadResult(
            ticker=symbol,
            frame=adjust_for_corporate_actions(frame) if len(frame) else frame,
            report=report,
            provider=self.provider.name,
            from_cache=from_cache,
        )

    def _refresh(self, symbol: str, start: date, end: date | None, force_refresh: bool) -> None:
        coverage = None if force_refresh else self.cache.coverage(symbol)
        fetch_start = start
        if coverage is not None and coverage[0] <= start:
            fetch_start = coverage[1] - timedelta(days=_REFETCH_OVERLAP_DAYS)

        logger.info("fetching %s from %s (%s)", symbol, fetch_start, self.provider.name)
        try:
            fresh = self.provider.get_ohlcv(symbol, fetch_start, end)
        except ProviderError:
            if self.cache.coverage(symbol) is None:
                raise
            logger.warning("%s: fetch failed, serving cached data", symbol)
            return
        self.cache.merge(symbol, fresh)

    def load_many(
        self,
        tickers: list[str],
        start: date | None = None,
        end: date | None = None,
        force_refresh: bool = False,
        skip_failures: bool = True,
    ) -> dict[str, LoadResult]:
        results: dict[str, LoadResult] = {}
        for ticker in tickers:
            try:
                results[normalise_ticker(ticker)] = self.load(
                    ticker, start=start, end=end, force_refresh=force_refresh
                )
            except Exception as exc:
                if not skip_failures:
                    raise
                logger.error("skipping %s: %s", ticker, exc)
        return results


def default_loader(config: DataConfig, cache_root: Path | None = None) -> MarketDataLoader:
    provider = build_provider(config.provider, config)
    cache = ParquetCache(
        root=cache_root or config.cache_path,
        provider=provider.name,
        ttl_hours=config.cache_ttl_hours,
    )
    return MarketDataLoader(config, provider=provider, cache=cache)
