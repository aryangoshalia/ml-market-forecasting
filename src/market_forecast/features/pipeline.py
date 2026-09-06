"""Assembles the feature groups into one matrix per ticker."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_forecast.config import FeatureConfig, UniverseConfig
from market_forecast.data.loader import LoadResult, MarketDataLoader
from market_forecast.features import calendar, market, momentum, price, trend, volatility, volume
from market_forecast.features.market import MarketContext
from market_forecast.features.registry import BuildResult, FeatureRegistry, FeatureSpec
from market_forecast.logging import get_logger

logger = get_logger(__name__)

_SIMPLE_GROUPS = {
    "returns": price.build,
    "momentum": momentum.build,
    "volatility": volatility.build,
    "volume": volume.build,
    "trend": trend.build,
}


@dataclass
class FeatureMatrix:
    ticker: str
    frame: pd.DataFrame
    registry: FeatureRegistry
    version: str
    warmup_rows: int
    first_session: pd.Timestamp | None = None
    last_session: pd.Timestamp | None = None
    context_gaps: dict[str, int] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return list(self.frame.columns)

    def describe(self) -> str:
        span = (
            f"{self.first_session:%Y-%m-%d}..{self.last_session:%Y-%m-%d}"
            if self.first_session is not None
            else "empty"
        )
        return (
            f"{self.ticker}: {len(self.frame)} rows x {len(self.frame.columns)} features "
            f"{span} (warm-up {self.warmup_rows})"
        )


class FeaturePipeline:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def build(
        self, ticker: str, prices: pd.DataFrame, context: MarketContext | None = None
    ) -> FeatureMatrix:
        registry = FeatureRegistry()
        parts: list[pd.DataFrame] = []

        for name, builder in _SIMPLE_GROUPS.items():
            if not self.config.is_enabled(name):
                continue
            result = builder(prices, self.config.group(name))
            self._collect(result, registry, parts)

        context_gaps: dict[str, int] = {}
        if self.config.is_enabled("market"):
            if context is None:
                raise ValueError(
                    f"{ticker}: market features are enabled but no context was supplied"
                )
            context_gaps = _context_gaps(context, prices.index)
            result = market.build(
                prices,
                self.config.group("market"),
                context,
                forward_fill_limit=self.config.max_forward_fill_sessions,
            )
            self._collect(result, registry, parts)

        if self.config.is_enabled("calendar"):
            result = calendar.build(prices, self.config.group("calendar"))
            self._collect(result, registry, parts)

        frame = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=prices.index)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.reindex(columns=registry.names)

        warmup = registry.max_lookback()
        if self.config.warmup_policy == "drop" and warmup:
            frame = frame.iloc[warmup:]

        for label, gaps in context_gaps.items():
            if gaps:
                logger.warning(
                    "%s: %s context missing on %d sessions, carried forward at most %d",
                    ticker,
                    label,
                    gaps,
                    self.config.max_forward_fill_sessions,
                )

        return FeatureMatrix(
            ticker=ticker,
            frame=frame,
            registry=registry,
            version=self.config.version,
            warmup_rows=warmup,
            first_session=frame.index[0] if len(frame) else None,
            last_session=frame.index[-1] if len(frame) else None,
            context_gaps=context_gaps,
        )

    @staticmethod
    def _collect(result: BuildResult, registry: FeatureRegistry, parts: list[pd.DataFrame]) -> None:
        _assert_scale_free(result.specs)
        registry.extend(result.specs)
        parts.append(result.frame)


def _assert_scale_free(specs: list[FeatureSpec]) -> None:
    offenders = [s.name for s in specs if not s.scale_free]
    if offenders:
        raise ValueError(f"features are not scale free and cannot be pooled: {offenders}")


def _context_gaps(context: MarketContext, index: pd.Index) -> dict[str, int]:
    gaps = {context.benchmark_symbol: market.staleness(context.benchmark["close"], index)}
    if context.volatility_index is not None and not context.volatility_index.empty:
        label = context.volatility_index_symbol or "vix"
        gaps[label] = market.staleness(context.volatility_index["close"], index)
    if context.sector is not None and not context.sector.empty:
        gaps[context.sector_symbol or "sector"] = market.staleness(context.sector["close"], index)
    return gaps


class ContextBuilder:
    """Loads and caches the shared benchmark, volatility index and sector series."""

    def __init__(self, loader: MarketDataLoader, universe: UniverseConfig) -> None:
        self.loader = loader
        self.universe = universe
        self._cache: dict[str, LoadResult] = {}

    def _series(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            self._cache[symbol] = self.loader.load(symbol, raise_on_error=False)
        return self._cache[symbol].frame

    def for_ticker(self, ticker: str) -> MarketContext:
        sector_etf = self.universe.sector_etf_for(ticker)
        return MarketContext(
            benchmark_symbol=self.universe.benchmark,
            benchmark=self._series(self.universe.benchmark),
            volatility_index=self._series(self.universe.volatility_index),
            volatility_index_symbol=self.universe.volatility_index,
            sector=self._series(sector_etf) if sector_etf else None,
            sector_symbol=sector_etf,
        )
