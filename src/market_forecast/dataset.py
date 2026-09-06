"""Assembles the pooled panel, indexed by (date, ticker) so splits are taken on dates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecast.config import AppConfig
from market_forecast.data.loader import MarketDataLoader
from market_forecast.features.pipeline import ContextBuilder, FeaturePipeline
from market_forecast.features.registry import FeatureRegistry
from market_forecast.logging import get_logger
from market_forecast.targets import build_targets

logger = get_logger(__name__)

META_COLUMNS = ("group", "sector")


@dataclass
class Panel:
    frame: pd.DataFrame
    feature_names: list[str]
    target_names: list[str]
    registry: FeatureRegistry
    feature_version: str
    horizons: list[int]
    tickers: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame.index.get_level_values("date").unique()).sort_values()

    def features(self) -> pd.DataFrame:
        return self.frame[self.feature_names]

    def subset(self, group: str) -> Panel:
        mask = self.frame["group"] == group
        return Panel(
            frame=self.frame[mask],
            feature_names=self.feature_names,
            target_names=self.target_names,
            registry=self.registry,
            feature_version=self.feature_version,
            horizons=self.horizons,
            tickers=sorted(self.frame[mask].index.get_level_values("ticker").unique()),
        )

    def describe(self) -> str:
        dates = self.dates
        return (
            f"{len(self.frame):,} rows | {len(self.tickers)} tickers | "
            f"{len(self.feature_names)} features | "
            f"{dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}"
        )

    def to_parquet(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(path, engine="pyarrow")
        metadata = {
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "feature_version": self.feature_version,
            "horizons": self.horizons,
            "tickers": self.tickers,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_panel(
    config: AppConfig,
    loader: MarketDataLoader,
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> Panel:
    universe = config.universe
    selected = tickers or universe.all_tickers

    pipeline = FeaturePipeline(config.features)
    context_builder = ContextBuilder(loader, universe)
    benchmark = loader.load(universe.benchmark, start=start, end=end, raise_on_error=False).frame

    blocks: list[pd.DataFrame] = []
    registry: FeatureRegistry | None = None
    feature_names: list[str] = []
    target_names: list[str] = []
    skipped: dict[str, str] = {}
    included: list[str] = []

    for ticker in selected:
        try:
            prices = loader.load(ticker, start=start, end=end).frame
            matrix = pipeline.build(ticker, prices, context_builder.for_ticker(ticker))
            targets = build_targets(
                ticker,
                prices,
                config.targets.horizons,
                benchmark_prices=benchmark,
            )
        except Exception as exc:
            logger.error("skipping %s: %s", ticker, exc)
            skipped[ticker] = str(exc)
            continue

        if matrix.frame.empty:
            skipped[ticker] = "no rows survived the feature warm-up"
            continue

        if registry is None:
            registry = matrix.registry
            feature_names = matrix.feature_names
            target_names = list(targets.frame.columns)
        elif matrix.feature_names != feature_names:
            missing = set(feature_names) ^ set(matrix.feature_names)
            skipped[ticker] = f"feature set differs: {sorted(missing)}"
            logger.error("skipping %s: %s", ticker, skipped[ticker])
            continue

        block = matrix.frame.join(targets.frame, how="left")
        block["group"] = universe.group_of(ticker) or "unassigned"
        block["sector"] = universe.sector_map.get(ticker, "unknown")
        block["ticker"] = ticker
        block = block.set_index("ticker", append=True)
        block.index.names = ["date", "ticker"]
        blocks.append(block)
        included.append(ticker)
        logger.info("%s | %s", matrix.describe(), block["group"].iloc[0])

    if not blocks:
        raise RuntimeError("no tickers produced a usable feature matrix")
    if registry is None:
        raise RuntimeError("feature registry was never populated")

    frame = pd.concat(blocks).sort_index()
    ordered = feature_names + target_names + list(META_COLUMNS)
    frame = frame[[c for c in ordered if c in frame.columns]]

    return Panel(
        frame=frame,
        feature_names=feature_names,
        target_names=target_names,
        registry=registry,
        feature_version=config.features.version,
        horizons=config.targets.horizons,
        tickers=included,
        skipped=skipped,
    )


def label_coverage(panel: Panel, target: str, horizon: int) -> dict[str, Any]:
    column = f"{target}_{horizon}d"
    labels = panel.frame[column]
    valid = labels.dropna()
    return {
        "target": column,
        "labelled_rows": int(len(valid)),
        "unlabelled_rows": int(labels.isna().sum()),
        "base_rate": float(valid.mean()) if len(valid) else float("nan"),
    }


def load_panel(path: Path) -> Panel:
    """Read a panel written by :meth:`Panel.to_parquet`, with its sidecar metadata."""
    frame = pd.read_parquet(path)
    sidecar = path.with_suffix(".meta.json")
    if not sidecar.exists():
        raise FileNotFoundError(f"missing panel metadata: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    return Panel(
        frame=frame,
        feature_names=metadata["feature_names"],
        target_names=metadata["target_names"],
        registry=FeatureRegistry(),
        feature_version=metadata["feature_version"],
        horizons=metadata["horizons"],
        tickers=metadata["tickers"],
    )
