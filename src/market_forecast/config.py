"""Typed configuration loaded from the YAML files in ``configs/``."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


def project_root() -> Path:
    env = os.environ.get("MARKET_FORECAST_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a mapping at the top level of {path}")
    return loaded


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataValidationConfig(Frozen):
    min_rows: int = 400
    max_missing_session_ratio: float = 0.02
    max_abs_daily_return: float = 0.60
    allow_zero_volume_ratio: float = 0.02


class DataConfig(Frozen):
    provider: str = "yfinance"
    fallback_provider: str | None = "stooq"
    start_date: date
    end_date: date | None = None
    cache_dir: str = "data/cache"
    cache_ttl_hours: float = 12.0
    exchange_calendar: str = "XNYS"
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    validation: DataValidationConfig = Field(default_factory=DataValidationConfig)

    @property
    def cache_path(self) -> Path:
        raw = Path(self.cache_dir)
        return raw if raw.is_absolute() else project_root() / raw


class UniverseConfig(Frozen):
    benchmark: str
    volatility_index: str
    groups: dict[str, list[str]]
    sector_etfs: dict[str, str]
    sector_map: dict[str, str]

    @field_validator("groups")
    @classmethod
    def _disjoint_groups(cls, groups: dict[str, list[str]]) -> dict[str, list[str]]:
        seen: dict[str, str] = {}
        for name, tickers in groups.items():
            for ticker in tickers:
                if ticker in seen:
                    raise ValueError(f"{ticker} appears in both '{seen[ticker]}' and '{name}'")
                seen[ticker] = name
        return groups

    @property
    def all_tickers(self) -> list[str]:
        return sorted({t for tickers in self.groups.values() for t in tickers})

    def group_of(self, ticker: str) -> str | None:
        for name, tickers in self.groups.items():
            if ticker in tickers:
                return name
        return None

    def sector_etf_for(self, ticker: str) -> str | None:
        sector = self.sector_map.get(ticker)
        return self.sector_etfs.get(sector) if sector else None

    @property
    def support_tickers(self) -> list[str]:
        return sorted({self.benchmark, self.volatility_index, *self.sector_etfs.values()})


class ThresholdConfig(Frozen):
    method: str
    ewma_lambda: float = 0.94
    min_periods: int = 60
    target_neutral_fraction: float = 1.0 / 3.0
    kappa_search: tuple[float, float] = (0.05, 1.5)
    lower_quantile: float = 1.0 / 3.0
    upper_quantile: float = 2.0 / 3.0


class TargetDefinition(Frozen):
    kind: str
    return_basis: str
    description: str
    benchmark: str | None = None
    threshold: ThresholdConfig | None = None


class TargetsConfig(Frozen):
    horizons: list[int]
    definitions: dict[str, TargetDefinition]
    primary: str
    secondary: str | None = None

    @field_validator("horizons")
    @classmethod
    def _positive(cls, horizons: list[int]) -> list[int]:
        if not horizons or any(h < 1 for h in horizons):
            raise ValueError("horizons must be positive integers")
        return horizons


class FeatureConfig(Frozen):
    version: str
    groups: dict[str, dict[str, Any]]
    warmup_policy: str = "drop"
    max_forward_fill_sessions: int = 3

    def group(self, name: str) -> dict[str, Any]:
        return self.groups.get(name, {})

    def is_enabled(self, name: str) -> bool:
        return bool(self.groups.get(name, {}).get("enabled", False))


class WalkForwardConfig(Frozen):
    scheme: str = "anchored"
    rolling_train_sessions: int = 1260
    initial_train_sessions: int = 1512
    test_sessions: int = 126
    step_sessions: int = 126
    purge_sessions: int | None = None
    embargo_sessions: int = 5
    inner_validation_sessions: int = 120
    seed: int = 17

    def purge_for(self, horizon: int) -> int:
        return horizon if self.purge_sessions is None else self.purge_sessions


class AppConfig(Frozen):
    data: DataConfig
    universe: UniverseConfig
    targets: TargetsConfig
    features: FeatureConfig
    walkforward: WalkForwardConfig

    @classmethod
    def load(cls, directory: Path | None = None) -> AppConfig:
        base = directory or config_dir()
        return cls(
            data=DataConfig(**load_yaml(base / "data.yaml")),
            universe=UniverseConfig(**load_yaml(base / "universe.yaml")),
            targets=TargetsConfig(**load_yaml(base / "targets.yaml")),
            features=FeatureConfig(**load_yaml(base / "features.yaml")),
            walkforward=WalkForwardConfig(**load_yaml(base / "walkforward.yaml")),
        )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig.load()
