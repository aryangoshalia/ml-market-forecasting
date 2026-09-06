"""Synthetic fixtures. The unit suite runs offline; network tests are marked separately."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_forecast.config import (
    AppConfig,
    DataConfig,
    FeatureConfig,
    TargetsConfig,
    UniverseConfig,
    WalkForwardConfig,
    config_dir,
    load_yaml,
)
from market_forecast.features.market import MarketContext
from tests.factories import synthetic_prices


@pytest.fixture
def prices() -> pd.DataFrame:
    return synthetic_prices()


@pytest.fixture
def benchmark() -> pd.DataFrame:
    return synthetic_prices(seed=21, initial_price=380.0, daily_volatility=0.009)


@pytest.fixture
def volatility_index() -> pd.DataFrame:
    return synthetic_prices(seed=33, initial_price=18.0, daily_volatility=0.06, drift=0.0)


@pytest.fixture
def sector() -> pd.DataFrame:
    return synthetic_prices(seed=44, initial_price=150.0, daily_volatility=0.011)


@pytest.fixture
def context(benchmark, volatility_index, sector) -> MarketContext:
    return MarketContext(
        benchmark_symbol="SPY",
        benchmark=benchmark,
        volatility_index=volatility_index,
        volatility_index_symbol="^VIX",
        sector=sector,
        sector_symbol="XLK",
    )


@pytest.fixture
def feature_config() -> FeatureConfig:
    return FeatureConfig(**load_yaml(config_dir() / "features.yaml"))


@pytest.fixture
def targets_config() -> TargetsConfig:
    return TargetsConfig(**load_yaml(config_dir() / "targets.yaml"))


@pytest.fixture
def universe_config() -> UniverseConfig:
    return UniverseConfig(**load_yaml(config_dir() / "universe.yaml"))


@pytest.fixture
def walkforward_config() -> WalkForwardConfig:
    return WalkForwardConfig(**load_yaml(config_dir() / "walkforward.yaml"))


@pytest.fixture
def data_config(tmp_path) -> DataConfig:
    payload = load_yaml(config_dir() / "data.yaml")
    payload["cache_dir"] = str(tmp_path / "cache")
    payload["start_date"] = date(2015, 1, 1)
    return DataConfig(**payload)


@pytest.fixture
def app_config(
    data_config, universe_config, targets_config, feature_config, walkforward_config
) -> AppConfig:
    return AppConfig(
        data=data_config,
        universe=universe_config,
        targets=targets_config,
        features=feature_config,
        walkforward=walkforward_config,
    )
