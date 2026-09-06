from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_forecast.config import UniverseConfig, get_config


class TestShippedConfig:
    def test_loads(self):
        assert get_config().features.version

    def test_asset_groups_are_disjoint(self, universe_config):
        seen: set[str] = set()
        for tickers in universe_config.groups.values():
            assert not seen & set(tickers)
            seen |= set(tickers)

    def test_every_ticker_has_a_sector_and_an_etf(self, universe_config):
        for ticker in universe_config.all_tickers:
            assert ticker in universe_config.sector_map, ticker
            assert universe_config.sector_etf_for(ticker), ticker

    def test_no_group_is_dominated_by_one_sector(self, universe_config):
        for name, tickers in universe_config.groups.items():
            sectors = [universe_config.sector_map[t] for t in tickers]
            share = max(sectors.count(s) for s in set(sectors)) / len(sectors)
            assert share <= 0.6, f"{name} is {share:.0%} one sector"

    def test_support_tickers_are_not_in_any_group(self, universe_config):
        assert not set(universe_config.support_tickers) & set(universe_config.all_tickers)

    def test_horizons_are_the_two_declared(self, targets_config):
        assert targets_config.horizons == [1, 5]

    def test_purge_defaults_to_the_horizon(self, walkforward_config):
        assert walkforward_config.purge_for(5) == 5
        assert walkforward_config.purge_for(1) == 1


class TestValidation:
    def test_rejects_a_ticker_in_two_groups(self):
        with pytest.raises(ValidationError):
            UniverseConfig(
                benchmark="SPY",
                volatility_index="^VIX",
                groups={"dev": ["AAA"], "test": ["AAA"]},
                sector_etfs={},
                sector_map={},
            )

    def test_rejects_a_non_positive_horizon(self, targets_config):
        with pytest.raises(ValidationError):
            targets_config.model_copy(update={"horizons": [0]}).model_validate(
                {**targets_config.model_dump(), "horizons": [0]}
            )
