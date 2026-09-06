from __future__ import annotations

import pandas as pd
import pytest

from market_forecast.config import DataConfig, UniverseConfig
from market_forecast.data.csv_provider import CsvProvider
from market_forecast.data.loader import MarketDataLoader
from market_forecast.dataset import build_panel, label_coverage
from tests.factories import synthetic_prices

TICKERS = {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}
SUPPORT = {"SPY": 11, "^VIX": 12, "XLK": 13, "XLF": 14}


@pytest.fixture
def csv_directory(tmp_path):
    directory = tmp_path / "csv"
    directory.mkdir()
    for ticker, seed in {**TICKERS, **SUPPORT}.items():
        frame = synthetic_prices(sessions=1400, seed=seed, start="2016-01-04")
        frame.drop(columns=["adj_factor"]).to_csv(
            directory / f"{ticker.replace('^', 'INDEX_')}.csv"
        )
    return directory


@pytest.fixture
def panel_config(app_config):
    universe = UniverseConfig(
        benchmark="SPY",
        volatility_index="^VIX",
        groups={"dev": ["AAA", "BBB"], "val": ["CCC"], "test": ["DDD"]},
        sector_etfs={"technology": "XLK", "financials": "XLF"},
        sector_map={
            "AAA": "technology",
            "BBB": "technology",
            "CCC": "financials",
            "DDD": "financials",
        },
    )
    return app_config.model_copy(update={"universe": universe})


@pytest.fixture
def loader(csv_directory, panel_config):
    config: DataConfig = panel_config.data
    return MarketDataLoader(config, provider=CsvProvider(csv_directory))


@pytest.fixture
def panel(panel_config, loader):
    return build_panel(panel_config, loader)


class TestPanelStructure:
    def test_indexed_by_date_and_ticker(self, panel):
        assert list(panel.frame.index.names) == ["date", "ticker"]

    def test_contains_every_requested_ticker(self, panel):
        assert set(panel.tickers) == set(TICKERS)
        assert not panel.skipped

    def test_all_tickers_share_one_feature_set(self, panel):
        counts = panel.features().groupby(level="ticker").apply(lambda block: block.shape[1])
        assert counts.nunique() == 1

    def test_sorted_chronologically(self, panel):
        dates = panel.frame.index.get_level_values("date")
        assert dates.is_monotonic_increasing

    def test_group_and_sector_are_attached(self, panel):
        assert set(panel.frame["group"].unique()) == {"dev", "val", "test"}
        assert panel.frame.loc[(slice(None), "AAA"), "sector"].eq("technology").all()

    def test_subset_selects_one_group(self, panel):
        dev = panel.subset("dev")
        assert set(dev.tickers) == {"AAA", "BBB"}
        assert dev.feature_names == panel.feature_names

    def test_every_feature_is_documented_in_the_registry(self, panel):
        assert set(panel.feature_names) <= set(panel.registry.names)


class TestPanelLabels:
    @pytest.mark.parametrize("horizon", [1, 5])
    def test_targets_are_present_for_each_horizon(self, panel, horizon):
        for name in ("direction", "excess_direction", "fwd_ret"):
            assert f"{name}_{horizon}d" in panel.frame.columns

    @pytest.mark.parametrize("horizon", [1, 5])
    def test_each_ticker_loses_exactly_h_labels_at_the_end(self, panel, horizon):
        for ticker in panel.tickers:
            block = panel.frame.xs(ticker, level="ticker")
            assert int(block[f"direction_{horizon}d"].isna().sum()) == horizon

    def test_base_rates_are_plausible(self, panel):
        for horizon in (1, 5):
            rate = label_coverage(panel, "direction", horizon)["base_rate"]
            assert 0.3 < rate < 0.7

    def test_a_label_is_never_attached_to_the_wrong_ticker(self, panel, loader):
        block = panel.frame.xs("AAA", level="ticker")
        prices = loader.load("AAA").frame
        expected = prices["close"].shift(-5) / prices["close"] - 1.0
        aligned = expected.reindex(block.index).dropna()
        pd.testing.assert_series_equal(
            block["fwd_ret_5d"].reindex(aligned.index), aligned, check_names=False
        )


class TestDateSplittingIsSafe:
    def test_every_date_appears_for_several_tickers(self, panel):
        counts = panel.frame.groupby(level="date").size()
        assert counts.max() == len(TICKERS)

    def test_a_date_cut_puts_no_ticker_on_both_sides(self, panel):
        cut = panel.dates[len(panel.dates) // 2]
        before = panel.frame[panel.frame.index.get_level_values("date") <= cut]
        after = panel.frame[panel.frame.index.get_level_values("date") > cut]
        assert (
            before.index.get_level_values("date").max() < after.index.get_level_values("date").min()
        )
