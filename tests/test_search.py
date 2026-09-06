"""The hyperparameter search must only ever see early data."""

from __future__ import annotations

from datetime import date

import numpy as np
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
from market_forecast.data.csv_provider import CsvProvider
from market_forecast.data.loader import MarketDataLoader
from market_forecast.dataset import build_panel
from market_forecast.experiments.search import SEARCH_SPACES, sample_params, search_model
from market_forecast.models.base import ModelSpec
from market_forecast.validation.splits import WalkForwardSplitter
from tests.factories import synthetic_prices

TICKERS = {"AAA": 1, "BBB": 2}
SUPPORT = {"SPY": 11, "^VIX": 12, "XLK": 13}


@pytest.fixture(scope="module")
def search_config(tmp_path_factory):
    data_payload = load_yaml(config_dir() / "data.yaml")
    data_payload["cache_dir"] = str(tmp_path_factory.mktemp("cache"))
    data_payload["start_date"] = date(2010, 1, 1)
    walkforward_payload = load_yaml(config_dir() / "walkforward.yaml")
    walkforward_payload.update(
        initial_train_sessions=800,
        test_sessions=100,
        step_sessions=100,
        inner_validation_sessions=80,
    )
    return AppConfig(
        data=DataConfig(**data_payload),
        universe=UniverseConfig(
            benchmark="SPY",
            volatility_index="^VIX",
            groups={"dev": ["AAA", "BBB"]},
            sector_etfs={"technology": "XLK"},
            sector_map={"AAA": "technology", "BBB": "technology"},
        ),
        targets=TargetsConfig(**load_yaml(config_dir() / "targets.yaml")),
        features=FeatureConfig(**load_yaml(config_dir() / "features.yaml")),
        walkforward=WalkForwardConfig(**walkforward_payload),
    )


@pytest.fixture(scope="module")
def search_panel(search_config, tmp_path_factory):
    directory = tmp_path_factory.mktemp("csv")
    for ticker, seed in {**TICKERS, **SUPPORT}.items():
        frame = synthetic_prices(sessions=1900, seed=seed, start="2010-01-04")
        frame.drop(columns=["adj_factor"]).to_csv(
            directory / f"{ticker.replace('^', 'INDEX_')}.csv"
        )
    loader = MarketDataLoader(search_config.data, provider=CsvProvider(directory))
    return build_panel(search_config, loader)


class TestSampling:
    def test_draws_only_declared_values(self):
        rng = np.random.default_rng(0)
        space = SEARCH_SPACES["xgboost"]
        for _ in range(30):
            drawn = sample_params(space, rng)
            assert set(drawn) == set(space)
            for name, value in drawn.items():
                assert value in space[name]

    def test_is_reproducible(self):
        space = SEARCH_SPACES["random_forest"]
        first = sample_params(space, np.random.default_rng(5))
        second = sample_params(space, np.random.default_rng(5))
        assert first == second

    def test_every_tunable_model_has_a_space(self):
        from market_forecast.models.registry import default_specs

        tunable = [s for s in default_specs() if not s.is_baseline]
        assert {s.kind for s in tunable} <= set(SEARCH_SPACES)


class TestSearchStaysInThePast:
    @pytest.fixture(scope="class")
    def result(self, search_panel, search_config):
        return search_model(
            search_panel,
            search_config,
            ModelSpec("logistic", "logistic", needs_scaling=True),
            target="direction",
            horizon=1,
            groups=["dev"],
            n_trials=4,
            n_folds=2,
            seed=17,
        )

    def test_returns_a_best_trial(self, result):
        assert result.best in result.trials
        assert result.best.mean_auc == max(t.mean_auc for t in result.trials)

    def test_uses_only_the_requested_number_of_folds(self, result):
        assert result.folds_used == 2

    def test_cutoff_precedes_every_test_window(self, result, search_panel, search_config):
        """Settings are applied to all folds, so the search must not see any scored session."""
        rows = search_panel.frame[search_panel.frame["direction_1d"].notna()]
        sessions = pd.DatetimeIndex(rows.index.get_level_values("date").unique()).sort_values()
        folds = WalkForwardSplitter(search_config.walkforward, 1).split(sessions)
        earliest_test = min(fold.test_dates[0] for fold in folds)
        assert pd.Timestamp(result.cutoff) < earliest_test

    def test_gap_before_the_first_test_window_exceeds_the_horizon(
        self, result, search_panel, search_config
    ):
        rows = search_panel.frame[search_panel.frame["direction_1d"].notna()]
        sessions = pd.DatetimeIndex(rows.index.get_level_values("date").unique()).sort_values()
        folds = WalkForwardSplitter(search_config.walkforward, 1).split(sessions)
        cutoff_position = int(sessions.get_loc(pd.Timestamp(result.cutoff)))
        assert folds[0].test[0] - cutoff_position > 1

    def test_reports_a_score_per_trial(self, result):
        for trial in result.trials:
            assert len(trial.fold_scores) == 2
            assert np.isfinite(trial.mean_auc)

    def test_table_is_sorted_best_first(self, result):
        scores = result.table()["mean_auc"].to_numpy()
        assert (np.diff(scores) <= 1e-12).all()

    def test_rejects_a_model_with_no_search_space(self, search_panel, search_config):
        with pytest.raises(ValueError, match="no search space"):
            search_model(
                search_panel,
                search_config,
                ModelSpec("majority", "majority", is_baseline=True),
                target="direction",
                horizon=1,
                groups=["dev"],
            )
