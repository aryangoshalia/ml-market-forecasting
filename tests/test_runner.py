"""End-to-end checks on the walk-forward loop, focused on what it must never do."""

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
from market_forecast.experiments.runner import RunConfig, WalkForwardRunner, aggregate
from market_forecast.models.base import ModelSpec
from tests.factories import synthetic_prices

TICKERS = {"AAA": 1, "BBB": 2, "CCC": 3}
SUPPORT = {"SPY": 11, "^VIX": 12, "XLK": 13}
SESSIONS = 2600


@pytest.fixture(scope="module")
def csv_directory(tmp_path_factory):
    directory = tmp_path_factory.mktemp("csv")
    for ticker, seed in {**TICKERS, **SUPPORT}.items():
        frame = synthetic_prices(sessions=SESSIONS, seed=seed, start="2010-01-04")
        frame.drop(columns=["adj_factor"]).to_csv(
            directory / f"{ticker.replace('^', 'INDEX_')}.csv"
        )
    return directory


@pytest.fixture(scope="module")
def runner_config(tmp_path_factory):
    data_payload = load_yaml(config_dir() / "data.yaml")
    data_payload["cache_dir"] = str(tmp_path_factory.mktemp("cache"))
    data_payload["start_date"] = date(2010, 1, 1)

    walkforward_payload = load_yaml(config_dir() / "walkforward.yaml")
    walkforward_payload.update(
        initial_train_sessions=900,
        test_sessions=120,
        step_sessions=120,
        inner_validation_sessions=100,
    )

    return AppConfig(
        data=DataConfig(**data_payload),
        universe=UniverseConfig(
            benchmark="SPY",
            volatility_index="^VIX",
            groups={"dev": ["AAA", "BBB"], "test": ["CCC"]},
            sector_etfs={"technology": "XLK"},
            sector_map={"AAA": "technology", "BBB": "technology", "CCC": "technology"},
        ),
        targets=TargetsConfig(**load_yaml(config_dir() / "targets.yaml")),
        features=FeatureConfig(**load_yaml(config_dir() / "features.yaml")),
        walkforward=WalkForwardConfig(**walkforward_payload),
    )


@pytest.fixture(scope="module")
def panel(runner_config, csv_directory):
    loader = MarketDataLoader(runner_config.data, provider=CsvProvider(csv_directory))
    return build_panel(runner_config, loader)


@pytest.fixture(scope="module")
def result(panel, runner_config):
    run_config = RunConfig(
        target="direction",
        horizon=5,
        models=[
            ModelSpec("majority", "majority", is_baseline=True, needs_imputation=False),
            ModelSpec("logistic", "logistic", needs_scaling=True),
        ],
        groups=["dev"],
        seed=17,
    )
    return WalkForwardRunner(panel, runner_config, run_config).run()


class TestOutputShape:
    def test_produces_several_folds(self, result):
        assert len(result.folds) >= 3

    def test_one_row_per_model_per_test_observation(self, result):
        counts = result.predictions.groupby("model").size()
        assert counts.nunique() == 1

    def test_carries_both_raw_and_calibrated_probabilities(self, result):
        assert {"prob_raw", "prob", "y_true", "threshold", "fold"} <= set(
            result.predictions.columns
        )

    def test_probabilities_are_valid(self, result):
        for column in ("prob_raw", "prob"):
            values = result.predictions[column]
            assert values.between(0.0, 1.0).all()
            assert values.notna().all()

    def test_labels_are_never_missing(self, result):
        assert result.predictions["y_true"].notna().all()

    def test_every_fold_reports_metrics_for_every_model(self, result):
        table = result.fold_metrics.pivot_table(index="fold", columns="model", values="roc_auc")
        assert table.notna().all().all()


class TestPredictionsComeOnlyFromTestWindows:
    def test_no_prediction_falls_outside_a_test_window(self, result):
        predicted = set(result.predictions.index.get_level_values("date").unique())
        allowed: set[pd.Timestamp] = set()
        for fold in result.folds:
            allowed |= set(fold.dates[fold.test[0] : fold.test[1] + 1])
        assert predicted <= allowed

    def test_no_prediction_date_lies_inside_its_own_fit_window(self, result):
        for fold in result.folds:
            block = result.predictions[result.predictions["fold"] == fold.index]
            dates = block.index.get_level_values("date")
            fit_start, fit_end = fold.fit_dates
            assert not ((dates >= fit_start) & (dates <= fit_end)).any()

    def test_no_prediction_date_lies_inside_its_own_inner_window(self, result):
        for fold in result.folds:
            block = result.predictions[result.predictions["fold"] == fold.index]
            dates = block.index.get_level_values("date")
            inner_start, inner_end = fold.inner_dates
            assert not ((dates >= inner_start) & (dates <= inner_end)).any()

    def test_folds_never_predict_the_same_observation_twice(self, result):
        one_model = result.predictions[result.predictions["model"] == "logistic"]
        assert not one_model.index.duplicated().any()

    def test_predictions_advance_in_time_with_the_fold_index(self, result):
        starts = result.predictions.groupby("fold").apply(
            lambda block: block.index.get_level_values("date").min(), include_groups=False
        )
        assert list(starts) == sorted(starts)


class TestBaselineBehaviour:
    def test_majority_is_constant_within_a_fold(self, result):
        for _, block in result.predictions[result.predictions["model"] == "majority"].groupby(
            "fold"
        ):
            assert block["prob_raw"].nunique() == 1

    def test_majority_scores_half_auc(self, result):
        scores = result.fold_metrics[result.fold_metrics["model"] == "majority"]["roc_auc"]
        assert np.allclose(scores.dropna(), 0.5)

    def test_no_model_achieves_an_implausible_score(self, result):
        """On a random walk nothing can be predictable. A high AUC here means a leak."""
        summary = aggregate(result.fold_metrics)
        assert summary["roc_auc"].max() < 0.60, summary["roc_auc"].to_dict()


class TestReproducibility:
    def test_same_seed_gives_identical_predictions(self, panel, runner_config):
        run_config = RunConfig(
            target="direction",
            horizon=1,
            models=[ModelSpec("logistic", "logistic", needs_scaling=True)],
            groups=["dev"],
            max_folds=2,
            seed=17,
        )
        first = WalkForwardRunner(panel, runner_config, run_config).run()
        second = WalkForwardRunner(panel, runner_config, run_config).run()
        pd.testing.assert_frame_equal(first.predictions, second.predictions)


class TestSelection:
    def test_group_filter_restricts_the_tickers(self, panel, runner_config):
        run_config = RunConfig(
            target="direction",
            horizon=1,
            models=[ModelSpec("majority", "majority", is_baseline=True, needs_imputation=False)],
            groups=["dev"],
            max_folds=1,
        )
        result = WalkForwardRunner(panel, runner_config, run_config).run()
        assert set(result.predictions.index.get_level_values("ticker").unique()) == {"AAA", "BBB"}

    def test_unknown_target_raises(self, panel, runner_config):
        run_config = RunConfig(target="momentum", horizon=1, groups=["dev"], max_folds=1)
        with pytest.raises(KeyError, match="momentum"):
            WalkForwardRunner(panel, runner_config, run_config).run()


class TestHeldOutEvaluation:
    """Training on one set of tickers and scoring on another is the unseen-asset test.
    It is only meaningful if no scored ticker ever appears in a fitting window."""

    @pytest.fixture(scope="class")
    def held_out(self, panel, runner_config):
        run_config = RunConfig(
            target="direction",
            horizon=1,
            models=[
                ModelSpec("majority", "majority", is_baseline=True, needs_imputation=False),
                ModelSpec("logistic", "logistic", needs_scaling=True),
            ],
            groups=["dev"],
            eval_groups=["test"],
            max_folds=3,
            seed=17,
        )
        return WalkForwardRunner(panel, runner_config, run_config).run()

    def test_scores_only_the_held_out_tickers(self, held_out):
        scored = set(held_out.predictions.index.get_level_values("ticker").unique())
        assert scored == {"CCC"}

    def test_training_tickers_are_never_scored(self, held_out):
        scored = set(held_out.predictions.index.get_level_values("ticker").unique())
        assert not scored & {"AAA", "BBB"}

    def test_dataset_record_separates_train_from_scored(self, held_out):
        assert held_out.dataset["held_out"] is True
        assert held_out.dataset["groups"] == ["dev"]
        assert held_out.dataset["eval_groups"] == ["test"]
        assert set(held_out.dataset["train_tickers"]) == {"AAA", "BBB"}
        assert set(held_out.dataset["tickers"]) == {"CCC"}

    def test_base_rate_is_measured_on_the_scored_group(self, held_out, panel):
        expected = panel.frame[panel.frame["group"] == "test"]["direction_1d"].mean()
        assert held_out.dataset["base_rate"] == pytest.approx(float(expected), abs=1e-9)

    def test_baseline_still_scores_half_auc(self, held_out):
        scores = held_out.fold_metrics[held_out.fold_metrics["model"] == "majority"]["roc_auc"]
        assert np.allclose(scores.dropna(), 0.5)

    def test_no_implausible_score_on_unseen_assets(self, held_out):
        summary = aggregate(held_out.fold_metrics)
        assert summary["roc_auc"].max() < 0.60, summary["roc_auc"].to_dict()

    def test_same_group_on_both_sides_is_not_flagged_held_out(self, panel, runner_config):
        run_config = RunConfig(
            target="direction",
            horizon=1,
            models=[ModelSpec("majority", "majority", is_baseline=True, needs_imputation=False)],
            groups=["dev"],
            eval_groups=["dev"],
            max_folds=1,
        )
        result = WalkForwardRunner(panel, runner_config, run_config).run()
        assert result.dataset["held_out"] is False
