from __future__ import annotations

import json

import pandas as pd
import pytest

from market_forecast.experiments.tracker import (
    ExperimentTracker,
    RunRecord,
    config_hash,
    environment,
)


@pytest.fixture
def tracker(tmp_path) -> ExperimentTracker:
    return ExperimentTracker(tmp_path / "runs")


@pytest.fixture
def record(tracker) -> RunRecord:
    identity = {"target": "direction", "horizon": 1, "seed": 17}
    return RunRecord(
        run_id=tracker.new_run_id("unit", identity),
        created_at=pd.Timestamp("2026-01-02T10:00:00").isoformat(),
        experiment="unit",
        config_hash=config_hash(identity),
        seed=17,
        target="direction",
        horizon=1,
        feature_version="v1",
        n_features=58,
        dataset={"rows": 1000, "base_rate": 0.52},
        folds={"count": 5},
        models=["majority", "logistic"],
        metrics={"roc_auc": 0.51},
        environment=environment(),
    )


class TestConfigHash:
    def test_is_order_independent(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_changes_when_a_value_changes(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_is_short_and_stable(self):
        digest = config_hash({"seed": 17})
        assert len(digest) == 12
        assert digest == config_hash({"seed": 17})


class TestEnvironmentCapture:
    def test_records_what_is_needed_to_reproduce_a_run(self):
        captured = environment()
        assert captured["python"]
        assert captured["platform"]
        assert "numpy" in captured["packages"]
        assert "scikit-learn" in captured["packages"]

    def test_records_the_commit_and_whether_the_tree_was_dirty(self):
        captured = environment()
        assert "commit" in captured
        assert captured["dirty"] in ("true", "false")


class TestSaveAndLoad:
    def test_writes_the_record(self, tracker, record):
        directory = tracker.save(record)
        payload = json.loads((directory / "record.json").read_text())
        assert payload["run_id"] == record.run_id
        assert payload["target"] == "direction"

    def test_writes_predictions_and_fold_metrics(self, tracker, record):
        predictions = pd.DataFrame(
            {"prob": [0.5, 0.6], "y_true": [0.0, 1.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-01-02"), "AAA"), (pd.Timestamp("2026-01-03"), "AAA")],
                names=["date", "ticker"],
            ),
        )
        fold_metrics = pd.DataFrame({"fold": [0], "model": ["logistic"], "roc_auc": [0.51]})
        tracker.save(record, predictions, fold_metrics)

        pd.testing.assert_frame_equal(tracker.load_predictions(record.run_id), predictions)
        pd.testing.assert_frame_equal(tracker.load_fold_metrics(record.run_id), fold_metrics)

    def test_appends_to_the_index(self, tracker, record):
        tracker.save(record)
        second = RunRecord(**{**record.to_dict(), "run_id": record.run_id + "-b"})
        tracker.save(second)

        index = tracker.index()
        assert len(index) == 2
        assert set(index["run_id"]) == {record.run_id, second.run_id}

    def test_index_is_empty_before_any_run(self, tracker):
        assert tracker.index().empty

    def test_run_ids_are_unique_per_config(self, tracker):
        first = tracker.new_run_id("unit", {"seed": 1})
        second = tracker.new_run_id("unit", {"seed": 2})
        assert first.split("-")[-1] != second.split("-")[-1]

    def test_refuses_to_write_outside_its_directory(self, tracker):
        with pytest.raises(ValueError, match="outside"):
            tracker.directory("../escape")

    def test_config_is_stored_alongside_the_record(self, tracker, record):
        directory = tracker.save(record, config={"target": "direction", "seed": 17})
        stored = json.loads((directory / "config.json").read_text())
        assert stored["seed"] == 17
