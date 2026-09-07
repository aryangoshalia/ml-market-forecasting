"""Reads the artefacts that ship with the repository.

The dashboard is meant to work on a fresh clone, so prediction history, fold metrics and
the regime timeline are read from disk rather than recomputed. Everything here is
read-only and cached, because these files do not change while the server is running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecast.config import project_root
from market_forecast.experiments.tracker import ExperimentTracker
from market_forecast.logging import get_logger

logger = get_logger(__name__)

DEMO_DIR = "data/demo"


@dataclass(frozen=True)
class Coverage:
    """What a shipped file actually covers, so the interface can say so."""

    first_session: str
    last_session: str
    rows: int
    folds: int
    models: tuple[str, ...]
    run_id: str


class ArtifactStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or project_root())
        self.demo_dir = self.root / DEMO_DIR
        self._predictions: dict[str, pd.DataFrame] = {}
        self._manifest: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return (self.demo_dir / "manifest.json").exists()

    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            path = self.demo_dir / "manifest.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"no shipped artefacts at {path}; run 'make demo-data' or 'make walkforward'"
                )
            self._manifest = json.loads(path.read_text(encoding="utf-8"))
        return self._manifest

    def _entry(self, target: str, horizon: int) -> dict[str, Any]:
        for entry in self.manifest()["predictions"]:
            if entry["target"] == target and entry["horizon"] == horizon:
                return entry
        raise KeyError(f"no shipped predictions for {target} at horizon {horizon}")

    def formulations(self) -> list[dict[str, Any]]:
        return list(self.manifest()["predictions"])

    def predictions(self, target: str, horizon: int) -> pd.DataFrame:
        key = f"{target}_{horizon}"
        if key not in self._predictions:
            entry = self._entry(target, horizon)
            self._predictions[key] = pd.read_parquet(self.demo_dir / entry["file"])
        return self._predictions[key]

    def coverage(self, target: str, horizon: int) -> Coverage:
        entry = self._entry(target, horizon)
        return Coverage(
            first_session=entry["first_session"],
            last_session=entry["last_session"],
            rows=int(entry["rows"]),
            folds=int(entry["folds"]),
            models=tuple(entry["models"]),
            run_id=entry["run_id"],
        )

    def predictions_for_ticker(
        self, ticker: str, target: str, horizon: int, model: str | None = None
    ) -> pd.DataFrame:
        frame = self.predictions(target, horizon)
        selected = frame[frame.index.get_level_values("ticker") == ticker]
        if model:
            selected = selected[selected["model"] == model]
        return selected.droplevel("ticker").sort_index()

    def regimes(self) -> pd.DataFrame:
        path = self.demo_dir / "regimes.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no shipped regime series at {path}")
        return pd.read_parquet(path)

    def regime_timeline(self) -> pd.DataFrame:
        path = self.demo_dir / "regime_timeline.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no shipped regime timeline at {path}")
        return pd.read_parquet(path)

    def regime_names(self) -> dict[int, str]:
        payload = self.manifest().get("regimes", {})
        return {int(k): v for k, v in payload.get("names", {}).items()}

    def fold_metrics(self, experiment: str = "main") -> pd.DataFrame:
        tracker = ExperimentTracker(self.root / "experiments" / "runs")
        index = tracker.index()
        if index.empty:
            raise FileNotFoundError("no experiment records found")

        selected = index[index["experiment"] == experiment]
        blocks = []
        for key, group in selected.groupby(["target", "horizon"]):
            target, horizon = str(key[0]), int(str(key[1]))
            newest = group.iloc[-1]
            block = tracker.load_fold_metrics(newest["run_id"])
            block = block.assign(target=target, horizon=horizon, run_id=str(newest["run_id"]))
            blocks.append(block)
        if not blocks:
            raise KeyError(f"no runs recorded for experiment {experiment!r}")
        return pd.concat(blocks, ignore_index=True)

    def experiments(self) -> pd.DataFrame:
        tracker = ExperimentTracker(self.root / "experiments" / "runs")
        index = tracker.index()
        if index.empty:
            return pd.DataFrame()
        return index[
            ["run_id", "experiment", "target", "horizon", "created_at", "folds", "dataset"]
        ]


@lru_cache(maxsize=1)
def get_store() -> ArtifactStore:
    return ArtifactStore()
