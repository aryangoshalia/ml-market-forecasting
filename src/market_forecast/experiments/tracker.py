"""Local experiment tracking. One directory per run, plus an append-only index."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecast.config import project_root
from market_forecast.logging import get_logger

logger = get_logger(__name__)

TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "shap", "hmmlearn")
INDEX_NAME = "index.jsonl"


def git_revision() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                args, cwd=project_root(), capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except Exception:
            return ""

    revision = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    return {"commit": revision, "dirty": str(dirty).lower()}


def environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "absent"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        **git_revision(),
    }


def config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    experiment: str
    config_hash: str
    seed: int
    target: str
    horizon: int
    feature_version: str
    n_features: int
    dataset: dict[str, Any] = field(default_factory=dict)
    folds: dict[str, Any] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or project_root() / "experiments" / "runs")
        self.root.mkdir(parents=True, exist_ok=True)

    def new_run_id(self, experiment: str, payload: dict[str, Any]) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{experiment}-{config_hash(payload)}"

    def directory(self, run_id: str) -> Path:
        path = (self.root / run_id).resolve()
        if path.parent != self.root.resolve():
            raise ValueError("refusing to write outside the runs directory")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        record: RunRecord,
        predictions: pd.DataFrame | None = None,
        fold_metrics: pd.DataFrame | None = None,
        config: dict[str, Any] | None = None,
    ) -> Path:
        directory = self.directory(record.run_id)
        (directory / "record.json").write_text(
            json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        if config is not None:
            (directory / "config.json").write_text(
                json.dumps(config, indent=2, default=str), encoding="utf-8"
            )
        if predictions is not None and len(predictions):
            predictions.to_parquet(directory / "predictions.parquet", engine="pyarrow")
        if fold_metrics is not None and len(fold_metrics):
            fold_metrics.to_parquet(directory / "fold_metrics.parquet", engine="pyarrow")

        with (self.root / INDEX_NAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), default=str) + "\n")
        logger.info("saved run %s", record.run_id)
        return directory

    def index(self) -> pd.DataFrame:
        path = self.root / INDEX_NAME
        if not path.exists():
            return pd.DataFrame()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return pd.DataFrame(rows)

    def load_predictions(self, run_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.directory(run_id) / "predictions.parquet")

    def load_fold_metrics(self, run_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.directory(run_id) / "fold_metrics.parquet")
