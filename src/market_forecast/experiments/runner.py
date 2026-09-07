"""The walk-forward loop.

For every fold the model is fitted on the fit window only. The calibrator and the
decision threshold are fitted on the inner window, which the model never saw. The test
window is touched exactly once, to predict. Nothing fitted anywhere is refitted on data
that comes later in time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from market_forecast.config import AppConfig
from market_forecast.dataset import Panel
from market_forecast.evaluation.metrics import evaluate
from market_forecast.logging import get_logger
from market_forecast.models.base import ModelSpec
from market_forecast.models.calibration import ISOTONIC, ProbabilityCalibrator, select_threshold
from market_forecast.models.registry import build_model, default_specs
from market_forecast.validation.splits import Fold, WalkForwardSplitter, assert_no_overlap

logger = get_logger(__name__)


@dataclass
class RunConfig:
    target: str = "direction"
    horizon: int = 1
    models: list[ModelSpec] = field(default_factory=default_specs)
    calibration: str = ISOTONIC
    threshold_metric: str = "balanced_accuracy"
    groups: list[str] = field(default_factory=lambda: ["dev"])
    eval_groups: list[str] | None = None
    start: pd.Timestamp | None = None
    max_folds: int | None = None
    seed: int = 17

    @property
    def label_column(self) -> str:
        return f"{self.target}_{self.horizon}d"

    @property
    def scored_groups(self) -> list[str]:
        """Groups the model is scored on. Defaults to the groups it trains on."""
        return self.eval_groups or self.groups

    @property
    def is_held_out(self) -> bool:
        return self.eval_groups is not None and set(self.eval_groups) != set(self.groups)

    def identity(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "horizon": self.horizon,
            "models": [spec.signature() for spec in self.models],
            "calibration": self.calibration,
            "threshold_metric": self.threshold_metric,
            "groups": sorted(self.groups),
            "eval_groups": sorted(self.scored_groups),
            "start": str(self.start) if self.start is not None else None,
            "max_folds": self.max_folds,
            "seed": self.seed,
        }


@dataclass
class RunResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    folds: list[Fold]
    dataset: dict[str, Any]
    elapsed_seconds: float


class WalkForwardRunner:
    def __init__(self, panel: Panel, config: AppConfig, run_config: RunConfig) -> None:
        self.panel = panel
        self.config = config
        self.run_config = run_config

    def _selected_rows(self, groups: list[str]) -> pd.DataFrame:
        frame = self.panel.frame
        if groups:
            frame = frame[frame["group"].isin(groups)]
        if self.run_config.start is not None:
            dates = frame.index.get_level_values("date")
            frame = frame[dates >= self.run_config.start]
        label = self.run_config.label_column
        if label not in frame.columns:
            raise KeyError(f"panel has no target column {label!r}")
        return frame[frame[label].notna()]

    def run(self) -> RunResult:
        started = time.perf_counter()
        train_rows = self._selected_rows(self.run_config.groups)
        scored_rows = self._selected_rows(self.run_config.scored_groups)
        features = self.panel.feature_names
        label = self.run_config.label_column

        train_dates = pd.Series(train_rows.index.get_level_values("date"), index=train_rows.index)
        scored_dates = pd.Series(
            scored_rows.index.get_level_values("date"), index=scored_rows.index
        )
        sessions = pd.DatetimeIndex(train_dates.unique()).sort_values()

        splitter = WalkForwardSplitter(self.config.walkforward, self.run_config.horizon)
        folds = splitter.split(sessions)
        for fold in folds:
            assert_no_overlap(fold)
        if self.run_config.max_folds:
            folds = folds[: self.run_config.max_folds]

        logger.info(
            "%s h=%d | %s | train %d rows / %d tickers | score %d rows / %d tickers%s",
            self.run_config.target,
            self.run_config.horizon,
            splitter.describe(sessions),
            len(train_rows),
            train_rows.index.get_level_values("ticker").nunique(),
            len(scored_rows),
            scored_rows.index.get_level_values("ticker").nunique(),
            " [held out]" if self.run_config.is_held_out else "",
        )

        prediction_blocks: list[pd.DataFrame] = []
        metric_rows: list[dict[str, Any]] = []

        for fold in folds:
            fold_started = time.perf_counter()
            # Fitting and calibration draw from the training groups; the test window is
            # taken from the scored groups, which may be entirely different tickers.
            parts = {
                "fit": train_rows[fold.mask(train_dates, "fit").to_numpy()],
                "inner": train_rows[fold.mask(train_dates, "inner").to_numpy()],
                "test": scored_rows[fold.mask(scored_dates, "test").to_numpy()],
            }
            if any(len(block) == 0 for block in parts.values()):
                logger.warning("fold %d has an empty window, skipping", fold.index)
                continue

            targets = {
                part: block[label].to_numpy(dtype="float64") for part, block in parts.items()
            }
            matrices = {part: block[features] for part, block in parts.items()}

            for spec in self.run_config.models:
                model = build_model(spec, seed=self.run_config.seed)
                model.fit(matrices["fit"], parts["fit"][label])

                raw_inner = model.predict_proba(matrices["inner"])
                raw_test = model.predict_proba(matrices["test"])

                calibrator = ProbabilityCalibrator(self.run_config.calibration)
                calibrator.fit(raw_inner, targets["inner"])
                calibrated_test = calibrator.transform(raw_test)
                calibrated_inner = calibrator.transform(raw_inner)

                choice = select_threshold(
                    targets["inner"], calibrated_inner, self.run_config.threshold_metric
                )

                block = pd.DataFrame(
                    {
                        "fold": fold.index,
                        "model": spec.name,
                        "prob_raw": raw_test,
                        "prob": calibrated_test,
                        "y_true": targets["test"],
                        "threshold": choice.threshold,
                        "is_baseline": spec.is_baseline,
                    },
                    index=parts["test"].index,
                )
                prediction_blocks.append(block)

                at_half = evaluate(targets["test"], calibrated_test, threshold=0.5)
                at_tuned = evaluate(targets["test"], calibrated_test, threshold=choice.threshold)
                uncalibrated = evaluate(targets["test"], raw_test, threshold=0.5)

                metric_rows.append(
                    {
                        "fold": fold.index,
                        "model": spec.name,
                        "is_baseline": spec.is_baseline,
                        "test_start": fold.test_dates[0],
                        "test_end": fold.test_dates[1],
                        "n_fit": len(parts["fit"]),
                        "n_inner": len(parts["inner"]),
                        "n_test": len(parts["test"]),
                        "tuned_threshold": choice.threshold,
                        "ece_uncalibrated": uncalibrated.ece,
                        "brier_uncalibrated": uncalibrated.brier,
                        **{f"{k}": v for k, v in at_half.to_dict().items() if k != "extra"},
                        **{
                            f"tuned_{k}": v
                            for k, v in at_tuned.to_dict().items()
                            if k in ("accuracy", "precision", "recall", "f1", "positive_rate")
                        },
                    }
                )

            logger.info(
                "fold %2d/%d  test %s..%s  %d models  %.1fs",
                fold.index + 1,
                len(folds),
                f"{fold.test_dates[0]:%Y-%m-%d}",
                f"{fold.test_dates[1]:%Y-%m-%d}",
                len(self.run_config.models),
                time.perf_counter() - fold_started,
            )

        if not prediction_blocks:
            raise RuntimeError("no folds produced predictions")

        predictions = pd.concat(prediction_blocks).sort_index()
        fold_metrics = pd.DataFrame(metric_rows)

        return RunResult(
            predictions=predictions,
            fold_metrics=fold_metrics,
            folds=folds,
            dataset={
                "rows": int(len(scored_rows)),
                "train_rows": int(len(train_rows)),
                "train_tickers": sorted(train_rows.index.get_level_values("ticker").unique()),
                "tickers": sorted(scored_rows.index.get_level_values("ticker").unique()),
                "first_session": str(sessions[0].date()),
                "last_session": str(sessions[-1].date()),
                "n_sessions": int(len(sessions)),
                "groups": sorted(self.run_config.groups),
                "eval_groups": sorted(self.run_config.scored_groups),
                "held_out": self.run_config.is_held_out,
                "label": label,
                "base_rate": float(scored_rows[label].mean()),
            },
            elapsed_seconds=time.perf_counter() - started,
        )


def aggregate(fold_metrics: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Mean and standard error across folds, which is the unit of independent observation."""
    measures = columns or [
        "accuracy",
        "accuracy_over_base_rate",
        "roc_auc",
        "pr_auc",
        "brier",
        "ece",
        "f1",
        "log_loss",
    ]
    available = [c for c in measures if c in fold_metrics.columns]
    grouped = fold_metrics.groupby("model")[available]
    summary = grouped.mean()
    counts = fold_metrics.groupby("model").size().rename("folds")
    errors = grouped.sem().add_suffix("_se")
    baseline = fold_metrics.groupby("model")["is_baseline"].first()
    out = pd.concat([counts, baseline, summary, errors], axis=1)
    return out.sort_values("roc_auc", ascending=False)
