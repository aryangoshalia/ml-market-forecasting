"""Build the compact artefacts that ship with the repository.

A fresh clone should show a working dashboard without running the multi-hour evaluation
chain, so the out-of-sample prediction history and the regime timeline are committed.
Recast to float32 with a categorical model column they cost about a megabyte, which is
small enough that no subsetting is needed and the shipped history is complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from market_forecast.config import get_config, project_root
from market_forecast.data.loader import default_loader
from market_forecast.dataset import load_panel
from market_forecast.experiments.tracker import ExperimentTracker
from market_forecast.logging import configure, get_logger
from market_forecast.regime.describe import describe_states, label_states, regime_timeline
from market_forecast.regime.features import breadth_from_panel, build_regime_features
from market_forecast.regime.pipeline import assign_by_fold
from market_forecast.validation.splits import WalkForwardSplitter

logger = get_logger(__name__)

KEEP = ["fold", "model", "prob", "y_true", "threshold", "is_baseline"]
FLOAT32 = ["prob", "y_true", "threshold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--out", default=None)
    parser.add_argument("--regime-states", type=int, default=3)
    parser.add_argument("--skip-regimes", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def compact(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[[c for c in KEEP if c in frame.columns]].copy()
    for column in FLOAT32:
        if column in out.columns:
            out[column] = out[column].astype("float32")
    out["model"] = out["model"].astype("category")
    out["fold"] = out["fold"].astype("int16")
    return out


def main() -> int:
    args = parse_args()
    configure(args.log_level)
    config = get_config()

    out_dir = Path(args.out) if args.out else project_root() / "data" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker()
    index = tracker.index()
    runs = index[index["experiment"] == args.experiment]
    if runs.empty:
        logger.error("no runs recorded for experiment %r", args.experiment)
        return 1

    manifest: dict[str, object] = {"experiment": args.experiment, "predictions": []}
    for (target, horizon), group in runs.groupby(["target", "horizon"]):
        newest = group.iloc[-1]
        frame = compact(tracker.load_predictions(newest["run_id"]))
        name = f"predictions_{target}_h{horizon}.parquet"
        path = out_dir / name
        frame.to_parquet(path, compression="zstd")

        dates = frame.index.get_level_values("date")
        manifest["predictions"].append(  # type: ignore[union-attr]
            {
                "file": name,
                "target": target,
                "horizon": int(horizon),
                "run_id": newest["run_id"],
                "rows": int(len(frame)),
                "models": sorted(frame["model"].unique().tolist()),
                "folds": int(frame["fold"].nunique()),
                "first_session": str(pd.Timestamp(dates.min()).date()),
                "last_session": str(pd.Timestamp(dates.max()).date()),
                "size_bytes": int(path.stat().st_size),
            }
        )
        logger.info("%s: %d rows, %.2f MB", name, len(frame), path.stat().st_size / 1e6)

    if not args.skip_regimes:
        panel = load_panel(
            project_root() / "artifacts" / f"panel_{config.features.version}.parquet"
        )
        loader = default_loader(config.data)
        benchmark = loader.load(config.universe.benchmark).frame
        volatility_index = loader.load(config.universe.volatility_index, raise_on_error=False).frame
        dev = panel.frame[panel.frame["group"] == "dev"]
        features = build_regime_features(benchmark, volatility_index, breadth_from_panel(dev))

        sessions = pd.DatetimeIndex(dev.index.get_level_values("date").unique()).sort_values()
        folds = WalkForwardSplitter(config.walkforward, 1).split(sessions)
        assigned = assign_by_fold(
            features, folds, "hmm", args.regime_states, seed=config.walkforward.seed
        )

        summary = describe_states(features.loc[assigned.labels.index], assigned.labels)
        names = label_states(summary)

        payload = assigned.probabilities.astype("float32")
        payload["regime"] = assigned.labels.astype("int8")
        payload.to_parquet(out_dir / "regimes.parquet", compression="zstd")
        timeline = regime_timeline(assigned.labels, names)
        timeline.to_parquet(out_dir / "regime_timeline.parquet", compression="zstd")

        manifest["regimes"] = {
            "model": "hmm",
            "n_states": args.regime_states,
            "inference": "filtered",
            "sessions": int(len(assigned.labels)),
            "names": {str(k): v for k, v in names.items()},
            "episodes": int(len(timeline)),
        }
        logger.info("regimes: %d sessions, %d episodes", len(assigned.labels), len(timeline))

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(p.stat().st_size for p in out_dir.glob("*")) / 1e6
    logger.info("wrote %s (%.2f MB total)", out_dir, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
