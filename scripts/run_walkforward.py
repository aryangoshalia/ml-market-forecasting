"""Run walk-forward evaluation and record the results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from market_forecast.config import get_config, project_root
from market_forecast.dataset import load_panel
from market_forecast.evaluation.stats import block_bootstrap_ci, paired_fold_test
from market_forecast.experiments.runner import RunConfig, WalkForwardRunner, aggregate
from market_forecast.experiments.tracker import ExperimentTracker, RunRecord, environment
from market_forecast.logging import configure, get_logger
from market_forecast.models.registry import default_specs

logger = get_logger(__name__)

LEADERBOARD_COLUMNS = [
    "folds",
    "is_baseline",
    "accuracy",
    "accuracy_over_base_rate",
    "roc_auc",
    "roc_auc_se",
    "pr_auc",
    "brier",
    "ece",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=None)
    parser.add_argument("--targets", nargs="*", default=["direction", "excess_direction"])
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--groups", nargs="*", default=["dev"], help="groups the model trains on")
    parser.add_argument(
        "--eval-groups",
        nargs="*",
        default=None,
        help="groups to score on; defaults to --groups. Use to score held-out assets.",
    )
    parser.add_argument("--start", default=None, help="e.g. 2010-01-01 for the robustness check")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--calibration", default="isotonic", choices=["isotonic", "platt", "none"])
    parser.add_argument("--scheme", default=None, choices=["anchored", "rolling"])
    parser.add_argument("--params", default=None, help="yaml of tuned model parameters")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure(args.log_level)
    config = get_config()
    if args.scheme:
        config = config.model_copy(
            update={"walkforward": config.walkforward.model_copy(update={"scheme": args.scheme})}
        )

    panel_path = (
        Path(args.panel) if args.panel else project_root() / "artifacts" / "panel_v1.parquet"
    )
    panel = load_panel(panel_path)
    logger.info(
        "panel %s: %d rows, %d features",
        panel_path.name,
        len(panel.frame),
        len(panel.feature_names),
    )

    specs = default_specs()
    if args.params:
        tuned = yaml.safe_load(Path(args.params).read_text(encoding="utf-8"))["params"]
        specs = [
            spec.__class__(**{**spec.__dict__, "params": {**spec.params, **tuned[spec.name]}})
            if spec.name in tuned
            else spec
            for spec in specs
        ]
        logger.info("applied tuned parameters for %s", sorted(tuned))

    tracker = ExperimentTracker()
    horizons = args.horizons or config.targets.horizons
    start = pd.Timestamp(args.start) if args.start else None

    for target in args.targets:
        for horizon in horizons:
            run_config = RunConfig(
                target=target,
                horizon=horizon,
                models=specs,
                calibration=args.calibration,
                groups=args.groups,
                eval_groups=args.eval_groups,
                start=start,
                max_folds=args.max_folds,
                seed=config.walkforward.seed,
            )
            result = WalkForwardRunner(panel, config, run_config).run()
            leaderboard = aggregate(result.fold_metrics)

            reference = result.fold_metrics[result.fold_metrics["model"] == "majority"]
            comparisons = []
            for model in leaderboard.index:
                scores = result.fold_metrics[result.fold_metrics["model"] == model]
                merged = scores.merge(reference, on="fold", suffixes=("", "_ref"))
                test = paired_fold_test(
                    merged["roc_auc"].to_numpy(),
                    merged["roc_auc_ref"].to_numpy(),
                    model,
                    "majority",
                )
                low, high = block_bootstrap_ci(scores["roc_auc"].to_numpy())
                comparisons.append(
                    {
                        "model": model,
                        "auc_ci_low": low,
                        "auc_ci_high": high,
                        "vs_majority_p": test.p_value,
                        "vs_majority_wins": test.wins,
                    }
                )
            leaderboard = leaderboard.join(pd.DataFrame(comparisons).set_index("model"))

            identity = {
                **run_config.identity(),
                "scheme": config.walkforward.scheme,
                "walkforward": config.walkforward.model_dump(),
                "feature_version": panel.feature_version,
            }
            record = RunRecord(
                run_id=tracker.new_run_id(f"{args.experiment}-{target}-h{horizon}", identity),
                created_at=pd.Timestamp.now().isoformat(),
                experiment=args.experiment,
                config_hash=str(identity),
                seed=run_config.seed,
                target=target,
                horizon=horizon,
                feature_version=panel.feature_version,
                n_features=len(panel.feature_names),
                dataset=result.dataset,
                folds={
                    "count": len(result.folds),
                    "scheme": config.walkforward.scheme,
                    "gap": result.folds[0].gap,
                    "first_test": str(result.folds[0].test_dates[0].date()),
                    "last_test": str(result.folds[-1].test_dates[1].date()),
                },
                models=[spec.name for spec in run_config.models],
                metrics=leaderboard.reset_index().to_dict(orient="records"),
                environment=environment(),
            )
            tracker.save(record, result.predictions, result.fold_metrics, identity)

            print(
                f"\n=== {target} h={horizon} | base rate {result.dataset['base_rate']:.4f} "
                f"| {len(result.folds)} folds | {result.elapsed_seconds / 60:.1f} min ==="
            )
            print(
                leaderboard[LEADERBOARD_COLUMNS + ["auc_ci_low", "auc_ci_high", "vs_majority_p"]]
                .round(4)
                .to_string()
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
