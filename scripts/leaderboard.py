"""Compare recorded runs and test whether any model beats its baselines."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from market_forecast.evaluation.stats import (
    benjamini_hochberg,
    block_bootstrap_ci,
    paired_fold_test,
    permutation_null_pooled,
)
from market_forecast.experiments.runner import aggregate
from market_forecast.experiments.tracker import ExperimentTracker
from market_forecast.logging import configure, get_logger

logger = get_logger(__name__)

SHOWN = [
    "folds",
    "accuracy",
    "accuracy_over_base_rate",
    "roc_auc",
    "auc_lo",
    "auc_hi",
    "pr_auc",
    "brier",
    "ece",
    "vs_majority",
    "p_adj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--permutation-draws", type=int, default=200)
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure(args.log_level)
    tracker = ExperimentTracker()

    index = tracker.index()
    if index.empty:
        print("no recorded runs; run scripts/run_walkforward.py first")
        return 1
    if args.experiment:
        index = index[index["experiment"] == args.experiment]

    index = index.drop_duplicates(subset=["target", "horizon", "experiment"], keep="last")

    for row in index.sort_values(["target", "horizon"]).itertuples():
        fold_metrics = tracker.load_fold_metrics(row.run_id)
        predictions = tracker.load_predictions(row.run_id)
        base_rate = float(row.dataset["base_rate"])

        summary = aggregate(fold_metrics)
        reference = fold_metrics[fold_metrics["model"] == "majority"]

        rows = []
        for model in summary.index:
            scores = fold_metrics[fold_metrics["model"] == model]
            merged = scores.merge(reference, on="fold", suffixes=("", "_ref"))
            test = paired_fold_test(
                merged["roc_auc"].to_numpy(), merged["roc_auc_ref"].to_numpy(), model, "majority"
            )
            low, high = block_bootstrap_ci(scores["roc_auc"].to_numpy(), seed=7)
            rows.append(
                {
                    "model": model,
                    "auc_lo": low,
                    "auc_hi": high,
                    "vs_majority": f"{test.wins}/{test.n_folds}",
                    "p_raw": test.p_value,
                }
            )
        extra = pd.DataFrame(rows).set_index("model")
        extra["p_adj"] = benjamini_hochberg(extra["p_raw"].to_numpy())
        table = summary.join(extra)

        print(f"\n{'=' * 100}")
        print(
            f"{row.target}  h={row.horizon}  |  base rate {base_rate:.4f}  |  "
            f"{row.folds['count']} folds  |  {row.dataset['n_sessions']} sessions  |  "
            f"{len(row.dataset['tickers'])} tickers"
        )
        print(f"{'=' * 100}")
        print(table[[c for c in SHOWN if c in table.columns]].round(4).to_string())

        best = table[~table["is_baseline"]]["roc_auc"].idxmax()
        pooled = predictions[predictions["model"] == best]
        flat = pooled.reset_index()
        null = permutation_null_pooled(
            flat["y_true"].to_numpy(),
            flat["prob"].to_numpy(),
            dates=flat["date"].to_numpy(),
            folds=flat["fold"].to_numpy(),
            draws=args.permutation_draws,
            seed=3,
        )
        print(
            f"\npermutation null for the best non-baseline model ({best}), pooled over all folds:"
        )
        print(f"  {null.summary()}")
        verdict = (
            "above the null" if null.observed > null.quantile_95 else "inside the null distribution"
        )
        print(f"  n={len(pooled)} predictions; observed AUC is {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
