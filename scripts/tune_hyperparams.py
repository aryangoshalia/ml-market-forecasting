"""Random hyperparameter search on the earliest folds only, written to configs/models/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from market_forecast.config import get_config, project_root
from market_forecast.dataset import load_panel
from market_forecast.experiments.search import SEARCH_SPACES, search_model
from market_forecast.logging import configure, get_logger
from market_forecast.models.registry import default_specs

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=None)
    parser.add_argument("--target", default="direction")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--groups", nargs="*", default=["dev"])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure(args.log_level)
    config = get_config()

    panel_path = (
        Path(args.panel) if args.panel else project_root() / "artifacts" / "panel_v1.parquet"
    )
    panel = load_panel(panel_path)

    tunable = [spec for spec in default_specs() if spec.kind in SEARCH_SPACES]
    chosen: dict[str, dict] = {}
    report: dict[str, list] = {}

    for spec in tunable:
        result = search_model(
            panel,
            config,
            spec,
            target=args.target,
            horizon=args.horizon,
            groups=args.groups,
            n_trials=args.trials,
            n_folds=args.folds,
            seed=config.walkforward.seed,
        )
        chosen[spec.name] = result.best.params
        report[spec.name] = result.table().head(5).to_dict(orient="records")
        print(
            f"\n{spec.name}: best inner auc {result.best.mean_auc:.4f} "
            f"(+/- {result.best.std_auc:.4f}) over {result.folds_used} folds up to {result.cutoff}"
        )
        print(f"  {result.best.params}")
        spread = result.table()["mean_auc"]
        print(
            f"  search spread: {spread.min():.4f} to {spread.max():.4f} over {len(spread)} trials"
        )

    out = Path(args.out) if args.out else project_root() / "configs" / "models" / "tuned.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_on": {
            "target": args.target,
            "horizon": args.horizon,
            "groups": args.groups,
            "folds": args.folds,
            "trials": args.trials,
            "note": (
                "Chosen on the fit and inner windows of the earliest folds only, then frozen "
                "for every later fold. No test window influenced this choice."
            ),
        },
        "params": chosen,
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (out.parent / "tuned_trials.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
