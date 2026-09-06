"""Build the pooled feature and target panel, and write it to artifacts/."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from market_forecast.config import get_config, project_root
from market_forecast.data.loader import default_loader
from market_forecast.dataset import build_panel, label_coverage
from market_forecast.logging import configure, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--groups", nargs="*", help="dev, val or test")
    parser.add_argument("--start", help="override the configured start date, e.g. 2010-01-01")
    parser.add_argument(
        "--end", help="pin the last session, e.g. 2026-09-04, to reproduce a published run"
    )
    parser.add_argument("--out", default=None, help="output parquet path")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure(args.log_level)
    config = get_config()

    tickers = args.tickers
    if not tickers and args.groups:
        tickers = [t for g in args.groups for t in config.universe.groups.get(g, [])]

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    loader = default_loader(config.data)
    panel = build_panel(config, loader, tickers=tickers, start=start, end=end)

    logger.info("panel: %s", panel.describe())
    if panel.skipped:
        logger.warning("skipped %d tickers: %s", len(panel.skipped), panel.skipped)

    for horizon in panel.horizons:
        for target in ("direction", "excess_direction"):
            coverage = label_coverage(panel, target, horizon)
            logger.info(
                "%s: %d labelled rows, base rate %.4f",
                coverage["target"],
                coverage["labelled_rows"],
                coverage["base_rate"],
            )

    suffix = f"_{args.start}" if args.start else ""
    suffix += f"_to{args.end}" if args.end else ""
    out = (
        Path(args.out)
        if args.out
        else project_root() / "artifacts" / f"panel_{panel.feature_version}{suffix}.parquet"
    )
    panel.to_parquet(out)
    logger.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
