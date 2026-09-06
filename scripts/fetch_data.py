"""Warm the local cache for the configured universe."""

from __future__ import annotations

import argparse
import sys
import time

from market_forecast.config import get_config
from market_forecast.data.loader import default_loader
from market_forecast.logging import configure, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", help="override the configured universe")
    parser.add_argument("--groups", nargs="*", default=None, help="dev, val or test")
    parser.add_argument(
        "--skip-support", action="store_true", help="skip benchmark, VIX and sector ETFs"
    )
    parser.add_argument("--force", action="store_true", help="ignore the cache and refetch")
    parser.add_argument("--pause", type=float, default=0.4, help="seconds between requests")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure(args.log_level)
    config = get_config()
    universe = config.universe

    if args.tickers:
        targets = list(args.tickers)
    elif args.groups:
        targets = [t for g in args.groups for t in universe.groups.get(g, [])]
    else:
        targets = universe.all_tickers

    if not args.skip_support:
        targets = targets + [t for t in universe.support_tickers if t not in targets]

    loader = default_loader(config.data)
    failures: list[str] = []
    warned: list[str] = []

    for position, ticker in enumerate(targets, start=1):
        try:
            result = loader.load(ticker, force_refresh=args.force, raise_on_error=False)
        except Exception as exc:
            logger.error("%s failed: %s", ticker, exc)
            failures.append(ticker)
            continue

        status = "cache" if result.from_cache else "fetch"
        logger.info("[%d/%d] %s (%s)", position, len(targets), result.report.summary(), status)
        if not result.report.ok:
            failures.append(ticker)
        elif result.report.warnings:
            warned.append(ticker)
        if not result.from_cache:
            time.sleep(args.pause)

    logger.info(
        "done: %d requested, %d failed, %d with warnings",
        len(targets),
        len(failures),
        len(warned),
    )
    if failures:
        logger.error("failed: %s", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
