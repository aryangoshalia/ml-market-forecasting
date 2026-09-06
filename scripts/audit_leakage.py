"""Run the leakage, truncation and scale audits across the universe. Non-zero exit on failure."""

from __future__ import annotations

import argparse
import sys

from market_forecast.config import get_config
from market_forecast.data.loader import default_loader
from market_forecast.features.pipeline import ContextBuilder, FeaturePipeline
from market_forecast.logging import configure, get_logger
from market_forecast.targets import binary_direction, forward_return
from market_forecast.validation.leakage import (
    AuditReport,
    audit_perturbation,
    audit_scale_invariance,
    audit_target_causality,
    audit_truncation,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", help="default is a sample from every group")
    parser.add_argument("--sample", type=int, default=2, help="tickers per group when none given")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure(args.log_level)
    config = get_config()

    tickers = args.tickers
    if not tickers:
        tickers = [t for group in config.universe.groups.values() for t in group[: args.sample]]

    loader = default_loader(config.data)
    pipeline = FeaturePipeline(config.features)
    context_builder = ContextBuilder(loader, config.universe)

    failures: list[AuditReport] = []
    for ticker in tickers:
        prices = loader.load(ticker, raise_on_error=False).frame
        context = context_builder.for_ticker(ticker)

        def build(frame, ctx, ticker=ticker):
            return pipeline.build(ticker, frame, ctx).frame

        reports = [
            audit_perturbation(build, prices, context),
            audit_truncation(build, prices, context),
            audit_scale_invariance(build, prices, context),
        ]
        for horizon in config.targets.horizons:
            reports.append(
                audit_target_causality(
                    prices,
                    horizon,
                    lambda frame, h=horizon: binary_direction(forward_return(frame, h)),
                )
            )

        verdict = "pass" if all(r.ok for r in reports) else "FAIL"
        checks = sum(r.checks_run for r in reports)
        features = reports[0].features_checked
        print(f"{ticker:6s} {verdict:4s}  {checks:3d} checks  {features} features")
        for report in reports:
            if report.ok:
                continue
            failures.append(report)
            for finding in report.errors[:5]:
                print(f"         {report.name}: {finding.detail}")

    print(f"\n{len(tickers)} tickers audited, {len(failures)} failing reports")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
