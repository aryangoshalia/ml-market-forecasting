"""Generate the feature reference table from the registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from market_forecast.config import get_config, project_root
from market_forecast.data.loader import default_loader
from market_forecast.features.pipeline import ContextBuilder, FeaturePipeline
from market_forecast.logging import configure, get_logger

logger = get_logger(__name__)

GROUP_ORDER = ("returns", "momentum", "volatility", "volume", "trend", "market", "calendar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=None, help="any ticker; only the schema is used")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure("WARNING")
    config = get_config()

    ticker = args.ticker or config.universe.groups["dev"][0]
    loader = default_loader(config.data)
    matrix = FeaturePipeline(config.features).build(
        ticker,
        loader.load(ticker, raise_on_error=False).frame,
        ContextBuilder(loader, config.universe).for_ticker(ticker),
    )
    registry = matrix.registry

    lines = [
        f"# Feature reference (`{config.features.version}`)",
        "",
        f"{len(registry)} features across {len(GROUP_ORDER)} groups. "
        f"Longest lookback is {registry.max_lookback()} sessions, which sets the warm-up "
        "period dropped from the start of every ticker.",
        "",
        "Generated from the feature registry by `scripts/feature_docs.py`; do not edit by hand.",
        "",
    ]

    for group in GROUP_ORDER:
        specs = registry.by_group(group)
        if not specs:
            continue
        lines += [f"## {group} ({len(specs)})", ""]
        lines += [
            "| Feature | Lookback | Definition | Why it might inform the forecast |",
            "|---|---|---|---|",
        ]
        for spec in specs:
            lines.append(
                f"| `{spec.name}` | {spec.lookback} | {spec.description}<br>`{spec.formula}` | "
                f"{spec.rationale} |"
            )
        lines.append("")

        notes = {s.leakage_note for s in specs}
        default = "Backward-looking rolling window ending at t inclusive."
        extra = sorted(n for n in notes if n != default)
        if extra:
            lines += ["Leakage notes:", ""]
            lines += [f"- {note}" for note in extra]
            lines.append("")

    out = Path(args.out) if args.out else project_root() / "docs" / "FEATURES.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(registry)} features)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
