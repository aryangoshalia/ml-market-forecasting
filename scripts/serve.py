"""Run the local API, optionally pre-warming the serving model first."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from market_forecast.api.app import create_app
from market_forecast.api.service import get_service
from market_forecast.logging import configure, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="localhost by default, on purpose")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-warm", action="store_true", help="skip training the serving model")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure(args.log_level)

    service = get_service()
    if not args.no_warm:
        logger.info("warming the serving model so the first request does not pay for training")
        for version in service.warm():
            logger.info("ready: %s", version)

    uvicorn.run(
        create_app(service),
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        timeout_keep_alive=30,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
