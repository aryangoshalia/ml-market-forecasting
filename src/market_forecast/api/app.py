"""Local FastAPI application.

Bound to localhost and intended to be run by one person on their own machine, but the
usual protections are still in place: the ticker is validated against a strict pattern
before it reaches the filesystem or a vendor URL, cross-origin requests are limited to
the local dev server, requests are rate limited, and internal errors are logged rather
than returned to the caller.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from market_forecast.api import schemas
from market_forecast.api.service import ForecastService, get_service
from market_forecast.data.base import InvalidTickerError, ProviderError, normalise_ticker
from market_forecast.logging import get_logger

logger = get_logger(__name__)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
]
RATE_LIMIT_REQUESTS = 120
RATE_LIMIT_WINDOW_SECONDS = 60.0
MAX_TICKERS_PER_REQUEST = 1

_TARGETS = ("direction", "excess_direction")
_HORIZONS = (1, 5)


class RateLimiter:
    """Fixed-window limiter keyed by client host. In-memory, which is right for one process."""

    def __init__(self, limit: int, window: float) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True


limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def validated_ticker(ticker: str) -> str:
    try:
        return normalise_ticker(ticker)
    except InvalidTickerError as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def validated_target(target: str) -> str:
    if target not in _TARGETS:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, detail=f"target must be one of {_TARGETS}"
        )
    return target


def validated_horizon(horizon: int) -> int:
    if horizon not in _HORIZONS:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, detail=f"horizon must be one of {_HORIZONS}"
        )
    return horizon


def create_app(service: ForecastService | None = None) -> FastAPI:
    app = FastAPI(
        title="Market direction forecasting",
        description="Research API. Model output, not financial advice.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    resolved = service or get_service()

    def current() -> ForecastService:
        return resolved

    @app.middleware("http")
    async def rate_limit(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "too many requests, slow down"},
            )
        return await call_next(request)

    @app.exception_handler(ProviderError)
    async def provider_error(request: Request, exc: ProviderError) -> JSONResponse:
        del request
        logger.warning("provider failure: %s", exc)
        return JSONResponse(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "market data provider is unavailable, try again shortly"},
        )

    @app.exception_handler(FileNotFoundError)
    async def missing_artifact(request: Request, exc: FileNotFoundError) -> JSONResponse:
        del request
        logger.error("missing artefact: %s", exc)
        return JSONResponse(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "required data is missing; run 'make panel' and 'make demo-data'"},
        )

    @app.exception_handler(KeyError)
    async def missing_key(request: Request, exc: KeyError) -> JSONResponse:
        del request
        return JSONResponse(status_code=HTTP_404_NOT_FOUND, content={"detail": str(exc).strip("'")})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s", request.url.path)
        del exc
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal error"},
        )

    @app.get("/api/health", response_model=schemas.Health)
    def health() -> schemas.Health:
        return current().health()

    @app.get("/api/universe", response_model=schemas.Universe)
    def universe() -> schemas.Universe:
        return current().universe()

    @app.get("/api/analyze/{ticker}", response_model=schemas.Analysis)
    def analyze(ticker: str, explain: bool = Query(default=True)) -> schemas.Analysis:
        return current().analyze(validated_ticker(ticker), explain=explain)

    @app.get("/api/history/{ticker}", response_model=schemas.PriceHistory)
    def history(
        ticker: str, sessions: int = Query(default=504, ge=60, le=5200)
    ) -> schemas.PriceHistory:
        return current().history(validated_ticker(ticker), sessions=sessions)

    @app.get("/api/predictions/{ticker}", response_model=schemas.PredictionHistory)
    def predictions(
        ticker: str,
        target: str = Query(default="excess_direction"),
        horizon: int = Query(default=1),
        model: str = Query(default="lightgbm"),
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> schemas.PredictionHistory:
        return current().predictions(
            validated_ticker(ticker),
            validated_target(target),
            validated_horizon(horizon),
            model=model,
            limit=limit,
        )

    @app.get("/api/performance", response_model=schemas.Performance)
    def performance(
        target: str = Query(default="excess_direction"),
        horizon: int = Query(default=1),
        experiment: str = Query(default="main"),
    ) -> schemas.Performance:
        return current().performance(
            validated_target(target), validated_horizon(horizon), experiment=experiment
        )

    @app.get("/api/regimes", response_model=schemas.RegimeTimeline)
    def regimes() -> schemas.RegimeTimeline:
        return current().regime_timeline()

    @app.get("/api/explain/{ticker}", response_model=schemas.Explanation)
    def explain(
        ticker: str,
        target: str = Query(default="excess_direction"),
        horizon: int = Query(default=1),
    ) -> schemas.Explanation:
        return current().explain(
            validated_ticker(ticker), validated_target(target), validated_horizon(horizon)
        )

    @app.get("/api/experiments")
    def experiments() -> list[dict[str, Any]]:
        return current().experiments()

    @app.get("/api/model-card", response_model=schemas.ModelCard)
    def model_card(
        target: str = Query(default="excess_direction"),
        horizon: int = Query(default=1),
    ) -> schemas.ModelCard:
        return current().model_card(validated_target(target), validated_horizon(horizon))

    return app
