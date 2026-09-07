"""Response models. Every payload states what data it came from and over what period."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    shipped_artifacts: bool
    universe_size: int
    feature_version: str


class Coverage(BaseModel):
    first_session: str
    last_session: str
    rows: int
    folds: int
    models: list[str]
    run_id: str
    source: str = Field(description="shipped with the repository, or produced locally")


class UniverseEntry(BaseModel):
    ticker: str
    group: str
    sector: str


class Universe(BaseModel):
    benchmark: str
    volatility_index: str
    tickers: list[UniverseEntry]
    groups: dict[str, int]


class Snapshot(BaseModel):
    ticker: str
    name: str | None = None
    exchange: str | None = None
    last_session: str
    sessions_stale: int
    close: float
    change: float
    change_pct: float
    volume: float
    in_training_universe: bool
    sector_etf: str
    sector_is_proxy: bool


class Forecast(BaseModel):
    target: str
    horizon: int
    probability: float
    direction: str
    threshold: float
    baseline_rate: float = Field(description="what always predicting the majority class scores")
    model: str
    model_version: str


class Contribution(BaseModel):
    feature: str
    direction: str
    contribution: float
    value: float


class Explanation(BaseModel):
    base_value: float
    logodds: float
    raw_probability: float = Field(
        description="sigmoid of the log-odds; this is what the contributions reconstruct"
    )
    probability: float = Field(
        description="the reported forecast, after calibration is applied to the raw value"
    )
    contributions: list[Contribution]
    space: str = "log-odds"
    caveat: str


class RegimeState(BaseModel):
    state: int
    name: str
    probability: float


class RegimeNow(BaseModel):
    state: int
    name: str
    probability: float
    states: list[RegimeState]
    inference: str = Field(description="filtered, so only data up to the session is used")


class Analysis(BaseModel):
    snapshot: Snapshot
    forecasts: list[Forecast]
    regime: RegimeNow | None = None
    explanation: Explanation | None = None
    disclaimer: str


class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceHistory(BaseModel):
    ticker: str
    points: list[PricePoint]
    indicators: dict[str, list[float | None]]


class PredictionRow(BaseModel):
    date: str
    model: str
    probability: float
    predicted: int
    actual: int
    correct: bool
    fold: int
    regime: int | None = None


class PredictionHistory(BaseModel):
    ticker: str
    target: str
    horizon: int
    coverage: Coverage
    rows: list[PredictionRow]
    accuracy: float
    n: int


class ModelScore(BaseModel):
    model: str
    is_baseline: bool
    folds: int
    accuracy: float
    accuracy_over_base_rate: float
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float


class Performance(BaseModel):
    target: str
    horizon: int
    experiment: str
    coverage: Coverage
    scores: list[ModelScore]
    note: str


class RegimeEpisode(BaseModel):
    state: int
    name: str
    start: str
    end: str
    sessions: int


class RegimeTimeline(BaseModel):
    episodes: list[RegimeEpisode]
    names: dict[str, str]
    n_states: int
    model: str
    inference: str


class ModelCard(BaseModel):
    card: dict[str, Any]
    universe: dict[str, Any]
    limitations: list[str]
