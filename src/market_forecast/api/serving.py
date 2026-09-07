"""The model that scores a ticker on request.

Trained on first use and cached, rather than shipped as a pickle: a stored estimator
goes stale against the live data the application is meant to use, and pickles break
across library versions on someone else's machine.

Training keeps the same discipline as a walk-forward fold. The calibrator and the
decision threshold are fitted on a held-out tail that the estimator never saw, with a
purge and embargo between the two so no label straddles the boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from market_forecast.config import AppConfig, project_root
from market_forecast.data.base import AssetMetadata, normalise_ticker
from market_forecast.data.loader import MarketDataLoader
from market_forecast.dataset import Panel
from market_forecast.experiments.tracker import environment
from market_forecast.features.pipeline import ContextBuilder, FeaturePipeline
from market_forecast.logging import get_logger
from market_forecast.models.base import SklearnModel
from market_forecast.models.calibration import ISOTONIC, ProbabilityCalibrator, select_threshold
from market_forecast.models.registry import build_model, default_specs

logger = get_logger(__name__)

SERVING_DIR = "artifacts/serving"
DEFAULT_MODEL = "lightgbm"


@dataclass
class ServingArtifact:
    model: SklearnModel
    calibrator: ProbabilityCalibrator
    threshold: float
    target: str
    horizon: int
    model_name: str
    feature_names: list[str]
    feature_version: str
    trained_through: str
    n_train_rows: int
    n_tickers: int
    created_at: str
    environment: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return (
            f"{self.model_name}-{self.target}-h{self.horizon}"
            f"-{self.feature_version}-{self.trained_through}"
        )

    def card(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "target": self.target,
            "horizon": self.horizon,
            "version": self.version,
            "feature_version": self.feature_version,
            "n_features": len(self.feature_names),
            "trained_through": self.trained_through,
            "n_train_rows": self.n_train_rows,
            "n_tickers": self.n_tickers,
            "calibration": self.calibrator.method,
            "decision_threshold": round(self.threshold, 4),
            "created_at": self.created_at,
            "packages": self.environment.get("packages", {}),
        }


def train_serving_model(
    panel: Panel,
    config: AppConfig,
    target: str = "excess_direction",
    horizon: int = 1,
    model_name: str = DEFAULT_MODEL,
    groups: tuple[str, ...] = ("dev",),
) -> ServingArtifact:
    label = f"{target}_{horizon}d"
    rows = panel.frame[panel.frame["group"].isin(groups)]
    rows = rows[rows[label].notna()]
    if rows.empty:
        raise ValueError(f"no labelled rows for {label}")

    sessions = pd.DatetimeIndex(rows.index.get_level_values("date").unique()).sort_values()
    gap = config.walkforward.purge_for(horizon) + config.walkforward.embargo_sessions
    inner_length = config.walkforward.inner_validation_sessions

    inner_start = sessions[-inner_length]
    fit_end = sessions[-(inner_length + gap + 1)]

    dates = pd.Series(rows.index.get_level_values("date"), index=rows.index)
    fit = rows[(dates <= fit_end).to_numpy()]
    inner = rows[(dates >= inner_start).to_numpy()]

    spec = next(s for s in default_specs() if s.name == model_name)
    model = build_model(spec, seed=config.walkforward.seed)
    model.fit(fit[panel.feature_names], fit[label])

    raw = model.predict_proba(inner[panel.feature_names])
    truth = inner[label].to_numpy(dtype="float64")
    calibrator = ProbabilityCalibrator(ISOTONIC).fit(raw, truth)
    choice = select_threshold(truth, calibrator.transform(raw), "balanced_accuracy")

    logger.info(
        "trained serving model %s on %d rows through %s, calibrated on %d",
        model_name,
        len(fit),
        fit_end.date(),
        len(inner),
    )
    return ServingArtifact(
        model=model,
        calibrator=calibrator,
        threshold=choice.threshold,
        target=target,
        horizon=horizon,
        model_name=model_name,
        feature_names=list(panel.feature_names),
        feature_version=panel.feature_version,
        trained_through=str(sessions[-1].date()),
        n_train_rows=int(len(fit)),
        n_tickers=int(rows.index.get_level_values("ticker").nunique()),
        created_at=pd.Timestamp.now().isoformat(),
        environment=environment(),
    )


class ServingModelStore:
    """Loads a cached artefact when it is still valid, otherwise trains a fresh one."""

    def __init__(self, config: AppConfig, root: Path | None = None) -> None:
        self.config = config
        self.directory = Path(root or project_root() / SERVING_DIR)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ServingArtifact] = {}

    def _paths(self, target: str, horizon: int, model_name: str) -> tuple[Path, Path]:
        stem = f"{model_name}_{target}_h{horizon}"
        return self.directory / f"{stem}.joblib", self.directory / f"{stem}.json"

    def _is_valid(self, meta: dict[str, Any], panel: Panel) -> bool:
        if meta.get("feature_version") != panel.feature_version:
            return False
        current = environment()["packages"]
        stored = meta.get("packages", {})
        if any(stored.get(name) != version for name, version in current.items()):
            logger.info("cached serving model was built against different package versions")
            return False
        latest = str(pd.DatetimeIndex(panel.frame.index.get_level_values("date")).max().date())
        return meta.get("trained_through") == latest

    def get(
        self,
        panel: Panel,
        target: str = "excess_direction",
        horizon: int = 1,
        model_name: str = DEFAULT_MODEL,
    ) -> ServingArtifact:
        key = f"{model_name}_{target}_{horizon}"
        if key in self._cache:
            return self._cache[key]

        # joblib deserialisation executes code, so the cache directory is trusted the same
        # way the source tree is. The path is derived from fixed names rather than request
        # input, the file is only ever written by this process, and it is not distributed.
        blob, meta_path = self._paths(target, horizon, model_name)
        if blob.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if self._is_valid(meta, panel):
                    artifact = joblib.load(blob)
                    self._cache[key] = artifact
                    logger.info("loaded cached serving model %s", artifact.version)
                    return artifact
            except Exception as exc:
                logger.warning("cached serving model unusable (%s), retraining", exc)

        artifact = train_serving_model(panel, self.config, target, horizon, model_name)
        joblib.dump(artifact, blob)
        meta_path.write_text(json.dumps(artifact.card(), indent=2), encoding="utf-8")
        self._cache[key] = artifact
        return artifact


@dataclass
class LiveFeatures:
    ticker: str
    features: pd.DataFrame
    prices: pd.DataFrame
    metadata: AssetMetadata
    sector_etf: str
    sector_is_proxy: bool
    last_session: pd.Timestamp
    sessions_stale: int


def resolve_sector_etf(
    ticker: str, config: AppConfig, metadata: AssetMetadata | None
) -> tuple[str, bool]:
    """Sector ETF for a ticker, falling back to the benchmark when the sector is unknown.

    Falling back keeps the feature vector the same shape as the one the model was trained
    on. The substitution is reported so the interface can say the sector features are a
    market proxy rather than a true sector comparison.
    """
    mapped = config.universe.sector_etf_for(ticker)
    if mapped:
        return mapped, False

    sector = (metadata.sector if metadata else None) or ""
    key = sector.strip().lower().replace(" ", "_")
    aliases = {
        "technology": "technology",
        "information_technology": "technology",
        "communication_services": "technology",
        "financial_services": "financials",
        "financials": "financials",
        "healthcare": "healthcare",
        "consumer_cyclical": "consumer_discretionary",
        "consumer_discretionary": "consumer_discretionary",
        "consumer_defensive": "consumer_staples",
        "consumer_staples": "consumer_staples",
        "energy": "energy",
        "industrials": "industrials",
        "utilities": "utilities",
        "basic_materials": "materials",
        "materials": "materials",
    }
    resolved = config.universe.sector_etfs.get(aliases.get(key, ""), "")
    if resolved:
        return resolved, False
    return config.universe.benchmark, True


def build_live_features(
    ticker: str, config: AppConfig, loader: MarketDataLoader, start: date | None = None
) -> LiveFeatures:
    """Fetch the latest data for one ticker and run the same pipeline used in training."""
    symbol = normalise_ticker(ticker)
    result = loader.load(symbol, start=start)
    metadata = loader.provider.get_metadata(symbol)
    sector_etf, is_proxy = resolve_sector_etf(symbol, config, metadata)

    context_builder = ContextBuilder(loader, config.universe)
    context = context_builder.for_ticker(symbol)
    if context.sector is None or context.sector.empty:
        context.sector = loader.load(sector_etf, raise_on_error=False).frame
        context.sector_symbol = sector_etf

    matrix = FeaturePipeline(config.features).build(symbol, result.frame, context)
    return LiveFeatures(
        ticker=symbol,
        features=matrix.frame,
        prices=result.frame,
        metadata=metadata,
        sector_etf=sector_etf,
        sector_is_proxy=is_proxy,
        last_session=result.frame.index[-1],
        sessions_stale=result.report.sessions_stale,
    )
