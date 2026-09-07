"""Business logic behind the endpoints.

Heavy objects are built once on first use: the panel, the serving model, the regime
model and the shipped artefacts. Requests after that are cheap.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from market_forecast.api import schemas
from market_forecast.api.serving import ServingArtifact, ServingModelStore, build_live_features
from market_forecast.api.store import ArtifactStore
from market_forecast.config import AppConfig, get_config, project_root
from market_forecast.data.loader import MarketDataLoader, default_loader
from market_forecast.dataset import Panel, load_panel
from market_forecast.explain.shap_explainer import ShapExplainer
from market_forecast.logging import get_logger
from market_forecast.regime.features import breadth_from_panel, build_regime_features
from market_forecast.regime.models import HmmRegime

logger = get_logger(__name__)

DISCLAIMER = (
    "Model output for research and education. Not financial advice, and not a "
    "recommendation to trade."
)
SHAP_CAVEAT = (
    "Contributions describe how this model combined correlated inputs. They are not "
    "evidence that a feature causes future returns, and where features move together the "
    "credit split between them is arbitrary. They sum to the raw probability; the "
    "reported forecast applies calibration on top, which is monotone and so preserves "
    "the ordering of contributions while shifting the level."
)
LIMITATIONS = [
    "The universe contains only companies still listed today, which biases every "
    "historical result optimistic.",
    "Adjusted prices are restated whenever a dividend or split occurs, so the history "
    "seen today is not what an observer saw at the time.",
    "Labels at a five-session horizon overlap, so the effective sample is far smaller "
    "than the row count suggests.",
    "Measured edges are small enough that transaction costs remove them; see the "
    "break-even analysis in docs/ANALYSIS.md.",
]

INDICATOR_COLUMNS = ("px_to_sma_20", "px_to_sma_50", "rsi_14", "vol_20", "beta_60")


def _dated_rows(frame: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, pd.Series]]:
    """iterrows with the index typed as the Timestamp it actually is."""
    for index, row in frame.iterrows():
        yield pd.Timestamp(str(index)), row


@dataclass
class ForecastSpec:
    target: str
    horizon: int


DEFAULT_FORECASTS = (
    ForecastSpec("excess_direction", 1),
    ForecastSpec("excess_direction", 5),
    ForecastSpec("direction", 1),
    ForecastSpec("direction", 5),
)


class ForecastService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._panel: Panel | None = None
        self._loader: MarketDataLoader | None = None
        self._serving: ServingModelStore | None = None
        self._store = ArtifactStore()
        self._regime_model: HmmRegime | None = None
        self._regime_features: pd.DataFrame | None = None

    @property
    def store(self) -> ArtifactStore:
        return self._store

    @property
    def loader(self) -> MarketDataLoader:
        if self._loader is None:
            self._loader = default_loader(self.config.data)
        return self._loader

    @property
    def panel(self) -> Panel:
        if self._panel is None:
            path = project_root() / "artifacts" / f"panel_{self.config.features.version}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"no panel at {path}; run 'make panel' first")
            self._panel = load_panel(path)
        return self._panel

    @property
    def serving(self) -> ServingModelStore:
        if self._serving is None:
            self._serving = ServingModelStore(self.config)
        return self._serving

    def artifact(self, target: str, horizon: int) -> ServingArtifact:
        return self.serving.get(self.panel, target=target, horizon=horizon)

    def warm(self) -> list[str]:
        versions = []
        for spec in DEFAULT_FORECASTS:
            versions.append(self.artifact(spec.target, spec.horizon).version)
        self._ensure_regime_model()
        return versions

    def _ensure_regime_model(self) -> tuple[HmmRegime, pd.DataFrame]:
        if self._regime_model is None or self._regime_features is None:
            dev = self.panel.frame[self.panel.frame["group"] == "dev"]
            benchmark = self.loader.load(self.config.universe.benchmark).frame
            volatility_index = self.loader.load(
                self.config.universe.volatility_index, raise_on_error=False
            ).frame
            features = build_regime_features(benchmark, volatility_index, breadth_from_panel(dev))
            states = int(self.store.manifest().get("regimes", {}).get("n_states", 3))
            self._regime_model = HmmRegime(n_states=states, seed=self.config.walkforward.seed).fit(
                features
            )
            self._regime_features = features
            logger.info("fitted regime model on %d sessions", len(features))
        return self._regime_model, self._regime_features

    # endpoints -----------------------------------------------------------------

    def health(self) -> schemas.Health:
        return schemas.Health(
            status="ok",
            shipped_artifacts=self.store.available,
            universe_size=len(self.config.universe.all_tickers),
            feature_version=self.config.features.version,
        )

    def universe(self) -> schemas.Universe:
        universe = self.config.universe
        entries = [
            schemas.UniverseEntry(
                ticker=ticker,
                group=universe.group_of(ticker) or "unassigned",
                sector=universe.sector_map.get(ticker, "unknown"),
            )
            for ticker in universe.all_tickers
        ]
        return schemas.Universe(
            benchmark=universe.benchmark,
            volatility_index=universe.volatility_index,
            tickers=entries,
            groups={name: len(t) for name, t in universe.groups.items()},
        )

    def analyze(self, ticker: str, explain: bool = True) -> schemas.Analysis:
        live = build_live_features(ticker, self.config, self.loader)
        prices = live.prices
        latest = prices.iloc[-1]
        previous = prices.iloc[-2] if len(prices) > 1 else latest

        snapshot = schemas.Snapshot(
            ticker=live.ticker,
            name=live.metadata.name,
            exchange=live.metadata.exchange,
            last_session=str(live.last_session.date()),
            sessions_stale=live.sessions_stale,
            close=float(latest["close"]),
            change=float(latest["close"] - previous["close"]),
            change_pct=float(latest["close"] / previous["close"] - 1.0),
            volume=float(latest["volume"]),
            in_training_universe=live.ticker in self.config.universe.all_tickers,
            sector_etf=live.sector_etf,
            sector_is_proxy=live.sector_is_proxy,
        )

        row = live.features.iloc[[-1]]
        forecasts = []
        primary: ServingArtifact | None = None
        for spec in DEFAULT_FORECASTS:
            artifact = self.artifact(spec.target, spec.horizon)
            primary = primary or artifact
            probability = float(artifact.calibrator.transform(artifact.model.predict_proba(row))[0])
            label = f"{spec.target}_{spec.horizon}d"
            base = self.panel.frame[label].dropna()
            forecasts.append(
                schemas.Forecast(
                    target=spec.target,
                    horizon=spec.horizon,
                    probability=round(probability, 4),
                    direction="bullish" if probability >= 0.5 else "bearish",
                    threshold=round(artifact.threshold, 4),
                    baseline_rate=round(float(max(base.mean(), 1 - base.mean())), 4),
                    model=artifact.model_name,
                    model_version=artifact.version,
                )
            )

        explanation = self.explain(ticker, live=live) if explain and primary else None
        return schemas.Analysis(
            snapshot=snapshot,
            forecasts=forecasts,
            regime=self.regime_now(),
            explanation=explanation,
            disclaimer=DISCLAIMER,
        )

    def explain(
        self, ticker: str, target: str = "excess_direction", horizon: int = 1, live: Any = None
    ) -> schemas.Explanation:
        resolved = live or build_live_features(ticker, self.config, self.loader)
        artifact = self.artifact(target, horizon)
        row = resolved.features.iloc[[-1]]
        local = ShapExplainer(artifact.model).explain_local(row)[0]
        raw = float(local.probability)
        calibrated = float(artifact.calibrator.transform(np.array([raw]))[0])
        return schemas.Explanation(
            base_value=round(local.base_value, 5),
            logodds=round(local.logodds, 5),
            raw_probability=round(raw, 4),
            probability=round(calibrated, 4),
            contributions=[
                schemas.Contribution(
                    feature=str(r["feature"]),
                    direction=str(r["direction"]),
                    contribution=float(r["contribution"]),
                    value=float(r["value"]),
                )
                for r in local.as_records(6)
            ],
            space="log-odds",
            caveat=SHAP_CAVEAT,
        )

    def regime_now(self) -> schemas.RegimeNow | None:
        try:
            model, features = self._ensure_regime_model()
        except Exception as exc:
            logger.warning("regime unavailable: %s", exc)
            return None

        filtered = model.filtered_proba(features)[-1]
        names = self.store.regime_names() if self.store.available else {}
        order = int(filtered.argmax())
        states = [
            schemas.RegimeState(
                state=i, name=names.get(i, f"state {i}"), probability=round(float(p), 4)
            )
            for i, p in enumerate(filtered)
        ]
        return schemas.RegimeNow(
            state=order,
            name=names.get(order, f"state {order}"),
            probability=round(float(filtered[order]), 4),
            states=states,
            inference="filtered",
        )

    def history(self, ticker: str, sessions: int = 504) -> schemas.PriceHistory:
        live = build_live_features(ticker, self.config, self.loader)
        prices = live.prices.tail(sessions)
        features = live.features.reindex(prices.index)

        points = [
            schemas.PricePoint(
                date=str(index.date()),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for index, row in _dated_rows(prices)
        ]
        indicators = {
            column: [None if pd.isna(v) else float(v) for v in features[column]]
            for column in INDICATOR_COLUMNS
            if column in features.columns
        }
        return schemas.PriceHistory(ticker=live.ticker, points=points, indicators=indicators)

    def _coverage(self, target: str, horizon: int) -> schemas.Coverage:
        found = self.store.coverage(target, horizon)
        return schemas.Coverage(
            first_session=found.first_session,
            last_session=found.last_session,
            rows=found.rows,
            folds=found.folds,
            models=list(found.models),
            run_id=found.run_id,
            source="shipped with the repository",
        )

    def predictions(
        self,
        ticker: str,
        target: str = "excess_direction",
        horizon: int = 1,
        model: str = "lightgbm",
        limit: int = 500,
    ) -> schemas.PredictionHistory:
        frame = self.store.predictions_for_ticker(ticker, target, horizon, model)
        if frame.empty:
            raise KeyError(f"no shipped predictions for {ticker}")

        regimes = None
        with contextlib.suppress(FileNotFoundError):
            regimes = self.store.regimes()["regime"]

        recent = frame.tail(limit)
        rows = []
        for index, row in _dated_rows(recent):
            predicted = int(row["prob"] >= 0.5)
            actual = int(row["y_true"])
            rows.append(
                schemas.PredictionRow(
                    date=str(index.date()),
                    model=str(row["model"]),
                    probability=round(float(row["prob"]), 4),
                    predicted=predicted,
                    actual=actual,
                    correct=predicted == actual,
                    fold=int(row["fold"]),
                    regime=(
                        int(regimes.loc[index])
                        if regimes is not None and index in regimes.index
                        else None
                    ),
                )
            )
        accuracy = float(((frame["prob"] >= 0.5).astype(int) == frame["y_true"]).mean())
        return schemas.PredictionHistory(
            ticker=ticker.upper(),
            target=target,
            horizon=horizon,
            coverage=self._coverage(target, horizon),
            rows=rows,
            accuracy=round(accuracy, 4),
            n=int(len(frame)),
        )

    def performance(
        self, target: str = "excess_direction", horizon: int = 1, experiment: str = "main"
    ) -> schemas.Performance:
        metrics = self.store.fold_metrics(experiment)
        selected = metrics[(metrics["target"] == target) & (metrics["horizon"] == horizon)]
        if selected.empty:
            raise KeyError(f"no recorded metrics for {target} at horizon {horizon}")

        columns = [
            "accuracy",
            "accuracy_over_base_rate",
            "roc_auc",
            "pr_auc",
            "brier",
            "ece",
        ]
        grouped = selected.groupby("model")
        summary = grouped[columns].mean()
        counts = grouped.size()
        baseline = grouped["is_baseline"].first()

        scores = [
            schemas.ModelScore(
                model=str(name),
                is_baseline=bool(baseline.loc[name]),
                folds=int(counts.loc[name]),
                **{c: round(float(summary.loc[name, c]), 4) for c in columns},
            )
            for name in summary.sort_values("roc_auc", ascending=False).index
        ]
        return schemas.Performance(
            target=target,
            horizon=horizon,
            experiment=experiment,
            coverage=self._coverage(target, horizon),
            scores=scores,
            note=(
                "Averages over walk-forward folds. Baselines are included so accuracy can "
                "be read against the floor it has to clear."
            ),
        )

    def regime_timeline(self) -> schemas.RegimeTimeline:
        timeline = self.store.regime_timeline()
        payload = self.store.manifest().get("regimes", {})
        episodes = [
            schemas.RegimeEpisode(
                state=int(row["state"]),
                name=str(row["name"]),
                start=str(pd.Timestamp(row["start"]).date()),
                end=str(pd.Timestamp(row["end"]).date()),
                sessions=int(row["sessions"]),
            )
            for _, row in timeline.iterrows()
        ]
        return schemas.RegimeTimeline(
            episodes=episodes,
            names={str(k): v for k, v in self.store.regime_names().items()},
            n_states=int(payload.get("n_states", 3)),
            model=str(payload.get("model", "hmm")),
            inference=str(payload.get("inference", "filtered")),
        )

    def model_card(self, target: str = "excess_direction", horizon: int = 1) -> schemas.ModelCard:
        artifact = self.artifact(target, horizon)
        universe = self.config.universe
        return schemas.ModelCard(
            card=artifact.card(),
            universe={
                "benchmark": universe.benchmark,
                "groups": {name: len(t) for name, t in universe.groups.items()},
                "trained_on": sorted(universe.groups.get("dev", [])),
            },
            limitations=LIMITATIONS,
        )

    def experiments(self) -> list[dict[str, Any]]:
        index = self.store.experiments()
        if index.empty:
            return []
        return [{str(k): v for k, v in row.items()} for row in index.to_dict(orient="records")]


@lru_cache(maxsize=1)
def get_service() -> ForecastService:
    return ForecastService()
