"""Model construction from a name and a parameter dictionary."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from market_forecast.models.base import ModelSpec, SklearnModel
from market_forecast.models.baselines import AlwaysLong, MajorityClass, Persistence

BASELINE_KINDS = ("majority", "always_long", "persistence")


def build_model(spec: ModelSpec, seed: int) -> SklearnModel:
    kind = spec.kind
    params: dict[str, Any] = dict(spec.params)

    if kind == "majority":
        return SklearnModel(spec, MajorityClass())
    if kind == "always_long":
        return SklearnModel(spec, AlwaysLong())
    if kind == "persistence":
        return SklearnModel(spec, Persistence(**params))

    if kind == "logistic":
        params.setdefault("max_iter", 2000)
        params.setdefault("solver", "lbfgs")
        return SklearnModel(spec, LogisticRegression(random_state=seed, **params))

    if kind == "elastic_net_logistic":
        # sklearn 1.8 deprecated `penalty`; the mix is now set by l1_ratio alone.
        params.setdefault("max_iter", 5000)
        params.setdefault("solver", "saga")
        params.setdefault("l1_ratio", 0.5)
        return SklearnModel(spec, LogisticRegression(random_state=seed, **params))

    if kind == "random_forest":
        params.setdefault("n_estimators", 400)
        params.setdefault("min_samples_leaf", 50)
        params.setdefault("n_jobs", -1)
        return SklearnModel(spec, RandomForestClassifier(random_state=seed, **params))

    if kind == "xgboost":
        from xgboost import XGBClassifier

        params.setdefault("n_estimators", 400)
        params.setdefault("max_depth", 4)
        params.setdefault("learning_rate", 0.03)
        params.setdefault("subsample", 0.8)
        params.setdefault("colsample_bytree", 0.8)
        params.setdefault("reg_lambda", 5.0)
        params.setdefault("tree_method", "hist")
        params.setdefault("n_jobs", -1)
        params.setdefault("eval_metric", "logloss")
        return SklearnModel(spec, XGBClassifier(random_state=seed, **params))

    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        params.setdefault("n_estimators", 400)
        params.setdefault("num_leaves", 15)
        params.setdefault("learning_rate", 0.03)
        params.setdefault("subsample", 0.8)
        params.setdefault("subsample_freq", 1)
        params.setdefault("colsample_bytree", 0.8)
        params.setdefault("reg_lambda", 5.0)
        params.setdefault("min_child_samples", 100)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbose", -1)
        return SklearnModel(spec, LGBMClassifier(random_state=seed, **params))

    raise ValueError(f"unknown model kind: {kind!r}")


def default_specs() -> list[ModelSpec]:
    return [
        ModelSpec("majority", "majority", is_baseline=True, needs_imputation=False),
        ModelSpec("always_long", "always_long", is_baseline=True, needs_imputation=False),
        ModelSpec("persistence", "persistence", is_baseline=True, needs_imputation=False),
        ModelSpec("logistic", "logistic", needs_scaling=True),
        ModelSpec(
            "elastic_net",
            "elastic_net_logistic",
            params={"C": 0.1, "l1_ratio": 0.5},
            needs_scaling=True,
        ),
        ModelSpec("random_forest", "random_forest"),
        ModelSpec("xgboost", "xgboost", handles_nan=True),
        ModelSpec("lightgbm", "lightgbm", handles_nan=True),
    ]
