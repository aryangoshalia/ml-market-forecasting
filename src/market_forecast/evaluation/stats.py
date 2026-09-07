"""Significance testing for fold-level results.

Two properties of this data make naive testing wrong. Labels at horizon h overlap, so
consecutive observations are not independent and the effective sample is far smaller
than the row count. And returns are correlated across tickers on the same day, so rows
within a session are not independent either. Everything here therefore treats the fold,
not the row, as the unit of observation, and resamples folds in contiguous blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


@dataclass
class PairedTest:
    model: str
    reference: str
    n_folds: int
    mean_difference: float
    wins: int
    statistic: float
    p_value: float

    def summary(self) -> str:
        return (
            f"{self.model} vs {self.reference}: {self.mean_difference:+.4f} "
            f"over {self.n_folds} folds, wins {self.wins}/{self.n_folds}, p={self.p_value:.3f}"
        )


def paired_fold_test(
    scores: np.ndarray,
    reference: np.ndarray,
    model: str = "model",
    reference_name: str = "baseline",
) -> PairedTest:
    """Wilcoxon signed-rank over per-fold scores.

    Paired and non-parametric: folds differ in difficulty, and the distribution of the
    per-fold difference is not close to normal at this sample size.
    """
    paired = np.isfinite(scores) & np.isfinite(reference)
    a, b = scores[paired], reference[paired]
    if len(a) < 3:
        return PairedTest(
            model, reference_name, len(a), float("nan"), 0, float("nan"), float("nan")
        )

    difference = a - b
    if np.allclose(difference, 0.0):
        return PairedTest(model, reference_name, len(a), 0.0, 0, float("nan"), 1.0)

    statistic, p_value = stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
    return PairedTest(
        model=model,
        reference=reference_name,
        n_folds=int(len(a)),
        mean_difference=float(difference.mean()),
        wins=int((difference > 0).sum()),
        statistic=float(statistic),
        p_value=float(p_value),
    )


def block_bootstrap_ci(
    values: np.ndarray, block_size: int = 4, draws: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Confidence interval for a fold-level mean, resampling contiguous blocks of folds."""
    clean = np.asarray(values, dtype="float64")
    clean = clean[np.isfinite(clean)]
    if len(clean) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(clean)
    size = int(min(block_size, n))
    starts = np.arange(n - size + 1)
    needed = int(np.ceil(n / size))

    means = np.empty(draws)
    for draw in range(draws):
        chosen = rng.choice(starts, size=needed, replace=True)
        sample = np.concatenate([clean[s : s + size] for s in chosen])[:n]
        means[draw] = sample.mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


@dataclass
class PermutationNull:
    observed: float
    mean: float
    std: float
    p_value: float
    quantile_95: float
    draws: int

    def summary(self) -> str:
        return (
            f"observed {self.observed:.4f}, null {self.mean:.4f} +/- {self.std:.4f}, "
            f"95th percentile {self.quantile_95:.4f}, p={self.p_value:.4f}"
        )


def permutation_null_auc(
    target: np.ndarray,
    probabilities: np.ndarray,
    blocks: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    draws: int = 500,
    seed: int = 0,
) -> PermutationNull:
    """Distribution of AUC under the null that predictions carry no information.

    Answers the question an AUC of 0.52 actually raises: given this sample size and this
    dependence structure, how large would AUC be by chance alone?

    ``strata`` matters when predictions from several folds are pooled. A model that only
    ever predicts its own fold's base rate is constant within a fold, yet scores above
    0.5 on the pooled set whenever training and test base rates move together, which they
    do here. Permuting labels inside each stratum keeps that structure in the null, so the
    test measures skill rather than base-rate drift.
    """
    y = np.asarray(target, dtype="float64")
    p = np.asarray(probabilities, dtype="float64")
    usable = np.isfinite(y) & np.isfinite(p)
    y, p = y[usable], p[usable]
    if len(np.unique(y)) < 2:
        return PermutationNull(
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0
        )

    observed = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)

    if strata is not None:
        stratum_ids = np.asarray(strata)[usable]
        within = [np.flatnonzero(stratum_ids == s) for s in np.unique(stratum_ids)]
    else:
        within = None

    if blocks is not None:
        block_ids = np.asarray(blocks)[usable]
        groups = [np.flatnonzero(block_ids == b) for b in np.unique(block_ids)]
    else:
        groups = None

    null = np.empty(draws)
    for draw in range(draws):
        if within is not None:
            shuffled = y.copy()
            for members in within:
                shuffled[members] = rng.permutation(y[members])
        elif groups is None:
            shuffled = rng.permutation(y)
        else:
            shuffled = y.copy()
            order = rng.permutation(len(groups))
            for source, destination in zip(order, range(len(groups)), strict=True):
                take = groups[source]
                place = groups[destination]
                size = min(len(take), len(place))
                shuffled[place[:size]] = y[take[:size]]
        null[draw] = roc_auc_score(shuffled, p)

    return PermutationNull(
        observed=observed,
        mean=float(null.mean()),
        std=float(null.std(ddof=1)),
        p_value=float((null >= observed).mean()),
        quantile_95=float(np.quantile(null, 0.95)),
        draws=draws,
    )


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Adjusted p-values. We compare many models across several targets and horizons, so
    the nominal values overstate significance."""
    p = np.asarray(p_values, dtype="float64")
    finite = np.isfinite(p)
    adjusted = np.full_like(p, np.nan)
    if not finite.any():
        return adjusted

    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    scaled = ranked * n / np.arange(1, n + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]

    result = np.empty(n)
    result[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite] = result
    del alpha
    return adjusted


def permutation_null_pooled(
    target: np.ndarray,
    probabilities: np.ndarray,
    dates: np.ndarray,
    folds: np.ndarray,
    draws: int = 300,
    seed: int = 0,
) -> PermutationNull:
    """Null for predictions pooled across walk-forward folds.

    Two dependencies have to survive into the null or the test overstates significance.
    Base rates drift between folds, and a predictor that only tracks its own fold's base
    rate already scores above 0.5 on the pooled set. And on any one session the tickers
    move together, so their labels are not independent draws.

    Whole sessions are therefore permuted within each fold: the label vector for one day
    is swapped with another day's from the same fold. Fold base rates and same-day
    cross-sectional structure both survive; only the link between a prediction and its
    own outcome is broken.
    """
    y = np.asarray(target, dtype="float64")
    p = np.asarray(probabilities, dtype="float64")
    usable = np.isfinite(y) & np.isfinite(p)
    y, p = y[usable], p[usable]
    day = np.asarray(dates)[usable]
    fold = np.asarray(folds)[usable]

    if len(np.unique(y)) < 2:
        return PermutationNull(*(float("nan"),) * 5, 0)

    observed = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)

    sessions_by_fold: list[list[np.ndarray]] = []
    for f in np.unique(fold):
        in_fold = fold == f
        sessions_by_fold.append(
            [np.flatnonzero(in_fold & (day == d)) for d in np.unique(day[in_fold])]
        )

    null = np.empty(draws)
    for draw in range(draws):
        shuffled = y.copy()
        for sessions in sessions_by_fold:
            order = rng.permutation(len(sessions))
            for source, destination in enumerate(order):
                take, place = sessions[destination], sessions[source]
                size = min(len(take), len(place))
                shuffled[place[:size]] = y[take[:size]]
        null[draw] = roc_auc_score(shuffled, p)

    return PermutationNull(
        observed=observed,
        mean=float(null.mean()),
        std=float(null.std(ddof=1)),
        p_value=float((null >= observed).mean()),
        quantile_95=float(np.quantile(null, 0.95)),
        draws=draws,
    )
