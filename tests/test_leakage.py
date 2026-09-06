"""The audits that make the no-leakage claim testable rather than asserted.

Each auditor is checked twice: it must pass on the real pipeline, and it must fail on a
deliberately broken feature. A test that only ever passes proves nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_forecast.features.pipeline import FeaturePipeline
from market_forecast.targets import binary_direction, forward_return
from market_forecast.validation.leakage import (
    audit_perturbation,
    audit_scale_invariance,
    audit_target_causality,
    audit_truncation,
    corrupt_after,
)


@pytest.fixture
def build(feature_config):
    pipeline = FeaturePipeline(feature_config)
    return lambda frame, context: pipeline.build("TEST", frame, context).frame


class TestCorruptionHelper:
    def test_leaves_history_untouched(self, prices):
        corrupted = corrupt_after(prices, 500)
        pd.testing.assert_frame_equal(prices.iloc[:501], corrupted.iloc[:501])

    def test_actually_changes_the_future(self, prices):
        corrupted = corrupt_after(prices, 500)
        assert not np.allclose(prices["close"].iloc[501:], corrupted["close"].iloc[501:])

    def test_produces_bars_that_still_make_sense(self, prices):
        corrupted = corrupt_after(prices, 500)
        assert (corrupted["high"] >= corrupted["low"]).all()
        assert (corrupted[["open", "high", "low", "close"]] > 0).all().all()


class TestPipelinePassesEveryAudit:
    def test_no_feature_reads_forward(self, build, prices, context):
        report = audit_perturbation(build, prices, context)
        assert report.ok, [f.detail for f in report.errors[:5]]
        assert report.checks_run > 0

    def test_features_do_not_depend_on_data_arriving_later(self, build, prices, context):
        report = audit_truncation(build, prices, context)
        assert report.ok, [f.detail for f in report.errors[:5]]

    def test_every_feature_is_scale_free(self, build, prices, context):
        report = audit_scale_invariance(build, prices, context)
        assert report.ok, [f.detail for f in report.errors[:5]]

    @pytest.mark.parametrize("horizon", [1, 5])
    def test_targets_look_exactly_h_sessions_forward(self, prices, horizon):
        report = audit_target_causality(
            prices, horizon, lambda frame: binary_direction(forward_return(frame, horizon))
        )
        assert report.ok, [f.detail for f in report.errors[:5]]


class TestAuditsCatchPlantedFaults:
    """Negative controls. Each fault is one an ordinary pipeline could plausibly contain."""

    def test_catches_a_feature_built_from_tomorrow(self, build, prices, context):
        def leaky(frame, ctx):
            out = build(frame, ctx)
            out["next_day_return"] = (frame["close"].shift(-1) / frame["close"] - 1).reindex(
                out.index
            )
            return out

        report = audit_perturbation(leaky, prices, context)
        assert not report.ok
        assert report.errors[0].feature == "next_day_return"

    def test_catches_a_centre_weighted_rolling_window(self, build, prices, context):
        def centred(frame, ctx):
            out = build(frame, ctx)
            average = frame["close"].rolling(21, center=True).mean()
            out["centred_ma"] = (frame["close"] / average).reindex(out.index)
            return out

        assert not audit_perturbation(centred, prices, context).ok

    def test_catches_normalisation_fitted_on_the_whole_sample(self, build, prices, context):
        def global_z(frame, ctx):
            out = build(frame, ctx)
            returns = frame["close"].pct_change()
            out["full_sample_z"] = ((returns - returns.mean()) / returns.std()).reindex(out.index)
            return out

        # the full-sample mean shifts once later rows are removed, which truncation exposes
        assert not audit_truncation(global_z, prices, context).ok

    def test_catches_a_leak_through_the_market_context(self, build, prices, context):
        def leaky_context(frame, ctx):
            out = build(frame, ctx)
            benchmark = ctx.benchmark["close"].reindex(frame.index).ffill()
            out["benchmark_tomorrow"] = (benchmark.shift(-1) / benchmark - 1).reindex(out.index)
            return out

        assert not audit_perturbation(leaky_context, prices, context).ok

    def test_catches_a_backward_fill(self, build, prices, context):
        def back_filled(frame, ctx):
            out = build(frame, ctx)
            sparse = frame["close"].copy()
            sparse.iloc[::3] = np.nan
            out["back_filled"] = (sparse.bfill() / frame["close"]).reindex(out.index)
            return out

        assert not audit_perturbation(back_filled, prices, context).ok

    def test_catches_a_raw_price_level(self, build, prices, context):
        def raw_level(frame, ctx):
            out = build(frame, ctx)
            out["raw_sma_50"] = frame["close"].rolling(50).mean().reindex(out.index)
            return out

        report = audit_scale_invariance(raw_level, prices, context)
        assert not report.ok
        assert report.errors[0].feature == "raw_sma_50"

    def test_catches_a_raw_volume_level(self, build, prices, context):
        def raw_volume(frame, ctx):
            out = build(frame, ctx)
            out["raw_volume"] = frame["volume"].reindex(out.index)
            return out

        assert not audit_scale_invariance(raw_volume, prices, context).ok

    def test_catches_a_target_that_reaches_past_its_horizon(self, prices):
        report = audit_target_causality(
            prices, 5, lambda frame: binary_direction(forward_return(frame, 15))
        )
        assert not report.ok
        assert "reaches too far forward" in report.errors[0].detail

    def test_catches_a_target_that_does_not_look_forward_at_all(self, prices):
        report = audit_target_causality(
            prices, 5, lambda frame: binary_direction(frame["close"].pct_change())
        )
        assert not report.ok
        assert "does not depend on its own horizon" in report.errors[0].detail
