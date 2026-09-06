"""The splitter is where most time-series projects leak. These tests are the guard."""

from __future__ import annotations

import pandas as pd
import pytest

from market_forecast.validation.splits import (
    ROLLING,
    WalkForwardSplitter,
    assert_no_overlap,
    evaluated_sessions,
    label_reach,
)

SESSIONS = 5200


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2006-01-02", periods=SESSIONS, name="date")


@pytest.fixture(params=[1, 5])
def horizon(request) -> int:
    return request.param


@pytest.fixture
def folds(walkforward_config, dates, horizon):
    return WalkForwardSplitter(walkforward_config, horizon).split(dates)


class TestGeometry:
    def test_produces_several_folds(self, folds):
        assert len(folds) > 10

    def test_windows_never_touch(self, folds):
        for fold in folds:
            assert_no_overlap(fold)

    def test_ordering_is_fit_then_inner_then_test(self, folds):
        for fold in folds:
            assert fold.fit[1] < fold.inner[0] < fold.inner[1] < fold.test[0] < fold.test[1]

    def test_gap_exceeds_the_horizon(self, folds, horizon, walkforward_config):
        for fold in folds:
            assert fold.gap == horizon + walkforward_config.embargo_sessions
            assert fold.gap > horizon

    def test_test_windows_are_the_configured_length(self, folds, walkforward_config):
        for fold in folds:
            assert fold.n_test == walkforward_config.test_sessions

    def test_inner_windows_are_the_configured_length(self, folds, walkforward_config):
        for fold in folds:
            assert fold.n_inner == walkforward_config.inner_validation_sessions

    def test_folds_advance_in_time(self, folds):
        starts = [fold.test[0] for fold in folds]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)

    def test_test_windows_do_not_overlap_each_other(self, folds):
        coverage = evaluated_sessions(folds)
        assert len(coverage) == coverage.nunique()

    def test_every_fold_stays_inside_the_available_sessions(self, folds, dates):
        for fold in folds:
            assert fold.fit[0] >= 0
            assert fold.test[1] < len(dates)


class TestNoLabelLeaks:
    """The property that matters: no label used for fitting is defined by a later window."""

    def test_last_fit_label_resolves_before_the_inner_window(self, folds, dates, horizon):
        for fold in folds:
            reach = label_reach(dates, fold.fit[1], horizon)
            assert reach < dates[fold.inner[0]], f"fold {fold.index}"

    def test_last_inner_label_resolves_before_the_test_window(self, folds, dates, horizon):
        for fold in folds:
            reach = label_reach(dates, fold.inner[1], horizon)
            assert reach < dates[fold.test[0]], f"fold {fold.index}"

    def test_removing_the_purge_would_leak(self, walkforward_config, dates):
        """Negative control: with no purge or embargo the boundary label overlaps the test set."""
        horizon = 5
        unguarded = walkforward_config.model_copy(
            update={"purge_sessions": 0, "embargo_sessions": 0}
        )
        folds = WalkForwardSplitter(unguarded, horizon).split(dates)
        leaked = [
            fold.index
            for fold in folds
            if label_reach(dates, fold.inner[1], horizon) >= dates[fold.test[0]]
        ]
        assert leaked, "the unguarded splitter should leak, otherwise this test proves nothing"

    def test_a_longer_horizon_pushes_the_fit_window_back(self, walkforward_config, dates):
        short = WalkForwardSplitter(walkforward_config, 1).split(dates)
        long = WalkForwardSplitter(walkforward_config, 5).split(dates)
        assert len(short) == len(long)
        for a, b in zip(short, long, strict=True):
            assert a.test == b.test
            assert b.fit[1] < a.fit[1]


class TestSchemes:
    def test_anchored_folds_all_start_at_the_beginning(self, walkforward_config, dates):
        for fold in WalkForwardSplitter(walkforward_config, 5).split(dates):
            assert fold.fit[0] == 0

    def test_anchored_training_data_grows(self, walkforward_config, dates):
        sizes = [f.n_fit for f in WalkForwardSplitter(walkforward_config, 5).split(dates)]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_rolling_windows_are_fixed_width(self, walkforward_config, dates):
        rolling = walkforward_config.model_copy(update={"scheme": ROLLING})
        sizes = {f.n_fit for f in WalkForwardSplitter(rolling, 5).split(dates)}
        assert sizes == {walkforward_config.rolling_train_sessions}

    def test_both_schemes_evaluate_the_same_test_windows(self, walkforward_config, dates):
        rolling = walkforward_config.model_copy(update={"scheme": ROLLING})
        anchored_folds = WalkForwardSplitter(walkforward_config, 5).split(dates)
        rolling_folds = WalkForwardSplitter(rolling, 5).split(dates)
        assert [f.test for f in anchored_folds] == [f.test for f in rolling_folds]

    def test_rejects_an_unknown_scheme(self, walkforward_config):
        with pytest.raises(ValueError, match="scheme"):
            WalkForwardSplitter(walkforward_config.model_copy(update={"scheme": "sliding"}), 5)

    def test_rejects_a_non_positive_horizon(self, walkforward_config):
        with pytest.raises(ValueError, match="horizon"):
            WalkForwardSplitter(walkforward_config, 0)

    def test_raises_when_the_series_is_too_short(self, walkforward_config):
        with pytest.raises(ValueError, match="no folds"):
            WalkForwardSplitter(walkforward_config, 5).split(
                pd.bdate_range("2020-01-01", periods=200)
            )


class TestMasksOnAPooledPanel:
    @pytest.fixture
    def panel_dates(self, dates) -> pd.Series:
        tickers = ["AAA", "BBB", "CCC"]
        repeated = dates.repeat(len(tickers))
        return pd.Series(repeated, index=range(len(repeated)))

    def test_masks_select_disjoint_rows(self, folds, panel_dates):
        for fold in folds[:3]:
            fit = fold.mask(panel_dates, "fit")
            inner = fold.mask(panel_dates, "inner")
            test = fold.mask(panel_dates, "test")
            assert not (fit & inner).any()
            assert not (inner & test).any()
            assert not (fit & test).any()

    def test_masks_select_whole_sessions_across_every_ticker(self, folds, panel_dates):
        fold = folds[0]
        selected = panel_dates[fold.mask(panel_dates, "test")]
        assert selected.value_counts().nunique() == 1

    def test_no_session_appears_on_both_sides_of_a_boundary(self, folds, panel_dates):
        for fold in folds[:3]:
            fit_dates = set(panel_dates[fold.mask(panel_dates, "fit")])
            test_dates = set(panel_dates[fold.mask(panel_dates, "test")])
            assert not fit_dates & test_dates
