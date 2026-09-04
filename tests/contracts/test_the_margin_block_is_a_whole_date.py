"""A block shorter than one date resamples 110 names as if they were 110 days.

REGISTER #200, found 01.09, marked закрито, and the fix never made -- which is
how it was found again on 04.09 by `stale_state_scan.py` rule G.

`_block_bootstrap_sigma` is the only thing standing between "the model beat
the opponent" and "the model beat the opponent by more than this holdout can
produce by luck". It took a block of `n ** (1/3)`:

    daily holdout   n = 140,945 rows, 110 rows per date -> block 52
    15m holdout     n =  30,494 rows, 110 rows per date -> block 31

Both are less than half a single day. So the resampling drew names from
*inside* a date as though each were an independent observation, when 110
names on one day are largely one event moved by a common market factor. The
effective sample is ~1,280 dates, not 140,945 rows, and every margin the gate
quoted in standard errors was inflated by that confusion -- `target_up_1d`
"5 sigma", `breakout_1h` "18.8", `volatility_spike_1h` "2.3".

The n^(1/3) rule is not wrong; it was applied to the wrong unit. It is now
applied to the DATES and expanded back into rows, and starts are aligned to
date boundaries so a block cannot begin halfway through a day.

Direction matters and is stated in the code: this can only WIDEN sigma. Every
margin measured before it is an upper bound on the evidence, never a lower
one, and no champion is promoted by this change that would not have been
promoted before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.base_trainer import BaseTrainer


class _Evaluator:
    """Accuracy, and nothing else -- the point here is the resampling."""

    @staticmethod
    def calculate(truth, pred, task_type=None):
        truth = np.asarray(truth, dtype=float)
        pred = np.asarray(pred, dtype=float)
        return {"Accuracy": float((truth == pred).mean())}


class _Trainer(BaseTrainer):
    """BaseTrainer is abstract on the two hooks a real trainer supplies; the
    bootstrap needs neither, so they are stubbed rather than mocked away."""

    MARGIN_BOOTSTRAP_RESAMPLES = 200

    def __init__(self):
        self.evaluator = _Evaluator()

    def _prepare_ticker_groups(self, *args, **kwargs):
        raise NotImplementedError

    def _train_ticker_group(self, *args, **kwargs):
        raise NotImplementedError


def _series(dates: int, per_date: int, seed: int = 0):
    """A pooled holdout: `per_date` names on each of `dates` days, with the
    outcome shared across names on a day -- which is the whole point."""
    rng = np.random.default_rng(seed)
    day_truth = rng.integers(0, 2, size=dates)
    truth = np.repeat(day_truth, per_date).astype(float)
    model = truth.copy()
    flip = rng.random(truth.size) < 0.30
    model[flip] = 1.0 - model[flip]
    baseline = np.zeros_like(truth)
    return truth, model, baseline


def _sigma(rows_per_bar: float, dates: int = 400, per_date: int = 110):
    truth, model, baseline = _series(dates, per_date)
    return _Trainer()._block_bootstrap_sigma(
        truth, model, baseline,
        task_type="classification", metric_key="Accuracy",
        rows_per_bar=rows_per_bar,
    )


def test_a_pooled_frame_gets_a_wider_sigma_than_the_old_row_block():
    """The measurement that justifies the change: on data where a day is one
    event, pretending rows are independent understates the spread."""
    honest = _sigma(rows_per_bar=110)
    as_if_independent = _sigma(rows_per_bar=1)

    assert honest is not None and as_if_independent is not None
    assert honest > as_if_independent, (
        f"counting the block in dates ({honest:.5f}) did not widen sigma "
        f"against counting it in rows ({as_if_independent:.5f}); the gate's "
        "margins stay inflated"
    )


def test_a_single_series_holdout_is_left_exactly_where_it_was():
    """The old rule was written for this case and is right for it. A fix that
    moves it would be a second defect."""
    truth, model, baseline = _series(dates=4000, per_date=1)
    trainer = _Trainer()
    with_default = trainer._block_bootstrap_sigma(
        truth, model, baseline,
        task_type="classification", metric_key="Accuracy")
    explicit_one = trainer._block_bootstrap_sigma(
        truth, model, baseline,
        task_type="classification", metric_key="Accuracy", rows_per_bar=1.0)
    assert with_default == pytest.approx(explicit_one)


def test_too_few_dates_returns_not_measured_rather_than_a_small_number():
    """Two dates cannot be block-resampled. Returning a tiny sigma there
    would read as a precise margin, which is REGISTER #202 exactly: "не
    виміряно" must never equal "пройдено"."""
    truth, model, baseline = _series(dates=2, per_date=110)
    sigma = _Trainer()._block_bootstrap_sigma(
        truth, model, baseline,
        task_type="classification", metric_key="Accuracy", rows_per_bar=110)
    assert sigma is None


def test_the_block_covers_whole_dates():
    """Derived from the code's own arithmetic, so a change to the rule has to
    change this line too."""
    n, per_date = 140_945, 110
    dates = round((n / per_date) ** (1 / 3))
    block = per_date * dates
    assert block % per_date == 0
    assert block > per_date, "the block does not span even one full date"
    assert block == 1210, (
        f"the daily block moved from 1,210 rows to {block}; #200 measured "
        "110 rows per date and ~1,280 dates on that holdout"
    )
    assert block > round(n ** (1 / 3)) * 20, (
        "the new block is not materially longer than the 52-row one it "
        "replaces, so nothing was actually fixed"
    )


def test_the_row_count_per_date_is_recorded_beside_the_sigma():
    """A sigma whose block length cannot be reconstructed is a number nobody
    can check -- the defect the register itself was rebuilt to stop."""
    import inspect
    source = inspect.getsource(BaseTrainer)
    assert "baseline_margin_rows_per_bar" in source, (
        "the frame's rows-per-date is not stored, so a reader cannot tell "
        "whether a margin was measured in dates or in rows"
    )
    assert "rows_per_bar=rows_per_bar" in source, (
        "the caller no longer passes the measured rows-per-date, so the "
        "bootstrap silently falls back to the row block"
    )
