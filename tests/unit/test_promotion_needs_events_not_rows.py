"""The gate counted rows and called it evidence.

`_evaluate_promotion_gate` asked three things: was there a real holdout, was it
at least 20 rows, and did the winner beat the naive baseline on it. It never
asked whether the holdout contained enough of the thing being predicted for
the answer to mean anything.

So a binary target with 331 holdout rows and three positive events passed the
row check trivially, and its precision — an estimate with a 95% interval about
thirty points wide — went into every aggregate beside figures that meant
something. Measured on the 2026-08-15 batch: 36 of the 58 promoted contexts
carried fewer than ten events.

This is not something more data fixes, which is the point. Extending the
hourly history from 183 to 725 days moved rows per context from 86 to 331 and
events per context from 5 to 7, because the binding constraint is how finely
the data is partitioned (ticker x timeframe x target x regime), not how much
of it there is. Hence a gate.

Ten is a floor, not a target. Below it a proportion cannot separate skill from
chance; above it the comparison against the baseline starts to carry weight.
`min_holdout_events: 0` restores the previous behaviour.
"""
import numpy as np
import pandas as pd
import pytest

from src.training.base_trainer import BaseTrainer


class _Trainer(BaseTrainer):
    """Concrete only so the abstract base can be exercised."""

    def _prepare_ticker_groups(self, *args, **kwargs):  # pragma: no cover
        return {}

    def _train_ticker_group(self, *args, **kwargs):  # pragma: no cover
        return {}


@pytest.fixture
def trainer():
    return _Trainer()


def _metrics(events=None, rows=331, score=0.8, baseline=0.5):
    out = {
        "status": "measured",
        "holdout_sample_count": rows,
        "score": score,
        "baseline_score": baseline,
    }
    if events is not None:
        out["holdout_event_count"] = events
    return {"winner_holdout_metrics": out}


# --- counting the events -------------------------------------------------


def test_a_binary_holdout_reports_how_often_it_happened():
    counted = BaseTrainer._holdout_event_count(np.array([1] * 7 + [0] * 324), True)

    assert counted["holdout_event_count"] == 7
    assert counted["holdout_event_rate"] == pytest.approx(7 / 331)


def test_a_multiclass_holdout_reports_its_scarcest_class():
    """The rarest class is what bounds the estimate, not the commonest."""
    counted = BaseTrainer._holdout_event_count(
        np.array([0] * 50 + [1] * 30 + [2] * 5), True
    )

    assert counted["holdout_event_count"] == 5


def test_a_regression_holdout_reports_nothing_rather_than_zero():
    """'Event' has no meaning for a continuous target.

    Absent, not zero: the gate must be able to tell "not applicable" from
    "nothing happened", or every regression context fails a check that does
    not apply to it.
    """
    assert BaseTrainer._holdout_event_count(np.random.randn(100), False) == {}


# --- the gate ------------------------------------------------------------


def test_a_handful_of_events_no_longer_promotes(trainer):
    verdict = trainer._evaluate_promotion_gate(_metrics(events=3))

    assert verdict["passed"] is False
    assert "3 events" in " ".join(verdict["reasons"])


def test_the_median_context_of_the_last_run_is_refused(trainer):
    """Seven was the median after quadrupling the hourly history."""
    assert trainer._evaluate_promotion_gate(_metrics(events=7))["passed"] is False


def test_enough_events_still_promotes(trainer):
    assert trainer._evaluate_promotion_gate(_metrics(events=40))["passed"] is True


def test_the_boundary_is_inclusive(trainer):
    assert trainer._evaluate_promotion_gate(_metrics(events=10))["passed"] is True
    assert trainer._evaluate_promotion_gate(_metrics(events=9))["passed"] is False


def test_a_regression_winner_is_not_judged_on_events(trainer):
    """No event count means the check does not apply, not that it failed."""
    assert trainer._evaluate_promotion_gate(_metrics(events=None))["passed"] is True


def test_rows_and_events_are_different_questions(trainer):
    """331 rows is plenty; three events is not, and only one used to count."""
    plenty_of_rows = _metrics(events=3, rows=5000)

    verdict = trainer._evaluate_promotion_gate(plenty_of_rows)

    assert verdict["passed"] is False
    assert "rows" not in " ".join(verdict["reasons"]).split("events")[0][-40:]
