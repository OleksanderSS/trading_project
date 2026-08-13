"""Absence of evidence must not arrive at the agents as evidence of health.

PipelineBridge guards every threshold with ``is not None``, so a Stage 7 result
carrying no metrics failed no check and used to reach the agents as
``verdict="clear"`` -- an agent reading that concluded the model was fine when
nothing had been examined. The bridge had no tests at all, which is how it
survived.

While stages 0-3 are under repair, partial and empty Stage 7 output is the normal
case rather than the exception, so this is the shape that would mislead most
often. These tests pin the distinction between "checked and fine", "checked and
bad", and "never checked".
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.agents.pipeline_bridge import PipelineBridge


def _fresh_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _performance(result: dict) -> dict:
    return PipelineBridge().from_pipeline_result(result).metadata["model_performance"]


def test_empty_stage7_result_is_not_clear():
    performance = _performance({})

    assert performance["verdict"] == "caution"
    assert performance["evidence_status"] == "insufficient_evidence"
    assert "missing_evaluation_metrics" in performance["threshold_failures"]
    assert performance["performance_score"] is None


def test_empty_stage7_result_names_every_missing_input():
    performance = _performance({})

    assert performance["missing_evidence"] == [
        "validation_score",
        "sample_count",
        "evaluation_timestamp",
    ]


def test_a_metric_that_passes_its_threshold_cannot_clear_on_its_own():
    """A lone healthy sharpe is not a verdict about the model."""
    performance = _performance({"sharpe_ratio": 1.4})

    assert performance["verdict"] == "caution"
    assert performance["evidence_status"] == "insufficient_evidence"
    assert "validation_score" in performance["missing_evidence"]


def test_missing_timestamp_blocks_clear_because_staleness_cannot_be_checked():
    performance = _performance({"validation_score": 0.71, "n_samples": 400})

    assert performance["verdict"] == "caution"
    assert performance["missing_evidence"] == ["evaluation_timestamp"]


def test_complete_and_healthy_evidence_clears():
    performance = _performance(
        {"validation_score": 0.71, "n_samples": 400, "evaluated_at": _fresh_timestamp()}
    )

    assert performance["verdict"] == "clear"
    assert performance["evidence_status"] == "evaluated"
    assert performance["missing_evidence"] == []
    assert performance["threshold_failures"] == []


def test_checked_and_bad_is_distinguishable_from_never_checked():
    """Both are caution; only evidence_status separates them."""
    examined = _performance(
        {"validation_score": 0.31, "n_samples": 400, "evaluated_at": _fresh_timestamp()}
    )
    unexamined = _performance({})

    assert examined["verdict"] == unexamined["verdict"] == "caution"
    assert examined["evidence_status"] == "evaluated"
    assert unexamined["evidence_status"] == "insufficient_evidence"
    assert "validation_score_below_threshold" in examined["threshold_failures"]


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"sharpe_ratio": 1.4},
        {"validation_score": 0.71, "n_samples": 400},
    ],
)
def test_no_incomplete_result_ever_reaches_the_agents_as_clear(result):
    """The property that matters, stated once over every incomplete shape."""
    assert _performance(result)["verdict"] != "clear"


def test_consumers_that_only_understand_caution_still_react():
    """dean_os tuning and operations agents compare verdict == "caution" literally.

    A third verdict value would read as "not caution" -- i.e. as clear -- to every
    consumer that has not been updated, so insufficient evidence has to keep
    presenting as caution.
    """
    performance = _performance({})

    assert performance["verdict"] == "caution"
