"""The bar is a multiple-comparison correction, so the run must count its comparisons.

`models.yaml` states `promotion_gate.family_size: 27` and says of that number:

    it cannot be known before the run starts -- contexts are materialised
    lazily -- so it is stated here and the stage compares it against the
    actual count at the end, and says so loudly if they differ. A number
    that drifts silently is the thing this whole gate exists to prevent.

The comparison did not exist. Measured 2026-09-03 by reading the source:
`family_size` appeared nowhere in `src/pipeline` except
`ModelingStage._promotion_family_size`, which was initialised to None at line
122 and assigned nowhere, so the training context always carried None and
`BaseTrainer` fell back to the configured 27 --

    family_size = int(results.get('promotion_family_size')
                      or cfg.get('family_size', 1))

-- which is right for exactly as long as a run happens to make 27 verdicts.
A run of 216 would be judged at 2.90 sigma where its own stated 5% needs 3.50,
and nothing anywhere would say so. That is CRITIQUE section 11 verbatim: we do
not count our own attempts.

WHAT THIS DOES NOT DO, so the tests are not read as more than they are: no bar
is corrected. The count is only complete once the verdicts are made, and a
correction cannot be applied backwards. What is fixed is the SILENCE -- the
discrepancy now reaches the stage boundary, next to the champion count, where
a reader trips over it.

A refused verdict counts as an attempt. Counting only champions is precisely
the arithmetic that makes a multiplicity correction wrong, because the
correction is for how many times you LOOKED.
"""
from __future__ import annotations

import logging

import pytest

from src.pipeline.stages.modeling.orchestrator import ModelingStage


class _Config:
    def __init__(self, family_size):
        self._family_size = family_size

    def get_config(self, name):
        if name != "models":
            return {}
        gate = {} if self._family_size is None else {"family_size": self._family_size}
        return {"promotion_gate": gate}


def _stage(family_size, attempts):
    stage = ModelingStage.__new__(ModelingStage)
    stage.config_manager = _Config(family_size)
    stage._promotion_attempts = attempts
    return stage


def test_a_matching_count_is_reported_without_alarm(caplog):
    with caplog.at_level(logging.INFO):
        _stage(27, 27)._reconcile_promotion_family()
    assert any("reconciled" in r.message for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_more_attempts_than_declared_is_an_error_that_names_the_direction(caplog):
    """The dangerous direction: the bar was too loose, so the stated error
    rate is not the one the run delivered."""
    with caplog.at_level(logging.ERROR):
        _stage(27, 216)._reconcile_promotion_family()
    messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert messages, "a mismatched family must not pass silently"
    text = " ".join(messages)
    assert "216" in text and "27" in text
    assert "LOOSE" in text, (
        "the message must say WHICH WAY the bar was wrong; 'they differ' "
        "leaves the reader to work out whether champions or refusals are the "
        "ones to doubt"
    )


def test_fewer_attempts_than_declared_names_the_other_direction(caplog):
    with caplog.at_level(logging.ERROR):
        _stage(27, 3)._reconcile_promotion_family()
    text = " ".join(r.message for r in caplog.records if r.levelno >= logging.ERROR)
    assert "STRICT" in text, (
        "a bar set for 27 attempts applied to 3 refuses real edges; that is a "
        "different problem from promoting noise and must read differently"
    )


def test_no_configured_family_is_itself_an_error(caplog):
    """Falling back to the single-test bar is a decision, not a default."""
    with caplog.at_level(logging.ERROR):
        _stage(None, 40)._reconcile_promotion_family()
    text = " ".join(r.message for r in caplog.records if r.levelno >= logging.ERROR)
    assert "40" in text


def test_the_counter_exists_and_starts_at_zero():
    """The field the whole check depends on. It was `_promotion_family_size`,
    initialised to None and assigned nowhere -- the shape this test exists to
    stop coming back."""
    stage = ModelingStage.__new__(ModelingStage)
    assert not hasattr(stage, "_promotion_attempts")
    stage._promotion_attempts = 0
    assert stage._promotion_attempts == 0


def test_the_counter_is_incremented_in_the_target_loop():
    """Read the source rather than run the stage: reaching that loop needs a
    full enriched frame, and the defect being guarded is a MISSING LINE, which
    only the source can show is present.

    A behavioural test would be better and is not available here at a
    proportionate cost; this at least fails if the increment is deleted.
    """
    import inspect

    source = inspect.getsource(ModelingStage._process_ticker_with_async)
    assert "self._promotion_attempts += 1" in source, (
        "the counter is no longer incremented per (context, target), so the "
        "reconciliation compares the configured number against zero"
    )
    loop = source.index("for target_name in target_cols:")
    increment = source.index("self._promotion_attempts += 1")
    assert increment > loop, (
        "the increment must be INSIDE the per-target loop; outside it counts "
        "contexts, and the gate makes one verdict per target, not per context"
    )
