"""Feature drift has to be measurable, and a sample has to say it is one.

WHY. REGISTER #221 has stood open since 2026-09-01: feature drift has never
been measured in this project's history. The mechanism is worth stating in
full because every piece of it looked correct on its own.

    The skip logic is right -- `feature_drift` reads a frame the context loop
    does not vary, so running it once is not a shortcut, 330 identical
    computations cannot say more than one.

    The single run timed out at 90s.

    The other 329 contexts then reported `skipped_inputs_unchanged`, a state
    whose comment reads "a second run can only repeat the first answer" --
    true, and the first answer was a failure.

So a check that never ran once read, from outside, as a check that ran and
found nothing.

WHAT THE MEASUREMENT FOUND. `MAX_DRIFT_FEATURES = 100` caps COLUMNS, and its
own comment says it exists "so Evidently does not stall for minutes". Nothing
capped ROWS. The cap was put on one axis of a two-axis cost. Timed against
the real monitor with 100 features:

     5,000 rows   22.3s      50,000 rows    31.1s
    20,000 rows   20.3s     200,000 rows    over five minutes

And capping the rows was not enough on its own: taking `[cols].copy()` and
sampling AFTER it duplicated 623,398 x 100 float32 to keep 50,000 rows, and
still measured 109.8s. Selecting, slicing, then copying takes the same rows
in 66.9s -- the fourth defect family in the audit method, found inside the fix
for the third.

WHAT IS PINNED HERE: that both caps exist, that the row sample is EVENLY
SPACED rather than the head (these frames are time-ordered, so a head sample
compares the oldest slice of one period against the oldest of another), and
that the result says how much it looked at. A 50,000-row answer presented as
the whole frame is the same lie as "no drift detected" over 100 of 1,940
features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.feature_drift_monitor import FeatureDriftMonitor


def test_both_axes_are_capped():
    """The defect this file exists for: one axis was, the other was not."""
    assert isinstance(FeatureDriftMonitor.MAX_DRIFT_FEATURES, int)
    assert isinstance(FeatureDriftMonitor.MAX_DRIFT_ROWS, int), (
        "MAX_DRIFT_ROWS is gone, so a 623,398-row frame goes to Evidently "
        "whole again and the single drift run times out as it did for the "
        "life of the project (REGISTER #221)"
    )
    assert 0 < FeatureDriftMonitor.MAX_DRIFT_ROWS <= 100_000, (
        "the row cap is outside the range that was measured to fit the budget"
    )


def test_the_row_sample_spans_the_frame_rather_than_its_head():
    """Head sampling on a time-ordered frame compares the oldest rows of one
    period with the oldest of another and calls the difference drift."""
    monitor = FeatureDriftMonitor.__new__(FeatureDriftMonitor)
    frame = pd.DataFrame({"x": np.arange(FeatureDriftMonitor.MAX_DRIFT_ROWS * 4)})

    sample = monitor._even_row_sample(frame)

    assert len(sample) == FeatureDriftMonitor.MAX_DRIFT_ROWS
    # A head sample would end a quarter of the way in.
    assert sample["x"].iloc[-1] > len(frame) * 0.9, (
        "the sample stops early, so it is a head slice rather than a stride "
        "across the frame"
    )
    assert sample["x"].iloc[0] < len(frame) * 0.01


def test_a_small_frame_is_returned_untouched():
    """A cap that copies or reindexes when it does not need to is a cost with
    no benefit, and this one runs on every check."""
    monitor = FeatureDriftMonitor.__new__(FeatureDriftMonitor)
    frame = pd.DataFrame({"x": np.arange(100)})
    assert monitor._even_row_sample(frame) is frame


def test_the_sample_keeps_the_column_it_was_given():
    monitor = FeatureDriftMonitor.__new__(FeatureDriftMonitor)
    frame = pd.DataFrame(
        {"a": np.arange(200_000), "b": np.arange(200_000) * 2.0})
    sample = monitor._even_row_sample(frame)
    assert list(sample.columns) == ["a", "b"]
    assert (sample["b"] == sample["a"] * 2.0).all(), (
        "rows were mixed between columns, which would fabricate drift"
    )


def test_the_monitor_selects_before_it_copies():
    """Pinned because the naive order measured 109.8s against a 90s budget
    WITH the row cap in place -- copying 623,398 rows to keep 50,000."""
    import inspect

    source = inspect.getsource(FeatureDriftMonitor.check_drift)
    sampled = source.index("_even_row_sample")
    # The copy that matters is the one building the frames handed to Evidently.
    assert "_even_row_sample(self.reference_data[valid_common]).copy()" in source, (
        "the drift frames are copied before being sampled again; that "
        "duplicates the whole frame to keep a fraction of it"
    )
    assert sampled > 0


@pytest.mark.parametrize("key", ["rows_checked", "rows_available", "rows_sampled"])
def test_the_result_states_how_much_it_looked_at(key):
    """The feature cap already carries this honesty; the row cap must too. A
    50,000-row answer read as the whole frame is the same lie as "no drift"
    over 100 of 1,940 features."""
    import inspect

    source = inspect.getsource(FeatureDriftMonitor.check_drift)
    assert f"'{key}'" in source, (
        f"{key} is no longer reported, so a sampled answer is indistinguishable "
        f"from a complete one"
    )


def test_the_invariant_analyzers_get_their_own_budget():
    """A budget paid once is not a budget paid 330 times. The 90s per-context
    limit exists to bound the stage; the invariant analyzers run in the first
    context only, and judging them by it is what kept #221 open."""
    import inspect

    from src.pipeline.stages.evaluation.orchestrator import EvaluationStage

    source = inspect.getsource(EvaluationStage)
    assert "invariant_timeout_seconds" in source, (
        "the first context is back on the per-context budget, so the one run "
        "of feature_drift is judged by a limit sized for 330 of them"
    )
    assert "context_timeout_seconds" in source, (
        "the per-context budget is gone; without it a slow analyzer can run "
        "for the invariant budget in every one of the contexts"
    )
