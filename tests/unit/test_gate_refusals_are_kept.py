"""The gate says why it refused and used to keep no record of it.

Answering "why has no return target ever produced a champion" meant finding
the run's log and parsing 446 refusal lines out of it. That worked only
because the log happened to still exist, and what it found was worth having
as an artifact: 342 of the 446 were "does not beat the naive baseline", 24
were "too few events", and the numbers in each line are what separate "no
edge" from "not enough data to tell".

The next run writes them next to the holdout predictions instead.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.stages.modeling.orchestrator import ModelingStage


@pytest.fixture
def stage():
    instance = ModelingStage.__new__(ModelingStage)
    instance._gate_refusals = []
    return instance


def _result(target, reasons, **numbers):
    """Shaped like the real thing, which is the whole point.

    The first version of this helper put the numbers inside `promotion_gate`,
    because that is where the code looked for them. Both were wrong together,
    so the tests passed and the 2026-08-23 artifact came out with 497 rows and
    every numeric column null. The gate carries `passed` and `reasons`; the
    figures live in `winner_holdout_metrics`, under `score`,
    `holdout_sample_count` and `holdout_event_count`.
    """
    return {
        "ticker": "AAPL",
        "timeframe": "1d",
        "target_name": target,
        "winner": "catboost",
        "promotion_gate": {"passed": False, "reasons": reasons},
        "winner_holdout_metrics": {"status": "measured", **numbers},
    }


def test_a_refusal_keeps_the_reason_not_just_the_verdict(stage):
    stage._collect_gate_refusal(
        _result("target_return_1d", ["holdout score -0.0147 does not beat the naive baseline -0.0024"]),
        "AAPL_1d_target_return_1d",
    )
    kept = stage._gate_refusals[0]
    assert kept["target"] == "target_return_1d"
    assert "naive baseline" in kept["reasons"]
    assert kept["context"] == "AAPL_1d_target_return_1d"


def test_several_reasons_are_all_kept(stage):
    stage._collect_gate_refusal(
        _result("target_up_1d", ["too few events", "below chance on 2 of 4 folds"]),
        "AAPL_1d_target_up_1d",
    )
    reasons = stage._gate_refusals[0]["reasons"]
    assert "too few events" in reasons and "below chance" in reasons


def test_the_numbers_that_separate_no_edge_from_no_data_are_kept(stage):
    stage._collect_gate_refusal(
        _result("target_return_5d", ["x"], holdout_sample_count=18,
                holdout_event_count=4, score=-0.02, baseline_score=0.58,
                baseline_kind="persistence", baseline_constant_score=0.0,
                baseline_persistence_score=0.58),
        "AAPL_1d_target_return_5d",
    )
    kept = stage._gate_refusals[0]
    assert kept["holdout_rows"] == 18
    assert kept["holdout_events"] == 4
    assert kept["holdout_score"] == -0.02
    assert kept["baseline_score"] == 0.58
    assert kept["model_type"] == "catboost"
    # Which opponent won matters now that persistence was found to have been
    # reading the future on every multi-bar target.
    assert kept["baseline_kind"] == "persistence"
    assert kept["baseline_constant_score"] == 0.0
    assert kept["baseline_persistence_score"] == 0.58


def test_a_note_is_recorded_when_the_gate_itself_said_nothing(stage):
    """Indicator targets are refused by policy, not by the gate."""
    stage._collect_gate_refusal(
        {"ticker": "AAPL", "timeframe": "1d", "target_name": "target_sma_20_f1"},
        "AAPL_1d_target_sma_20_f1",
        note="indicator_prediction targets are measured but not promoted",
    )
    assert "not promoted" in stage._gate_refusals[0]["reasons"]


def test_nothing_is_written_when_every_context_won(stage):
    assert ModelingStage._write_gate_refusals([]) is None


def test_the_artifact_carries_one_row_per_refusal(tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    for target in ("target_return_1d", "target_up_1d", "target_return_1d"):
        stage._collect_gate_refusal(_result(target, ["no edge"]), f"AAPL_1d_{target}")

    path = ModelingStage._write_gate_refusals(stage._gate_refusals)

    assert path is not None and path.exists()
    frame = pd.read_parquet(path)
    assert len(frame) == 3
    assert set(frame["target"]) == {"target_return_1d", "target_up_1d"}
    assert frame["target"].value_counts()["target_return_1d"] == 2


def test_a_context_that_never_reached_training_is_still_counted(stage):
    """"No data to tell" and "no edge" are opposite answers, not one answer.

    `prepare_data_for_models` returns nothing when no split can be built, and
    the loop simply moved on. Those contexts appeared nowhere: not among the
    champions, not among the refusals. So the artifact quietly answered "every
    context that failed, failed on skill" -- which is the exact confusion it
    was built to prevent.
    """
    stage._record_unprepared_context(
        ticker="AAPL", timeframe="15m", target_name="target_return_1d",
    )
    assert len(stage._gate_refusals) == 1
    row = stage._gate_refusals[0]
    assert row["ticker"] == "AAPL"
    assert row["target"] == "target_return_1d"
    assert "training never ran" in row["reasons"]
    # A score would be a lie here: nothing was scored.
    assert row["holdout_score"] is None
    assert row["model_type"] is None


def test_every_path_that_abandons_a_context_records_why():
    """A `continue` that keeps no record is how the artifact lost a category.

    Two of the four abandon-paths in the training loop logged and persisted
    nothing: the walk-forward stability refusal (66 of 446 in the run that was
    parsed by hand) and the unprepared-data path. Both looked fine in the log
    and were absent from the file.

    This reads the loop rather than the two functions, because both functions
    were correct while the loop skipped them.
    """
    import ast
    import inspect

    from src.pipeline.stages.modeling import orchestrator as module

    source = inspect.getsource(module)
    lines = source.splitlines()
    tree = ast.parse(source)

    loops = [
        node
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
        for node in ast.walk(function)
        if isinstance(node, ast.For | ast.AsyncFor)
        and "_collect_gate_refusal" in ast.unparse(node)
    ]
    assert loops, "the training loop no longer mentions the refusal recorder"

    unrecorded = []
    for loop in loops:
        for node in ast.walk(loop):
            if not isinstance(node, ast.Continue):
                continue
            window = "\n".join(lines[max(0, node.lineno - 16): node.lineno])
            if not ("_collect_gate_refusal" in window
                    or "_record_unprepared_context" in window):
                unrecorded.append(node.lineno)

    assert not unrecorded, (
        "these `continue` statements abandon a context without recording why, "
        f"so it will be missing from gate_refusals: lines {unrecorded}"
    )


def test_a_note_is_added_to_the_gate_reasons_not_swallowed_by_them(stage):
    """Written as `reasons or [note]`, the note lost whenever the gate spoke.

    The walk-forward stability refusal passes its reason as a note. If the
    promotion gate had also filled in `reasons`, the note was dropped and the
    row was labelled with the gate's text instead -- and in the 2026-08-23
    artifact that text was `holdout_measured_and_beats_baseline` for 99 of 497
    rows. A refusal whose stated reason is that the model beat its baseline
    tells the reader nothing and actively misleads them.
    """
    stage._collect_gate_refusal(
        _result("target_up_5d", ["holdout_measured_and_beats_baseline"]),
        "AAPL_1d_target_up_5d",
        note="balanced accuracy beats chance on only 2 of 4 folds",
    )
    reasons = stage._gate_refusals[0]["reasons"]
    assert "2 of 4 folds" in reasons, "the real reason was dropped"
    assert "beats_baseline" in reasons, "the gate's own text was dropped"


def test_a_refusal_with_nothing_to_say_says_so(stage):
    """An empty string in `reasons` reads as a missing value, not as a fact."""
    stage._collect_gate_refusal(
        {"ticker": "AAPL", "timeframe": "1d", "target_name": "target_return_1d"},
        "AAPL_1d_target_return_1d",
    )
    assert stage._gate_refusals[0]["reasons"] == "no reason given"


def test_the_keys_read_here_are_the_keys_the_trainer_writes():
    """The two were wrong together once; this makes them fail together instead.

    `_collect_gate_refusal` reads `winner_holdout_metrics` by key name. Nothing
    connected those names to the ones `BaseTrainer` actually writes, so when
    they disagreed the result was silence: null columns, no error, and a test
    suite that agreed with the mistake because the fixture had been built from
    the same assumption.
    """
    import inspect

    from src.training import base_trainer

    source = inspect.getsource(base_trainer)
    for key in ("'score'", "'baseline_score'", "'holdout_sample_count'",
                "'holdout_event_count'", "'baseline_kind'",
                "'baseline_constant_score'", "'baseline_persistence_score'"):
        assert key in source, (
            f"{key} is read from winner_holdout_metrics but BaseTrainer no "
            "longer writes it; the refusal artifact would silently go null"
        )
