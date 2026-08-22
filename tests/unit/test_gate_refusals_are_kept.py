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


def _result(target, reasons, **extra):
    return {
        "ticker": "AAPL",
        "timeframe": "1d",
        "target_name": target,
        "model_type": "catboost",
        "promotion_gate": {"passed": False, "reasons": reasons, **extra},
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
        _result("target_return_5d", ["x"], holdout_rows=18, holdout_events=4,
                holdout_score=-0.02, baseline_score=0.58),
        "AAPL_1d_target_return_5d",
    )
    kept = stage._gate_refusals[0]
    assert kept["holdout_rows"] == 18
    assert kept["holdout_events"] == 4
    assert kept["holdout_score"] == -0.02
    assert kept["baseline_score"] == 0.58


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
