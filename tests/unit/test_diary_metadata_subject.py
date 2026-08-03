"""record_decision_metadata hardcoded whose metadata it was recording.

agent_id="consensus_engine" and ticker="CONSENSUS" were literals, so any
other writer would have its subject silently replaced. That is why
compare_layers.py invented `diary.add_entry(ticker=..., regime=...,
event_type=..., data=...)` -- a method that has never existed on DiaryEngine,
so a full experiment run raised AttributeError at the very end, after doing
all the work.
"""
from __future__ import annotations

import json
import logging

import pytest

from src.meta_learning.memory.diary_engine import DiaryEngine


class _Recorder:
    def __init__(self):
        self.rows = []

    def upsert(self, table, frame, unique_on=None):
        self.rows.append(frame.iloc[0].to_dict())


@pytest.fixture()
def engine():
    instance = object.__new__(DiaryEngine)
    instance.logger = logging.getLogger("diary-metadata-test")
    instance.data_manager = _Recorder()
    instance.table_name = "experience_diary"
    return instance


def test_the_default_subject_is_unchanged(engine):
    engine.record_decision_metadata({"fingerprint": "1|0"})

    row = engine.data_manager.rows[0]
    assert row["agent_id"] == "consensus_engine"
    assert row["ticker"] == "CONSENSUS"


def test_another_writer_can_name_its_own_subject(engine):
    engine.record_decision_metadata(
        {"sharpe": 1.4}, agent_id="layer_experiment:stacked", ticker="AAPL"
    )

    row = engine.data_manager.rows[0]
    assert row["agent_id"] == "layer_experiment:stacked"
    assert row["ticker"] == "AAPL"


def test_metadata_rows_stay_out_of_every_rate_calculation(engine):
    """outcome='metadata' is excluded by _RESOLVED_OUTCOMES, which is what
    makes this a safe home for a measurement rather than a trade."""
    engine.record_decision_metadata({"sharpe": 1.4}, ticker="AAPL")

    assert engine.data_manager.rows[0]["outcome"] == "metadata"
    assert "metadata" not in DiaryEngine._RESOLVED_OUTCOMES


def test_a_measurement_does_not_land_in_model_prediction(engine):
    """Averaging an unbounded metric with predictions produced a performance
    score of -13,820 for `linear`. A Sharpe passed as payload must not
    repeat it."""
    engine.record_decision_metadata(
        {"sharpe": 1.4, "total_return": 0.31}, ticker="AAPL"
    )

    row = engine.data_manager.rows[0]
    assert row["model_prediction"] == 0.0
    assert json.loads(row["reasoning"])["sharpe"] == 1.4


def test_nothing_live_calls_a_diary_method_that_does_not_exist():
    """compare_layers.py called diary.add_entry(ticker=, regime=, event_type=,
    data=). No such method has ever existed on DiaryEngine.

    It was archived rather than repaired: the script also imports
    devtools.experimentation.base.BaseExperiment, which exists nowhere in the
    project, so it raised ModuleNotFoundError on import and never reached the
    add_entry line at all. Writing the missing base class is new
    functionality, not a fix.
    """
    from pathlib import Path

    assert not hasattr(DiaryEngine, "add_entry")

    root = Path(__file__).resolve().parents[2] / "src"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "archive" not in path.parts
        and ".add_entry(" in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert offenders == [], f"calls a DiaryEngine method that does not exist: {offenders}"
