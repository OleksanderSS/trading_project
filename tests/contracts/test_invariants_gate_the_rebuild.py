"""A corrupt checkpoint stops the rebuild; a thin one does not.

REGISTER #170 and #275. `batch_invariants.py` declared in its own docstring
that its exit code can gate a run, and had ZERO callers for its whole life --
holding, unrun, the answers to a full day of measurement. Two things had to be
true before it could be wired, and both were measured on 2026-09-04:

  IT HAD TO STOP FAILING ON A KNOWN NORM. It exited 1 on 13 columns piled on a
  truthful zero and 69 features constant in training -- both the same fact,
  that news and several sources have no history before 2024 (CLAIMS R29). A
  check that fires on a condition everyone agreed to is a check nobody runs,
  and this one was switched off in the quietest way available: nothing called
  it. Failures are now CORRUPTION or ADVISORY, and only the first sets the
  exit code.

  IT HAD TO BE SAFE ON THE FRAME THAT MATTERS. The daily frame passes with the
  whole scale to spare on three blocking checks and 1.2 points on the fourth
  (indicator recomputation, 99.2% against a floor of 98%). Gating it costs
  nothing today. The intraday frames fail two blocking checks right now --
  indicators at 96.6%, and news_freshness_hours_60m 77% on the sentinel 999 --
  so they are deliberately NOT gated: intraday is out of analysis entirely
  (R26), and killing every such run on defects nobody is fixing is how a gate
  gets switched off again.

WHAT THIS PINS: that the gate exists and is called, that it blocks only on
corruption, that it never turns "could not run" into a pass, and that the
intraday exclusion is a named decision rather than an oversight.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

from src.core.exceptions import DataProcessingError
from src.pipeline.stages.feature_engineering.orchestrator import (
    FeatureEngineeringStage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Stage:
    """The gate alone, without building the whole stage."""

    _GATED_TIMEFRAMES = FeatureEngineeringStage._GATED_TIMEFRAMES
    _INVARIANT_TIMEOUT_SECONDS = FeatureEngineeringStage._INVARIANT_TIMEOUT_SECONDS
    _gate_on_invariants = FeatureEngineeringStage._gate_on_invariants

    def __init__(self):
        import logging

        self.logger = logging.getLogger("gate-under-test")


def test_the_gate_is_actually_called_after_the_checkpoint():
    """The defect family this whole file answers: a mechanism that exists and
    is invoked nowhere. `critical: true` in twenty collectors (#252),
    `family_size` promised in one config and verified in none (#235), and this
    script for its entire life (#170)."""
    source = inspect.getsource(FeatureEngineeringStage)
    assert "self._gate_on_invariants(" in source, (
        "the gate is defined and never called, which is the exact state "
        "REGISTER #170 recorded for four months"
    )
    checkpoint_at = source.index("self._checkpoint_enriched(tf, enriched_df)")
    gate_at = source.index("self._gate_on_invariants(tf)")
    assert gate_at > checkpoint_at, (
        "the gate runs before the checkpoint it is supposed to read"
    )


def test_only_the_daily_frame_is_gated():
    """Not timidity, a measurement: 60m fails two blocking checks today."""
    assert FeatureEngineeringStage._GATED_TIMEFRAMES == ("1d",), (
        "the gated set changed. Adding an intraday frame means every such run "
        "dies on indicators at 96.6% and a 999 sentinel (REGISTER #275); "
        "removing 1d means nothing is gated at all"
    )


def test_an_ungated_frame_is_skipped_loudly(caplog, monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a))
    with caplog.at_level("INFO"):
        _Stage()._gate_on_invariants("60m")
    assert not called, "an ungated frame still ran the checker"
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "not gated" in said and "#275" in said, (
        "the skip is silent, so a reader cannot tell a frame was not checked "
        "from a frame that passed"
    )


def test_a_missing_checkpoint_is_an_error_not_a_pass(caplog, monkeypatch):
    """The lesson of #221: `skipped_inputs_unchanged` for a check that never
    ran once."""
    monkeypatch.chdir(PROJECT_ROOT / "docs")  # no scripts/ or data/ here
    with caplog.at_level("ERROR"):
        _Stage()._gate_on_invariants("1d")
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "UNCHECKED, not clean" in said, (
        "a gate that could not run reported nothing, which reads downstream as "
        "a clean batch"
    )


@pytest.mark.parametrize("code", [0, 2, 137])
def test_only_exit_one_stops_the_run(code, monkeypatch, caplog, tmp_path):
    """Exit 1 is the script's declared contract for a blocking failure. Any
    other non-zero code means the checker itself broke, and a broken checker
    must be loud without pretending to be a verdict."""
    monkeypatch.setattr(
        Path, "exists", lambda self: True)
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, code, "report", "err"))
    with caplog.at_level("INFO"):
        _Stage()._gate_on_invariants("1d")   # must not raise
    said = " ".join(r.getMessage() for r in caplog.records)
    if code == 0:
        assert "passed" in said
    else:
        assert "UNCHECKED" in said


def test_exit_one_raises_and_says_why(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 1, "[FAIL] rows are time-ordered", ""))

    with pytest.raises(DataProcessingError) as raised:
        _Stage()._gate_on_invariants("1d")

    message = str(raised.value)
    assert "blocking invariant" in message
    assert "time-ordered" in message, (
        "the report is dropped, so whoever sees the failure has to re-run the "
        "checker by hand to learn what failed"
    )
    assert "report success on it" in message, (
        "the consequence is not stated; that sentence is why the gate exists"
    )
