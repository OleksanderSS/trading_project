"""Scan unit P4, mode A: every `--mode` must be dispatched, implemented, and DESCRIBED.

The coverage ledger's argument for scanning this unit at all: all four
cross-cutting units have produced a defect BY ACCIDENT, which is what "nobody
has looked here" looks like from outside. P4 already had two on record before
today -- `pool_tickers` silently broke four things, and `--mode light` was
recommended twice as a way to exercise stages 5-7 that it has never touched
(REGISTER #205, CLAIMS P8) -- and 2026-09-03's #229, a lost timeframe reported
at ERROR and then declared a success, is the same shape found the same way.

The inventory pass on 2026-09-03 found no DEAD switch: all thirteen are read
somewhere and every declared mode has a dispatch branch, so the `--mode
calibrate` defect has not come back. What it found instead is subtler and is
what these tests hold:

    the switch that most changes what a run does made no checkable claim.

`help='Pipeline execution mode'` cannot be wrong, and cannot be verified. The
#205 defect lived exactly there: "light" implies a smaller version of the
whole thing, and it is `stages_to_run=[4]` -- stage 4 and nothing after it.
Nobody misread the code; there was nothing to read.

So each mode now states its stages in the help text, and these tests hold that
statement to the code. Mode A's purpose in one line: a switch with no stated
promise cannot be found to have broken one.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARSER = PROJECT_ROOT / "src" / "cli" / "argument_parser.py"
RUNNER = PROJECT_ROOT / "run_hybrid_pipeline.py"
EXECUTOR = PROJECT_ROOT / "src" / "cli" / "pipeline_executor.py"


@pytest.fixture(scope="module")
def modes() -> list[str]:
    from src.cli.argument_parser import create_argument_parser

    for action in create_argument_parser()._actions:
        if action.dest == "mode":
            return list(action.choices)
    pytest.fail("--mode has no choices; the modes are no longer enumerable")


@pytest.fixture(scope="module")
def mode_help() -> str:
    from src.cli.argument_parser import create_argument_parser

    for action in create_argument_parser()._actions:
        if action.dest == "mode":
            return action.help or ""
    return ""


def test_every_declared_mode_is_dispatched(modes):
    """The `--mode calibrate` defect: advertised on the command line, wired to
    nothing, raising AttributeError on every invocation."""
    runner = RUNNER.read_text(encoding="utf-8")
    dispatched = set(re.findall(r"args\.mode == '([^']+)'", runner))
    missing = [m for m in modes if m not in dispatched]
    assert not missing, (
        f"these modes are offered and never dispatched: {missing}. A user who "
        f"passes one gets whatever the fall-through does, which has been "
        f"NotImplementedError since the last time this happened."
    )


def test_every_dispatched_mode_has_an_executor(modes):
    executor = EXECUTOR.read_text(encoding="utf-8")
    missing = [
        mode for mode in modes
        if f"def execute_{mode}_mode" not in executor
    ]
    assert not missing, (
        f"dispatched to a method that does not exist: {missing}"
    )


def test_every_mode_is_named_in_the_help(modes, mode_help):
    """Adding a mode without saying what it runs fails here."""
    unmentioned = [m for m in modes if m not in mode_help]
    assert not unmentioned, (
        f"these modes are offered with no description: {unmentioned}. "
        f"'Pipeline execution mode' is what the help said when --mode light "
        f"was twice recommended for stages it has never run."
    )


def test_the_help_says_light_is_stage_four_only(mode_help):
    """The specific claim #205 cost twelve hours to learn, pinned.

    Not a style preference: `light_models_trainer.py` passes
    `stages_to_run=[4]`, so a run of it produces no predictions, no trades and
    no evaluation, and the argument "let the run finish, because stages 5-7
    have never seen live champions" was false twice on the same day.
    """
    assert "STAGE 4 ONLY" in mode_help.upper(), (
        "the help no longer says that --mode light stops after training"
    )


def test_the_claim_matches_the_code():
    """`light` really is stage 4 alone -- checked against the source, so the
    help and the behaviour cannot drift apart in either direction."""
    trainer = (PROJECT_ROOT / "src" / "pipeline" / "hybrid" /
               "light_models_trainer.py").read_text(encoding="utf-8")
    stage_sets = re.findall(r"stages_to_run\s*=\s*\[([^\]]*)\]", trainer)
    assert stage_sets, "light_models_trainer no longer pins its stages"
    for found in stage_sets:
        assert found.strip() == "4", (
            f"light mode now runs stages [{found}]; the help text says stage 4 "
            f"only. Change both together or neither."
        )
