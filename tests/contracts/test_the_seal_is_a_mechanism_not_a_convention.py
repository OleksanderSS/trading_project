"""The seal must be something the code does, not something we remember to do.

REGISTER #264, the half it recorded as unfixed and left named: "`apply_seal` --
the function SEALED_HOLDOUT.md describes as the enforcement mechanism, with an
`allow_sealed` flag for saying so out loud -- has ZERO callers. Every
diagnostic filters for itself and prints how much it dropped; that works, but
it means the seal rests on a convention rather than a mechanism."

Re-read on 2026-09-05 and the remainder turned out to be bigger than the
sentence. Twelve diagnostics filter by the ABSOLUTE date; exactly one --
`opponent_ladder` -- uses the module's per-frame rule. And `apply_seal` itself
used the bare date, so routing callers through it would have SPREAD the defect
rather than fixed it.

MEASURED ON THE LIVE BATCH:

    1d    705,492 rows   absolute withholds  82,094   per-frame  82,094
    60m   380,938 rows   absolute withholds 380,938   per-frame  69,502

One hundred percent of the hourly frame. A diagnostic handed zero rows reports
"nothing to measure", which reads as a statement about the market rather than
about the seal -- and the module's docstring already recorded it happening:
`opponent_ladder --interval 15m` returned "159,149 rows withheld; 0 remain".

So `apply_seal` was fixed FIRST, before anything was pointed at it.
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.pipeline.sealed_period import (
    SEAL_START, apply_seal, seal_start_for,
)


def _frame(start: str, periods: int, freq: str, interval: str = "1d"):
    return pd.DataFrame({
        "datetime": pd.date_range(start, periods=periods, freq=freq, tz="UTC"),
        "interval": interval,
        "value": range(periods),
    })


def test_a_long_frame_is_sealed_at_the_declared_date():
    """The daily frame reaches back past the seal AND forward past it, so the
    absolute date governs and nothing about it changes.

    The first version used 7,000 calendar days from 1996-08-26, which ends in
    2015 -- before the seal. Every assertion then passed vacuously: nothing
    was sealed because nothing reached the seal. The neighbouring test caught
    it by asserting the two rules agree, which they cannot on a frame that
    stops early.
    """
    frame = _frame("1996-08-26", 11_000, "D")
    assert frame["datetime"].max() > SEAL_START, "the fixture stops short again"

    kept, dropped = apply_seal(frame)
    assert dropped > 0
    assert kept["datetime"].max() < SEAL_START
    assert kept["datetime"].min() == frame["datetime"].min(), (
        "the early history was withheld too, so this is not the declared seal"
    )


def test_a_short_frame_keeps_something():
    """The defect this closes. A frame whose history begins after the seal was
    withheld ENTIRELY -- not protected, deleted."""
    frame = _frame("2024-08-19", 5000, "h", interval="60m")
    kept, dropped = apply_seal(frame)

    assert len(kept) > 0, (
        "the seal withheld every row of a frame that starts after the declared "
        "date; a seal that leaves nothing to explore is not a stricter seal, "
        "it is a broken one"
    )
    assert 0 < dropped < len(frame)
    assert len(kept) + dropped == len(frame)


def test_the_share_withheld_from_a_short_frame_is_the_configured_one():
    from src.pipeline.sealed_period import SEAL_SHARE

    frame = _frame("2024-08-19", 5000, "h", interval="60m")
    _, dropped = apply_seal(frame)
    assert dropped / len(frame) == pytest.approx(SEAL_SHARE, abs=0.02)


def test_a_stacked_frame_seals_each_cadence_on_its_own_span():
    """One seal for two cadences is the same mistake at a smaller scale: the
    long frame drags the short one's seal past its own history."""
    daily = _frame("1996-08-26", 11_000, "D", interval="1d")
    hourly = _frame("2024-08-19", 5000, "h", interval="60m")
    stacked = pd.concat([daily, hourly], ignore_index=True)

    together, dropped_together = apply_seal(stacked, by="interval")
    separate = sum(len(apply_seal(part)[0]) for part in (daily, hourly))

    assert len(together) == separate, (
        "sealing a stacked frame as one gives a different answer from sealing "
        "each cadence, which means one of the two is wrong"
    )
    assert dropped_together == len(stacked) - separate


def test_the_confirmation_flag_still_lets_a_run_through():
    """`allow_sealed` is the one door, and it exists so the single
    confirmation run can be made deliberately rather than by accident."""
    frame = _frame("1996-08-26", 11_000, "D")
    kept, dropped = apply_seal(frame, allow_sealed=True)
    assert dropped == 0
    assert len(kept) == len(frame)


def test_a_frame_without_the_column_is_returned_untouched_not_emptied():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    kept, dropped = apply_seal(frame)
    assert dropped == 0
    assert len(kept) == 3


def test_apply_seal_uses_the_per_frame_rule_and_not_the_bare_date():
    """Routing callers through a function that used the bare date would have
    SPREAD the defect. The order mattered: fix the mechanism, then point
    anything at it."""
    source = inspect.getsource(apply_seal)
    assert "seal_start_for" in source, (
        "apply_seal compares against SEAL_START directly again, so every "
        "caller it gains inherits the whole-frame withholding"
    )
    assert "< SEAL_START" not in source


def test_the_two_rules_agree_where_the_frame_is_long_enough():
    """Not a new decision, an extension of the existing one: on a frame that
    reaches back past the seal, per-frame and absolute must be the same
    answer."""
    stamps = _frame("1996-08-26", 11_000, "D")["datetime"]
    assert stamps.max() > SEAL_START
    assert pd.Timestamp(seal_start_for(stamps)) == SEAL_START


def test_the_report_names_the_boundary_that_was_actually_applied():
    """Caught one command after the diagnostics were routed through the rule.

    `conditional_pattern_report --interval 60m` printed "sealed from
    2023-09-01 onward ... exploration sees earlier data only" while keeping
    rows through 2026-04-22. Every word of that sentence was false about the
    run that printed it, and a reader checking the output would have concluded
    the hourly report was reading sealed data.
    """
    from src.pipeline.sealed_period import describe_for

    hourly = _frame("2024-08-19", 5000, "h", interval="60m")
    note = describe_for(hourly["datetime"])
    assert str(SEAL_START.date()) not in note.split("declared")[0], (
        "the description still leads with the absolute date on a frame that "
        "was not sealed at the absolute date"
    )
    boundary = seal_start_for(hourly["datetime"])
    assert str(pd.Timestamp(boundary).date()) in note

    daily = _frame("1996-08-26", 11_000, "D")
    assert str(SEAL_START.date()) in describe_for(daily["datetime"]), (
        "the daily frame IS sealed at the declared date and its report must "
        "keep saying so"
    )


def test_sealing_and_describing_is_one_call_that_cannot_be_misordered():
    """The reason `seal_and_describe` exists rather than two functions.

    The first call site passed the ALREADY-SEALED frame to the description,
    whose own tail is by then the boundary -- so it reported a date LATER than
    the one applied. Two correct functions in the wrong order is exactly the
    convention this module was written to replace.
    """
    from src.pipeline.sealed_period import seal_and_describe

    hourly = _frame("2024-08-19", 5000, "h", interval="60m")
    kept, withheld, note = seal_and_describe(hourly)

    truthful = pd.Timestamp(seal_start_for(hourly["datetime"])).date()
    misordered = pd.Timestamp(seal_start_for(kept["datetime"])).date()
    assert truthful != misordered, (
        "the fixture no longer reproduces the trap, so this test would pass "
        "whichever frame the description was given"
    )
    assert str(truthful) in note
    assert str(misordered) not in note
    assert withheld == len(hourly) - len(kept)


def test_a_stacked_frame_is_described_once_per_cadence():
    from src.pipeline.sealed_period import seal_and_describe

    stacked = pd.concat([_frame("1996-08-26", 11_000, "D", interval="1d"),
                         _frame("2024-08-19", 5000, "h", interval="60m")],
                        ignore_index=True)
    _, _, note = seal_and_describe(stacked, by="interval")

    assert "[1d]" in note and "[60m]" in note, (
        "one line for two cadences sealed at two different dates is false "
        "about at least one of them"
    )
    assert str(SEAL_START.date()) in note.split("[60m]")[0]


def test_the_open_door_says_so_in_the_report_not_only_in_the_flag():
    from src.pipeline.sealed_period import seal_and_describe

    frame = _frame("1996-08-26", 11_000, "D")
    kept, withheld, note = seal_and_describe(frame, allow_sealed=True)
    assert withheld == 0 and len(kept) == len(frame)
    assert "OPENED" in note, (
        "the confirmation run consumed the holdout and its own output does "
        "not say so; `allow_sealed` exists to be visible in the record"
    )
