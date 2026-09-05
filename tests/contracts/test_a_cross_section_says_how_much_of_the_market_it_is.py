"""A number measured on our names must say it was measured on our names.

WHY THIS EXISTS, in one measurement. Р41 compared a fitted cross-sectional
book against "buy everything" at Sharpe 0.987 and treated that as the floor.
Р47 then measured the floor: the equal-weighted basket of our 105 names beats
SPY -- which carries every constituent that died, at the price it died at --
by 6.86% a year, Sharpe 1.222, t +6.32. Two thirds of the opponent was the
list having been written in 2026, and no report said so.

Three diagnostics DID print `survivorship-inflated: an upper bound, not the
market` (commit db1754aa). True, unquantified, and written out three times --
so a correction would have landed in one copy and the other two would have
lived on, which is this repository's commonest defect shape. The sentence now
lives once, in `universe_membership`, and carries the number.

WHAT THIS FILE FORBIDS:

  * the caveat existing as a literal in a script again;
  * `scale_note` losing the ability to refuse, which is the only thing that
    stops it from answering a date it cannot know about;
  * the two panel diagnostics printing a cross-section without its scale.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.data.universe_membership import (
    COLUMNS, OPPONENT_CAVEAT, NoCoverage, scale_note, universe_as_of,
)

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS = ROOT / "scripts" / "diagnostics"

#: The scripts whose cross-sectional numbers are cited in CLAIMS.md. Scoped by
#: measurement rather than by sweep: eighteen diagnostics touch a Sharpe, and
#: the rest report per-feature or per-ticker quantities nobody would read as a
#: statement about the market.
PANELS = ("net_test_every_survivor.py",
          "does_a_combination_beat_the_best_single_feature.py")


def _store(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    for column in ("ipo_date", "delisting_date", "fetched_at"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


@pytest.fixture
def store() -> pd.DataFrame:
    return _store([
        ("ALIVE", "Alive Co", "NYSE", "Stock", "1990-01-02", None,
         "Active", "test", "", "2026-09-05"),
        ("ALSO", "Also Co", "NYSE", "Stock", "1990-01-02", None,
         "Active", "test", "", "2026-09-05"),
        ("DEAD", "Dead Co", "NYSE", "Stock", "1990-01-02", "2005-06-01",
         "Delisted", "test", "", "2026-09-05"),
        ("LATER", "Later Co", "NYSE", "Stock", "1990-01-02", "2015-06-01",
         "Delisted", "test", "", "2026-09-05"),
    ])


def test_the_note_states_both_counts_and_the_share(store):
    # Inside the fixture's coverage window: after the first delisting it
    # observed (2005-06-01) and before the last (2015-06-01). Picking 2004
    # first was the module refusing correctly, not a bug -- a store whose
    # earliest death is 2005 cannot speak about 2004.
    when = "2006-01-02"
    existed = universe_as_of(when, store)
    note = scale_note(when, 2, store)

    assert str(len(existed)) in note, "the note does not say how many existed"
    assert "2 names" in note
    assert "%" in note, (
        "a ratio the reader has to compute is a ratio the reader will not "
        "compute; that is why the number was missing for a month"
    )


def test_the_note_refuses_rather_than_guessing_outside_coverage(store):
    """The refusal is the whole mechanism. A note that answers a date the
    store cannot know about would print today's survivors under a
    point-in-time label -- the bias the module exists to remove, wearing the
    fix's name."""
    with pytest.raises(NoCoverage):
        universe_as_of("1991-01-02", store)

    note = scale_note("1991-01-02", 2, store)
    assert "UNKNOWN" in note
    assert "%" not in note.split("UNKNOWN")[0], (
        "it printed a share before admitting it does not know the date"
    )


def test_the_note_never_raises_so_a_report_still_prints_its_numbers(store):
    """A report that cannot state its scale must SAY so and keep going. If
    this raised, the missing sentence would take the whole measurement with
    it, and a guard that turns an absent file into a crash is worse than no
    guard."""
    for when in ("1991-01-02", "2004-01-02", "2030-01-02"):
        assert isinstance(scale_note(when, 2, store), str)


def test_the_caveat_has_a_number_in_it():
    """`an upper bound, not the market` was true and unquantified for a month.
    A reader could not tell 0.05 of a Sharpe from 1.5 of one."""
    assert re.search(r"\d", OPPONENT_CAVEAT), "the caveat is qualitative again"
    assert "1.22" in OPPONENT_CAVEAT and "6.86" in OPPONENT_CAVEAT, (
        "the caveat no longer carries Р47's measurement, so it is back to "
        "being a feeling with a citation"
    )


def test_no_script_writes_the_caveat_out_for_itself():
    """Three copies is how a fix lands in one and the other two live on."""
    offenders = []
    for path in DIAGNOSTICS.glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "survivorship-inflated" in text and "OPPONENT_CAVEAT" not in text:
            offenders.append(path.name)
    assert not offenders, (
        "these state the survivorship caveat as their own literal instead of "
        "importing the one that carries the measurement: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize("name", PANELS)
def test_a_panel_report_prints_the_scale_beside_its_panel(name):
    path = DIAGNOSTICS / name
    if not path.exists():
        pytest.skip(f"{name} was removed")
    text = path.read_text(encoding="utf-8")
    assert "scale_note" in text, (
        f"{name} prints a cross-sectional panel and no longer says what "
        f"share of the market it is; without that line its Sharpe reads as a "
        f"statement about the market (Р47)"
    )


def test_the_caveat_is_imported_where_it_is_used_not_inside_a_try():
    """The bug this caught on the day it was written.

    The rewiring attached the import to the first `universe_membership` line
    in each file -- which in two of them sat INSIDE the scale mark's
    try/except. On a machine with no membership store the except branch runs,
    the name is never bound, and the caveat print forty lines later raises
    NameError. The guard would have turned a missing file into a crash.
    """
    for name in PANELS:
        path = DIAGNOSTICS / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8")
                                      .splitlines(), start=1):
            if "import OPPONENT_CAVEAT" in line or "OPPONENT_CAVEAT," in line:
                assert not line.startswith(" "), (
                    f"{name}:{number} imports the caveat inside a block, so "
                    f"the name is unbound whenever that block does not run"
                )
