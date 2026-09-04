""""Verified" must not be a word someone typed once.

The three tables only grow -- REGISTER.md alone is 557 KiB across 277 rows --
so a row is written once and read rarely. The failure that produces is not a
wrong entry. It is a RIGHT entry whose state was never updated, and there is
no way to tell one from the other by reading the state.

Measured 2026-09-04, when the owner asked whether everything verified is
actually marked so. Four rows carried `закрито` while their own closing
sentences described the fix in the future tense and the code confirmed it had
never been made:

    #188  "**Виправлення:** щабель має брати 5-10 найкорельованіших
          кандидатів" -- and `_score_single_feature_baseline` still picks
          `corr.idxmax()`. A GATE RUNG, read as verified for four days.
    #200  "блок кратний рядкам-на-дату" -- and `_block_bootstrap_sigma` still
          takes `n ** (1/3)`, so every "N standard errors" the gate quotes is
          still inflated.
    #142  the FRED dedup key -- no ON CONFLICT anywhere in src/.
    #205  "повідомлення має перелічувати виконані стадії" -- still a bare
          "Pipeline completed successfully".

And one row ran the other way: #192 stood at `знято` ("checked, did not hold")
while its own text CONFIRMS the defect and names an unmade fix. That state
hid real work behind the word "checked".

`test_register_counter_matches_the_table.py` already guards the ARITHMETIC --
that the printed counts equal the rows. Nothing guarded the rows against the
EVIDENCE. This does, for the three rules that survived review with no false
positives; B, C, E and F stay advisory in the scanner because a closed entry
being CITED is not the same as it being resolved, and judging that is a
person's job.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / "scripts" / "diagnostics" / "stale_state_scan.py"

sys.path.insert(0, str(SCANNER.parent))
import stale_state_scan as scan  # noqa: E402


def _findings(rule: str) -> list[str]:
    """Run the scanner's own logic rather than a copy of it.

    A test that reimplements the check tests the copy. This calls the script,
    so the rule can only be weakened in one place.
    """
    result = subprocess.run(
        [sys.executable, str(SCANNER), "--rule", rule],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT),
        check=True,
    )
    lines = [line for line in result.stdout.splitlines()
             if line.startswith("  ") and not line.startswith("  None")]
    return [line.strip() for line in lines if line.strip()]


def test_no_closed_entry_still_describes_its_fix_in_the_future():
    """Rule G. The expensive direction: work that reads as done."""
    hits = _findings("G")
    assert not hits, (
        "these rows are marked закрито while their closing text proposes a "
        "fix and never reports making it. Either the fix landed and the row "
        "must say so, or the row is not closed:\n  " + "\n  ".join(hits)
    )


def test_no_unknown_entry_carries_its_own_verdict():
    """Rule D. #175 sat at `?` while its own text said the candidate was
    effectively closed."""
    hits = _findings("D")
    assert not hits, (
        "these rows answer their own question and still say the state is "
        "unknown:\n  " + "\n  ".join(hits)
    )


def test_no_roadmap_item_is_unticked_while_saying_it_was_done():
    """Rule A."""
    hits = _findings("A")
    assert not hits, (
        "these ROADMAP boxes are unticked and their own text reports the work "
        "finished:\n  " + "\n  ".join(hits)
    )


def test_the_scanner_can_still_read_both_files():
    """A scanner that silently matches nothing passes every rule above. That
    is REGISTER #202 exactly -- "не виміряно" is not "пройдено"."""
    rows = scan.register_rows()
    assert len(rows) > 250, f"the register parser found only {len(rows)} rows"
    assert scan.roadmap_items(), "the roadmap parser found no unticked items"

    states = {state for state, _ in rows.values()}
    assert states <= {"закрито", "відкрито", "знято", "неперевірюване", "?"}, (
        f"an unrecognised state appeared: {states}"
    )


@pytest.mark.parametrize("rule", sorted(scan.RULE_NAMES))
def test_every_rule_runs(rule):
    """Including the advisory ones: an advisory rule that crashes stops being
    read, and then it is not advisory, it is absent."""
    subprocess.run(
        [sys.executable, str(SCANNER), "--rule", rule],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT),
        check=True,
    )
