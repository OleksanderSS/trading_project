"""The register's headline counts must be derived from its own rows.

Until 2026-09-01 the header of `docs/REGISTER.md` declared a `стан` field and
the table had no such column: state was written in bold inside the `тип` cell,
and only sometimes. The headline read **126 / 27 / 17 / 4** -- a hand-typed
number that nothing could reproduce and that did not even sum to the number of
rows (174 against 220). It sat at the top of the file the whole project uses
to know what is done.

That is the same defect this audit has been finding all day, in the document
that records the finding of it: a number whose origin nobody can check. So the
counts are now computed from the table, and this test fails when the header
and the rows disagree.

It also holds the shape steady. Every entry must carry one of the four states
or an explicit `?`, because "no state" and "state not yet established" are
different facts, and only the second one is honest about what we know.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

REGISTER = Path(__file__).resolve().parents[2] / "docs" / "REGISTER.md"

#: The states an entry may carry. `?` is deliberate: 143 entries pre-date the
#: column and their text does not say what happened, so they are marked
#: unknown rather than guessed. Guessing the state of 62% of the register
#: would be exactly the failure the register exists to record.
STATES = {"закрито", "відкрито", "знято", "неперевірюване", "?"}

ID_ROW = re.compile(r"^\|\s*(\d+)\s*\|")


def _rows() -> list[list[str]]:
    text = REGISTER.read_text(encoding="utf-8")
    return [
        [cell.strip() for cell in line.split("|")]
        for line in text.splitlines()
        if ID_ROW.match(line)
    ]


@pytest.fixture(scope="module")
def rows():
    return _rows()


def test_every_entry_carries_a_state(rows):
    unknown_shape = [
        (row[1], row[2]) for row in rows if row[2] not in STATES
    ]
    assert not unknown_shape, (
        "these entries have something other than a state in the second "
        f"column, so the counts below them cannot be trusted: {unknown_shape[:10]}"
    )


def test_the_header_counts_match_the_rows(rows):
    counts = collections.Counter(row[2] for row in rows)
    header = REGISTER.read_text(encoding="utf-8")

    claimed = re.search(
        r"закрито \*\*(\d+)\*\* · відкрито \*\*(\d+)\*\* · знято \*\*(\d+)\*\* ·"
        r" неперевірюване \*\*(\d+)\*\* · \*\*стан не записаний: (\d+)\*\* з (\d+)",
        header,
    )
    assert claimed, (
        "the register header no longer states its counts in the form this "
        "test can read; update both together or the number goes back to being "
        "someone's memory"
    )

    closed, open_, withdrawn, unverifiable, unknown, total = (
        int(g) for g in claimed.groups()
    )
    assert closed == counts["закрито"]
    assert open_ == counts["відкрито"]
    assert withdrawn == counts["знято"]
    assert unverifiable == counts["неперевірюване"]
    assert unknown == counts["?"]
    assert total == len(rows)


def test_ids_are_unique(rows):
    ids = [row[1] for row in rows]
    duplicates = [item for item, n in collections.Counter(ids).items() if n > 1]
    assert not duplicates, (
        f"the same id appears more than once, so a reference to it is "
        f"ambiguous: {duplicates}"
    )


def test_the_unknown_count_only_falls():
    """`?` means "we have not established this", and it is work, not a state.

    Pinned so the backlog cannot quietly grow: a new entry written today has
    no excuse for an unrecorded state, and the 143 inherited ones can only be
    resolved by reading them.
    """
    counts = collections.Counter(row[2] for row in _rows())
    assert counts["?"] <= 143, (
        f"entries with no recorded state rose to {counts['?']}. A new entry "
        "must say what happened to it; only the pre-2026-09-01 backlog is "
        "allowed to be unknown."
    )
