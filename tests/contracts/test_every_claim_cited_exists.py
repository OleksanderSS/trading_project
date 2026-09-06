"""Every claim cited as a basis must exist, and the numbering must be whole.

REGISTER #284. Р22 was cited by FOUR claims -- Р23 as "the net-test method",
Р24 and Р25 as "the cost model", Р26 as "gross by horizon" -- and by a ticked
ROADMAP item, while its own entry had never been written. `git log -S` found
the text in no commit at all: the number first appears as a CITATION in the
same commit that added Р23.

That is a number whose origin nobody can check, in the file that exists to
stop exactly that. It went unseen for three days because nothing counted the
entries against the numbers used.

Found by counting, not by reading: CLAIMS declared Р1..Р47 and held 46 bodies.
So the count is what this pins.
"""
from __future__ import annotations

import re
from pathlib import Path

CLAIMS = Path(__file__).resolve().parents[2] / "docs" / "CLAIMS.md"

HEADING = re.compile(r"^#+\s*Р(\d+)\.", re.M)
#: A citation, not a heading: "Р22 (модель витрат)", "спирається на Р17".
CITATION = re.compile(r"Р(\d+)\b")


def _text() -> str:
    return CLAIMS.read_text(encoding="utf-8")


def test_the_numbering_has_no_holes():
    """A gap means either a claim was deleted or one was never written, and
    the second is what happened."""
    numbers = sorted({int(n) for n in HEADING.findall(_text())})
    assert numbers, "no claims at all"
    missing = [n for n in range(1, max(numbers) + 1) if n not in numbers]
    assert not missing, (
        f"CLAIMS declares Р1..Р{max(numbers)} and is missing {missing}. A "
        f"number that other claims cite but nobody wrote is a basis nobody "
        f"can read (#284)."
    )


def test_every_cited_claim_exists():
    """The failure that actually bit: four claims resting on one that was
    never written."""
    text = _text()
    have = {int(n) for n in HEADING.findall(text)}
    cited = {int(n) for n in CITATION.findall(text)}
    phantom = sorted(cited - have)
    assert not phantom, (
        f"these claim numbers are cited but have no entry: {phantom}. Write "
        f"the entry or correct the citation; a claim cannot rest on something "
        f"unreadable (#284)."
    )


def test_each_claim_number_is_used_once_as_a_heading():
    """Two bodies under one number is the same defect wearing the other face:
    a citation would not say which one it meant."""
    numbers = HEADING.findall(_text())
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    assert not duplicates, f"more than one entry numbered: {duplicates}"
