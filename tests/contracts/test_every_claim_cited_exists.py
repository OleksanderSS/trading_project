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

#: Where a `#N` citation can appear. Docs and code both cite register rows;
#: the scan skips the row's own line, which naturally starts with its number.
CITING_ROOTS = ("docs", "src", "scripts", "tests")
SKIP_DIRS = {"__pycache__", ".git", "node_modules", "archive", "dean_os",
             "data", ".venv", "venv", "logs", "mlruns", "diagnostic_reports"}

REGISTER_ROW = re.compile(r"^\|\s*(\d+)\s*\|")
REGISTER_REF = re.compile(r"#(\d{1,3})\b")


def _register_numbers() -> set[int]:
    numbers: set[int] = set()
    for name in ("REGISTER.md", "AUDIT_HISTORY.md"):
        path = CLAIMS.parent / name
        for line in path.read_text(encoding="utf-8").splitlines():
            match = REGISTER_ROW.match(line)
            if match:
                numbers.add(int(match.group(1)))
    return numbers


def _files():
    root = CLAIMS.parent.parent
    for top in CITING_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in (".md", ".py"):
                continue
            if set(path.parts) & SKIP_DIRS:
                continue
            yield path


def test_every_register_citation_resolves():
    """The other direction of #284, pinned at ZERO where it already is.

    Measured 2026-09-06 across 1,135 files: not one `#N` pointed at a missing
    register row. A rule introduced at its current value binds immediately; a
    rule introduced with a budget never binds at all, which is why the seal's
    RISK_FREE rule went in at zero and why this one does too.
    """
    known = _register_numbers()
    assert known, "no register rows found at all"

    phantom: dict[int, str] = {}
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if REGISTER_ROW.match(line):
                continue
            for match in REGISTER_REF.finditer(line):
                value = int(match.group(1))
                if value in known or value == 0:
                    continue
                phantom.setdefault(value, f"{path.name}:{number}  {line.strip()[:90]}")

    assert not phantom, (
        "these #N citations point at a register row that does not exist:\n"
        + "\n".join(f"  #{n}  {where}" for n, where in sorted(phantom.items()))
        + "\n\nWrite the row, or correct the citation. A reference nobody can "
        "follow is the shape that let Р22 be cited four times without existing "
        "(#284)."
    )
