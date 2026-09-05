"""Archiving must lose nothing: the question and the result stay in the register.

The owner's requirement on 2026-09-04, in their words: a passed item must stay
remembered -- what was asked, what came out -- so the same ground is not
walked twice. Archiving that dropped either half would trade an unreadable
ledger for an amnesiac one.

REGISTER.md was 559 KiB across 277 rows, 90% of it settled entries. That size
is not cosmetic: it is why states rot. Nobody reaches row #142, so nobody
notices that #142 said `закрито` while its fix was never made -- one of four
such rows found that day, one of them a gate rung.

So the split is: one line per settled entry stays in the register (number,
state, the question verbatim, the result quoted from the entry itself), and
the full body moves to AUDIT_HISTORY.md. Numbers never change, so every `#N`
reference keeps pointing at the same entry.

The dangerous part is the rebuild, and it already bit once. The first version
of the archiver read only the register; run a second time it would have taken
the one-line INDEX rows as the entries and rewritten the archive with them --
destroying 253 bodies silently, while its own verification confirmed each row
matched itself. Idempotency is therefore a contract here, not a nicety.
"""
from __future__ import annotations

import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "REGISTER.md"
ARCHIVE = ROOT / "docs" / "AUDIT_HISTORY.md"
SCRIPT = ROOT / "scripts" / "maintenance" / "archive_settled_register_rows.py"

ID_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]*?)\s*\|")
STATES = {"закрито", "відкрито", "знято", "неперевірюване", "?"}
SETTLED = {"закрито", "знято", "неперевірюване"}


def _rows(path: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    for line in io.open(path, encoding="utf-8").read().splitlines():
        match = ID_ROW.match(line)
        if match and match.group(2).strip() in STATES:
            rows.setdefault(int(match.group(1)), line)
    return rows


@pytest.fixture(scope="module")
def register():
    return _rows(REGISTER)


@pytest.fixture(scope="module")
def archive():
    return _rows(ARCHIVE)


def test_every_archived_entry_still_has_a_line_in_the_register(register, archive):
    """The register remains the one place that lists every entry."""
    missing = sorted(set(archive) - set(register))
    assert not missing, (
        f"{len(missing)} entries live only in the archive, so the register no "
        f"longer knows they happened: {missing[:15]}"
    )


def test_no_settled_entry_keeps_its_full_body_in_the_register(register):
    """Otherwise the file grows back and the reason for the split returns."""
    fat = [number for number, line in register.items()
           if ID_ROW.match(line).group(2).strip() in SETTLED and len(line) > 900]
    assert not fat, (
        f"settled entries still carrying their full text in the register: {fat}"
    )


def test_each_index_line_carries_a_result_not_an_empty_cell(register, archive):
    """"What came out" is the half that is easy to lose, and the half the
    owner asked for by name."""
    empty = []
    for number in archive:
        cells = [cell.strip() for cell in register[number].split("|")]
        if len(cells) < 6 or len(cells[5]) < 15:
            empty.append(number)
    assert not empty, (
        f"{len(empty)} settled entries have no readable result in the "
        f"register, so the ledger remembers the question and forgets the "
        f"answer: {empty[:15]}"
    )


def test_the_result_is_quoted_from_the_entry_and_not_composed(register, archive):
    """A summary written by the archiver would be text nobody measured, in the
    file that exists to stop exactly that. Checked by requiring the words to
    appear in the entry itself."""
    invented = []
    for number, row in list(archive.items())[:60]:
        cells = [cell.strip() for cell in register[number].split("|")]
        if len(cells) < 6:
            continue
        quoted = cells[5].rstrip("…").strip()
        head = " ".join(quoted.split()[:6])
        if head and head not in " ".join(row.split()):
            invented.append(number)
    assert not invented, (
        f"the result column does not appear verbatim in the entry for "
        f"{invented[:10]} -- it is a paraphrase, which is not evidence"
    )


def test_running_the_archiver_again_changes_nothing():
    """It ran destructively once in development. It must not be able to
    again."""
    before = (REGISTER.read_bytes(), ARCHIVE.read_bytes())
    subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT),
                   capture_output=True, text=True, encoding="utf-8", check=True)
    after = (REGISTER.read_bytes(), ARCHIVE.read_bytes())
    assert before == after, (
        "a second run rewrote the files; the archiver is reading its own "
        "output as input"
    )


def test_the_numbers_never_moved(register):
    """Every `#N` in code, commits and the other documents depends on this."""
    numbers = sorted(register)
    assert len(numbers) == len(set(numbers))
    assert max(numbers) >= 277, (
        "the highest entry number fell, so entries were dropped rather than "
        "moved"
    )


def test_a_row_written_into_the_index_section_is_not_silently_deleted():
    """The index is REBUILT from the archive, so anything else in it is lost.

    That is the design, and the section's own preamble says editing it by hand
    makes no sense. But "makes no sense" and "is destroyed without a word" are
    different promises. On 2026-09-05 a freshly written #278 was inserted into
    the index section by mistake; the archiver printed "0 newly archived" and
    wrote a register without it. The row was gone from both files, and nothing
    in the output said so -- it was found only because the next command looked
    for the number.

    A maintenance script that deletes work while reporting success is worse
    than one that refuses, which is the whole argument of REGISTER #260 and of
    the family this suite guards: silence must carry a mark.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # Run against a COPY of the repository's docs, never the real ones: a
        # test that plants a row in REGISTER.md and crashes leaves it there.
        work = Path(tmp) / "docs"
        work.mkdir()
        shutil.copy(ROOT / "docs" / "REGISTER.md", work / "REGISTER.md")
        shutil.copy(ROOT / "docs" / "AUDIT_HISTORY.md", work / "AUDIT_HISTORY.md")

        text = (work / "REGISTER.md").read_text(encoding="utf-8").splitlines()
        anchor = next(i for i, line in enumerate(text)
                      if re.match(r"\|\s*277\s*\|", line))
        text.insert(anchor, "| 999 | закрито | дефект | планта | результат |")
        (work / "REGISTER.md").write_text("\n".join(text), encoding="utf-8")

        script = (ROOT / "scripts" / "maintenance"
                  / "archive_settled_register_rows.py").read_text(encoding="utf-8")
        script = script.replace('"REGISTER.md"', f'r"{work / "REGISTER.md"}"')
        script = script.replace('"AUDIT_HISTORY.md"',
                                f'r"{work / "AUDIT_HISTORY.md"}"')
        runner = Path(tmp) / "run.py"
        runner.write_text(script, encoding="utf-8")

        done = subprocess.run(
            [sys.executable, str(runner), "--dry-run"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT),
        )
        output = (done.stdout or "") + (done.stderr or "")

    assert done.returncode != 0, (
        "the archiver accepted an index-section row it does not have "
        "archived; rebuilding the section deletes it and reports success"
    )
    assert "999" in output, (
        "it refused, but without naming the row that would have been lost, so "
        "the text is not recoverable from the message"
    )
