"""A non-standard run may not write over the standard record.

Three times now. Twice on 2026-09-06 -- a two-hold smoke test over
`net_test_fdr.csv`, a lowered varies threshold over `net_test_varying.csv` --
and the fix was to encode the changed parameters in the filename. Then a third
time on 2026-09-10, by the same gap: the encoding listed the flags that existed
when it was written, and `--rotations 50 --only <34 columns>` wrote 34 rows
straight over a 243-row screen. Recovered from git, which is luck.

The lesson is not "add two more flags". It is that an ENUMERATION of flags rots
the moment a flag is added, and nothing notices until a record is gone. So this
test compares the two lists directly: every argument that changes WHAT IS
MEASURED must appear in the filename builder.

Arguments that change only how the run is reported or how fast it goes are
exempt, and named here rather than guessed at.
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diagnostics" / "net_test_every_survivor.py"

#: Flags that do NOT change what is measured, so absence from the filename is
#: correct. Each needs a reason, the way known_unread_risk_settings.txt does.
NOT_IN_THE_RECORD = {
    "universe": "already the stem of the filename, not a suffix",
    "attempts": ("changes the THRESHOLD, not the measurement -- and it exists "
                 "precisely to carry the screen's multiplicity into a "
                 "shortlist run, which the _onlyN suffix already marks"),
}


def _flags() -> set[str]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            continue
        name = node.args[0].value
        if isinstance(name, str) and name.startswith("--"):
            found.add(name[2:].replace("-", "_"))
    return found


def _suffix_source() -> str:
    """The block that builds the output filename."""
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index("suffix = ")
    end = text.index("report.to_csv", start)
    return text[start:end]


def test_every_flag_that_changes_the_measurement_reaches_the_filename():
    flags = _flags()
    assert flags, "no --flags found; the parser moved and this test is blind"
    block = _suffix_source()
    missing = sorted(
        flag for flag in flags
        if flag not in NOT_IN_THE_RECORD and f"args.{flag}" not in block)
    assert not missing, (
        f"{missing} change what the run measures and do not appear in the "
        "filename it writes, so a run using them overwrites the standard "
        "record. That has happened three times. Add the flag to the suffix in "
        "the same commit that adds the flag, or list it in NOT_IN_THE_RECORD "
        "with the reason it does not change the measurement.")


def test_the_exemption_list_does_not_name_flags_that_are_gone():
    """An exemption outliving its flag is how the next gap opens."""
    stale = sorted(set(NOT_IN_THE_RECORD) - _flags())
    assert not stale, (
        f"{stale} are exempted from the filename and no longer exist as "
        "flags. Remove the lines: a list nobody prunes is a list nobody reads.")


def test_the_standard_run_still_writes_the_standard_name():
    """The whole point is that a DEFAULT run keeps the cited filename.

    If the suffix ever fired on defaults, every artefact the register cites
    would move and the guard would have broken the thing it protects.
    """
    block = _suffix_source()
    assert 'suffix = "" if args.min_varies == MIN_VARIES' in block, (
        "the suffix no longer starts empty for a default run; check that "
        "net_test_varying.csv is still what a plain invocation writes")
    for flag in ("rotations", "only", "rotate_all_holds"):
        assert f"if args.{flag}" in block or f"args.{flag} !=" in block, (
            f"{flag} reaches the filename unconditionally rather than only "
            "when it differs from the default, which would rename the "
            "standard record")
