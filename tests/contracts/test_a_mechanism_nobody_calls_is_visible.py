"""The scanner that would have found two of today's three zero-caller defects.

Advisory, not blocking, and the count is a reading list rather than a defect
count -- the same shape as `row_order_dependency_scan`. What IS pinned is that
the scanner still runs, still discriminates, and still refuses to see what it
cannot see.

WHY IT EXISTS. On 2026-09-05 the same shape surfaced three times by accident:
`apply_seal` (#264), `universe_as_of` (Р46), `validation_predictions` (#47).
Each was a mechanism that was built, tested green, and called by nothing.

WHY IT IS ADVISORY. Commit dabe5540 archived `CriticalSignalDetector` for
having "zero callers" and was WRONG -- analysis.yaml constructs it by module
path, and Stage 7 silently lost a working analyzer. A blocking rule on this
signal would repeat that on the next config-driven class.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / "scripts" / "diagnostics" / "what_has_no_caller.py"


@pytest.fixture(scope="module")
def scanner():
    spec = importlib.util.spec_from_file_location("what_has_no_caller", SCANNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["what_has_no_caller"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result(scanner):
    import os
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return scanner.scan()
    finally:
        os.chdir(cwd)


def test_the_scanner_still_runs_and_returns_the_three_parts(result):
    unmarked, marked, by_config = result
    assert isinstance(unmarked, list) and isinstance(marked, list)
    assert isinstance(by_config, int)


def test_it_separates_a_declared_absence_from_an_undeclared_one(result):
    """The whole point. `stages/modeling/utils.py` says in its own docstring
    that ModelingStage never adopted it; `apply_seal` said nothing and looked
    wired. One is a decision, the other was a defect."""
    unmarked, marked, _ = result
    assert marked, (
        "nothing is recognised as a DECLARED absence any more, so a module "
        "that documents being unused now reads the same as one that hides it"
    )
    marked_names = {name for _, name, _, _ in marked}
    unmarked_names = {name for _, name, _, _ in unmarked}
    assert not (marked_names & unmarked_names)


def test_it_does_not_flag_what_the_config_constructs(result, scanner):
    """The documented blind spot, pinned by name.

    `CriticalSignalDetector` is live -- analysis.yaml builds it by module path
    -- and the naive version of this scan put it third in the list. If it
    reappears, the blind-spot filter has stopped working and the next reader
    is one commit away from repeating dabe5540.
    """
    unmarked, marked, by_config = result
    names = {name for _, name, _, _ in unmarked + marked}
    assert "CriticalSignalDetector" not in names, (
        "a config-constructed analyzer is being reported as uncalled again; "
        "that exact mistake cost Stage 7 a working analyzer"
    )
    assert by_config >= 0


def test_the_marked_phrases_recognise_a_real_declaration(scanner):
    """A filter that cannot match its own case would silently reclassify every
    documented decision as a finding."""
    declared = (
        "NOTE: despite the name, ModelingStage (orchestrator.py) does NOT call\n"
        "these - it was rewritten around `_process_ticker_with_async` and never\n"
        "adopted them."
    )
    assert any(phrase in declared for phrase in scanner.MARKED)
    assert not any(phrase in "def apply_seal(frame, column='datetime'):"
                   for phrase in scanner.MARKED), (
        "the phrase list now matches ordinary code, so every module would "
        "count as having declared itself unused"
    )


def test_intra_module_use_counts_as_use(scanner):
    """SEAL_SHARE and NOT_MEASURED were both flagged by the first version
    while being used three lines below where they are defined. Excluding the
    defining file wholesale asks a different question from the one that
    matters."""
    import os
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        unmarked, marked, _ = scanner.scan()
    finally:
        os.chdir(cwd)
    names = {name for _, name, _, _ in unmarked + marked}
    for used_at_home in ("SEAL_SHARE", "NOT_MEASURED", "SEALED_ON"):
        assert used_at_home not in names, (
            f"{used_at_home} is reported as uncalled while its own module "
            f"uses it; the scan is counting definitions, not uses"
        )
