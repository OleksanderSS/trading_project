"""The seal date is defined once. Everything else imports it.

WHY. `docs/SEALED_HOLDOUT.md` says moving the seal EARLIER is "always safe"
and moving it later "destroys the guarantee and must be recorded in the
register with the reason". Both sentences assume there is one date to move.

On 2026-09-04 there were NINE: `SEAL_START` in `src/pipeline/sealed_period.py`
and eight separate `pd.Timestamp("2023-09-01", tz="UTC")` constants across
`scripts/diagnostics/`, two of them written that same day by me while
measuring something else. Moving the seal would have been safe in one file
and silently ignored in the other eight, which continue reading data the
policy says nobody may look at -- and they would print the old date in their
own headers while doing it, so the output would look correct.

This is the duplication family the audit method names first: a fix lands in
one copy, the other lives on, and everything looks repaired. It is how the
calendar config and `news_impact` each survived their own repairs.

WHAT THIS FORBIDS: the literal date anywhere but its definition. Not a style
rule -- a policy that cannot be enforced in one edit is not a policy.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFINITION = PROJECT_ROOT / "src" / "pipeline" / "sealed_period.py"

#: Where a date literal is legitimate: the definition itself, and tests that
#: assert on a specific boundary. Documentation quotes the date freely --
#: prose is not what silently keeps reading sealed rows.
SEARCHED = ("src", "scripts")

#: Any ISO date used as a seal boundary, not just today's. Matching only
#: "2023-09-01" would let the next copy in under a different value, which is
#: the same defect with a new number.
SEAL_LITERAL = re.compile(
    r'pd\.Timestamp\(\s*["\']\d{4}-\d{2}-\d{2}["\']\s*,\s*tz\s*=\s*["\']UTC["\']'
)


def _python_files():
    for root in SEARCHED:
        for path in (PROJECT_ROOT / root).rglob("*.py"):
            if "__pycache__" in path.parts or "archive" in path.parts:
                continue
            yield path


def test_only_the_seal_module_states_the_date():
    offenders = []
    for path in _python_files():
        if path == DEFINITION:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in SEAL_LITERAL.finditer(text):
            line = text[: match.start()].count("\n") + 1
            context = text.splitlines()[line - 1].strip()
            # A UTC timestamp is not automatically a seal: a script may build
            # any date. Only flag ones assigned to a seal-named constant, which
            # is what the eight copies were.
            if re.search(r"SEAL|SEALED", context, re.IGNORECASE):
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line}  {context}")

    assert not offenders, (
        "the seal date is stated outside src/pipeline/sealed_period.py. "
        "Moving the seal would take effect in one file and be silently "
        "ignored here:\n"
        + "\n".join(f"  {entry}" for entry in offenders)
        + "\n\nImport it instead:  "
        "from src.pipeline.sealed_period import SEAL_START"
    )


def test_the_definition_is_still_where_this_test_points():
    """A ratchet whose target moved passes by looking at nothing."""
    assert DEFINITION.exists(), (
        f"{DEFINITION.relative_to(PROJECT_ROOT)} is gone; this test has been "
        f"guarding an empty rule"
    )
    source = DEFINITION.read_text(encoding="utf-8")
    assert SEAL_LITERAL.search(source), (
        "the seal module no longer states a date literal, so either the "
        "definition moved -- and this test now forbids it everywhere while "
        "guarding nothing -- or the seal is gone"
    )


def test_the_scanner_catches_a_planted_copy(tmp_path):
    """A scanner that cannot fail its own case proves nothing."""
    planted = 'SEALED = pd.Timestamp("2019-01-01", tz="UTC")\n'
    assert SEAL_LITERAL.search(planted), (
        "the pattern no longer recognises a seal constant, so the test above "
        "would report a clean repository however many copies exist"
    )
    assert not SEAL_LITERAL.search('start = pd.Timestamp("2019-01-01")\n'), (
        "the pattern now matches ordinary timestamps; it will fire on scripts "
        "that merely build a date, and a check that fires on ordinary code "
        "gets switched off"
    )


def test_the_diagnostics_that_read_the_batch_import_the_seal():
    """The eight that used to keep copies. Named individually: a count would
    pass if one were deleted and a new copy appeared elsewhere."""
    expected = [
        "book_from_one_feature.py",
        "does_lower_turnover_save_it.py",
        "does_the_price_lever_pay.py",
        "is_the_edge_bigger_on_dear_names.py",
        "net_test_every_survivor.py",
        "what_has_never_been_measured.py",
        "what_the_clock_fix_changes_on_60m.py",
        "who_pays_the_friction.py",
    ]
    missing = []
    for name in expected:
        path = PROJECT_ROOT / "scripts" / "diagnostics" / name
        if not path.exists():
            continue  # deleting a diagnostic is allowed; forgetting is not
        if "sealed_period import" not in path.read_text(encoding="utf-8"):
            missing.append(name)

    assert not missing, (
        "these read the batch and no longer import the seal, so they decide "
        "for themselves what is sealed:\n"
        + "\n".join(f"  {name}" for name in missing)
    )


@pytest.mark.parametrize("name", ["SEAL_START", "SEALED_ON", "apply_seal",
                                  "seal_start_for", "describe"])
def test_the_seal_module_still_exports_what_callers_import(name):
    from src.pipeline import sealed_period

    assert hasattr(sealed_period, name), (
        f"sealed_period.{name} is gone; every caller importing it fails at "
        f"collection, which CI blocks on -- but a caller that silently stopped "
        f"importing it would not"
    )
