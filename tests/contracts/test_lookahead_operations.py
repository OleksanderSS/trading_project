"""No new operation may read forward in time without saying why.

`TemporalLeakageGuard` claimed to catch `.shift(-1)` and `bfill`, but matched
those strings against DataFrame COLUMN NAMES -- and a column is never called
"close.shift(-1)". Measured on the 2026-08-02 export: 0 of 1,189 names
contained "shift(", 0 contained "bfill". The check ran on every batch and
could not fire.

The check moved here, to the source, where the expressions exist. It is a
ratchet like the silent-failure and formula scanners: the count may fall,
never rise. Legitimate forward-looking code -- building a label, masking one
-- carries the project's existing `# audit-ignore: NEGATIVE_SHIFT_*` marker
and is not counted.
"""
from __future__ import annotations

import pytest

from tests.contracts._lookahead_scan import counts, scan

#: Every remaining occurrence is marked; these are the shapes with no
#: unmarked instances left in src/. Raising a ceiling is a decision, and it
#: belongs in a commit message, not in a passing test run.
CEILINGS = {
    "NEGATIVE_SHIFT": 0,
    "BACKFILL": 0,
    "CENTERED_WINDOW": 0,
}


@pytest.mark.parametrize("kind", sorted(CEILINGS))
def test_lookahead_shapes_do_not_spread(kind):
    found = [finding for finding in scan() if finding.kind == kind]
    ceiling = CEILINGS[kind]

    assert len(found) <= ceiling, (
        f"{kind} rose from {ceiling} to {len(found)}.\n"
        + "\n".join(f"    {finding}" for finding in found)
        + "\n\nIf the new occurrence builds or masks a TARGET, mark it "
        "'# audit-ignore: NEGATIVE_SHIFT_INTENTIONAL <why>' and it stops "
        "counting. If it computes a FEATURE, it is lookahead bias: the model "
        "will score well in backtest and fail live."
    )


def test_the_scanner_recognises_all_three_shapes(tmp_path):
    """A scanner nobody tests is a scanner that quietly stops finding things."""
    sample = tmp_path / "src" / "sample.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "import pandas as pd\n"
        "def build(frame):\n"
        "    a = frame['close'].shift(-1)\n"
        "    b = frame['macro'].bfill()\n"
        "    c = frame['x'].rolling(20, center=True).mean()\n"
        "    return a, b, c\n",
        encoding="utf-8",
    )

    assert counts(sample.parent) == {
        "NEGATIVE_SHIFT": 1,
        "BACKFILL": 1,
        "CENTERED_WINDOW": 1,
    }


def test_prose_about_lookahead_is_not_reported_as_lookahead(tmp_path):
    """The scanner reported its own documentation before this was fixed."""
    sample = tmp_path / "src" / "documented.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        '"""Never use .bfill() here, and never rolling(center=True).\n\n'
        'A .shift(-1) would reach into the future.\n"""\n'
        "def clean(frame):\n"
        "    # .shift(-1) is wrong for features\n"
        "    return frame.ffill()\n",
        encoding="utf-8",
    )

    assert counts(sample.parent) == {}


def test_a_marked_occurrence_stops_counting(tmp_path):
    sample = tmp_path / "src" / "targets.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "def label(frame):\n"
        "    return frame['close'].shift(-1)  "
        "# audit-ignore: NEGATIVE_SHIFT_INTENTIONAL target\n",
        encoding="utf-8",
    )

    assert counts(sample.parent) == {}


def test_a_positive_shift_is_never_reported(tmp_path):
    """shift(+n) reads the past. Flagging it would train people to ignore
    this scanner, which is worse than not having it."""
    sample = tmp_path / "src" / "safe.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "def previous(frame):\n"
        "    return frame['close'].shift(1), frame['x'].rolling(20).mean()\n",
        encoding="utf-8",
    )

    assert counts(sample.parent) == {}


def test_the_macro_cleaner_no_longer_backfills():
    """The find that justified the CENTERED_WINDOW shape.

    clean_macro_data filled leading gaps with later values AND clipped each
    row against bounds computed from a window centred on it -- so an outlier
    was judged partly against bars that had not happened. It has no callers,
    which is the only reason this never reached a model.
    """
    import inspect
    import textwrap

    from src.processing.cleaners import DataCleaner
    from tests.contracts._lookahead_scan import _code_only

    source = textwrap.dedent(inspect.getsource(DataCleaner.clean_macro_data))
    # Through the scanner's own blanking, not a "starts with #" filter: the
    # explanation of what was removed lives in the docstring, and this test
    # failed on its own documentation before -- the same defect the scanner
    # had.
    code = "\n".join(_code_only(source).values())

    assert ".bfill()" not in code
    assert "center=True" not in code
    assert ".ffill()" in code
