"""Deep copies of a caller's frame may fall in number, never rise.

This is a ratchet, not a clean sweep. 116 functions still copy a parameter
frame deeply when a shallow copy would do; each is cheap to fix but not all
are worth touching, because most run on frames small enough for it never to
matter. The ceiling exists so the shape stops spreading into the hot path,
where it has already killed three runs -- see `_frame_copy_scan` for the
three and for what one such copy costs on the stage-3 frame (~4.25 GiB).

The scan only reports copies it can prove are unnecessary: a parameter frame,
copied without `deep=False`, in a function that never writes into a slice. If
a fix needs a deep copy because it writes partially, the scan stops reporting
it and the ceiling should come down by one.
"""
from __future__ import annotations

import pytest

from tests.contracts._frame_copy_scan import scan

#: Measured 2026-08-22, after eight were fixed on the pipeline's hot path.
#: Lower this when findings are fixed; never raise it.
CEILING = 116

#: Modules where the frame is the wide stage-3 feature frame or the full news
#: frame. A deep copy here is not a wart, it is an out-of-memory crash: v6 died
#: in one with "Unable to allocate 4.25 GiB". These must stay at zero.
HOT_PATH_PREFIXES = (
    "src/pipeline/stages/feature_engineering/",
    "src/pipeline/hybrid/feature_processor.py",
    "src/pipeline/timeframe_lineage.py",
    "src/pipeline/stages/modeling/walk_forward_validation.py",
)


@pytest.fixture(scope="module")
def findings():
    return scan()


def test_unnecessary_deep_copies_do_not_multiply(findings):
    assert len(findings) <= CEILING, (
        f"{len(findings)} unnecessary deep frame copies, ceiling is {CEILING}. "
        "A new one was added: use `.copy(deep=False)`, or write into a slice "
        "if the copy really must own its data.\n"
        + "\n".join(f"  {finding}" for finding in findings)
    )


def test_the_hot_path_carries_none(findings):
    """The wide frames are where this stops being a wart and becomes a crash."""
    offenders = [
        finding for finding in findings
        if finding.path.startswith(HOT_PATH_PREFIXES)
    ]
    assert not offenders, (
        "Deep frame copy on the stage-3 hot path, where the frame is "
        "259,133 rows by 2,238 columns and the copy costs ~4.25 GiB:\n"
        + "\n".join(f"  {finding}" for finding in offenders)
    )


def test_the_scan_still_recognises_the_shape():
    """A scan that silently stops matching would pass every other test here."""
    import ast

    from tests.contracts._frame_copy_scan import _scan_function

    unsafe = ast.parse(
        "def f(df):\n"
        "    result = df.copy()\n"
        "    result['new'] = 1\n"
        "    return result\n"
    ).body[0]
    assert _scan_function(unsafe, "probe.py"), "scan missed a plain offender"

    shallow = ast.parse(
        "def f(df):\n"
        "    result = df.copy(deep=False)\n"
        "    result['new'] = 1\n"
        "    return result\n"
    ).body[0]
    assert not _scan_function(shallow, "probe.py"), "scan flagged a shallow copy"

    # A partial write makes the deep copy load-bearing: pandas 2.3.3 writes
    # `s.loc[0, "a"] = 99` straight through a shallow copy into the original.
    partial = ast.parse(
        "def f(df):\n"
        "    result = df.copy()\n"
        "    result.loc[0, 'a'] = 99\n"
        "    return result\n"
    ).body[0]
    assert not _scan_function(partial, "probe.py"), "scan flagged a needed copy"
