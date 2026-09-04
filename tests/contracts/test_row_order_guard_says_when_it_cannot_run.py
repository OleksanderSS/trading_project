"""A guard that could not run must not be silent about it.

REGISTER #170. `_restore_input_row_order` enforces one invariant at the
boundary every enricher passes through: the rows come back in the order they
went in. It has FIVE exits, and only one of them means "nothing needed":

    the hashes already match                 -- nothing needed
    the row count changed                    -- COULD NOT CHECK
    no `hash` column on one side             -- COULD NOT CHECK
    `hash` is not unique                     -- COULD NOT CHECK
    the same count, a different set of rows  -- COULD NOT CHECK

All five returned in the same silence, so a guard that did not run looked
exactly like a guard that ran and found nothing. That is the shape this
repository keeps paying for: `skipped_inputs_unchanged` for a drift check that
never ran (#221), `+0 columns` reported with a green tick (#144), a VaR of 0.0
for no data (#261).

The row-count case was the worst of the four. The reorder warning required
`len(before) == len(after)`, so an enricher that DROPS a row disabled the
protection AND said nothing -- and a dropped-plus-reordered frame is exactly
what put 54,000 bars on the wrong dates in the 2026-08-06 batch.

WHAT IS FIXED: each of those four now says which one it was. Filtering rows
stays an enricher's right; doing it invisibly does not.

WHAT IS NOT FIXED, and is deliberately left to the owner: #170 also asks that
a failed invariant STOP the run rather than warn. That is a behaviour change
of the same class as #229 and #252 -- it decides when runs die -- and it
belongs to the owner, not to a fix. Making the failure audible is what makes
that decision available.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.features.feature_orchestrator import FeatureOrchestrator


class _Enricher:
    name = "test_enricher"


def _frame(n: int, *, hashes: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame({
        "hash": hashes if hashes is not None else [f"h{i}" for i in range(n)],
        "datetime": pd.date_range("2020-01-01", periods=n, tz="UTC"),
        "value": np.arange(n, dtype=float),
    })


def _warn(before, after, caplog) -> str:
    with caplog.at_level(logging.WARNING):
        FeatureOrchestrator._warn_if_row_identity_changed(_Enricher(), before, after)
    return " ".join(r.getMessage() for r in caplog.records)


def test_a_dropped_row_is_named(caplog):
    """The case that disabled the guard silently."""
    said = _warn(_frame(10), _frame(10).iloc[:9], caplog)
    assert "ROW COUNT" in said
    assert "10 -> 9" in said
    assert "-1" in said
    assert "guard cannot run" in said, (
        "the message says the count changed but not that the protection is "
        "off, which is the half that matters"
    )


def test_an_added_row_is_named_too(caplog):
    """Rows appearing is as unprotected as rows leaving."""
    said = _warn(_frame(10), _frame(11), caplog)
    assert "ROW COUNT" in said and "10 -> 11" in said


def test_a_missing_hash_says_the_guard_could_not_run(caplog):
    said = _warn(_frame(10), _frame(10).drop(columns=["hash"]), caplog)
    assert "could not run" in said
    assert "hash" in said


def test_duplicate_hashes_say_the_guard_could_not_run(caplog):
    duplicated = _frame(10, hashes=["h0"] * 10)
    said = _warn(_frame(10), duplicated, caplog)
    assert "could not run" in said
    assert "not unique" in said


def test_a_different_set_of_rows_says_the_guard_could_not_run(caplog):
    said = _warn(_frame(10), _frame(10, hashes=[f"x{i}" for i in range(10)]), caplog)
    assert "could not run" in said
    assert "different SET" in said


def test_a_reorder_is_still_reported(caplog):
    """The original job of this check, which must survive the new branches."""
    shuffled = _frame(10).iloc[::-1].reset_index(drop=True)
    said = _warn(_frame(10), shuffled, caplog)
    assert "DIFFERENT" in said and "ORDER" in said


def test_an_untouched_frame_says_nothing(caplog):
    """A check that fires on ordinary runs gets switched off -- `|| true` sat
    in ci.yml for six weeks for exactly that."""
    before = _frame(10)
    with caplog.at_level(logging.WARNING):
        FeatureOrchestrator._warn_if_row_identity_changed(_Enricher(), before, before.copy())
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, f"a clean frame warned anyway: {warnings}"


def test_adding_columns_alone_says_nothing(caplog):
    """What an enricher normally does."""
    before = _frame(10)
    after = before.copy()
    after["new_feature"] = 1.0
    with caplog.at_level(logging.WARNING):
        FeatureOrchestrator._warn_if_row_identity_changed(_Enricher(), before, after)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("case", ["dropped", "no_hash", "duplicate", "different_set"])
def test_the_restore_still_returns_the_frame_unchanged(case):
    """Saying so must not change what the guard DOES. It could not align the
    rows in any of these cases, and inventing an alignment would be worse than
    the silence this replaces."""
    before = _frame(10)
    after = {
        "dropped": _frame(10).iloc[:9],
        "no_hash": _frame(10).drop(columns=["hash"]),
        "duplicate": _frame(10, hashes=["h0"] * 10),
        "different_set": _frame(10, hashes=[f"x{i}" for i in range(10)]),
    }[case]

    result = FeatureOrchestrator._restore_input_row_order(_Enricher(), before, after)
    pd.testing.assert_frame_equal(result, after)
