"""FRED sends revisions; the pivot must take the latest, not an arbitrary one.

Measured on the stored table: 8,048 distinct (series_id, date) pairs, held
across up to 10 rows each -- the same value re-fetched under different
realtime_start vintages. 60 of those pairs carry genuinely DIFFERENT values,
which are real revisions.

_pivot_macro_data collapses them with aggfunc='last'. That handles the
duplication (no row multiplication downstream, checked), but "last" means
last in row order, which is arbitrary unless the vintages are sorted -- so
for the 60 revised pairs it could pick a superseded number. Sorting by
realtime_start first makes it mean "most recently published".

Also verified while here, and left alone:

- The macro series really are daily (median gap 1 day for DGS10 and
  BAMLC0A0CM), so MacroScoreCalculator's pct_change(periods=252/12=21) is
  about a month, as intended.
- _calculate_weighted_composite renormalises by the weight actually
  AVAILABLE, so a missing indicator does not drag the score toward zero.
  That is the careful thing to do and it is done.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.enrichers.macro_features_enricher import MacroFeaturesEnricher


@pytest.fixture()
def enricher():
    return object.__new__(MacroFeaturesEnricher)


def _long(rows):
    """rows: (series_id, date, value, realtime_start)."""
    return pd.DataFrame(
        rows, columns=["series_id", "date", "value", "realtime_start"]
    )


def test_the_latest_vintage_wins(enricher):
    """The regression: row order decided which revision survived."""
    frame = _long([
        ("DGS10", "2026-01-05", 4.10, "2026-03-01"),   # newest, listed first
        ("DGS10", "2026-01-05", 4.00, "2026-01-06"),   # original print
    ])

    pivoted = enricher._pivot_macro_data(frame)

    assert pivoted.loc[pd.Timestamp("2026-01-05"), "FRED_DGS10"] == pytest.approx(4.10)


def test_the_result_holds_regardless_of_row_order(enricher):
    rows = [
        ("DGS10", "2026-01-05", 4.00, "2026-01-06"),
        ("DGS10", "2026-01-05", 4.10, "2026-03-01"),
    ]
    forward = enricher._pivot_macro_data(_long(rows))
    reversed_ = enricher._pivot_macro_data(_long(list(reversed(rows))))

    assert (
        forward.loc[pd.Timestamp("2026-01-05"), "FRED_DGS10"]
        == reversed_.loc[pd.Timestamp("2026-01-05"), "FRED_DGS10"]
    )


def test_identical_duplicates_collapse_to_one_row(enricher):
    """Row multiplication downstream is the thing that must not happen."""
    frame = _long([
        ("DGS10", "2026-01-05", 4.00, f"2026-0{i}-01") for i in range(1, 6)
    ])

    pivoted = enricher._pivot_macro_data(frame)

    assert len(pivoted) == 1
    assert pivoted.iloc[0]["FRED_DGS10"] == pytest.approx(4.00)


def test_several_series_become_several_columns(enricher):
    frame = _long([
        ("DGS10", "2026-01-05", 4.0, "2026-01-06"),
        ("VIXCLS", "2026-01-05", 18.0, "2026-01-06"),
    ])

    pivoted = enricher._pivot_macro_data(frame)

    assert sorted(pivoted.columns) == ["FRED_DGS10", "FRED_VIXCLS"]


def test_data_without_vintages_still_pivots(enricher):
    """Not every source carries realtime_start."""
    frame = pd.DataFrame(
        [("DGS10", "2026-01-05", 4.0)], columns=["series_id", "date", "value"]
    )

    pivoted = enricher._pivot_macro_data(frame)

    assert pivoted.loc[pd.Timestamp("2026-01-05"), "FRED_DGS10"] == pytest.approx(4.0)


def test_a_frame_without_the_expected_columns_is_returned_unchanged(enricher):
    frame = pd.DataFrame({"something": [1, 2]})

    assert enricher._pivot_macro_data(frame).equals(frame)


def test_the_live_table_still_looks_like_this():
    """If the duplication ever stops, this stops being worth guarding."""
    import duckdb

    connection = duckdb.connect("data/trading_data.duckdb", read_only=True)
    pairs, revised = connection.execute(
        "SELECT COUNT(*), SUM(CASE WHEN nv > 1 THEN 1 ELSE 0 END) FROM ("
        "  SELECT series_id, date, COUNT(DISTINCT value) AS nv "
        "  FROM fred_data GROUP BY 1, 2)"
    ).fetchone()
    connection.close()

    assert pairs > 0
    if not revised:
        pytest.skip("no revised (series, date) pairs stored any more")
    assert revised >= 1
