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


def test_each_vintage_appears_at_its_own_publication_date(enricher):
    """Rewritten 2026-08-13, when the pivot moved onto publication dates.

    It used to assert that the newest revision won a single row keyed on the
    observation date. That keying is what made CPI for July visible to bars
    dated 1 July, six weeks before its release, so the property changed
    shape: a revision does not overwrite the original, it becomes known
    later. Which value a bar sees is then decided by the merge, correctly,
    from when that bar is.
    """
    frame = _long([
        ("DGS10", "2026-01-05", 4.10, "2026-03-01"),   # revision, listed first
        ("DGS10", "2026-01-05", 4.00, "2026-01-06"),   # original print
    ])

    pivoted = enricher._pivot_macro_data(frame)

    assert pivoted.loc[pd.Timestamp("2026-01-06"), "FRED_DGS10"] == pytest.approx(4.00)
    assert pivoted.loc[pd.Timestamp("2026-03-01"), "FRED_DGS10"] == pytest.approx(4.10)
    # Nothing at all is known on the observation date itself.
    assert pd.Timestamp("2026-01-05") not in pivoted.index


def test_the_result_holds_regardless_of_row_order(enricher):
    rows = [
        ("DGS10", "2026-01-05", 4.00, "2026-01-06"),
        ("DGS10", "2026-01-05", 4.10, "2026-03-01"),
    ]
    forward = enricher._pivot_macro_data(_long(rows))
    reversed_ = enricher._pivot_macro_data(_long(list(reversed(rows))))

    assert forward["FRED_DGS10"].equals(reversed_["FRED_DGS10"])


def test_a_republished_value_never_multiplies_the_bars(enricher):
    """Row multiplication downstream is the thing that must not happen.

    Under publication-date keying five re-publications of one number are
    five rows, not one -- each is a real moment at which that number was
    the current print. What must not follow is bars multiplying, and
    merge_asof does not multiply: it picks the latest row at or before each
    bar. That is the property, checked where it lives.
    """
    frame = _long([
        ("DGS10", "2026-01-05", 4.00, f"2026-0{i}-01") for i in range(1, 6)
    ])

    pivoted = enricher._pivot_macro_data(frame)
    assert pivoted["FRED_DGS10"].tolist() == pytest.approx([4.00] * 5)

    bars = pd.DataFrame({
        "datetime": pd.date_range("2026-06-01", periods=10, freq="D"),
        "ticker": ["AAPL"] * 10,
        "close": range(10),
    })
    macro_wide = pivoted.copy()
    macro_wide.index.name = "datetime"
    merged = pd.merge_asof(
        bars.sort_values("datetime"),
        macro_wide.reset_index().sort_values("datetime"),
        on="datetime", direction="backward",
    )
    assert len(merged) == len(bars)
    assert merged["FRED_DGS10"].tolist() == pytest.approx([4.00] * 10)


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
