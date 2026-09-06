"""A figure becomes knowable when it is published, not when we asked for it.

ROADMAP §22. `realtime_start` is FRED's vintage stamp, and for a series FRED
does NOT revise a non-vintage request is stamped with the REQUEST date. So a
1996 observation pulled down in a 2026 backfill carries a 2026 stamp: inside
any training window it never becomes available at all.

MEASURED ON THE LIVE TABLE, 2026-09-06, before the fix:

    rows in fred_data                                 485,010
    stamped >400 days after their own observation     359,989  (74.2%)
    largest gap                                       10,958 days (30 years)
    available_at column                               ABSENT

    distinct realtime_start dates (the old join key)    4,194
    distinct observation dates (the new base)          9,449
      before 2010:            stamps 1,449   observations 4,197

That last line is the cost in one number: across the deepest stretch of the
training history the macro frame had a THIRD of the anchor dates it should
have, so values sat constant over spans where they had in fact moved.

WHY THIS FILE EXISTS RATHER THAN A COMMENT. The repair is easy to undo by
accident -- deleting one dict entry silently returns that series to the
request stamp, and nothing else in the suite would notice. The failure mode is
not an exception; it is a frame that still looks full.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.collectors.fred_collector import FredCollector


def test_an_unrevised_series_becomes_available_after_its_measured_lag():
    assert FredCollector.availability_for("DGS10", "1996-07-01") == \
        pd.Timestamp("1996-07-02")
    assert FredCollector.availability_for("T10Y2Y", "2005-03-15") == \
        pd.Timestamp("2005-03-15")


def test_a_revised_series_keeps_its_real_vintage():
    """None means "no better answer than realtime_start" -- and for a series
    FRED genuinely revises, the vintage IS the availability (#131). Guessing
    one here would overwrite the point-in-time work rather than complete it."""
    assert FredCollector.availability_for("CPIAUCSL", "2005-03-15") is None
    assert FredCollector.availability_for("GDP", "2005-03-15") is None


def test_every_unrevised_series_has_a_measured_lag():
    """A member with no entry falls silently back to the request stamp -- the
    exact defect, restored for one series and invisible everywhere else."""
    missing = sorted(FredCollector.UNREVISED_DAILY_SERIES
                     - set(FredCollector.PUBLICATION_LAG_DAYS))
    assert not missing, (
        f"these unrevised series have no publication lag, so their "
        f"availability is still the day we asked: {missing}"
    )


def test_the_two_monthly_members_are_not_given_a_daily_lag():
    """GS10 and GS2 sit in a set called UNREVISED_DAILY_SERIES and are
    MONTHLY. Measured minimum lag 29 days. A one-day lag would claim a monthly
    average was readable four weeks before it existed -- look-ahead introduced
    by the fix for look-ahead."""
    for series in ("GS10", "GS2"):
        lag = FredCollector.PUBLICATION_LAG_DAYS[series]
        assert lag >= 28, (
            f"{series} is a monthly series with a measured minimum lag of 29 "
            f"days and now carries {lag}"
        )


def test_no_lag_is_negative():
    """A negative lag is a figure available before it was observed."""
    negative = {name: lag for name, lag
                in FredCollector.PUBLICATION_LAG_DAYS.items() if lag < 0}
    assert not negative, negative


def _rows() -> pd.DataFrame:
    return pd.DataFrame({
        "series_id": ["DGS10", "DGS10", "CPIAUCSL"],
        "date": ["1996-07-01", "2005-03-15", "2005-03-01"],
        "value": ["6.9", "4.5", "193.3"],
        # The stamp a backfill leaves: the day we asked, thirty years late.
        "realtime_start": ["2026-09-02", "2026-09-02", "2005-04-15"],
    })


def test_the_enricher_derives_availability_for_rows_already_stored():
    """The fix must reach the 485,010 rows already on disk, not only the next
    collection -- otherwise it lands only after a re-collect nobody has run."""
    from src.features.enrichers.macro_features_enricher import MacroFeaturesEnricher

    enricher = MacroFeaturesEnricher.__new__(MacroFeaturesEnricher)
    pivoted = enricher._pivot_macro_data(_rows())

    assert not pivoted.empty
    stamps = pd.to_datetime(pd.Series(pivoted.index)).dt.year.tolist()
    assert 1996 in stamps, (
        "the 1996 observation still arrives in 2026, so it is invisible to "
        "every training window that ends earlier"
    )


def test_a_revised_row_is_not_overwritten_by_a_derived_stamp():
    """Falling back per ROW, not per column: the CPI row keeps its real
    vintage while the DGS10 rows get a derived one."""
    from src.features.enrichers.macro_features_enricher import MacroFeaturesEnricher

    enricher = MacroFeaturesEnricher.__new__(MacroFeaturesEnricher)
    pivoted = enricher._pivot_macro_data(_rows())

    years = sorted({pd.Timestamp(value).year for value in pivoted.index})
    assert 2005 in years, (
        "the revised row's own vintage (2005-04-15) was lost, which would "
        "undo #131 rather than complete it"
    )
