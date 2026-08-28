"""A macro reading that does not exist must stay missing.

Measured on the 110-ticker batch of 2026-08-27: 70.5% of every
`FRED_INDPRO_1d` value and 70.6% of every `FRED_VIXCLS_1d` value was exactly
the column's own median. Seven of every ten macro readings were a constant
rather than data, and the constant was computed over the whole frame -- so a
row dated 2018 carried an average that includes 2026. Every model that read a
FRED column read the future.

It also invented cross-sectional variation. A macro series is one number for
the whole economy on a date; with some rows real and others constant, the
column appeared to differ between tickers on 98% of dates, and on 2026-08-12
the split was exactly the 22 tickers of the old preset against the 88 added
later. That artefact ranked near the top of the leading-feature report,
because a ranking measures precisely that variation.

These tests exist so the fill can never fabricate again, and so a macro column
cannot silently start differing between names on one date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.macro_features_enricher import MacroFeaturesEnricher


@pytest.fixture
def enricher() -> MacroFeaturesEnricher:
    return MacroFeaturesEnricher.__new__(MacroFeaturesEnricher)


def _frame() -> pd.DataFrame:
    """Two tickers, one macro column, gaps in different places."""
    return pd.DataFrame({
        "ticker": ["AAPL"] * 5 + ["MSFT"] * 5,
        "FRED_INDPRO": [np.nan, 100.0, np.nan, np.nan, 101.0,
                        np.nan, np.nan, 200.0, np.nan, np.nan],
    })


def test_missing_readings_stay_missing(enricher):
    """The rows before a series begins have no value, and must say so."""
    out = enricher._post_process_fred_columns(_frame())

    # AAPL's first row precedes any reading; MSFT's first two do the same.
    assert np.isnan(out["FRED_INDPRO"].iloc[0])
    assert out["FRED_INDPRO"].iloc[5:7].isna().all()


def test_nothing_equals_the_column_median(enricher):
    """The specific line that broke this: fillna(median) over the frame."""
    out = enricher._post_process_fred_columns(_frame())
    values = out["FRED_INDPRO"].dropna()

    assert not np.isclose(values, values.median()).all()
    # 100, 101 and 200 are the only readings; nothing else may appear.
    assert set(np.unique(values)) <= {100.0, 101.0, 200.0}


def test_the_carry_forward_does_not_cross_tickers(enricher):
    """AAPL's last reading must never appear on MSFT's rows.

    A plain ffill over a frame with every ticker stacked carries the last row
    of one name into the first rows of the next, which is how a macro column
    came to hold different values for different names on the same date.
    """
    out = enricher._post_process_fred_columns(_frame())
    msft = out.loc[out["ticker"] == "MSFT", "FRED_INDPRO"]

    assert not (msft == 101.0).any(), "AAPL's value leaked into MSFT"
    assert msft.iloc[2:].tolist() == [200.0, 200.0, 200.0]


def test_a_reading_is_carried_forward_within_its_own_ticker(enricher):
    """The fill still has to work -- macro is published monthly, bars daily."""
    out = enricher._post_process_fred_columns(_frame())
    aapl = out.loc[out["ticker"] == "AAPL", "FRED_INDPRO"]

    assert aapl.tolist()[1:4] == [100.0, 100.0, 100.0]


def test_a_frame_without_tickers_still_works(enricher):
    """Some callers hand over a single name's frame; it must not raise."""
    frame = pd.DataFrame({"FRED_INDPRO": [np.nan, 100.0, np.nan]})
    out = enricher._post_process_fred_columns(frame)

    assert np.isnan(out["FRED_INDPRO"].iloc[0])
    assert out["FRED_INDPRO"].iloc[2] == 100.0


def test_non_fred_columns_are_left_alone(enricher):
    frame = _frame()
    frame["close"] = [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    out = enricher._post_process_fred_columns(frame)

    assert np.isnan(out["close"].iloc[1])


def test_the_carry_stops_after_its_limit(enricher):
    """60 rows, so a series that stopped publishing does not run forever."""
    frame = pd.DataFrame({
        "ticker": ["AAPL"] * 70,
        "FRED_INDPRO": [100.0] + [np.nan] * 69,
    })
    out = enricher._post_process_fred_columns(frame)

    assert out["FRED_INDPRO"].iloc[60] == 100.0
    assert np.isnan(out["FRED_INDPRO"].iloc[61])


# ----------------------------------------------------------------------
# The direction of the fill, which no reading of the code could reveal
# ----------------------------------------------------------------------

def _thirty_year_frame(descending: bool) -> pd.DataFrame:
    """Thirty years of bars for two tickers, in the given row order."""
    dates = pd.date_range("1996-01-01", "2026-01-01", freq="7D")
    frame = pd.concat(
        [pd.DataFrame({"datetime": dates, "ticker": name, "close": 1.0})
         for name in ("AAPL", "MSFT")],
        ignore_index=True,
    )
    if descending:
        frame = frame.sort_values(
            ["ticker", "datetime"], ascending=[True, False]
        ).reset_index(drop=True)
    return frame


def _two_years_of_macro() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2026-01-01", freq="30D")
    return pd.DataFrame({
        "datetime": dates,
        "series_id": "CPIAUCSL",
        "value": np.linspace(300.0, 330.0, len(dates)),
        "available_at": dates,
    })


def _merged(enricher, descending: bool) -> pd.DataFrame:
    pivoted = enricher._pivot_macro_data(_two_years_of_macro())
    pivoted.index.name = "datetime"
    frame = _thirty_year_frame(descending).set_index("datetime")
    out = enricher._merge_macro_data(frame, pivoted).reset_index()
    out["year"] = pd.to_datetime(out["datetime"]).dt.year
    return out


@pytest.mark.parametrize("descending", [False, True])
def test_the_past_is_not_filled_from_the_future(enricher, descending):
    """`ffill` walks rows, not dates, and there is no bfill in this code.

    On the 110-ticker batch of 2026-08-28, FRED_CPIAUCSL_1d held 313.569 on
    every row from 1996 to 2023 -- the 2024 level, against an actual 1996 CPI
    of about 157 -- with 0.5% NaN. Reproduced by running this same merge twice
    over the same data and changing only the row order:

        ascending    1996 = NaN     93.3% NaN
        descending   1996 = 300.0    0.0% NaN

    Thirty years of lookahead, and not one `bfill` anywhere to find by reading.
    """
    out = _merged(enricher, descending)
    column = next(c for c in out.columns if "CPIAUCSL" in c)
    early = out.loc[out["year"] <= 2023, column]

    assert early.isna().all(), (
        "rows dated before the series begins were handed a later value"
    )


@pytest.mark.parametrize("descending", [False, True])
def test_the_rows_that_do_have_a_reading_keep_it(enricher, descending):
    """The fill still has to work forwards, or the fix is just deletion."""
    out = _merged(enricher, descending)
    column = next(c for c in out.columns if "CPIAUCSL" in c)

    assert out.loc[out["year"] == 2025, column].notna().all()
    assert 300.0 <= out.loc[out["year"] == 2025, column].median() <= 330.0


def test_row_order_does_not_change_the_result(enricher):
    """Same data in either order must produce the same numbers."""
    ascending = _merged(enricher, descending=False)
    descending = _merged(enricher, descending=True)
    column = next(c for c in ascending.columns if "CPIAUCSL" in c)

    left = ascending.groupby("year")[column].median()
    right = descending.groupby("year")[column].median()
    pd.testing.assert_series_equal(left, right)
