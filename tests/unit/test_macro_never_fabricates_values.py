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
