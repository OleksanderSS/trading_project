"""A fingerprint of 185 columns identifies the row, not the situation.

Measured on the 2026-08-14 export, per timeframe, over rows that actually
carry that timeframe's data:

    15m   12,177 rows   12,170 distinct fingerprints   7 ever repeat
    60m   19,194 rows   19,175 distinct               6 ever repeat
    1d    11,368 rows   11,350 distinct               3 ever repeat

Everything built on top follows from that. `context_velocity` is a rolling
mean of "the fingerprint changed", and a fingerprint that never repeats
changes on every bar, so velocity measured 0.9994 and was reported as
constant. `context_anxiety_index` is `velocity > 0.6`, so it is the constant
1. `context_pattern_id` hashes a sequence of fingerprints, making it a row id
-- and the k-NN logic it exists to serve cannot match a pattern that occurs
once.

Width is the whole story. On the same rows, using non-calendar state columns:

    width 6   -> 215 repeated groups, median size 23
    width 8   -> 841 repeated groups, median size  5
    width 12  -> 1,611 groups, median size 3
    width 185 ->     7 groups, median size 1

Two things the numbers also settle. Ranked by entropy WITH calendar columns
present, day_of_year wins outright (H=5.06 against 1.58 for the best market
column) -- a fingerprint keyed on the date matches nothing but coincidence.
And among market columns the top eight are EMA_20, SMA_20, BB_Middle,
SMA_200, SMA_10, EMA_10, obv, EMA_50: seven of them ask "is price above a
moving average", so eight columns carry about one bit.

Which columns to use is a modelling decision and stays with the operator.
What this file fixes is that there was no way to express it, and no signal
that the default was useless.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.context_map_enricher import ContextMapEnricher


def _frame():
    n = 60
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "datetime": pd.date_range("2026-07-01", periods=n, freq="h", tz="UTC"),
        "state_trend": rng.choice([-1, 0, 1], n),
        "state_vol": rng.choice([-1, 0, 1], n),
        "state_volume": rng.choice([-1, 0, 1], n),
        "state_day_of_year": np.arange(n),          # a calendar column: unique
        "state_noise": np.arange(n),                # unique per row
    })


def test_a_narrow_fingerprint_repeats_and_a_wide_one_does_not():
    df = _frame()
    wide = df[["state_trend", "state_vol", "state_volume",
               "state_day_of_year", "state_noise"]].astype(str).agg("|".join, axis=1)
    narrow = df[["state_trend", "state_vol", "state_volume"]].astype(str).agg("|".join, axis=1)

    assert wide.nunique() == len(df), "one unique fingerprint per row"
    assert narrow.nunique() < len(df)
    assert (narrow.value_counts() > 1).sum() > 0, "situations must recur"


def test_the_configured_columns_are_used_in_the_order_given():
    """Position i of a fingerprint IS driver i, so order is part of the schema."""
    enricher = ContextMapEnricher({"fingerprint_columns":
                                   ["state_vol", "state_trend"]})
    df = _frame()

    drivers = enricher._fingerprint_drivers(df, ["state_trend", "state_vol"], [])

    assert drivers == ["state_vol", "state_trend"], "not re-sorted"


def test_a_name_that_does_not_exist_is_reported_not_skipped(caplog):
    """A typo would otherwise narrow the fingerprint invisibly."""
    import logging

    enricher = ContextMapEnricher({"fingerprint_columns":
                                   ["state_trend", "state_typo"]})

    with caplog.at_level(logging.ERROR):
        drivers = enricher._fingerprint_drivers(_frame(), ["state_trend"], [])

    assert drivers == ["state_trend"]
    assert "state_typo" in "\n".join(r.getMessage() for r in caplog.records)


def test_the_unconfigured_default_says_what_it_costs(caplog):
    """Behaviour is unchanged without a list -- but no longer silent."""
    import logging

    enricher = ContextMapEnricher()

    with caplog.at_level(logging.WARNING):
        drivers = enricher._fingerprint_drivers(
            _frame(), ["state_trend", "state_vol"], ["state_day_of_year"]
        )

    assert drivers == ["state_day_of_year", "state_trend", "state_vol"]
    assert "context_velocity" in "\n".join(r.getMessage() for r in caplog.records)


def test_velocity_does_not_average_across_two_tickers():
    """The change flag was per ticker; the rolling mean was not.

    Each ticker's first `velocity_window` bars mixed in the tail of whichever
    ticker happened to precede it in row order.
    """
    enricher = ContextMapEnricher({"velocity_window": 3})
    df = pd.DataFrame({
        "ticker": ["AAPL"] * 6 + ["MSFT"] * 6,
        # AAPL changes on every bar; MSFT never changes after its first.
        "context_fingerprint": [f"a{i}" for i in range(6)] + ["m"] * 6,
    })

    enricher._calculate_context_velocity(df)

    msft = df.loc[df["ticker"] == "MSFT", "context_velocity"]
    assert msft.iloc[-1] == pytest.approx(0.0), (
        "a ticker whose context never changes must show zero velocity, "
        "whatever the ticker before it was doing"
    )
