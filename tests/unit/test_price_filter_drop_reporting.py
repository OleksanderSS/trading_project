"""Every path that drops a timeframe must run, and must say why.

I added ERROR logging to PriceFilter's four exits in b42fd866 and tested
only the path where data PASSES -- so none of the new lines ever executed
under test. `self.logger` did not exist on PriceFilter (only a module-level
`logger` did), and the first real prepare run raised AttributeError inside
the very code meant to explain a drop.

These tests exercise each exit, which is what should have been written the
first time. Logging on an error path is code; code that only runs when
something has already gone wrong is exactly the code that must be tested
deliberately, because nothing else will reach it.

The drops themselves are real and correct. The 2026-08-05 prepare run
reported:

    Timeframe '15m' DROPPED on cross_ticker_duplicate_ohlcv,
    extreme_return_contamination (63085 rows). cadence_match=0.939,
    extreme_return_ratio=0.069

and the database bears it out: 4,668 rows of 15m carry prices belonging to
another instrument (KO above 200, INTC above 300), 16 of 24 tickers have a
15m range inconsistent with their own daily range, and 1d and 1h have none
of it.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.processing.filters.price_filter import PriceFilter


@pytest.fixture()
def price_filter():
    return PriceFilter({})


def _bars(rows=60, close=100.0, ticker="AAPL", interval="15m", start="2026-08-03"):
    index = pd.date_range(start, periods=rows, freq="15min", tz="UTC")
    return pd.DataFrame({
        "ticker": [ticker] * rows,
        "interval": [interval] * rows,
        "datetime": index,
        "open": [close] * rows,
        "high": [close * 1.001] * rows,
        "low": [close * 0.999] * rows,
        "close": [close] * rows,
        "volume": [1000.0] * rows,
    })


def test_the_filter_has_the_logger_the_drop_paths_use(price_filter):
    """The regression itself: self.logger did not exist."""
    assert hasattr(price_filter, "logger")
    assert price_filter.logger is not None


def test_an_empty_timeframe_is_dropped_and_reported(price_filter, caplog):
    with caplog.at_level(logging.ERROR):
        kept, report = price_filter.filter_price_data({"15m": pd.DataFrame()})

    assert "15m" not in kept
    assert report["15m"]["status"] == "empty"
    assert any("DROPPED" in record.message for record in caplog.records)


def test_too_few_candles_is_dropped_and_reported(price_filter, caplog):
    with caplog.at_level(logging.ERROR):
        kept, report = price_filter.filter_price_data({"15m": _bars(rows=5)})

    assert "15m" not in kept
    assert report["15m"]["status"] == "insufficient_data"
    assert any("minimum is" in record.message for record in caplog.records)


def test_extreme_returns_are_dropped_and_the_numbers_reported(price_filter, caplog):
    """A close that doubles bar to bar is another instrument's price, which
    is exactly what the 15m table holds."""
    frame = _bars(rows=60)
    frame.loc[frame.index[30:], "close"] = 900.0

    with caplog.at_level(logging.ERROR):
        kept, report = price_filter.filter_price_data({"15m": frame})

    assert "15m" not in kept
    assert "extreme_return_contamination" in report["15m"]["reason"]
    assert any("extreme_return_ratio" in record.message for record in caplog.records)


def test_clean_data_passes_and_is_not_reported(price_filter, caplog):
    """The path I did test the first time -- kept, so the drop paths are
    demonstrably distinguishable from it."""
    generator = np.random.default_rng(0)
    frame = _bars(rows=200)
    frame["close"] = 100.0 * (1.0 + generator.normal(0, 0.001, 200)).cumprod()
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] * 1.001
    frame["low"] = frame["close"] * 0.999

    with caplog.at_level(logging.ERROR):
        kept, _ = price_filter.filter_price_data({"15m": frame})

    assert "15m" in kept
    assert not [r for r in caplog.records if "DROPPED" in r.message]


def test_every_drop_path_logs():
    """Guards the shape rather than one instance: a future `continue` added
    beside these without a log would restore the silence."""
    import inspect
    import textwrap

    from tests.contracts._lookahead_scan import _code_only

    # Comments stripped. The block introducing these exits uses the word
    # "continue" to describe them, and counting it made this assertion fail
    # on its own documentation -- the third time in this audit that matching
    # source text without removing prose has produced a false result.
    source = textwrap.dedent(inspect.getsource(PriceFilter.filter_price_data))
    code = "\n".join(_code_only(source).values())
    continues = code.count("continue")
    logged = code.count("self.logger.error")

    assert logged >= continues, (
        f"{continues} exit(s) but only {logged} logged -- a timeframe can "
        "still disappear without a word"
    )
