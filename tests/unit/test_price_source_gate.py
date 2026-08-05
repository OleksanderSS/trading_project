"""No price row reaches the database without passing the gate.

The gate existed as a PRIVATE METHOD on the Yahoo collector -- one of 22 --
and BaseCollector has no validation hook, so any second price source would
have written into market_data_raw with nothing between it and the table.

The cost of that arrangement is measured, not hypothetical. A yfinance
shared-cache race filed one instrument's bars under another ticker. 63,038
rows sat in the database for four months; 4,668 carried impossible prices
(KO above 900 when it trades near 47, INTC above 900 when its range is
18-141). Nothing noticed until Stage 2's PriceFilter -- three stages
downstream -- refused the whole 15m timeframe, and even then the reason was
only visible after logging was added to the drop paths.

The gate now lives in one module, and DataManager.upsert applies it to every
PRICE_TABLES write. That is the version that cannot be bypassed by
forgetting to call it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import DataProcessingError
from src.data.management.data_manager import DataManager
from src.data.validation.price_source_gate import price_source_issues


def _bars(ticker="AAPL", interval="15m", rows=4, close=100.0):
    return pd.DataFrame({
        "datetime": pd.date_range("2026-08-03 13:30", periods=rows, freq="15min", tz="UTC"),
        "ticker": [ticker] * rows,
        "interval": [interval] * rows,
        "open": [close] * rows,
        "high": [close] * rows,
        "low": [close] * rows,
        "close": [close] * rows,
        "volume": [1000.0] * rows,
    })


def test_clean_bars_pass():
    assert price_source_issues(_bars()) == []


def test_the_shape_that_destroyed_15m_is_caught():
    """Identical OHLCV under two different tickers -- one instrument's bars
    filed under another's name."""
    contaminated = pd.concat([_bars("AAPL"), _bars("KO")], ignore_index=True)

    issues = price_source_issues(contaminated)

    assert any("cross_identity_ohlcv_rows" in issue for issue in issues)


def test_the_same_bar_twice_is_caught():
    duplicated = pd.concat([_bars(), _bars()], ignore_index=True)

    assert any("duplicate_identity_rows" in i for i in price_source_issues(duplicated))


def test_naive_timestamps_are_caught():
    """A timestamp without a zone cannot be aligned against anything else."""
    frame = _bars()
    frame["datetime"] = frame["datetime"].dt.tz_localize(None)

    assert "datetime_timezone_unresolved" in price_source_issues(frame)


def test_bars_spaced_wrongly_for_their_interval_are_caught():
    frame = _bars(interval="15m")
    frame["datetime"] = pd.date_range(
        "2026-08-03 13:30", periods=len(frame), freq="7min", tz="UTC"
    )

    assert any("cadence_mismatch" in i for i in price_source_issues(frame))


def test_missing_columns_are_reported_before_anything_else():
    assert price_source_issues(pd.DataFrame({"close": [1.0]})) == [
        "missing_columns=datetime,high,interval,low,open,ticker,volume"
    ]


def test_an_empty_frame_is_reported():
    assert price_source_issues(
        pd.DataFrame(columns=["datetime", "ticker", "interval", "open", "high",
                              "low", "close", "volume"])
    ) == ["empty_market_data"]


# --- the enforcement point ------------------------------------------------

class _Manager:
    """DataManager's gate without a database behind it."""

    PRICE_TABLES = DataManager.PRICE_TABLES
    _gate_price_source = DataManager._gate_price_source


def test_a_contaminated_write_to_a_price_table_is_refused():
    contaminated = pd.concat([_bars("AAPL"), _bars("KO")], ignore_index=True)

    with pytest.raises(DataProcessingError, match="source gate"):
        _Manager()._gate_price_source("market_data_raw", contaminated)


def test_a_clean_write_to_a_price_table_passes():
    _Manager()._gate_price_source("market_data_raw", _bars())


def test_non_price_tables_are_untouched():
    """The gate is about OHLCV; news and macro tables have other shapes and
    must not be judged by this one."""
    _Manager()._gate_price_source("google_news", pd.DataFrame({"title": ["x"]}))


def test_the_collector_uses_the_shared_gate_rather_than_a_copy():
    """A second implementation is how the two would drift apart."""
    import inspect

    from src.data.collectors.yf_collector import YFCollector

    source = inspect.getsource(YFCollector._validate_collected_price_data)

    assert "price_source_issues(frame)" in source
    assert "cross_identity" not in source, "the checks were copied, not shared"


# --- quarantine rather than reject the batch -------------------------------

def _quarantine(frame):
    from src.data.validation.price_source_gate import quarantine_bad_rows
    return quarantine_bad_rows(frame)


def test_a_few_bad_rows_do_not_bin_the_batch():
    """The 2026-08-05 run: 202,713 rows collected, all refused over 102.

    0.05% of the download, on a database that had already gone six days
    without a new row.
    """
    good = pd.concat([_bars("AAPL"), _bars("MSFT", close=250.0)], ignore_index=True)
    contaminated = pd.concat([good, _bars("KO")], ignore_index=True)

    clean, rejected, fatal = _quarantine(contaminated)

    assert fatal == []
    assert set(clean["ticker"]) == {"AAPL", "MSFT"}
    assert set(rejected["ticker"]) == {"KO"}


def test_what_survives_quarantine_passes_the_gate():
    """Otherwise the enforcement point at DataManager.upsert would refuse it."""
    contaminated = pd.concat([_bars("AAPL"), _bars("KO")], ignore_index=True)

    clean, _, _ = _quarantine(contaminated)

    assert price_source_issues(clean) == []


def test_one_side_of_a_duplicate_pair_is_kept():
    """One of them is the genuine bar. Dropping both loses real data to
    punish the copy."""
    contaminated = pd.concat([_bars("AAPL"), _bars("KO")], ignore_index=True)

    clean, rejected, _ = _quarantine(contaminated)

    assert len(clean) == len(_bars("AAPL"))
    assert len(rejected) == len(_bars("KO"))


def test_the_same_bar_recorded_twice_keeps_one():
    doubled = pd.concat([_bars("AAPL"), _bars("AAPL")], ignore_index=True)

    clean, rejected, _ = _quarantine(doubled)

    assert len(clean) == 4
    assert len(rejected) == 4


def test_a_frame_level_defect_still_fails_everything():
    """Absent columns leave nothing to separate the good rows from."""
    clean, rejected, fatal = _quarantine(pd.DataFrame({"close": [1.0]}))

    assert fatal and clean.empty


def test_naive_timestamps_are_frame_level():
    """A missing timezone is a property of the whole download, not of rows."""
    frame = _bars()
    frame["datetime"] = frame["datetime"].dt.tz_localize(None)

    _, _, fatal = _quarantine(frame)

    assert "datetime_timezone_unresolved" in fatal


def test_a_clean_batch_passes_through_untouched():
    frame = pd.concat([_bars("AAPL"), _bars("MSFT", close=250.0)], ignore_index=True)

    clean, rejected, fatal = _quarantine(frame)

    assert len(clean) == len(frame) and rejected.empty and fatal == []


def test_the_collector_quarantines_instead_of_raising():
    import inspect

    from src.data.collectors.yf_collector import YFCollector

    source = inspect.getsource(YFCollector.run)

    assert "quarantine_bad_rows(df_to_check)" in source


# --- a daily bar wearing an hourly label -----------------------------------

def _series(ticker, interval, rows, close, start, freq, volume=32074200.0):
    return pd.DataFrame({
        "datetime": pd.date_range(start, periods=rows, freq=freq, tz="UTC"),
        "ticker": [ticker] * rows,
        "interval": [interval] * rows,
        "open": [close] * rows, "high": [close] * rows,
        "low": [close] * rows, "close": [close] * rows,
        "volume": [volume] * rows,
    })


def test_a_daily_bar_copied_into_the_hourly_series_is_caught():
    """Found in the live database: AAPL's 2026-03-16 daily bar appeared in
    the 1h series at 02:00 (same timestamp) AND at 06:00 (different one).
    The existing cross-identity check compares rows sharing a timestamp, so
    it saw the first and missed the second. 80 such rows were present."""
    frame = pd.concat([
        _series("AAPL", "1h", 6, 100.0, "2026-03-16 16:00", "1h"),
        _series("AAPL", "1d", 1, 100.0, "2026-03-16 02:00", "1d"),
    ], ignore_index=True)

    assert any("cross_interval_ohlcv_rows" in i for i in price_source_issues(frame))


def test_the_hourly_copy_is_dropped_and_the_daily_bar_kept():
    """Direction matters. The daily bar is a genuine observation; an hourly
    row carrying a full day's volume is that bar mislabelled. Dropping the
    daily one would delete the real data and leave the intruder."""
    frame = pd.concat([
        _series("AAPL", "1h", 6, 100.0, "2026-03-16 16:00", "1h"),
        _series("AAPL", "1d", 1, 100.0, "2026-03-16 02:00", "1d"),
    ], ignore_index=True)

    clean, rejected, _ = _quarantine(frame)

    assert set(rejected["interval"]) == {"1h"}
    assert set(clean["interval"]) == {"1d"}


def test_different_prices_across_intervals_are_left_alone():
    """The normal case: an hour and a day simply do not match."""
    frame = pd.concat([
        _series("AAPL", "1h", 6, 100.0, "2026-03-16 16:00", "1h"),
        _series("AAPL", "1d", 1, 250.0, "2026-03-16 02:00", "1d"),
    ], ignore_index=True)

    clean, rejected, _ = _quarantine(frame)

    assert rejected.empty and len(clean) == len(frame)


def test_one_interval_alone_is_never_flagged():
    frame = _series("AAPL", "1h", 6, 100.0, "2026-03-16 16:00", "1h")

    assert not any("cross_interval" in i for i in price_source_issues(frame))
