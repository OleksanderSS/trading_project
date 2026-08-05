"""One delisted symbol must not discard an entire interval's download.

_single_ticker_download_with_retry raises RuntimeError after exhausting its
retries, and _blocking_download's loop had no handler. The exception unwound
past every ticker already downloaded, so all_ticker_data -- complete up to
that point -- was thrown away with it. The caller logs one line per interval,
"[YF] Download task failed", and the database silently gains nothing.

Observed on the 2026-08-05 prepare run: 'BLOCK' is no longer a valid Yahoo
symbol ("possibly delisted; no timezone found"). Its failure discarded 15m,
1h AND 1d for all 114 tickers. market_data_raw had not gained a row since
2026-07-30 -- five days of collection lost to one renamed instrument, and
nothing in the log said so.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import pytest

from src.data.collectors.yf_collector import YFCollector


class _Collector:
    """_blocking_download with the download call stubbed."""

    def __init__(self, dead: set[str]):
        self.dead = dead
        self.configs = {"max_retries": 1, "retry_delay": 0}
        self.logger = logging.getLogger("yf-one-bad-ticker-test")

    def _single_ticker_download_with_retry(self, ticker, interval, start, end, **kw):
        if ticker in self.dead:
            raise RuntimeError(f"Data download failed for {ticker}/{interval}")
        return pd.DataFrame({"Close": [1.0, 2.0]})

    def _process_single_ticker_dataframe(self, df, ticker, interval):
        return [{"ticker": ticker, "interval": interval, "close": c}
                for c in df["Close"]]

    _blocking_download = YFCollector._blocking_download


def _run(dead, tickers=("AAPL", "BLOCK", "MSFT", "NVDA")):
    collector = _Collector(set(dead))
    return collector._blocking_download(
        list(tickers), "15m", datetime(2026, 8, 1), datetime(2026, 8, 5)
    )


def test_the_healthy_tickers_survive_a_dead_one():
    rows = _run({"BLOCK"})

    assert {row["ticker"] for row in rows} == {"AAPL", "MSFT", "NVDA"}
    assert len(rows) == 6


def test_a_ticker_failing_first_does_not_discard_the_rest():
    """Position must not matter; before the fix, the earliest failure lost
    everything after it as well."""
    rows = _run({"AAPL"})

    assert {row["ticker"] for row in rows} == {"BLOCK", "MSFT", "NVDA"}


def test_all_tickers_healthy_returns_everything():
    rows = _run(set())

    assert len({row["ticker"] for row in rows}) == 4


def test_every_ticker_failing_returns_empty_rather_than_raising():
    assert _run({"AAPL", "BLOCK", "MSFT", "NVDA"}) == []


def test_the_failures_are_summarised_once_with_the_count(caplog):
    with caplog.at_level(logging.ERROR):
        _run({"BLOCK", "MSFT"})

    summaries = [r for r in caplog.records if "could not be downloaded" in r.message]
    assert len(summaries) == 1
    assert "2 of 4" in summaries[0].getMessage()


def test_each_skip_is_named(caplog):
    with caplog.at_level(logging.WARNING):
        _run({"BLOCK"})

    assert any("Skipping BLOCK/15m" in r.getMessage() for r in caplog.records)


def test_the_loop_still_raises_nothing_upward():
    """The caller gathers with return_exceptions=True and only logs, so an
    escape here is invisible apart from the missing data."""
    try:
        _run({"BLOCK"})
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"a dead ticker escaped the loop: {exc!r}")
