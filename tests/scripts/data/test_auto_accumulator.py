#!/usr/bin/env python3
"""Tests for the Integrity Guard (auto_accumulator).

The previous file in this path tested an API that has never existed: it
patched a class named `AutoAccumulator` (the class is `AutoAccumulatorGuard`)
and asserted calls to `run_accumulation_cycle`, `run_scheduled_accumulation`,
`run_continuous_accumulation` and `get_accumulation_report`, none of which are
defined anywhere. It could only ever have passed against mocks of itself.
"""
from __future__ import annotations

import asyncio
import datetime as dt
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.scripts.data.auto_accumulator import AutoAccumulatorGuard


def _guard(latest_rows: list[tuple[str, str, str]], *, staleness_days: int = 1):
    """Build a guard with the DB and config layers stubbed out."""
    guard = object.__new__(AutoAccumulatorGuard)
    guard.staleness_days = staleness_days
    guard.active_tickers = ["AAA", "BBB"]
    guard.timeframes = {"15m": {}, "1d": {}}

    frame = pd.DataFrame(latest_rows, columns=["ticker", "interval", "latest"])
    guard._latest_timestamps = lambda: frame  # type: ignore[method-assign]

    calendar = MagicMock()
    calendar.get_previous_trading_days.return_value = [dt.date(2026, 7, 30)]
    guard.calendar = calendar
    return guard


def test_missing_pair_is_reported_as_a_gap():
    guard = _guard([("AAA", "15m", "2026-07-30 20:00:00")])
    gaps = guard.find_gaps()

    assert "1d" in gaps["AAA"]                  # AAA has no 1d row at all
    assert set(gaps["BBB"]) == {"15m", "1d"}    # BBB has nothing
    assert "15m" not in gaps.get("AAA", [])     # ...but its 15m is current


def test_series_ending_on_the_cutoff_day_is_current():
    guard = _guard([
        ("AAA", "15m", "2026-07-30 22:45:00"),
        ("AAA", "1d", "2026-07-30 03:00:00"),
    ])
    assert "AAA" not in guard.find_gaps()


def test_series_ending_before_the_cutoff_day_is_stale():
    guard = _guard([
        ("AAA", "15m", "2026-07-20 22:45:00"),
        ("AAA", "1d", "2026-07-20 03:00:00"),
    ])
    assert set(guard.find_gaps()["AAA"]) == {"15m", "1d"}


def test_staleness_is_counted_in_trading_days_not_wall_clock_minutes():
    """An intraday series must not go stale simply because it is evening.

    An earlier draft measured staleness as "3 intervals of silence", which
    marks every 15m series stale 45 minutes after the close -- so overnight and
    at weekends it flagged the entire universe.
    """
    guard = _guard([
        ("AAA", "15m", "2026-07-30 16:00:00"),
        ("AAA", "1d", "2026-07-30 03:00:00"),
    ])
    assert "AAA" not in guard.find_gaps()


def test_cutoff_uses_the_oldest_of_the_requested_trading_days():
    guard = _guard(
        [("AAA", "15m", "2026-07-28 20:00:00"), ("AAA", "1d", "2026-07-28 03:00:00")],
        staleness_days=3,
    )
    guard.calendar.get_previous_trading_days.return_value = [
        dt.date(2026, 7, 28), dt.date(2026, 7, 29), dt.date(2026, 7, 30),
    ]
    # Tolerating 3 trading days of lag, a series ending on the oldest of them
    # is still acceptable.
    assert "AAA" not in guard.find_gaps()


def test_no_gaps_means_the_cycle_does_no_work():
    guard = _guard([
        ("AAA", "15m", "2026-07-30 22:45:00"), ("AAA", "1d", "2026-07-30 03:00:00"),
        ("BBB", "15m", "2026-07-30 22:45:00"), ("BBB", "1d", "2026-07-30 03:00:00"),
    ])
    with patch.object(AutoAccumulatorGuard, "_refetch") as refetch:
        assert guard.run_guard_cycle() is False
        refetch.assert_not_called()


def test_gaps_are_grouped_by_interval_so_one_run_covers_many_tickers():
    guard = _guard([])   # everything missing
    calls: list[tuple[list[str], list[str]]] = []

    async def fake_refetch(tickers, intervals):
        calls.append((sorted(tickers), sorted(intervals)))

    guard._refetch = fake_refetch  # type: ignore[method-assign]
    assert guard.run_guard_cycle() is True

    # One call per interval, each carrying both tickers -- not one call per pair.
    assert sorted(c[1][0] for c in calls) == ["15m", "1d"]
    for tickers, _ in calls:
        assert tickers == ["AAA", "BBB"]


@pytest.mark.parametrize("interval", ["15m", "1h", "1d"])
def test_refetch_restricts_the_collector_to_the_requested_timeframe(interval):
    guard = _guard([])
    guard.timeframes = {
        "15m": {"period": "60d"}, "1h": {"period": "60d"}, "1d": {"period": "2y"},
    }
    guard.config_manager = MagicMock()
    guard.config_manager.get_config.return_value = {"yahoo_finance": {"enabled": False}}
    guard.http_factory = MagicMock()
    guard.db_manager = MagicMock()

    captured: dict = {}

    class FakeCollector:
        def __init__(self, cfg, *_args, **_kwargs):
            captured.update(cfg)

        async def run(self, tickers):
            captured["tickers"] = tickers

    with patch("src.scripts.data.auto_accumulator.YFCollector", FakeCollector):
        asyncio.run(guard._refetch(["AAA"], [interval]))

    assert list(captured["timeframes"]) == [interval]
    assert captured["enabled"] is True
    assert captured["tickers"] == ["AAA"]
