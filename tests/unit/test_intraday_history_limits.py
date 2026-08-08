"""One 58-day clamp stood in for every intraday interval Yahoo serves.

_adjust_intraday_dates cut any interval ending in 'm' or 'h' back to 58
days. For 15m that is right -- Yahoo's limit is 60. For hourly it is wrong
by more than an order of magnitude: Yahoo serves 1h for 730 days, so 92%
of the obtainable history was discarded on every request.

Not an abstract loss. AAPL's hourly series reached the training batch with
363 bars, and target_hourly_breakout_1h carried 11 positive events -- all
of them inside one volatile stretch in July. The split is chronological,
so that stretch fell entirely in validation and the training portion had a
single class; six models were fitted to it and reported 80% accuracy for
predicting a constant.

A rare event needs a long enough window to stop being clustered. Two years
of hourly bars is roughly five times the observations, spread across many
regimes, and that is the actual fix for that context -- the guard added
alongside only stops the bad model being built.

The limit is a property of the provider and the interval, so it lives in a
table keyed by canonical timeframe rather than as a number in an if.
"""
from __future__ import annotations

import pytest

from src.data.collectors.yf_collector import YFCollector


# --------------------------------------------------------- the limits


def test_hourly_gets_two_years_not_two_months():
    assert YFCollector._intraday_limit_days("1h") > 700


def test_the_two_spellings_of_hourly_agree():
    """1h and 60m are the same interval under two names -- a distinction
    that has already cost this project a defect."""
    assert (YFCollector._intraday_limit_days("1h")
            == YFCollector._intraday_limit_days("60m"))


@pytest.mark.parametrize("interval", ["5m", "15m", "30m", "90m"])
def test_sub_hourly_intervals_keep_the_sixty_day_limit(interval):
    """These were correct before and must not move: Yahoo really does cap
    them at 60 days, and asking for more returns nothing at all."""
    days = YFCollector._intraday_limit_days(interval)

    assert 55 <= days <= 60


def test_one_minute_is_tighter_still():
    assert YFCollector._intraday_limit_days("1m") <= 7


def test_an_unknown_interval_gets_the_conservative_limit():
    """A new interval must not silently inherit 730 days that the provider
    would refuse."""
    days = YFCollector._intraday_limit_days("3m")

    assert 55 <= days <= 60


def test_every_limit_leaves_a_safety_margin():
    """Requesting exactly the boundary date fails intermittently on
    timezone edges."""
    for interval, limit in YFCollector._INTRADAY_HISTORY_LIMIT_DAYS.items():
        assert YFCollector._intraday_limit_days(interval) < limit, interval


# ---------------------------------------------------- the clamp itself


def _clamped(interval, requested_days):
    from datetime import datetime, timedelta
    import logging

    collector = object.__new__(YFCollector)
    collector.logger = logging.getLogger("yf-limit-test")

    now = datetime(2026, 8, 8, 12, 0, 0)
    start, end = collector._adjust_intraday_dates(
        interval, now - timedelta(days=requested_days), now, now
    )
    return (end - start).days


def test_a_two_year_hourly_request_survives_the_clamp():
    """The request the config now makes. Before this, it came back as 58."""
    assert _clamped("1h", 730) > 700


def test_a_two_year_fifteen_minute_request_is_still_clamped():
    assert _clamped("15m", 730) <= 60


def test_a_short_request_is_left_alone():
    """The clamp is a ceiling, not a target -- it must never EXTEND a
    request into history the caller did not ask for."""
    assert _clamped("1h", 30) == 30


# --------------------------------------------------------- the config


def test_the_config_asks_for_the_history_the_clamp_now_allows():
    """A raised ceiling changes nothing while the request stays at 60d.
    Both halves have to move, which is exactly the kind of pair that gets
    half-done."""
    import yaml

    with open("src/config/collectors.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    timeframes = config["collectors"]["yahoo_finance"]["timeframes"] \
        if "collectors" in config else config["yahoo_finance"]["timeframes"]
    period = str(timeframes["1h"]["period"])

    assert period.endswith("d")
    assert int(period.rstrip("d")) > 700, (
        f"1h period is {period}; the collector can now fetch ~728 days"
    )
