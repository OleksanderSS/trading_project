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


def _configured_hourly_period_days():
    import yaml

    with open("src/config/collectors.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    timeframes = config["collectors"]["yahoo_finance"]["timeframes"] \
        if "collectors" in config else config["yahoo_finance"]["timeframes"]
    period = str(timeframes["1h"]["period"])
    assert period.endswith("d"), period
    return int(period.rstrip("d"))


def test_the_config_asks_for_more_than_the_old_clamp_allowed():
    """The clamp used to make anything above 58 days meaningless. That it
    now asks for materially more is the point of raising the ceiling."""
    assert _configured_hourly_period_days() > 100


def test_the_config_stays_within_what_news_covers():
    """Price history must not outrun news history.

    144 of the feature columns come from news and sentiment, and nothing
    distinguishes "no news happened" from "no news collected" --
    sentiment_available_* is the constant 1.0 on all three timeframes.
    News starts 2026-03-14, so 730 days of bars would come with ~17 months
    of zero-filled news features that read as a quiet market.

    This is a ceiling on the CONFIG, not on the collector: the clamp
    correctly allows 728, and this number should rise as news accumulates.
    """
    assert _configured_hourly_period_days() <= 365, (
        "hourly history now reaches far past the news that is supposed to "
        "explain it -- check news coverage before raising this"
    )


def test_the_request_is_within_what_the_provider_serves():
    """A config asking for more than the clamp allows is silently truncated,
    which reads in the log as a limit nobody chose."""
    assert (_configured_hourly_period_days()
            <= YFCollector._intraday_limit_days("1h"))
