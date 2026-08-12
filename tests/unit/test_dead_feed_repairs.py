"""Two of the four silent collectors were ours to fix; two were not.

The 2026-08-11 run had 16 collectors enabled and 10 delivering rows. The
other four were reported as successfully processed, because a `None` return
incremented the same counter as a saved batch. Probing each endpoint sorted
them into two kinds:

  fear_greed           production.datapoint.cloud no longer completes a TLS
                       handshake. The host is gone. CNN publishes the index
                       elsewhere, and the project's own user agent gets HTTP
                       200 there -- the endpoint was the whole problem.
  wikimedia_attention  HTTP 403 with the body "Please set a user-agent and
                       respect our robot policy". The collector was sending
                       "DEAN_OS_Agent research@example.com" -- a placeholder
                       contact, which is what the policy rejects. Complying
                       with a stated policy fixed it.

  aaii_sentiment       HTTP 403 to this project's user agent. A deliberate
                       block, on a survey behind a paid membership. Not
                       worked around.
  put_call_ratio       CBOE returns 403 through bot protection. Their side.

Verified live on 2026-08-12: fear_greed 251 rows, wikimedia 90 rows.

These tests are offline. They pin the parsing decisions, not the network.
"""
from datetime import datetime

import pytest

from src.data.collectors.aaii_sentiment_collector import AIISentimentCollector
from src.data.collectors.fear_greed_collector import FearGreedCollector


# ------------------------------------------------------- fear & greed payload


def test_the_series_is_read_from_the_nested_historical_key():
    """CNN's shape: fear_and_greed_historical.data, not top-level data."""
    payload = {
        "fear_and_greed": {"score": 62.3, "rating": "greed"},
        "fear_and_greed_historical": {
            "data": [
                {"x": 1754956800000.0, "y": 62.25, "rating": "greed"},
                {"x": 1755043200000.0, "y": 61.10, "rating": "greed"},
            ]
        },
    }

    series = FearGreedCollector._series_from_payload(payload)

    assert len(series) == 2
    assert series[0]["y"] == pytest.approx(62.25)


def test_the_retired_top_level_shape_is_still_accepted():
    """The endpoint moved once without warning; it can move again."""
    payload = {"data": [{"x": 1754956800000.0, "y": 55.0}]}

    assert len(FearGreedCollector._series_from_payload(payload)) == 1


def test_a_payload_with_no_series_yields_nothing_rather_than_raising():
    assert FearGreedCollector._series_from_payload({}) == []
    assert FearGreedCollector._series_from_payload(
        {"fear_and_greed_historical": {}}
    ) == []
    assert FearGreedCollector._series_from_payload("not json") == []


def test_the_collector_no_longer_requests_the_dead_host():
    """The 404 branch in this collector could never fire.

    A host that stops completing a TLS handshake fails before there is a
    status code to inspect, so "URL may have changed (404)" was unreachable
    while the URL had, in fact, changed.
    """
    collector = FearGreedCollector({'enabled': True}, None, None, None)

    assert "datapoint.cloud" not in collector.base_url
    assert collector.base_url.startswith("https://production.dataviz.cnn.io")


def test_the_endpoint_stays_overridable_from_config():
    """It moved once without warning. Next time should not need a release."""
    collector = FearGreedCollector(
        {'enabled': True, 'base_url': 'https://example.invalid/feed'},
        None, None, None,
    )

    assert collector.base_url == 'https://example.invalid/feed'


# --------------------------------------------------------------- AAII pairing


def _parser():
    collector = object.__new__(AIISentimentCollector)
    import logging

    collector.logger = logging.getLogger("AAIITest")
    return collector


def test_three_shares_that_do_not_sum_to_a_hundred_are_refused():
    """Three regexes over one page pair by position, not by row.

    On the live page the first matches were bullish 49.5, bearish 52.0,
    neutral 31.4 -- three different weeks, which would have been stored as
    one survey reading. The survey partitions its respondents, so its three
    shares sum to 100. Anything else is a mispairing, and a wrong sentiment
    number is worse than a missing one.
    """
    html = (
        "Jan 15, 2026 "
        "Bullish 49.5% "
        "Bearish 52.0% "
        "Neutral 31.4% "
    )

    assert _parser()._extract_raw_data(html) == []


def test_a_consistent_reading_is_kept():
    html = (
        "Jan 15, 2026 "
        "Bullish 40.0% "
        "Bearish 35.0% "
        "Neutral 25.0% "
    )

    records = _parser()._extract_raw_data(html)

    assert len(records) == 1
    assert records[0]["bullish"] == pytest.approx(40.0)
    assert records[0]["spread"] == pytest.approx(5.0)
    assert records[0]["timestamp"] == datetime(2026, 1, 15)
