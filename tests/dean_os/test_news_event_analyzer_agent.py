from __future__ import annotations

import asyncio

import pandas as pd

from dean_os.agents.news_event_analyzer import NewsEvent, NewsEventAnalyzerAgent
from dean_os.schemas import MarketContext


def test_news_event_accepts_classification_overrides():
    # The VIX-injection path in NewsEventAnalyzerAgent.run() relies on being
    # able to pass a known-good classification instead of re-deriving it from
    # a synthetic headline via keyword matching.
    event = NewsEvent(
        headline="VIX Spike to 32.1",
        source="MacroCollector",
        published_at="2026-07-09T09:00:00+00:00",
        event_type="credit_financial",
        shock="negative",
        shock_confidence=0.95,
        impact=0.64,
        predictability=0.4,
        time_to_impact="1w",
        affected_sectors=["technology", "consumer_discretionary", "financials"],
    )
    assert event.event_type == "credit_financial"
    assert event.shock == "negative"
    assert event.shock_confidence == 0.95
    assert event.impact == 0.64
    assert event.affected_sectors == ["technology", "consumer_discretionary", "financials"]


def test_news_event_falls_back_to_classification_without_overrides():
    event = NewsEvent(headline="Fed hikes interest rates by 50bps")
    assert event.event_type is not None
    assert event.shock in ("positive", "negative", "neutral")


def test_news_event_analyzer_handles_real_news_record_shape():
    # Real news collectors (src/features/nlp) use a "title" column, not
    # "headline" -- NewsEvent(**item) used to blow up with TypeError on any
    # such record since the constructor only accepted headline/source/
    # published_at as exact keyword names.
    agent = NewsEventAnalyzerAgent("news_event_analyzer", {})
    context = MarketContext(
        as_of="2026-07-09T09:00:00+00:00",
        news=[
            {
                "title": "Fed signals rate cut amid slowing inflation",
                "source": "Reuters",
                "published_at": "2026-07-08T09:00:00+00:00",
                "sentiment": 0.3,
                "hash": "abc123",
            }
        ],
    )
    report = asyncio.run(agent.run(context))
    assert report.agent_name == "news_event_analyzer"


def test_news_event_analyzer_vix_injection_does_not_crash():
    agent = NewsEventAnalyzerAgent("news_event_analyzer", {})
    vix_df = pd.DataFrame({"vix_current": [32.1]})
    context = MarketContext(
        as_of="2026-07-09T09:00:00+00:00",
        news=[{"title": "Markets steady", "source": "AP", "published_at": "2026-07-08T09:00:00+00:00"}],
        dataframes={"vix_data": vix_df},
    )
    report = asyncio.run(agent.run(context))
    assert report.agent_name == "news_event_analyzer"
