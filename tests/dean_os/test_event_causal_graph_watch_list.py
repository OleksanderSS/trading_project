from __future__ import annotations

from dean_os.agents.news_event_analyzer import NewsEvent
from dean_os.event_causal_graph import EventCausalGraphBuilder


def test_build_populates_ticker_watch_list_from_context_tickers():
    # build() used to reference an undefined `watch_list` name (copy-paste
    # slip from the sibling `all_sectors` accumulator) -- any event with a
    # non-neutral shock or |impact| >= 0.2 raised NameError. No existing test
    # exercised the real builder end-to-end (they construct CausalGraph/
    # CausalNode directly), so this was never caught.
    builder = EventCausalGraphBuilder(context_tickers=["NVDA", "AMD"])
    event = NewsEvent(
        headline="Fed hikes interest rates sharply amid inflation concerns",
        source="Reuters",
    )
    graph = builder.build(event)
    assert isinstance(graph.ticker_watch_list, list)
    assert all(isinstance(t, str) for t in graph.ticker_watch_list)
    assert len(graph.ticker_watch_list) > 0
    assert graph.summary


def test_build_multi_does_not_crash_on_significant_events():
    builder = EventCausalGraphBuilder(context_tickers=["NVDA"])
    event = NewsEvent(headline="Major geopolitical conflict escalates in key chip-manufacturing region")
    graphs = builder.build_multi([event])
    assert isinstance(graphs, list)
