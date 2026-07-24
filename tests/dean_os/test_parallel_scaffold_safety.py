from __future__ import annotations

import asyncio
from datetime import UTC

import pytest

from dean_os.agents.freshness_audit import _parse_ts
from dean_os.agents.news_event_analyzer import NewsEventAnalyzerAgent
from dean_os.registry import AgentRegistry
from dean_os.schemas import MarketContext


def test_default_registry_keeps_parallel_scaffold_explicit_only():
    # As of this project's live rollout, domain analysts (semiconductor_analyst,
    # agriculture_analyst, ...) and the post-hoc synthesis agents
    # (historical_analogies, coherence_scan, freshness_audit) are genuinely
    # enabled, live, and load-bearing -- not scaffolding awaiting explicit
    # opt-in. news_event_analyzer stays disabled (a real, separate finding:
    # its NewsEvent(**item) contract doesn't match real news collector
    # records). pipeline_manager stays disabled because it duplicates
    # semiconductor_analyst for the same domain with no artifact paths of
    # its own configured -- see agent_registry.yaml's comment on that entry.
    registry = AgentRegistry("dean_os/config/agent_registry.yaml")
    pipeline_agents = registry.load_branch(
        "pipeline",
        MarketContext(
            phase="pre_trade",
            as_of="2026-07-09T09:00:00+00:00",
        ),
    )
    analytical_agents = registry.load_branch(
        "analytical",
        MarketContext(
            phase="pre_trade",
            as_of="2026-07-09T09:00:00+00:00",
        ),
    )
    active_names = {agent.name for agent in pipeline_agents + analytical_agents}

    assert "pipeline_manager" not in active_names
    assert "news_event_analyzer" not in active_names
    assert "semiconductor_analyst" in active_names
    assert "agriculture_analyst" in active_names
    assert "historical_analogies" in active_names
    assert "coherence_scan" in active_names
    assert "freshness_audit" in active_names


def test_composite_and_standalone_domain_agents_share_an_exclusive_group():
    # pipeline_manager (composite) and semiconductor_analyst (standalone)
    # both target domain_id=semiconductor_ai_infrastructure and share
    # execution_group=semiconductor_domain_analysis. Only one may ever be
    # enabled at a time -- AgentRegistry._validate_exclusive_groups raises
    # at load time otherwise. This is what actually prevents the two from
    # silently running in parallel and double-counting the same domain.
    with pytest.raises(ValueError, match="exclusive execution group"):
        AgentRegistry.__new__(AgentRegistry)._validate_exclusive_groups(
            {
                "semiconductor_analyst": {
                    "enabled": True,
                    "execution_group": "semiconductor_domain_analysis",
                    "run_phases": ["pre_trade"],
                },
                "pipeline_manager": {
                    "enabled": True,
                    "execution_group": "semiconductor_domain_analysis",
                    "run_phases": ["pre_trade"],
                },
            }
        )


def test_news_event_analyzer_does_not_register_outcomes_by_default():
    agent = NewsEventAnalyzerAgent("news_event_analyzer", {})

    def fail_if_called(_events):
        raise AssertionError("outcome registration must be explicitly enabled")

    agent._register_significant_events = fail_if_called

    report = asyncio.run(
        agent.run(
            MarketContext(
                as_of="2026-07-09T09:00:00+00:00",
                news=[
                    {
                        "headline": "Unexpected sanctions shock chip exports",
                        "source": "fixture",
                        "published_at": "2026-07-09T08:30:00+00:00",
                    }
                ],
            )
        )
    )

    assert report.agent_name == "news_event_analyzer"
    assert report.verdict in {"bearish", "neutral", "bullish"}


def test_freshness_timestamp_parser_handles_iso_offsets():
    parsed = _parse_ts("2026-07-09T08:30:00+00:00")

    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == UTC.utcoffset(parsed)
