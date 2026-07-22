from __future__ import annotations

import asyncio

from dean_os.agents.coherence_scan import AGENT_DOMAIN_MAP, OVERLAP_PAIRS, CoherenceScanAgent
from dean_os.schemas import MarketContext


def test_agent_domain_map_includes_key_agents():
    assert "semiconductor_analyst" in AGENT_DOMAIN_MAP
    assert "macro_policy" in AGENT_DOMAIN_MAP
    assert "geopolitical" in AGENT_DOMAIN_MAP


def test_overlap_pairs_listed():
    assert len(OVERLAP_PAIRS) > 0
    for a, b in OVERLAP_PAIRS:
        assert isinstance(a, str)
        assert isinstance(b, str)


def test_coherence_scan_returns_report():
    agent = CoherenceScanAgent("coherence_scan", {})
    report = asyncio.run(
        agent.run(
            MarketContext(
                as_of="2026-07-09T09:00:00+00:00",
                metadata={
                    "agent_reports": [],
                },
            )
        )
    )
    assert report.agent_name == "coherence_scan"
    assert report.verdict in ("pass", "caution", "neutral", "needs_more_data")


def test_coherence_scan_no_agent_reports():
    agent = CoherenceScanAgent("coherence_scan", {})
    report = asyncio.run(
        agent.run(
            MarketContext(
                as_of="2026-07-09T09:00:00+00:00",
                metadata={},
            )
        )
    )
    assert report.verdict == "needs_more_data"
    assert "No agent reports" in " ".join(report.reasons)
