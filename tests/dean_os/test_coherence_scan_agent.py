from __future__ import annotations

import asyncio

from dean_os.agents.coherence_scan import AGENT_DOMAIN_MAP, OVERLAP_PAIRS, CoherenceScanAgent
from dean_os.schemas import AnalyticalReport, MarketContext


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


def _report(agent_name: str, verdict: str, confidence: float = 0.7) -> AnalyticalReport:
    return AnalyticalReport(
        agent_name=agent_name,
        agent_version="0.1.0",
        verdict=verdict,
        confidence=confidence,
        data_quality_score=0.8,
    )


def test_coherence_scan_handles_pydantic_report_instances():
    # dean_os.orchestrator.DEANOrchestrator.run() sets context._agent_reports
    # to the raw PipelineReport/AnalyticalReport model instances (no .get()),
    # not dicts -- only the context.metadata["agent_reports"] fallback is
    # pre-dumped. Since `getattr(context, "_agent_reports", None) or ...`
    # picks _agent_reports whenever it's truthy, this is the shape the agent
    # actually receives on every real run and must not crash on.
    agent = CoherenceScanAgent("coherence_scan", {})
    context = MarketContext(as_of="2026-07-09T09:00:00+00:00", metadata={})
    # historical_analogies and value_screening both map to the "global"
    # domain in AGENT_DOMAIN_MAP, so dom_a == dom_b triggers the contradiction
    # check regardless of OVERLAP_PAIRS.
    context._agent_reports = [
        _report("historical_analogies", "bullish"),
        _report("value_screening", "bearish"),
    ]
    report = asyncio.run(agent.run(context))
    assert report.agent_name == "coherence_scan"
    assert report.verdict == "caution"
    assert "1 contradictions" in " ".join(report.reasons)
