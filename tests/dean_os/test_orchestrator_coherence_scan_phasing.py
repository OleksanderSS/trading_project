from __future__ import annotations

import asyncio

from dean_os.agents.coherence_scan import CoherenceScanAgent
from dean_os.base import AnalyticalAgent
from dean_os.orchestrator import DEANOrchestrator
from dean_os.registry import AgentRegistry
from dean_os.schemas import AnalyticalReport, MarketContext

REGISTRY_PATH = "dean_os/config/agent_registry.yaml"


class _StubVerdictAgent(AnalyticalAgent):
    def __init__(self, name: str, verdict: str, config: dict | None = None):
        super().__init__(name=name, config=config or {})
        self._verdict = verdict

    async def run(self, context: MarketContext) -> AnalyticalReport:
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=self._verdict,
            confidence=0.8,
            data_quality_score=0.8,
        )


def test_coherence_scan_sees_peer_verdicts_produced_in_the_same_run(monkeypatch):
    # Regression test: CoherenceScanAgent is registered as branch=analytical and
    # used to run inside the same asyncio.gather() batch as the peer agents
    # (historical_analogies, value_screening, ...) whose verdicts it's supposed
    # to reconcile. Since the merged report set was only assembled into
    # context after that whole batch returned, coherence_scan always saw an
    # empty/stale report list and could never detect a real contradiction.
    # DEANOrchestrator.PEER_SYNTHESIS_AGENTS now holds coherence_scan back for
    # an explicit second pass, after analytical_reports is known.
    registry = AgentRegistry(REGISTRY_PATH)
    # historical_analogies and value_screening both map to the "global" domain
    # in coherence_scan's AGENT_DOMAIN_MAP, so a bullish/bearish pair between
    # them is a detectable contradiction regardless of OVERLAP_PAIRS.
    bullish = _StubVerdictAgent("historical_analogies", "bullish")
    bearish = _StubVerdictAgent("value_screening", "bearish")
    coherence = CoherenceScanAgent("coherence_scan", {})

    monkeypatch.setattr(
        registry,
        "load_branch",
        lambda branch, context=None: (
            [bullish, bearish, coherence] if branch == "analytical" else []
        ),
    )

    context = MarketContext(
        phase="pre_pipeline",
        as_of="2026-07-05T10:00:00+00:00",
        tickers=["NVDA"],
    )
    asyncio.run(DEANOrchestrator(registry=registry, soft_mode=True).run(context))

    reports_by_name = {r["agent_name"]: r for r in context.metadata["agent_reports"]}
    assert "coherence_scan" in reports_by_name
    coherence_report = reports_by_name["coherence_scan"]
    assert coherence_report["verdict"] == "caution"
    assert "1 contradictions" in " ".join(coherence_report["reasons"])
