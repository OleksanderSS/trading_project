from __future__ import annotations

import asyncio

from dean_os.base import BaseAgent
from dean_os.consensus import ConsensusEngine
from dean_os.orchestrator import DEANOrchestrator
from dean_os.registry import AgentRegistry
from dean_os.schemas import EvidenceItem, MarketContext, PipelineReport


class _RiskAgent(BaseAgent):
    async def run(self, context: MarketContext) -> PipelineReport:
        return PipelineReport(
            agent_name="risk",
            agent_version="test",
            verdict="clear",
            confidence=1.0,
            data_quality_score=1.0,
            signal_strength=1.0,
            reasons=[f"reviewed_{context.phase}"],
            evidence=[
                EvidenceItem(
                    source_type="metric",
                    source="test",
                    key="phase",
                    value=context.phase,
                )
            ],
        )


class _Registry:
    def __init__(self, synthetic_reports=None):
        self.config_path = "test_registry"
        self.phases = []
        self.synthetic_reports = synthetic_reports or {}

    def load_branch(self, branch, context):
        self.phases.append((branch, context.phase))
        if branch == "pipeline":
            return [_RiskAgent(name="risk", config={"veto_level": "hard"})]
        return []

    def get_synthetic_reports(self):
        return self.synthetic_reports


def test_orchestrator_rechecks_pipeline_safety_after_pipeline_and_stays_watchlist():
    registry = _Registry()
    pipeline_phases = []

    def pipeline_runner(context):
        pipeline_phases.append(context.phase)
        return {"model_score": 1.0, "tickers": context.tickers, "timeframe": "1d"}

    orchestrator = DEANOrchestrator(
        registry=registry,
        pipeline_runner=pipeline_runner,
        consensus=ConsensusEngine(),
    )
    context = MarketContext(tickers=["NVDA"], timeframe="1d")

    decision = asyncio.run(orchestrator.run(context))

    assert pipeline_phases == ["pre_pipeline"]
    assert registry.phases == [
        ("pipeline", "pre_pipeline"),
        ("analytical", "post_pipeline"),
        ("pipeline", "pre_trade"),
    ]
    assert context.phase == "pre_trade"
    assert decision.decision == "watchlist"
    assert decision.trade_allowed is False
    assert any(reason == "reviewed_pre_trade" for reason in decision.reasons)
    assert all(reason != "reviewed_pre_pipeline" for reason in decision.reasons)


def test_synthetic_hard_block_prevents_pipeline_runner():
    synthetic = PipelineReport(
        agent_name="pipeline_audit",
        agent_version="test",
        verdict="blocked",
        confidence=1.0,
        data_quality_score=0.0,
        signal_strength=-1.0,
        reasons=["missing audit"],
        evidence=[
            EvidenceItem(
                source_type="audit_finding",
                source="test",
                key="missing",
                value=True,
            )
        ],
    )
    registry = _Registry({"pipeline_audit": synthetic})
    pipeline_calls = []
    orchestrator = DEANOrchestrator(
        registry=registry,
        pipeline_runner=lambda context: pipeline_calls.append(context.phase),
    )

    decision = asyncio.run(orchestrator.run(MarketContext(tickers=["NVDA"])))

    assert pipeline_calls == []
    assert decision.decision == "blocked"
    assert decision.blocking_agents == ["pipeline_audit"]


def test_registry_missing_hard_prerequisite_becomes_valid_block_report(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
agents:
  pipeline_audit:
    class_path: dean_os.agents.pipeline_audit:PipelineAuditAgent
    branch: pipeline
    veto_level: hard
    enabled: true
    error_behavior: block
    findings_path: missing_findings.json
""".strip(),
        encoding="utf-8",
    )
    registry = AgentRegistry(registry_path, project_root=tmp_path)

    agents = registry.load_branch("pipeline", MarketContext())
    synthetic = registry.get_synthetic_reports()["pipeline_audit"]

    assert agents == []
    assert synthetic.verdict == "blocked"
    assert synthetic.evidence[0].source == "agent_registry"

    (tmp_path / "missing_findings.json").write_text(
        '{"findings": []}',
        encoding="utf-8",
    )
    agents = registry.load_branch("pipeline", MarketContext())

    assert len(agents) == 1
    assert registry.get_synthetic_reports() == {}
