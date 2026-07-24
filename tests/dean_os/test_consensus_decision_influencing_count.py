from __future__ import annotations

from dean_os.anxiety_kill_switch import AnxietyConfig, AnxietyKillSwitch
from dean_os.consensus import ConsensusEngine
from dean_os.schemas import MarketContext, PipelineReport


def _pipeline_report(agent_name: str, *, decision_influence: bool | None = None, verdict: str = "clear") -> PipelineReport:
    metrics_snapshot = {} if decision_influence is None else {"decision_influence": decision_influence}
    return PipelineReport(
        agent_name=agent_name,
        agent_version="0.1.0",
        verdict=verdict,
        confidence=0.7,
        data_quality_score=0.7,
        metrics_snapshot=metrics_snapshot,
    )


def test_decision_influencing_agent_count_excludes_review_only_reports():
    # 8 review-only domain analysts (decision_influence=False) plus 2 real
    # decision-relevant pipeline agents -- agent_report_hashes counts all 10,
    # but decision_influencing_agent_count must only count the 2 that can
    # actually move the score.
    review_only = [
        _pipeline_report(f"domain_analyst_{i}", decision_influence=False)
        for i in range(8)
    ]
    decision_relevant = [
        _pipeline_report("risk"),
        _pipeline_report("regime"),
    ]
    decision = ConsensusEngine().combine(
        review_only + decision_relevant,
        {},
        [],
    )
    assert len(decision.agent_report_hashes) == 10
    assert decision.decision_influencing_agent_count == 2


def test_kill_switch_fires_on_too_few_decision_influencing_agents_despite_many_review_only_reports():
    # Regression test: before this fix, the kill-switch's "too few agents"
    # trigger read len(decision.agent_report_hashes), which counted every
    # review-only report -- meaning it could never fire as long as enough
    # domain analysts ran, even if zero decision-relevant guardians did.
    review_only = [
        _pipeline_report(f"domain_analyst_{i}", decision_influence=False)
        for i in range(8)
    ]
    decision = ConsensusEngine().combine(review_only, {}, [])
    assert len(decision.agent_report_hashes) == 8
    assert decision.decision_influencing_agent_count == 0

    result = AnxietyKillSwitch(config=AnxietyConfig(min_active_agents=3)).evaluate(
        MarketContext(as_of="2026-07-24T00:00:00+00:00"),
        decision,
    )
    assert result.triggered is True
    assert any("active agents" in reason for reason in result.reasons)
