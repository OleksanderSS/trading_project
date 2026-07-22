"""Integration checks for safe agent activation and orchestration."""
from __future__ import annotations

import asyncio

import pytest

from dean_os.branches import PipelineBranch
from dean_os.orchestrator import DEANOrchestrator
from dean_os.registry import AgentRegistry
from dean_os.schemas import MarketContext, PipelineReport


REGISTRY_PATH = "dean_os/config/agent_registry.yaml"
EXPENSIVE_OR_MUTATING_DEFAULT_OFF = {
    "model_performance",
    "tuning",
    "chief_review",
    "paper_portfolio",
    "diary_bridge",
    "source_routing",
    "operations_proposal",
}


def test_default_registry_does_not_auto_enable_expensive_agents():
    registry = AgentRegistry(REGISTRY_PATH)
    agents = registry.load_branch(
        "pipeline",
        MarketContext(
            phase="pre_trade",
            as_of="2026-07-05T10:00:00+00:00",
        ),
    )
    names = {agent.name for agent in agents}

    assert names.isdisjoint(EXPENSIVE_OR_MUTATING_DEFAULT_OFF)


def test_default_orchestrator_fails_safe_without_pipeline_runner():
    decision = asyncio.run(
        DEANOrchestrator(
            registry=AgentRegistry(REGISTRY_PATH)
        ).run(
            MarketContext(
                phase="pre_pipeline",
                as_of="2026-07-05T10:00:00+00:00",
                tickers=["NVDA", "AMD"],
            )
        )
    )

    assert decision.decision in {"blocked", "no_trade", "watchlist"}
    assert decision.decision not in {"candidate_long", "candidate_short"}


def test_one_explicit_composite_manager_can_be_selected(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
agents:
  pipeline_manager:
    class_path: dean_os.agents.pipeline_manager:PipelineManagerAgent
    branch: pipeline
    enabled: true
    veto_level: none
    error_behavior: skip
    domain_id: semiconductor_ai_infrastructure
    execution_group: semiconductor_domain_analysis
    run_phases: [pre_trade]
""".strip(),
        encoding="utf-8",
    )
    context = MarketContext(
        phase="pre_trade",
        as_of="2026-07-05T10:00:00+00:00",
    )
    agents = AgentRegistry(
        registry_path,
        project_root=tmp_path,
    ).load_branch("pipeline", context)

    assert [agent.name for agent in agents] == ["pipeline_manager"]
    reports = asyncio.run(PipelineBranch(agents).run(context))
    assert len(reports) == 1
    assert isinstance(reports[0], PipelineReport)
    assert reports[0].verdict == "needs_more_data"


def test_composite_and_standalone_same_domain_fail_closed(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
agents:
  standalone:
    class_path: dean_os.agents.domain_analyst:DomainAnalystAgent
    branch: pipeline
    enabled: true
    execution_group: semiconductor_domain_analysis
    run_phases: [pre_trade]
  composite:
    class_path: dean_os.agents.pipeline_manager:PipelineManagerAgent
    branch: pipeline
    enabled: true
    execution_group: semiconductor_domain_analysis
    run_phases: [pre_trade]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exclusive execution group"):
        AgentRegistry(registry_path, project_root=tmp_path)
