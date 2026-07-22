from __future__ import annotations

from pathlib import Path

from dean_os.draft.dean_os_agent_system_v7.dean_os.consensus import ConsensusEngine
from dean_os.draft.dean_os_agent_system_v7.dean_os.decision_logger import DecisionLogger
from dean_os.draft.dean_os_agent_system_v7.src.processing.filters.orchestrator import DEANOrchestrator, PipelineContextBridge, PipelineRunner
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_adapter import HybridMode, HybridPipelineAdapter
from dean_os.draft.dean_os_agent_system_v7.src.models.prototypes.registry import AgentRegistry


def create_dean_orchestrator(
    project_root: str | Path = ".",
    pipeline_runner: PipelineRunner | None = None,
    enable_logging: bool = False,
    soft_mode: bool = False,
    registry_path: str | Path | None = None,
    registry_overrides: dict[str, dict] | None = None,
    pipeline_context_bridge: PipelineContextBridge | None = None,
) -> DEANOrchestrator:
    """Build a DEAN orchestrator over the YAML agent registry.

    Args:
        project_root: Repo root (registry + logs resolved relative to it).
        pipeline_runner: Callable that runs the src/ pipeline. If None, the
            orchestrator only runs the agent branches (review/analysis only).
        enable_logging: Persist every decision to logs/dean_os/decisions.jsonl.
        soft_mode: If True, guardian agents (data_quality/risk/pipeline_audit)
            cannot block the decision - useful for smoke tests and first runs
            where pipeline data is absent. Production MUST keep soft_mode=False
            so the hard-veto invariant is preserved.
    """
    root = Path(project_root).resolve()
    resolved_registry = Path(registry_path) if registry_path else root / "dean_os" / "config" / "agent_registry.yaml"
    if not resolved_registry.is_absolute():
        resolved_registry = root / resolved_registry
    registry = AgentRegistry(
        resolved_registry,
        project_root=root,
        overrides=registry_overrides,
    )
    logger = DecisionLogger(root / "logs" / "dean_os" / "decisions.jsonl") if enable_logging else None
    hard_veto_agents = (
        set()
        if soft_mode
        else registry.hard_veto_agent_names()
    )
    return DEANOrchestrator(
        registry=registry,
        pipeline_runner=pipeline_runner,
        consensus=ConsensusEngine(
            hard_veto_agents=hard_veto_agents,
            soft_mode=soft_mode,
        ),
        decision_logger=logger,
        soft_mode=soft_mode,
        pipeline_context_bridge=pipeline_context_bridge,
    )


def create_default_orchestrator(
    project_root: str | Path = ".",
    enable_logging: bool = False,
    soft_mode: bool = False,
) -> DEANOrchestrator:
    """Recommended entry point: DEAN orchestrator without a live pipeline runner.

    Returns an orchestrator whose pipeline branch runs guardian agents against
    whatever context is supplied, but does NOT execute the src/ trading
    pipeline. This is the safe review-only configuration.

    Args:
        project_root: Repo root.
        enable_logging: Persist decisions.
        soft_mode: False (production) keeps hard veto on guardian agents.
            True (smoke/first-run) disables blocking so the orchestrator can
            return ``no_trade`` instead of ``blocked`` with empty context.
    """
    return create_dean_orchestrator(
        project_root=project_root,
        pipeline_runner=None,
        enable_logging=enable_logging,
        soft_mode=soft_mode,
    )


def create_hybrid_dean_orchestrator(
    project_root: str | Path = ".",
    mode: HybridMode = "local",
    batch_name: str = "main_database",
    enable_logging: bool = False,
    soft_mode: bool = False,
    stages_to_run: list[int] | None = None,
    prepare_kwargs: dict | None = None,
    registry_path: str | Path | None = None,
    registry_overrides: dict[str, dict] | None = None,
) -> DEANOrchestrator:
    adapter = HybridPipelineAdapter(
        mode=mode,
        batch_name=batch_name,
        project_root=project_root,
        stages_to_run=stages_to_run,
        prepare_kwargs=prepare_kwargs,
    )
    return create_dean_orchestrator(
        project_root=project_root,
        pipeline_runner=adapter,
        enable_logging=enable_logging,
        soft_mode=soft_mode,
        registry_path=registry_path,
        registry_overrides=registry_overrides,
    )
