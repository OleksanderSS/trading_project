from __future__ import annotations

from pathlib import Path

from dean_os.consensus import ConsensusEngine
from dean_os.decision_logger import DecisionLogger
from dean_os.orchestrator import DEANOrchestrator, PipelineRunner
from dean_os.pipeline_adapter import HybridMode, HybridPipelineAdapter
from dean_os.registry import AgentRegistry


def create_dean_orchestrator(
    project_root: str | Path = ".",
    pipeline_runner: PipelineRunner | None = None,
    enable_logging: bool = False,
) -> DEANOrchestrator:
    root = Path(project_root).resolve()
    registry = AgentRegistry(root / "dean_os" / "config" / "agent_registry.yaml", project_root=root)
    logger = DecisionLogger(root / "logs" / "dean_os" / "decisions.jsonl") if enable_logging else None
    return DEANOrchestrator(
        registry=registry,
        pipeline_runner=pipeline_runner,
        consensus=ConsensusEngine(),
        decision_logger=logger,
    )


def create_hybrid_dean_orchestrator(
    project_root: str | Path = ".",
    mode: HybridMode = "local",
    batch_name: str = "main_database",
    enable_logging: bool = False,
    stages_to_run: list[int] | None = None,
    prepare_kwargs: dict | None = None,
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
    )
