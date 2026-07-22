from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from dean_os.draft.dean_os_agent_system_v7.dean_os.context_grids import ContextIndicatorGridBuilder
from dean_os.draft.dean_os_agent_system_v7.src.scripts.optimization.factory import create_dean_orchestrator
from dean_os.draft.dean_os_agent_system_v7.src.processing.filters.orchestrator import DEANOrchestrator
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_adapter import HybridMode, HybridPipelineAdapter
from dean_os.schemas import ConsensusDecision, MarketContext
from dean_os.world_model.world_model_event_learning import WorldModelEventLearningPacket
from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state_store import (
    HistoricalWorldStateRetriever,
    SQLiteWorldStateStore,
    WorldStateSnapshotBuilder,
    WorldStateStoreProtocol,
)


class MinimalSystemRunResult(BaseModel):
    run_id: str = Field(default_factory=lambda: f"minimal_system_{uuid4().hex}")
    status: str
    domain_id: str
    review_only: bool = True
    decision: ConsensusDecision
    pipeline_result: dict[str, Any] = Field(default_factory=dict)
    pipeline_execution_policy: dict[str, Any] = Field(default_factory=dict)
    agent_reports: list[dict[str, Any]] = Field(default_factory=list)
    pipeline_metric_snapshot: dict[str, Any] = Field(default_factory=dict)
    context_grid: dict[str, Any] = Field(default_factory=dict)
    indicator_state_grid: dict[str, Any] = Field(default_factory=dict)
    world_model_event_learning: dict[str, Any] = Field(default_factory=dict)
    world_state_snapshot: dict[str, Any] = Field(default_factory=dict)
    world_state_persistence: dict[str, Any] = Field(default_factory=dict)
    historical_world_state_analogs: list[dict[str, Any]] = Field(default_factory=list)
    original_decision_before_review_clamp: str | None = None
    safety: dict[str, bool] = Field(
        default_factory=lambda: {
            "can_trade": False,
            "can_write_production_config": False,
            "can_promote_model": False,
            "can_write_learning_memory": False,
            "human_review_required": True,
        }
    )


class DEANMinimalSystem:
    """Composition root for the first minimally complete DEAN-OS runtime.

    Execution order:
      pipeline-control/safety preflight -> bounded pipeline runner -> domain
      analytical branch -> post-pipeline safety/tuning proposals -> consensus ->
      world-model event/scenario packet.
    """

    def __init__(
        self,
        orchestrator: DEANOrchestrator,
        *,
        domain_id: str = "semiconductor_ai_infrastructure",
        world_model_builder: WorldModelEventLearningPacket | None = None,
        context_grid_builder: ContextIndicatorGridBuilder | None = None,
        save_world_model_artifacts: bool = False,
        world_state_store: WorldStateStoreProtocol | None = None,
        world_state_builder: WorldStateSnapshotBuilder | None = None,
        historical_analog_limit: int = 5,
    ):
        self.orchestrator = orchestrator
        self.domain_id = domain_id
        self.world_model_builder = world_model_builder or WorldModelEventLearningPacket()
        self.context_grid_builder = context_grid_builder or ContextIndicatorGridBuilder()
        self.save_world_model_artifacts = save_world_model_artifacts
        self.world_state_store = world_state_store
        self.world_state_builder = world_state_builder or WorldStateSnapshotBuilder()
        self.historical_analog_limit = max(0, int(historical_analog_limit))

    async def run(self, context: MarketContext) -> MinimalSystemRunResult:
        context.as_of = context.as_of or datetime.now(UTC).isoformat()
        context.metadata.setdefault("domain_id", self.domain_id)
        context.metadata.setdefault("system_mode", "minimal_review_only")

        decision = await self.orchestrator.run(context)
        original_decision: str | None = None
        if decision.trade_allowed:
            original_decision = decision.decision
            decision.decision = "watchlist"
            decision.requires_human_approval = True
            decision.reasons.append(
                "Minimal-system review clamp converted a trade-capable decision to watchlist."
            )
            decision.risks.append(
                "Execution requires a separately approved paper/live lifecycle."
            )

        metric_snapshot = context.metadata.get("pipeline_metric_snapshot", {})
        try:
            grid_packet = self.context_grid_builder.build(
                context,
                domain_id=self.domain_id,
                agent_reports=context.metadata.get("agent_reports", []),
                pipeline_metric_snapshot=(
                    metric_snapshot if isinstance(metric_snapshot, dict) and metric_snapshot else None
                ),
            )
            context_grid = grid_packet.context_grid.model_dump(mode="json")
            indicator_state_grid = grid_packet.indicator_state_grid.model_dump(mode="json")
        except Exception as exc:  # preserve the main run if a new grid contract fails
            context_grid = {
                "schema_version": "dean_context_grid_v1",
                "domain_id": self.domain_id,
                "as_of": context.as_of,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            indicator_state_grid = {
                "schema_version": "dean_indicator_state_grid_v1",
                "domain_id": self.domain_id,
                "as_of": context.as_of,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        knowledge_cutoff = str(
            context.metadata.get("knowledge_cutoff") or context.as_of
        )
        parent_snapshot_id: str | None = None
        historical_analogs: list[dict[str, Any]] = []
        analog_query_error: str | None = None
        if (
            self.world_state_store is not None
            and context_grid.get("status") != "failed"
            and indicator_state_grid.get("status") != "failed"
        ):
            try:
                prior_snapshots = self.world_state_store.list_snapshots(
                    domain_id=self.domain_id,
                    before_as_of=context.as_of,
                    knowledge_cutoff=knowledge_cutoff,
                    limit=1,
                )
                parent_snapshot_id = (
                    prior_snapshots[0].snapshot_id if prior_snapshots else None
                )
                provisional_snapshot = self.world_state_builder.build(
                    domain_id=self.domain_id,
                    as_of=context.as_of,
                    knowledge_cutoff=knowledge_cutoff,
                    context_grid=context_grid,
                    indicator_state_grid=indicator_state_grid,
                    scenario_outcome_graph=None,
                    world_model_summary={
                        "packet_status": "pre_scenario_analog_query",
                    },
                    parent_snapshot_id=parent_snapshot_id,
                )
                if self.historical_analog_limit:
                    historical_analogs = [
                        item.model_dump(mode="json")
                        for item in HistoricalWorldStateRetriever(
                            self.world_state_store
                        ).find_analogs(
                            provisional_snapshot,
                            limit=self.historical_analog_limit,
                        )
                    ]
                context.metadata["historical_world_state_analogs"] = historical_analogs
            except Exception as exc:
                analog_query_error = f"{type(exc).__name__}: {exc}"
                context.metadata["historical_world_state_analog_error"] = analog_query_error

        try:
            world_model = self.world_model_builder.build(
                context,
                domain_id=self.domain_id,
                as_of=context.as_of,
                save=self.save_world_model_artifacts,
            )
            world_model_status = str(
                world_model.get("summary", {}).get("packet_status") or "completed"
            )
        except Exception as exc:  # fail-soft: preserve branch results for review
            world_model = {
                "mode": "world_model_event_learning_packet",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "safety": {"can_trade": False, "can_write_learning_memory": False},
            }
            world_model_status = "failed"

        world_state_snapshot: dict[str, Any] = {}
        world_state_persistence: dict[str, Any] = {
            "status": "disabled" if self.world_state_store is None else "pending",
        }
        world_state_failed = False
        if (
            self.world_state_store is not None
            and context_grid.get("status") != "failed"
            and indicator_state_grid.get("status") != "failed"
        ):
            try:
                snapshot = self.world_state_builder.build(
                    domain_id=self.domain_id,
                    as_of=context.as_of,
                    knowledge_cutoff=knowledge_cutoff,
                    context_grid=context_grid,
                    indicator_state_grid=indicator_state_grid,
                    scenario_outcome_graph=world_model.get("scenario_outcome_graph"),
                    world_model_summary=dict(world_model.get("summary", {})),
                    run_id=str(world_model.get("run_id") or "") or None,
                    parent_snapshot_id=parent_snapshot_id,
                    evidence_gaps=[
                        str(item.get("description") or item)
                        for item in world_model.get("evidence_gaps", [])
                        if item
                    ],
                )
                append_result = self.world_state_store.append(snapshot)
                world_state_snapshot = snapshot.model_dump(mode="json")
                context.metadata["canonical_world_state_snapshot_id"] = snapshot.snapshot_id
                context.metadata["canonical_world_state_content_hash"] = snapshot.integrity.content_hash
                world_state_persistence = append_result.model_dump(mode="json")
                if analog_query_error:
                    world_state_persistence["analog_query_warning"] = analog_query_error
                context.metadata["world_state_snapshot"] = world_state_snapshot
                context.metadata["historical_world_state_analogs"] = historical_analogs
            except Exception as exc:
                world_state_failed = True
                world_state_persistence = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        elif self.world_state_store is not None:
            world_state_failed = True
            world_state_persistence = {
                "status": "skipped",
                "reason": "context_or_indicator_grid_failed",
            }

        decision.world_state = dict(decision.world_state or {})
        decision.world_state["event_learning_summary"] = world_model.get("summary", {})
        decision.world_state["event_learning_status"] = world_model_status
        decision.world_state["context_grid"] = context_grid
        decision.world_state["indicator_state_grid"] = indicator_state_grid
        decision.world_state["pipeline_metric_snapshot"] = (
            metric_snapshot if isinstance(metric_snapshot, dict) else {}
        )
        decision.world_state["world_state_snapshot"] = world_state_snapshot
        decision.world_state["historical_world_state_analogs"] = historical_analogs

        grids_failed = context_grid.get("status") == "failed" or indicator_state_grid.get("status") == "failed"
        status = (
            "blocked"
            if decision.decision == "blocked"
            else "partial"
            if world_model_status == "failed" or grids_failed or world_state_failed
            else "completed"
        )
        return MinimalSystemRunResult(
            status=status,
            domain_id=self.domain_id,
            decision=decision,
            pipeline_result=dict(context.pipeline_result),
            pipeline_execution_policy=dict(
                context.metadata.get("pipeline_execution_policy", {})
            ),
            agent_reports=list(context.metadata.get("agent_reports", [])),
            pipeline_metric_snapshot=(
                dict(metric_snapshot) if isinstance(metric_snapshot, dict) else {}
            ),
            context_grid=context_grid,
            indicator_state_grid=indicator_state_grid,
            world_model_event_learning=world_model,
            world_state_snapshot=world_state_snapshot,
            world_state_persistence=world_state_persistence,
            historical_world_state_analogs=historical_analogs,
            original_decision_before_review_clamp=original_decision,
        )


def create_minimal_system(
    project_root: str | Path = ".",
    *,
    domain_id: str = "semiconductor_ai_infrastructure",
    horizon_days: int = 180,
    pipeline_enabled: bool = True,
    pipeline_mode: HybridMode = "local",
    batch_name: str = "main_database",
    stages_to_run: list[int] | None = None,
    soft_mode: bool = True,
    enable_logging: bool = False,
    save_world_model_artifacts: bool = False,
    save_world_state_snapshots: bool = True,
    world_state_store_path: str | Path | None = None,
    historical_analog_limit: int = 5,
    orchestrator: Any | None = None,
    registry_path: str | Path | None = None,
) -> DEANMinimalSystem:
    root = Path(project_root).resolve()
    if orchestrator is not None:
        pipeline_runner = HybridPipelineAdapter(
            mode=pipeline_mode,
            batch_name=batch_name,
            project_root=root,
            orchestrator=orchestrator,
            stages_to_run=stages_to_run,
        )
    elif pipeline_enabled:
        pipeline_runner = HybridPipelineAdapter(
            mode=pipeline_mode,
            batch_name=batch_name,
            project_root=root,
            stages_to_run=stages_to_run,
        )
    else:
        pipeline_runner = None

    registry_overrides = {
        "domain_analyst": {
            "domain_id": domain_id,
            "horizon_days": horizon_days,
        }
    }
    resolved_registry_path = (
        Path(registry_path).expanduser().resolve()
        if registry_path is not None
        else Path(__file__).resolve().parent / "config" / "minimal_system_registry.yaml"
    )
    dean_orchestrator = create_dean_orchestrator(
        project_root=root,
        pipeline_runner=pipeline_runner,
        enable_logging=enable_logging,
        soft_mode=soft_mode,
        registry_path=resolved_registry_path,
        registry_overrides=registry_overrides,
    )
    world_state_store = None
    if save_world_state_snapshots:
        resolved_store_path = (
            Path(world_state_store_path).expanduser().resolve()
            if world_state_store_path is not None
            else root / "reports" / "dean_os" / "world_state" / "world_states.sqlite3"
        )
        world_state_store = SQLiteWorldStateStore(resolved_store_path)

    return DEANMinimalSystem(
        dean_orchestrator,
        domain_id=domain_id,
        world_model_builder=WorldModelEventLearningPacket(
            root / "reports" / "dean_os" / f"{domain_id}_world_model_event_learning"
        ),
        save_world_model_artifacts=save_world_model_artifacts,
        world_state_store=world_state_store,
        historical_analog_limit=historical_analog_limit,
    )


def create_agent_only_system(
    project_root: str | Path = ".",
    *,
    domain_id: str = "semiconductor_ai_infrastructure",
    horizon_days: int = 180,
    soft_mode: bool = True,
    enable_logging: bool = False,
    save_world_model_artifacts: bool = False,
    save_world_state_snapshots: bool = True,
    world_state_store_path: str | Path | None = None,
    historical_analog_limit: int = 5,
    registry_path: str | Path | None = None,
) -> DEANMinimalSystem:
    """Create the current agent-first runtime with the heavy pipeline disabled.

    The pipeline branch and contracts remain present in the architecture, but
    no `src` pipeline stage is executed. This is the canonical entry point for
    building and validating the analytical, world-model, and replay planes
    before live pipeline integration.
    """
    return create_minimal_system(
        project_root=project_root,
        domain_id=domain_id,
        horizon_days=horizon_days,
        pipeline_enabled=False,
        soft_mode=soft_mode,
        enable_logging=enable_logging,
        save_world_model_artifacts=save_world_model_artifacts,
        save_world_state_snapshots=save_world_state_snapshots,
        world_state_store_path=world_state_store_path,
        historical_analog_limit=historical_analog_limit,
        registry_path=registry_path,
    )


__all__ = [
    "DEANMinimalSystem",
    "MinimalSystemRunResult",
    "create_agent_only_system",
    "create_minimal_system",
]
