from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dean_os.draft.dean_os_agent_system_v7.dean_os.daily_agent_run import DailyAgentRun, DailyAgentRunResult
from dean_os.draft.dean_os_agent_system_v7.dean_os.daily_run_store import SQLiteDailyRunStore
from dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_catalog import SQLiteEvidenceCatalog
from dean_os.draft.dean_os_agent_system_v7.dean_os.minimal_system import create_agent_only_system
from dean_os.draft.dean_os_agent_system_v7.dean_os.operator_review_inbox_v2 import SQLiteOperatorReviewInbox
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_stage03_bridge import PipelineStage03Bridge, PipelineStage03Packet
from dean_os.draft.dean_os_agent_system_v7.dean_os.replay_scheduler import ReplayScheduleItem
from dean_os.schemas import MarketContext
from dean_os.draft.dean_os_agent_system_v7.dean_os.system_topology import (
    BranchExecutionRecord,
    BranchId,
    BranchRunStatus,
    BranchSpec,
    BranchTimer,
    SystemRunManifest,
    SystemTopology,
    load_default_system_topology,
)


class AgentSystemRunResult(BaseModel):
    status: BranchRunStatus
    domain_id: str
    topology: SystemTopology
    manifest: SystemRunManifest
    pipeline_stage03_packet: PipelineStage03Packet
    daily_run: DailyAgentRunResult
    branch_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    safety: dict[str, bool] = Field(
        default_factory=lambda: {
            "can_trade": False,
            "can_write_production_config": False,
            "can_promote_model": False,
            "can_write_learning_memory": False,
            "human_review_required": True,
        }
    )


class DEANAgentSystemOrchestrator:
    """Canonical composition root for the complete agent-system skeleton.

    The runtime has two primary trunks:

    * pipeline/control: consume existing stages 0-3 outputs and apply guardrails;
    * analytical/world-model: evidence intelligence, domain analysis, scenarios,
      replay, and review.

    Governance and audit are cross-cutting branches. Domain instances are
    registry/profile configuration; the orchestrator is not copied per sector.
    """

    def __init__(
        self,
        daily_runner: DailyAgentRun,
        *,
        domain_id: str,
        topology: SystemTopology | None = None,
        stage03_bridge: PipelineStage03Bridge | None = None,
    ):
        self.daily_runner = daily_runner
        self.domain_id = domain_id
        self.topology = topology or load_default_system_topology()
        self.stage03_bridge = stage03_bridge or PipelineStage03Bridge()
        self._specs = {item.branch_id: item for item in self.topology.enabled_branches()}

    async def run(
        self,
        context: MarketContext,
        *,
        pipeline_stage03_result: dict[str, Any] | None = None,
        pipeline_artifact_references: list[dict[str, Any]] | None = None,
        evidence_payloads: list[dict[str, Any]] | None = None,
        prior_replay_tasks: list[ReplayScheduleItem | dict[str, Any]] | None = None,
    ) -> AgentSystemRunResult:
        started_at = datetime.now(UTC)
        context.as_of = context.as_of or started_at.isoformat()
        knowledge_cutoff = str(context.metadata.get("knowledge_cutoff") or context.as_of)
        context.metadata["knowledge_cutoff"] = knowledge_cutoff
        context.metadata["system_topology_id"] = self.topology.topology_id
        context.metadata["system_topology_hash"] = self.topology.topology_hash
        context.metadata["domain_id"] = self.domain_id
        context.metadata["system_mode"] = "full_agent_skeleton_review_only"
        context.metadata["execution_intent"] = "analysis_only"

        branch_records: dict[str, BranchExecutionRecord] = {}

        intake_spec = self._spec(BranchId.PIPELINE_STAGE03_INTAKE)
        intake_timer = BranchTimer()
        packet = self.stage03_bridge.build_packet(
            pipeline_stage03_result,
            as_of=context.as_of,
            knowledge_cutoff=knowledge_cutoff,
            artifact_references=pipeline_artifact_references,
        )
        self.stage03_bridge.attach_to_context(
            context,
            packet,
            raw_result=pipeline_stage03_result,
        )
        context.metadata["pipeline_operating_profile"] = (
            "stage03_data_only" if packet.status != "missing" else "agent_only_no_pipeline"
        )
        intake_status = (
            BranchRunStatus.SKIPPED
            if packet.status == "missing"
            else BranchRunStatus.FAILED
            if packet.status == "failed"
            else BranchRunStatus.PARTIAL
            if packet.status == "partial"
            else BranchRunStatus.COMPLETED
        )
        branch_records[intake_spec.branch_id] = intake_timer.finish(
            spec=intake_spec,
            status=intake_status,
            input_payload={
                "result_present": pipeline_stage03_result is not None,
                "artifact_reference_count": len(pipeline_artifact_references or []),
            },
            output_payload=packet,
            summary={
                "stages_present": packet.stages_present,
                "news_item_count": len(packet.news_items),
                "artifact_count": len(packet.artifact_references),
                "active_stage_boundary": [0, 1, 2, 3],
            },
            warnings=packet.warnings,
            safety=packet.safety,
        )

        # Include the immutable stage packet itself as provenance evidence. News
        # rows are already attached to context.news and will be point-in-time
        # filtered by DailyAgentRun before any analytical agent sees them.
        combined_evidence = list(evidence_payloads or [])
        if packet.status != "missing":
            combined_evidence.append(
                {
                    "evidence_id": f"pipeline_stage03_{packet.content_hash[:24]}",
                    "source_type": "pipeline_artifact",
                    "source": "pipeline_stage03_bridge",
                    "title": "Pipeline stages 0-3 normalized packet",
                    "value": packet.model_dump(mode="json"),
                    "available_at": packet.knowledge_cutoff,
                    "ingested_at": context.as_of,
                    "external_artifact_ref": "pipeline_stage03_packet",
                    "evidence_lanes": ["pipeline_data", "news_ingestion"],
                    "quality_score": 0.8,
                }
            )

        core_timer = BranchTimer()
        daily_result = await self.daily_runner.run(
            context,
            evidence_payloads=combined_evidence,
            prior_replay_tasks=prior_replay_tasks,
        )
        core_finished_at = datetime.now(UTC)

        self._project_core_branch_records(
            daily_result=daily_result,
            context=context,
            branch_records=branch_records,
            timer=core_timer,
        )

        ordered_records = [
            branch_records[spec.branch_id]
            for spec in self.topology.execution_order()
            if spec.branch_id in branch_records
        ]
        final_status, blocked_by, warnings = _aggregate_status(ordered_records)
        finished_at = datetime.now(UTC)
        manifest = SystemRunManifest(
            topology_id=self.topology.topology_id,
            topology_hash=self.topology.topology_hash,
            domain_id=self.domain_id,
            as_of=context.as_of,
            knowledge_cutoff=knowledge_cutoff,
            status=final_status,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            branch_records=ordered_records,
            blocked_by=blocked_by,
            warnings=warnings,
        )
        branch_outputs = _branch_output_index(daily_result, packet)
        branch_outputs["composite_runtime"] = {
            "daily_run_id": daily_result.daily_run_id,
            "composite_started_at": core_timer.started.isoformat(),
            "composite_finished_at": core_finished_at.isoformat(),
            "manifest_hash": manifest.content_hash,
        }
        return AgentSystemRunResult(
            status=final_status,
            domain_id=self.domain_id,
            topology=self.topology,
            manifest=manifest,
            pipeline_stage03_packet=packet,
            daily_run=daily_result,
            branch_outputs=branch_outputs,
        )

    def _project_core_branch_records(
        self,
        *,
        daily_result: DailyAgentRunResult,
        context: MarketContext,
        branch_records: dict[str, BranchExecutionRecord],
        timer: BranchTimer,
    ) -> None:
        system_result = dict(daily_result.system_result)
        agent_reports = list(system_result.get("agent_reports") or [])
        pipeline_reports = [item for item in agent_reports if item.get("branch") == "pipeline"]
        analytical_reports = [item for item in agent_reports if item.get("branch") == "analytical"]

        evidence_spec = self._spec(BranchId.EVIDENCE_INTELLIGENCE)
        evidence_status = _status_from_text(daily_result.evidence_manifest.status)
        branch_records[evidence_spec.branch_id] = timer.finish(
            spec=evidence_spec,
            status=evidence_status,
            input_payload={"context_news_count": len(context.news)},
            output_payload={
                "manifest": daily_result.evidence_manifest,
                "dedup": daily_result.evidence_dedup,
            },
            summary={
                "accepted_evidence_count": len(daily_result.evidence_records),
                "rejected_count": len(daily_result.evidence_manifest.rejected_items),
                "suppressed_count": len(daily_result.evidence_manifest.suppressed_items),
                "measurement_mode": "projected_from_daily_composite_run",
            },
        )

        pipeline_spec = self._spec(BranchId.PIPELINE_CONTROL)
        pipeline_status = _pipeline_control_status(system_result, pipeline_reports)
        branch_records[pipeline_spec.branch_id] = timer.finish(
            spec=pipeline_spec,
            status=pipeline_status,
            input_payload={
                "pipeline_stage03_hash": context.metadata.get("pipeline_stage03_source_hash"),
                "pipeline_metric_snapshot": system_result.get("pipeline_metric_snapshot", {}),
            },
            output_payload={
                "pipeline_execution_policy": system_result.get("pipeline_execution_policy", {}),
                "reports": pipeline_reports,
            },
            summary={
                "report_count": len(pipeline_reports),
                "blocked_agents": [
                    item.get("agent_name") for item in pipeline_reports if item.get("verdict") == "blocked"
                ],
                "proposal_only": True,
                "measurement_mode": "projected_from_core_orchestrator",
            },
        )

        analysis_spec = self._spec(BranchId.DOMAIN_ANALYSIS)
        analysis_status = (
            BranchRunStatus.PARTIAL if not analytical_reports else BranchRunStatus.COMPLETED
        )
        branch_records[analysis_spec.branch_id] = timer.finish(
            spec=analysis_spec,
            status=analysis_status,
            input_payload={
                "accepted_evidence_ids": daily_result.evidence_manifest.evidence_ids,
                "domain_id": self.domain_id,
            },
            output_payload=analytical_reports,
            summary={
                "analyst_instance_count": len(analytical_reports),
                "agent_names": [item.get("agent_name") for item in analytical_reports],
                "instances_shareable_by_registry": True,
                "measurement_mode": "projected_from_core_orchestrator",
            },
            warnings=[] if analytical_reports else ["No analytical report was produced"],
        )

        world_spec = self._spec(BranchId.WORLD_MODEL)
        world_status = _world_model_status(system_result)
        branch_records[world_spec.branch_id] = timer.finish(
            spec=world_spec,
            status=world_status,
            input_payload={
                "context_grid": system_result.get("context_grid", {}),
                "indicator_state_grid": system_result.get("indicator_state_grid", {}),
            },
            output_payload={
                "world_model": system_result.get("world_model_event_learning", {}),
                "snapshot": system_result.get("world_state_snapshot", {}),
            },
            summary={
                "world_state_snapshot_id": (
                    system_result.get("world_state_snapshot", {}).get("snapshot_id")
                ),
                "historical_analog_count": len(
                    system_result.get("historical_world_state_analogs", [])
                ),
                "measurement_mode": "projected_from_minimal_system",
            },
        )

        replay_spec = self._spec(BranchId.REPLAY_EVALUATION)
        branch_records[replay_spec.branch_id] = timer.finish(
            spec=replay_spec,
            status=BranchRunStatus.COMPLETED,
            input_payload=system_result.get("world_state_snapshot", {}),
            output_payload={
                "scheduled": daily_result.replay_schedule,
                "due": daily_result.due_replay_tasks,
            },
            summary={
                "scheduled_count": len(daily_result.replay_schedule),
                "due_count": len(daily_result.due_replay_tasks),
                "automatic_learning_promotion": False,
                "measurement_mode": "projected_from_daily_composite_run",
            },
        )

        review_spec = self._spec(BranchId.GOVERNANCE_REVIEW)
        review_status = (
            BranchRunStatus.PARTIAL
            if not daily_result.review_inbox_items
            else BranchRunStatus.COMPLETED
        )
        branch_records[review_spec.branch_id] = timer.finish(
            spec=review_spec,
            status=review_status,
            input_payload={
                "briefing_id": daily_result.briefing.briefing_id,
                "evidence_gap_plan": daily_result.evidence_gap_plan,
            },
            output_payload=daily_result.review_inbox_items,
            summary={
                "review_item_count": len(daily_result.review_inbox_items),
                "human_review_required": True,
                "hash_bound_review": True,
                "measurement_mode": "projected_from_daily_composite_run",
            },
        )

        audit_spec = self._spec(BranchId.DAILY_AUDIT)
        audit_status = (
            BranchRunStatus.COMPLETED
            if daily_result.persisted_run_record is not None
            else BranchRunStatus.PARTIAL
        )
        branch_records[audit_spec.branch_id] = timer.finish(
            spec=audit_spec,
            status=audit_status,
            input_payload={
                "daily_run_id": daily_result.daily_run_id,
                "rendered_artifacts": daily_result.rendered_artifacts,
            },
            output_payload=daily_result.persisted_run_record or daily_result.model_dump(mode="json"),
            summary={
                "daily_run_persisted": daily_result.persisted_run_record is not None,
                "rendered_artifacts": daily_result.rendered_artifacts,
                "measurement_mode": "projected_from_daily_composite_run",
            },
            warnings=(
                []
                if daily_result.persisted_run_record is not None
                else ["Daily run store is disabled; audit record was returned but not persisted"]
            ),
        )

    def _spec(self, branch_id: BranchId) -> BranchSpec:
        try:
            return self._specs[str(branch_id)]
        except KeyError as exc:
            raise ValueError(f"Required topology branch is not enabled: {branch_id}") from exc


def create_full_agent_system(
    project_root: str | Path = ".",
    *,
    domain_id: str = "semiconductor_ai_infrastructure",
    soft_mode: bool = True,
    persistence_enabled: bool = True,
    reports_root: str | Path | None = None,
    briefing_output_dir: str | Path | None = None,
    save_world_model_artifacts: bool = False,
    historical_analog_limit: int = 5,
    topology: SystemTopology | None = None,
) -> DEANAgentSystemOrchestrator:
    root = Path(project_root).resolve()
    reports = Path(reports_root) if reports_root else root / "reports" / "dean_os"
    if not reports.is_absolute():
        reports = root / reports

    evidence_catalog = None
    daily_store = None
    review_inbox = None
    world_state_path = None
    if persistence_enabled:
        evidence_catalog = SQLiteEvidenceCatalog(reports / "evidence" / "catalog.sqlite3")
        daily_store = SQLiteDailyRunStore(reports / "daily_runs" / "runs.sqlite3")
        review_inbox = SQLiteOperatorReviewInbox(reports / "operator_review" / "inbox.sqlite3")
        world_state_path = reports / "world_state" / "world_states.sqlite3"

    system = create_agent_only_system(
        project_root=root,
        domain_id=domain_id,
        soft_mode=soft_mode,
        save_world_model_artifacts=save_world_model_artifacts,
        save_world_state_snapshots=persistence_enabled,
        world_state_store_path=world_state_path,
        historical_analog_limit=historical_analog_limit,
    )
    daily_runner = DailyAgentRun(
        system,
        domain_id=domain_id,
        evidence_catalog=evidence_catalog,
        daily_run_store=daily_store,
        review_inbox=review_inbox,
        briefing_output_dir=(
            str(briefing_output_dir)
            if briefing_output_dir is not None
            else str(reports / "briefings")
            if persistence_enabled
            else None
        ),
    )
    return DEANAgentSystemOrchestrator(
        daily_runner,
        domain_id=domain_id,
        topology=topology,
    )


def _status_from_text(value: str) -> BranchRunStatus:
    normalized = str(value or "").lower()
    if normalized == "completed":
        return BranchRunStatus.COMPLETED
    if normalized == "failed":
        return BranchRunStatus.FAILED
    if normalized == "blocked":
        return BranchRunStatus.BLOCKED
    if normalized == "skipped":
        return BranchRunStatus.SKIPPED
    return BranchRunStatus.PARTIAL


def _pipeline_control_status(
    system_result: dict[str, Any],
    reports: list[dict[str, Any]],
) -> BranchRunStatus:
    if any(item.get("verdict") == "blocked" for item in reports):
        return BranchRunStatus.BLOCKED
    if not reports:
        return BranchRunStatus.PARTIAL
    if system_result.get("status") == "blocked":
        return BranchRunStatus.BLOCKED
    return BranchRunStatus.COMPLETED


def _world_model_status(system_result: dict[str, Any]) -> BranchRunStatus:
    world_model = system_result.get("world_model_event_learning") or {}
    persistence = system_result.get("world_state_persistence") or {}
    if world_model.get("status") == "failed" or persistence.get("status") == "failed":
        return BranchRunStatus.FAILED
    if not system_result.get("world_state_snapshot"):
        return BranchRunStatus.PARTIAL
    return BranchRunStatus.COMPLETED


def _aggregate_status(
    records: list[BranchExecutionRecord],
) -> tuple[BranchRunStatus, list[str], list[str]]:
    blocked = [item.branch_id for item in records if item.status == BranchRunStatus.BLOCKED]
    failed_required = [
        item.branch_id
        for item in records
        if item.required and item.status == BranchRunStatus.FAILED
    ]
    warnings = [warning for item in records for warning in item.warnings]
    if failed_required:
        return BranchRunStatus.FAILED, failed_required, warnings
    if blocked:
        return BranchRunStatus.BLOCKED, blocked, warnings
    if any(
        item.required and item.status in {BranchRunStatus.PARTIAL, BranchRunStatus.SKIPPED}
        for item in records
    ):
        return BranchRunStatus.PARTIAL, [], warnings
    if any(item.status in {BranchRunStatus.PARTIAL, BranchRunStatus.FAILED} for item in records):
        return BranchRunStatus.PARTIAL, [], warnings
    return BranchRunStatus.COMPLETED, [], warnings


def _branch_output_index(
    daily_result: DailyAgentRunResult,
    packet: PipelineStage03Packet,
) -> dict[str, dict[str, Any]]:
    system_result = daily_result.system_result
    reports = list(system_result.get("agent_reports") or [])
    return {
        str(BranchId.PIPELINE_STAGE03_INTAKE): {
            "packet_hash": packet.content_hash,
            "status": packet.status,
            "stages_present": packet.stages_present,
        },
        str(BranchId.PIPELINE_CONTROL): {
            "policy": system_result.get("pipeline_execution_policy", {}),
            "agent_reports": [item for item in reports if item.get("branch") == "pipeline"],
        },
        str(BranchId.EVIDENCE_INTELLIGENCE): {
            "manifest_hash": daily_result.evidence_manifest.content_hash,
            "accepted_evidence_ids": daily_result.evidence_manifest.evidence_ids,
        },
        str(BranchId.DOMAIN_ANALYSIS): {
            "agent_reports": [item for item in reports if item.get("branch") == "analytical"],
        },
        str(BranchId.WORLD_MODEL): {
            "snapshot": system_result.get("world_state_snapshot", {}),
            "scenario_graph": (
                system_result.get("world_model_event_learning", {}).get("scenario_outcome_graph", {})
            ),
        },
        str(BranchId.REPLAY_EVALUATION): {
            "scheduled": [item.model_dump(mode="json") for item in daily_result.replay_schedule],
            "due": [item.model_dump(mode="json") for item in daily_result.due_replay_tasks],
        },
        str(BranchId.GOVERNANCE_REVIEW): {
            "briefing_id": daily_result.briefing.briefing_id,
            "review_items": [item.model_dump(mode="json") for item in daily_result.review_inbox_items],
        },
        str(BranchId.DAILY_AUDIT): {
            "daily_run_id": daily_result.daily_run_id,
            "persisted": daily_result.persisted_run_record is not None,
            "rendered_artifacts": daily_result.rendered_artifacts,
        },
    }


__all__ = [
    "AgentSystemRunResult",
    "DEANAgentSystemOrchestrator",
    "create_full_agent_system",
]
