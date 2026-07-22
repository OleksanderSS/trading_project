from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

LayerPhase = Literal[
    "analysis",
    "indexing",
    "chief_review",
    "human_receipt",
    "dry_run_planning",
    "dry_run_result",
    "paper_planning",
    "paper_result",
    "system_summary",
]


class SystemLayer(BaseModel):
    order: int
    layer_id: str
    phase: LayerPhase
    purpose: str
    module_path: str
    agent_path: str | None = None
    cli_path: str | None = None
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    safety_boundary: list[str] = Field(default_factory=list)
    implementation_status: Literal["implemented", "planned"] = "implemented"


class SystemAuditSummary(BaseModel):
    summary_id: str = Field(default_factory=lambda: f"system_audit_summary_{uuid4().hex}")
    created_at: str = Field(default_factory=utc_now_iso)
    status: Literal["summary_ready", "summary_with_missing_files"] = "summary_ready"
    project: str = "DEAN-OS analyst/tuning supervised review system"
    layers: list[SystemLayer] = Field(default_factory=list)
    missing_files: list[str] = Field(default_factory=list)
    safety_summary: dict[str, bool] = Field(default_factory=dict)
    end_to_end_flow: list[str] = Field(default_factory=list)
    cli_commands: list[str] = Field(default_factory=list)
    next_recommended_blocks: list[str] = Field(default_factory=list)
    integration_notes: list[str] = Field(default_factory=list)

    review_required: bool = True
    live_execution_allowed: bool = False
    broker_access_allowed: bool = False
    production_config_write_allowed: bool = False
    learning_memory_write_allowed: bool = False
    model_promotion_allowed: bool = False


class SystemAuditSummaryBuilder:
    """Builds an end-to-end map of the current DEAN-OS review system.

    This layer does not make trading/tuning decisions. It documents the system,
    validates that key files exist, and writes a review artifact.
    """

    def __init__(
        self,
        project_root: str | Path = ".",
        output_dir: str | Path = "reports/dean_os/system_audit_summary",
    ):
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        layers = system_layers()
        missing = self._missing_files(layers)
        summary = SystemAuditSummary(
            status="summary_with_missing_files" if missing else "summary_ready",
            layers=layers,
            missing_files=missing,
            safety_summary={
                "review_only_architecture": True,
                "live_execution_allowed": False,
                "broker_access_allowed": False,
                "production_config_write_allowed": False,
                "learning_memory_write_allowed": False,
                "model_promotion_allowed": False,
                "execution_like_layers_are_plan_or_record_only": True,
            },
            end_to_end_flow=[layer.layer_id for layer in layers],
            cli_commands=_cli_commands(layers),
            next_recommended_blocks=[
                "Run focused tests for the newest layer after every iteration.",
                "Let Codex review import paths, registry compatibility, and integration placement.",
                "Keep all new agents disabled in registry until human review.",
                "Do not add live execution until a separate live-readiness architecture is designed.",
                "Before adding more execution-like layers, run a full project audit against safety boundaries.",
            ],
            integration_notes=[
                "agent_registry_analyst_tuning_draft.yaml is disabled-by-default and should not be copied blindly.",
                "Review artifacts are JSON/Markdown and intentionally local-file only.",
                "ReviewDecisionReceipt is the only layer that records human decisions.",
                "Dry-run and paper simulation layers are plan/result/review artifacts, not executors.",
                "PostPaperSimulationReview still ends at human review; it does not authorize live actions.",
            ],
        )

        payload = {
            "run_id": summary.summary_id,
            "mode": "system_audit_summary",
            "created_at": utc_now_iso(),
            "system_audit_summary": summary.model_dump(mode="json"),
            "safety": {
                "review_only": True,
                "summary_only": True,
                "live_execution_allowed": False,
                "broker_access_allowed": False,
                "production_config_write_allowed": False,
                "learning_memory_write_allowed": False,
                "model_promotion_allowed": False,
                "approval_performed": False,
            },
        }

        if save:
            markdown = render_system_audit_summary_markdown(payload)
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=markdown,
                run_id=summary.summary_id,
            )
            payload["saved_paths"] = saved_paths

        return json_ready(payload)

    def _missing_files(self, layers: list[SystemLayer]) -> list[str]:
        expected: list[str] = []
        for layer in layers:
            expected.append(layer.module_path)
            if layer.agent_path:
                expected.append(layer.agent_path)
            if layer.cli_path:
                expected.append(layer.cli_path)
        expected.extend(
            [
                "dean_os/config/agent_registry_analyst_tuning_draft.yaml",
                "docs/dean_os_analysts/WORK_LOG_CURRENT.md",
            ]
        )

        missing: list[str] = []
        for rel in expected:
            if not (self.project_root / rel).exists():
                missing.append(rel)
        return sorted(set(missing))


def system_layers() -> list[SystemLayer]:
    no_live = [
        "no_live_execution",
        "no_broker_access",
        "no_production_config_write",
        "no_learning_memory_write",
        "no_model_promotion",
        "human_review_required",
    ]

    return [
        SystemLayer(
            order=1,
            layer_id="domain_analyst",
            phase="analysis",
            purpose="Convert market context/evidence into a domain thesis and ticker basket review.",
            module_path="dean_os/agents/domain_analyst.py",
            agent_path="dean_os/agents/domain_analyst.py",
            cli_path="run_agent_domain_analyst.py",
            input_artifacts=["MarketContext/news/research notes"],
            output_artifacts=["reports/dean_os/domain_analyst/latest.json", "reports/dean_os/domain_analyst/latest.md"],
            decisions=["ready_for_review", "partial_ready_for_review", "needs_more_data", "blocked"],
            safety_boundary=no_live,
        ),
        SystemLayer(
            order=2,
            layer_id="pipeline_tuning_controller",
            phase="analysis",
            purpose="Create review-only tuning plans from model performance/control-surface metadata.",
            module_path="dean_os/agents/pipeline_tuning_controller.py",
            agent_path="dean_os/agents/pipeline_tuning_controller.py",
            cli_path="run_agent_pipeline_tuning_controller.py",
            input_artifacts=["context.metadata.model_performance", "context.metadata.pipeline_control_surface"],
            output_artifacts=["reports/dean_os/pipeline_tuning_controller/latest.json", "reports/dean_os/pipeline_tuning_controller/latest.md"],
            decisions=["tuning_candidate", "validate_first", "blocked", "no_action"],
            safety_boundary=no_live + ["no_real_training_or_tuning_run"],
        ),
        SystemLayer(
            order=3,
            layer_id="review_index",
            phase="indexing",
            purpose="Collect latest review artifacts into one discovery index.",
            module_path="dean_os/review_index.py",
            cli_path="run_agent_review_index.py",
            input_artifacts=["domain_analyst/latest.json", "pipeline_tuning_controller/latest.json"],
            output_artifacts=["reports/dean_os/review_index/latest.json", "reports/dean_os/review_index/latest.md"],
            decisions=["ready_for_chief_review", "missing_sources"],
            safety_boundary=no_live + ["read_existing_artifacts_only"],
        ),
        SystemLayer(
            order=4,
            layer_id="chief_review_index",
            phase="chief_review",
            purpose="Classify the review index into a supervised chief-review decision.",
            module_path="dean_os/chief_review_index.py",
            agent_path="dean_os/agents/chief_review_index.py",
            cli_path="run_agent_chief_review_index.py",
            input_artifacts=["reports/dean_os/review_index/latest.json"],
            output_artifacts=["reports/dean_os/chief_review_index/latest.json", "reports/dean_os/chief_review_index/latest.md"],
            decisions=["ready_for_human_review", "needs_more_evidence", "validate_before_tuning", "blocked"],
            safety_boundary=no_live,
        ),
        SystemLayer(
            order=5,
            layer_id="review_decision_receipt",
            phase="human_receipt",
            purpose="Record an explicit human review decision with scope and rationale.",
            module_path="dean_os/review_decision.py",
            cli_path="run_review_decision_receipt.py",
            input_artifacts=["reports/dean_os/chief_review_index/latest.json or later review artifact"],
            output_artifacts=["reports/dean_os/review_decisions/latest.json", "reports/dean_os/review_decisions/latest.md"],
            decisions=["mark_reviewed", "needs_more_data", "reject", "approve_dry_run", "approve_paper_only_simulation"],
            safety_boundary=no_live + ["human_decision_record_only"],
        ),
        SystemLayer(
            order=6,
            layer_id="dry_run_execution_plan",
            phase="dry_run_planning",
            purpose="Create a non-live dry-run plan only from approve_dry_run receipt.",
            module_path="dean_os/dry_run_plan.py",
            agent_path="dean_os/agents/dry_run_planner.py",
            cli_path="run_dry_run_execution_plan.py",
            input_artifacts=["reports/dean_os/review_decisions/latest.json"],
            output_artifacts=["reports/dean_os/dry_run_plans/latest.json", "reports/dean_os/dry_run_plans/latest.md"],
            decisions=["dry_run_plan_ready", "blocked_missing_receipt", "blocked_not_approved", "blocked_unsafe_receipt"],
            safety_boundary=no_live + ["plan_only", "no_dry_run_execution"],
        ),
        SystemLayer(
            order=7,
            layer_id="dry_run_result",
            phase="dry_run_result",
            purpose="Record externally executed isolated dry-run result.",
            module_path="dean_os/dry_run_result.py",
            agent_path="dean_os/agents/dry_run_result_recorder.py",
            cli_path="run_dry_run_result_record.py",
            input_artifacts=["reports/dean_os/dry_run_plans/latest.json", "external isolated dry-run output"],
            output_artifacts=["reports/dean_os/dry_run_results/latest.json", "reports/dean_os/dry_run_results/latest.md"],
            decisions=["ready_for_review", "rerun_dry_run", "needs_more_data", "reject"],
            safety_boundary=no_live + ["record_only", "external_executor_report_only"],
        ),
        SystemLayer(
            order=8,
            layer_id="post_dry_run_review",
            phase="dry_run_result",
            purpose="Review DryRunResult and decide whether human can review or rerun/reject.",
            module_path="dean_os/post_dry_run_review.py",
            agent_path="dean_os/agents/post_dry_run_review.py",
            cli_path="run_post_dry_run_review.py",
            input_artifacts=["reports/dean_os/dry_run_results/latest.json"],
            output_artifacts=["reports/dean_os/post_dry_run_review/latest.json", "reports/dean_os/post_dry_run_review/latest.md"],
            decisions=["ready_for_human_review", "rerun_dry_run", "reject", "needs_more_data"],
            safety_boundary=no_live,
        ),
        SystemLayer(
            order=9,
            layer_id="paper_simulation_plan",
            phase="paper_planning",
            purpose="Create a non-live paper simulation plan only from approve_paper_only_simulation receipt.",
            module_path="dean_os/paper_simulation_plan.py",
            agent_path="dean_os/agents/paper_simulation_planner.py",
            cli_path="run_paper_simulation_plan.py",
            input_artifacts=["reports/dean_os/review_decisions/latest.json"],
            output_artifacts=["reports/dean_os/paper_simulation_plans/latest.json", "reports/dean_os/paper_simulation_plans/latest.md"],
            decisions=["paper_simulation_plan_ready", "blocked_missing_receipt", "blocked_not_approved", "blocked_unsafe_receipt", "blocked_source_not_ready"],
            safety_boundary=no_live + ["plan_only", "no_paper_simulation_execution"],
        ),
        SystemLayer(
            order=10,
            layer_id="paper_simulation_result",
            phase="paper_result",
            purpose="Record externally executed isolated paper simulation result.",
            module_path="dean_os/paper_simulation_result.py",
            agent_path="dean_os/agents/paper_simulation_result_recorder.py",
            cli_path="run_paper_simulation_result_record.py",
            input_artifacts=["reports/dean_os/paper_simulation_plans/latest.json", "external isolated paper simulation output"],
            output_artifacts=["reports/dean_os/paper_simulation_results/latest.json", "reports/dean_os/paper_simulation_results/latest.md"],
            decisions=["ready_for_review", "rerun_paper_simulation", "needs_more_data", "reject"],
            safety_boundary=no_live + ["record_only", "external_executor_report_only"],
        ),
        SystemLayer(
            order=11,
            layer_id="post_paper_simulation_review",
            phase="paper_result",
            purpose="Review PaperSimulationResult and decide what human should inspect next.",
            module_path="dean_os/post_paper_simulation_review.py",
            agent_path="dean_os/agents/post_paper_simulation_review.py",
            cli_path="run_post_paper_simulation_review.py",
            input_artifacts=["reports/dean_os/paper_simulation_results/latest.json"],
            output_artifacts=["reports/dean_os/post_paper_simulation_review/latest.json", "reports/dean_os/post_paper_simulation_review/latest.md"],
            decisions=["ready_for_human_review", "rerun_paper_simulation", "reject", "needs_more_data"],
            safety_boundary=no_live,
        ),
        SystemLayer(
            order=12,
            layer_id="system_audit_summary",
            phase="system_summary",
            purpose="Summarize the full chain, commands, artifacts, and safety boundaries.",
            module_path="dean_os/system_audit_summary.py",
            agent_path="dean_os/agents/system_audit_summary.py",
            cli_path="run_system_audit_summary.py",
            input_artifacts=["source tree", "registry draft", "work log"],
            output_artifacts=["reports/dean_os/system_audit_summary/latest.json", "reports/dean_os/system_audit_summary/latest.md"],
            decisions=["summary_ready", "summary_with_missing_files"],
            safety_boundary=no_live + ["summary_only"],
        ),
    ]


def render_system_audit_summary_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("system_audit_summary") or {}
    layers = summary.get("layers") or []
    lines = [
        "# DEAN-OS System Audit Summary",
        "",
        f"- Summary ID: `{summary.get('summary_id')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Project: `{summary.get('project')}`",
        f"- Layer count: `{len(layers)}`",
        f"- Missing files: `{len(summary.get('missing_files') or [])}`",
        "",
        "## End-to-End Flow",
        "",
    ]

    for idx, layer_id in enumerate(summary.get("end_to_end_flow") or [], start=1):
        lines.append(f"{idx}. `{layer_id}`")

    lines.extend(["", "## Layers", "", "| # | Layer | Phase | Purpose | CLI |", "|---:|---|---|---|---|"])
    for layer in layers:
        lines.append(
            "| {order} | `{layer}` | `{phase}` | {purpose} | `{cli}` |".format(
                order=layer.get("order"),
                layer=layer.get("layer_id"),
                phase=layer.get("phase"),
                purpose=str(layer.get("purpose") or "").replace("|", "/"),
                cli=layer.get("cli_path") or "",
            )
        )

    lines.extend(["", "## CLI Commands", ""])
    for command in summary.get("cli_commands") or []:
        lines.append(f"```bash\n{command}\n```")

    lines.extend(["", "## Missing Files", ""])
    missing = summary.get("missing_files") or []
    if missing:
        for item in missing:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None.")

    lines.extend(["", "## Safety Summary", ""])
    for key, value in sorted((summary.get("safety_summary") or {}).items()):
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Integration Notes", ""])
    for item in summary.get("integration_notes") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Next Recommended Blocks", ""])
    for item in summary.get("next_recommended_blocks") or []:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Operator Note",
            "",
            "This is a system map and audit artifact only. It does not approve or execute any workflow.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _cli_commands(layers: list[SystemLayer]) -> list[str]:
    commands: list[str] = []
    for layer in layers:
        if layer.cli_path:
            if layer.layer_id == "domain_analyst":
                commands.append("python run_agent_domain_analyst.py --sample --tickers AMD TSM")
            elif layer.layer_id == "pipeline_tuning_controller":
                commands.append("python run_agent_pipeline_tuning_controller.py --sample --tickers AMD --timeframes 1d")
            elif layer.layer_id == "review_decision_receipt":
                commands.append('python run_review_decision_receipt.py --reviewer "Oleksandr" --decision mark_reviewed --rationale "Reviewed for planning only."')
            elif layer.layer_id == "dry_run_result":
                commands.append('python run_dry_run_result_record.py --executor isolated_executor --status completed --summary "Dry-run completed."')
            elif layer.layer_id == "paper_simulation_result":
                commands.append('python run_paper_simulation_result_record.py --executor isolated_paper_executor --status completed --summary "Paper simulation completed."')
            else:
                commands.append(f"python {layer.cli_path}")
    return commands
