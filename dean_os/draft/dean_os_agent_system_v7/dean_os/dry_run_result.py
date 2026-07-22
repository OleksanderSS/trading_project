from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DryRunResultStatus = Literal[
    "completed",
    "completed_with_warnings",
    "failed",
    "blocked_invalid_plan",
    "blocked_unsafe_plan",
]


class DryRunMetric(BaseModel):
    name: str
    value: Any
    unit: str | None = None
    interpretation: str | None = None


class DryRunResultArtifact(BaseModel):
    """Record of an externally executed isolated dry-run.

    This layer records the outcome only. It does not execute the dry-run.
    """

    result_id: str = Field(default_factory=lambda: f"dry_run_result_{uuid4().hex}")
    created_at: str = Field(default_factory=utc_now_iso)
    status: DryRunResultStatus
    executor: str
    source_plan_id: str | None = None
    source_plan_path: str | None = None
    source_receipt_id: str | None = None
    summary: str
    metrics: list[DryRunMetric] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    guardrail_checks: dict[str, bool] = Field(default_factory=dict)
    recommendation: Literal[
        "ready_for_review",
        "needs_more_data",
        "reject",
        "rerun_dry_run",
    ]

    review_required: bool = True
    dry_run_executed_by_this_layer: bool = False
    live_execution_allowed: bool = False
    broker_access_allowed: bool = False
    production_config_write_allowed: bool = False
    learning_memory_write_allowed: bool = False
    model_promotion_allowed: bool = False


class DryRunResultRecorder:
    """Records results from an external isolated dry-run."""

    def __init__(
        self,
        plan_artifact_path: str | Path,
        output_dir: str | Path = "reports/dean_os/dry_run_results",
    ):
        self.plan_artifact_path = Path(plan_artifact_path)
        self.output_dir = Path(output_dir)

    def record(
        self,
        *,
        executor: str,
        status: Literal["completed", "completed_with_warnings", "failed"],
        summary: str,
        metrics: list[DryRunMetric | dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
        artifacts: list[str] | None = None,
        guardrail_checks: dict[str, bool] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        plan_payload = self._load_plan()
        result = build_dry_run_result(
            plan_payload=plan_payload,
            plan_artifact_path=str(self.plan_artifact_path),
            executor=executor,
            requested_status=status,
            summary=summary,
            metrics=metrics or [],
            warnings=warnings or [],
            errors=errors or [],
            artifacts=artifacts or [],
            guardrail_checks=guardrail_checks or {},
        )
        payload = {
            "run_id": result.result_id,
            "mode": "dry_run_result",
            "created_at": utc_now_iso(),
            "dry_run_result": result.model_dump(mode="json"),
            "source_plan_summary": _plan_summary(plan_payload),
            "safety": {
                "review_only": True,
                "dry_run_result_recorded": True,
                "external_dry_run_reported": result.status in {"completed", "completed_with_warnings", "failed"},
                "dry_run_executed_by_this_layer": False,
                "live_execution_allowed": False,
                "broker_access_allowed": False,
                "production_config_write_allowed": False,
                "learning_memory_write_allowed": False,
                "model_promotion_allowed": False,
            },
        }

        if save:
            markdown = render_dry_run_result_markdown(payload)
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=markdown,
                run_id=result.result_id,
            )
            payload["saved_paths"] = saved_paths

        return json_ready(payload)

    def _load_plan(self) -> dict[str, Any]:
        if not self.plan_artifact_path.exists():
            return {
                "mode": "missing_dry_run_plan",
                "dry_run_plan": None,
                "errors": [f"Dry-run plan not found: {self.plan_artifact_path}"],
            }

        try:
            payload = json.loads(self.plan_artifact_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "mode": "invalid_dry_run_plan",
                "dry_run_plan": None,
                "errors": [repr(exc)],
            }

        return payload if isinstance(payload, dict) else {
            "mode": "invalid_dry_run_plan",
            "dry_run_plan": None,
            "errors": ["Plan JSON is not an object"],
        }


def build_dry_run_result(
    *,
    plan_payload: dict[str, Any],
    plan_artifact_path: str | None,
    executor: str,
    requested_status: Literal["completed", "completed_with_warnings", "failed"],
    summary: str,
    metrics: list[DryRunMetric | dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    artifacts: list[str],
    guardrail_checks: dict[str, bool],
) -> DryRunResultArtifact:
    plan = plan_payload.get("dry_run_plan")
    if not isinstance(plan, dict):
        return DryRunResultArtifact(
            status="blocked_invalid_plan",
            executor=executor,
            source_plan_path=plan_artifact_path,
            summary="Cannot record dry-run result because the dry-run plan artifact is missing or invalid.",
            errors=list(plan_payload.get("errors") or ["missing or invalid dry-run plan"]),
            recommendation="needs_more_data",
        )

    unsafe = _unsafe_plan_flags(plan, plan_payload)
    if unsafe:
        return DryRunResultArtifact(
            status="blocked_unsafe_plan",
            executor=executor,
            source_plan_id=plan.get("plan_id"),
            source_plan_path=plan_artifact_path,
            source_receipt_id=plan.get("source_receipt_id"),
            summary="Cannot record dry-run result because the source plan contains unsafe flags.",
            errors=unsafe,
            recommendation="reject",
        )

    if plan.get("status") != "dry_run_plan_ready":
        return DryRunResultArtifact(
            status="blocked_invalid_plan",
            executor=executor,
            source_plan_id=plan.get("plan_id"),
            source_plan_path=plan_artifact_path,
            source_receipt_id=plan.get("source_receipt_id"),
            summary=f"Cannot record dry-run result because plan status is {plan.get('status')!r}.",
            errors=[f"dry-run plan status is {plan.get('status')!r}, expected 'dry_run_plan_ready'"],
            recommendation="needs_more_data",
        )

    metric_items = [item if isinstance(item, DryRunMetric) else DryRunMetric(**item) for item in metrics]
    checks = _default_guardrail_checks()
    checks.update(guardrail_checks)

    final_status: DryRunResultStatus = requested_status
    if requested_status == "completed" and (warnings or not all(checks.values())):
        final_status = "completed_with_warnings"

    recommendation = _recommendation(final_status, errors, warnings, checks)
    return DryRunResultArtifact(
        status=final_status,
        executor=executor,
        source_plan_id=plan.get("plan_id"),
        source_plan_path=plan_artifact_path,
        source_receipt_id=plan.get("source_receipt_id"),
        summary=summary,
        metrics=metric_items,
        warnings=warnings,
        errors=errors,
        artifacts=artifacts,
        guardrail_checks=checks,
        recommendation=recommendation,
        review_required=True,
        dry_run_executed_by_this_layer=False,
        live_execution_allowed=False,
        broker_access_allowed=False,
        production_config_write_allowed=False,
        learning_memory_write_allowed=False,
        model_promotion_allowed=False,
    )


def render_dry_run_result_markdown(payload: dict[str, Any]) -> str:
    result = payload.get("dry_run_result") or {}
    plan = payload.get("source_plan_summary") or {}
    lines = [
        "# DEAN-OS Dry Run Result",
        "",
        f"- Result ID: `{result.get('result_id')}`",
        f"- Status: `{result.get('status')}`",
        f"- Recommendation: `{result.get('recommendation')}`",
        f"- Executor: `{result.get('executor')}`",
        f"- Source plan: `{result.get('source_plan_id')}`",
        f"- Source receipt: `{result.get('source_receipt_id')}`",
        "",
        "## Summary",
        "",
        str(result.get("summary") or ""),
        "",
        "## Source Plan Summary",
        "",
    ]

    for key, value in plan.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Metrics", ""])
    metrics = result.get("metrics") or []
    if metrics:
        lines.extend(["| Metric | Value | Unit | Interpretation |", "|---|---:|---|---|"])
        for metric in metrics:
            lines.append(
                "| {name} | {value} | {unit} | {interpretation} |".format(
                    name=metric.get("name"),
                    value=metric.get("value"),
                    unit=metric.get("unit") or "",
                    interpretation=metric.get("interpretation") or "",
                )
            )
    else:
        lines.append("- No metrics supplied.")

    lines.extend(["", "## Guardrail Checks", ""])
    for key, value in sorted((result.get("guardrail_checks") or {}).items()):
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Warnings", ""])
    for warning in result.get("warnings") or []:
        lines.append(f"- {warning}")
    if not result.get("warnings"):
        lines.append("- None.")

    lines.extend(["", "## Errors", ""])
    for error in result.get("errors") or []:
        lines.append(f"- {error}")
    if not result.get("errors"):
        lines.append("- None.")

    lines.extend(["", "## Produced Artifacts", ""])
    for artifact in result.get("artifacts") or []:
        lines.append(f"- `{artifact}`")
    if not result.get("artifacts"):
        lines.append("- None supplied.")

    lines.extend(["", "## Safety", ""])
    safety = payload.get("safety") or {}
    for key in sorted(safety):
        lines.append(f"- {key}: `{safety[key]}`")

    lines.extend(
        [
            "",
            "## Operator Note",
            "",
            "This result records an externally executed isolated dry-run. It does not approve promotion, paper trading, live trading, broker access, or production config writes.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _default_guardrail_checks() -> dict[str, bool]:
    return {
        "no_live_execution": True,
        "no_broker_access": True,
        "no_production_config_write": True,
        "no_learning_memory_write": True,
        "no_model_promotion": True,
        "review_artifact_written": True,
    }


def _unsafe_plan_flags(plan: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    dangerous = {
        "live_execution_allowed",
        "broker_access_allowed",
        "production_config_write_allowed",
        "learning_memory_write_allowed",
        "model_promotion_allowed",
        "training_or_tuning_run_allowed",
    }
    messages: list[str] = []
    safety = dict(payload.get("safety") or {})
    for flag in dangerous:
        if plan.get(flag) is True:
            messages.append(f"dry_run_plan.{flag}=true")
        if safety.get(flag) is True:
            messages.append(f"safety.{flag}=true")
    return messages


def _recommendation(
    status: DryRunResultStatus,
    errors: list[str],
    warnings: list[str],
    checks: dict[str, bool],
) -> str:
    if status in {"blocked_invalid_plan", "blocked_unsafe_plan", "failed"} or errors:
        return "reject"
    if warnings or not all(checks.values()) or status == "completed_with_warnings":
        return "rerun_dry_run"
    return "ready_for_review"


def _plan_summary(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("dry_run_plan") if isinstance(payload.get("dry_run_plan"), dict) else {}
    return {
        "mode": payload.get("mode"),
        "plan_id": plan.get("plan_id"),
        "status": plan.get("status"),
        "source_receipt_id": plan.get("source_receipt_id"),
        "source_decision": plan.get("source_decision"),
        "target": plan.get("target"),
        "step_count": len(plan.get("steps") or []),
    }
