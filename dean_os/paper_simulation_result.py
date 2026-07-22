from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.paper_lifecycle_contract import (
    PAPER_LIFECYCLE_SCHEMA_VERSION,
    file_sha256,
    fingerprint_matches,
    load_json_object,
    object_fingerprint,
    receipt_lineage_issues,
    valid_sha256,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

PaperSimulationResultStatus = Literal[
    "completed",
    "completed_with_warnings",
    "failed",
    "blocked_invalid_plan",
    "blocked_unsafe_plan",
    "blocked_missing_external_evidence",
    "blocked_external_evidence_mismatch",
]

PaperSimulationRecommendation = Literal[
    "ready_for_review",
    "rerun_paper_simulation",
    "needs_more_data",
    "reject",
]


class PaperSimulationMetric(BaseModel):
    name: str
    value: Any
    unit: str | None = None
    interpretation: str | None = None


class PaperSimulationResultArtifact(BaseModel):
    """Record of an externally executed isolated paper simulation.

    This layer records the outcome only. It does not execute paper simulation,
    submit orders, call brokers, promote models, or write production config.
    """

    result_id: str = Field(default_factory=lambda: f"paper_sim_result_{uuid4().hex}")
    created_at: str = Field(default_factory=utc_now_iso)
    status: PaperSimulationResultStatus
    executor: str
    source_plan_id: str | None = None
    source_plan_path: str | None = None
    source_plan_sha256: str | None = None
    source_plan_fingerprint: str | None = None
    source_receipt_id: str | None = None
    external_result_path: str | None = None
    external_result_sha256: str | None = None
    external_result_fingerprint: str | None = None
    lineage_verified: bool = False
    summary: str
    metrics: list[PaperSimulationMetric] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    guardrail_checks: dict[str, bool] = Field(default_factory=dict)
    recommendation: PaperSimulationRecommendation

    review_required: bool = True
    paper_simulation_executed_by_this_layer: bool = False
    live_execution_allowed: bool = False
    broker_access_allowed: bool = False
    production_config_write_allowed: bool = False
    learning_memory_write_allowed: bool = False
    model_promotion_allowed: bool = False


class PaperSimulationResultRecorder:
    """Records externally produced paper simulation results."""

    def __init__(
        self,
        plan_artifact_path: str | Path,
        output_dir: str | Path = "reports/dean_os/paper_simulation_results",
    ):
        self.plan_artifact_path = Path(plan_artifact_path)
        self.output_dir = Path(output_dir)

    def record(
        self,
        *,
        executor: str,
        status: Literal["completed", "completed_with_warnings", "failed"],
        summary: str,
        metrics: list[PaperSimulationMetric | dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
        artifacts: list[str] | None = None,
        guardrail_checks: dict[str, bool] | None = None,
        external_result_path: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        plan_payload = self._load_plan()
        result = build_paper_simulation_result(
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
            external_result_path=(
                str(external_result_path)
                if external_result_path is not None
                else None
            ),
        )
        result_payload = result.model_dump(mode="json")
        payload = {
            "run_id": result.result_id,
            "mode": "paper_simulation_result",
            "schema_version": PAPER_LIFECYCLE_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "paper_simulation_result": result_payload,
            "result_fingerprint": object_fingerprint(result_payload),
            "source_plan_summary": _plan_summary(plan_payload),
            "safety": {
                "review_only": True,
                "paper_simulation_result_recorded": True,
                "external_paper_simulation_reported": result.status in {"completed", "completed_with_warnings", "failed"},
                "paper_simulation_executed_by_this_layer": False,
                "live_execution_allowed": False,
                "broker_access_allowed": False,
                "production_config_write_allowed": False,
                "learning_memory_write_allowed": False,
                "model_promotion_allowed": False,
                "lineage_verified": result.lineage_verified,
            },
        }

        if save:
            markdown = render_paper_simulation_result_markdown(payload)
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
                "mode": "missing_paper_simulation_plan",
                "paper_simulation_plan": None,
                "errors": [f"Paper simulation plan not found: {self.plan_artifact_path}"],
            }

        try:
            payload = json.loads(self.plan_artifact_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "mode": "invalid_paper_simulation_plan",
                "paper_simulation_plan": None,
                "errors": [repr(exc)],
            }

        return payload if isinstance(payload, dict) else {
            "mode": "invalid_paper_simulation_plan",
            "paper_simulation_plan": None,
            "errors": ["Plan JSON is not an object"],
        }


def build_paper_simulation_result(
    *,
    plan_payload: dict[str, Any],
    plan_artifact_path: str | None,
    executor: str,
    requested_status: Literal["completed", "completed_with_warnings", "failed"],
    summary: str,
    metrics: list[PaperSimulationMetric | dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    artifacts: list[str],
    guardrail_checks: dict[str, bool],
    external_result_path: str | None = None,
) -> PaperSimulationResultArtifact:
    plan = plan_payload.get("paper_simulation_plan")
    if not isinstance(plan, dict):
        return PaperSimulationResultArtifact(
            status="blocked_invalid_plan",
            executor=executor,
            source_plan_path=plan_artifact_path,
            summary="Cannot record paper simulation result because the plan artifact is missing or invalid.",
            errors=list(plan_payload.get("errors") or ["missing or invalid paper simulation plan"]),
            recommendation="needs_more_data",
        )

    plan_sha256 = file_sha256(plan_artifact_path)
    plan_fingerprint = plan_payload.get("plan_fingerprint")
    plan_lineage_issues = _plan_lineage_issues(
        plan_payload=plan_payload,
        plan_artifact_path=plan_artifact_path,
    )
    if plan_lineage_issues:
        return PaperSimulationResultArtifact(
            status="blocked_invalid_plan",
            executor=executor,
            source_plan_id=plan.get("plan_id"),
            source_plan_path=plan_artifact_path,
            source_plan_sha256=plan_sha256,
            source_plan_fingerprint=plan_fingerprint,
            source_receipt_id=plan.get("source_receipt_id"),
            summary=(
                "Cannot record paper simulation result because plan "
                "lineage is invalid."
            ),
            errors=plan_lineage_issues,
            recommendation="reject",
        )

    unsafe = _unsafe_plan_flags(plan, plan_payload)
    if unsafe:
        return PaperSimulationResultArtifact(
            status="blocked_unsafe_plan",
            executor=executor,
            source_plan_id=plan.get("plan_id"),
            source_plan_path=plan_artifact_path,
            source_plan_sha256=plan_sha256,
            source_plan_fingerprint=plan_fingerprint,
            source_receipt_id=plan.get("source_receipt_id"),
            summary="Cannot record paper simulation result because the source plan contains unsafe flags.",
            errors=unsafe,
            recommendation="reject",
        )

    if plan.get("status") != "paper_simulation_plan_ready":
        return PaperSimulationResultArtifact(
            status="blocked_invalid_plan",
            executor=executor,
            source_plan_id=plan.get("plan_id"),
            source_plan_path=plan_artifact_path,
            source_plan_sha256=plan_sha256,
            source_plan_fingerprint=plan_fingerprint,
            source_receipt_id=plan.get("source_receipt_id"),
            summary=f"Cannot record paper simulation result because plan status is {plan.get('status')!r}.",
            errors=[f"paper simulation plan status is {plan.get('status')!r}, expected 'paper_simulation_plan_ready'"],
            recommendation="needs_more_data",
        )

    metric_items = [item if isinstance(item, PaperSimulationMetric) else PaperSimulationMetric(**item) for item in metrics]
    checks = _default_guardrail_checks()
    checks.update(guardrail_checks)
    normalized_metrics = [
        item.model_dump(mode="json") for item in metric_items
    ]
    external_payload = load_json_object(external_result_path)
    external_issues = _external_evidence_issues(
        payload=external_payload,
        external_result_path=external_result_path,
        executor=executor,
        source_plan_id=plan.get("plan_id"),
        source_plan_sha256=plan_sha256,
        requested_status=requested_status,
        summary=summary,
        metrics=normalized_metrics,
        warnings=warnings,
        errors=errors,
        artifacts=artifacts,
        guardrail_checks=checks,
    )
    if external_issues:
        blocked_status: PaperSimulationResultStatus = (
            "blocked_missing_external_evidence"
            if external_payload is None
            else "blocked_external_evidence_mismatch"
        )
        return PaperSimulationResultArtifact(
            status=blocked_status,
            executor=executor,
            source_plan_id=plan.get("plan_id"),
            source_plan_path=plan_artifact_path,
            source_plan_sha256=plan_sha256,
            source_plan_fingerprint=plan_fingerprint,
            source_receipt_id=plan.get("source_receipt_id"),
            external_result_path=external_result_path,
            external_result_sha256=file_sha256(
                external_result_path
            ),
            external_result_fingerprint=(
                external_payload.get("output_fingerprint")
                if external_payload
                else None
            ),
            summary=(
                "Cannot record a completed paper simulation without one "
                "matching immutable isolated-executor output."
            ),
            errors=external_issues,
            recommendation="needs_more_data",
        )

    final_status: PaperSimulationResultStatus = requested_status
    if requested_status == "completed" and (warnings or not all(checks.values())):
        final_status = "completed_with_warnings"

    recommendation = _recommendation(final_status, errors, warnings, checks)
    return PaperSimulationResultArtifact(
        status=final_status,
        executor=executor,
        source_plan_id=plan.get("plan_id"),
        source_plan_path=plan_artifact_path,
        source_plan_sha256=plan_sha256,
        source_plan_fingerprint=plan_fingerprint,
        source_receipt_id=plan.get("source_receipt_id"),
        external_result_path=external_result_path,
        external_result_sha256=file_sha256(external_result_path),
        external_result_fingerprint=(
            external_payload.get("output_fingerprint")
            if external_payload
            else None
        ),
        lineage_verified=True,
        summary=summary,
        metrics=metric_items,
        warnings=warnings,
        errors=errors,
        artifacts=artifacts,
        guardrail_checks=checks,
        recommendation=recommendation,
        review_required=True,
        paper_simulation_executed_by_this_layer=False,
        live_execution_allowed=False,
        broker_access_allowed=False,
        production_config_write_allowed=False,
        learning_memory_write_allowed=False,
        model_promotion_allowed=False,
    )


def render_paper_simulation_result_markdown(payload: dict[str, Any]) -> str:
    result = payload.get("paper_simulation_result") or {}
    plan = payload.get("source_plan_summary") or {}
    lines = [
        "# DEAN-OS Paper Simulation Result",
        "",
        f"- Result ID: `{result.get('result_id')}`",
        f"- Status: `{result.get('status')}`",
        f"- Recommendation: `{result.get('recommendation')}`",
        f"- Executor: `{result.get('executor')}`",
        f"- Source plan: `{result.get('source_plan_id')}`",
        f"- Source plan SHA256: `{result.get('source_plan_sha256')}`",
        f"- Source receipt: `{result.get('source_receipt_id')}`",
        f"- External result SHA256: `{result.get('external_result_sha256')}`",
        f"- Lineage verified: `{result.get('lineage_verified')}`",
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
            "This result records an external isolated paper simulation. It does not approve live trading, broker access, model promotion, production config writes, or learning-memory writes.",
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
            messages.append(f"paper_simulation_plan.{flag}=true")
        if safety.get(flag) is True:
            messages.append(f"safety.{flag}=true")
    if plan.get("paper_simulation_executed") is True or safety.get("paper_simulation_executed") is True:
        messages.append("paper_simulation_executed=true")
    return messages


def _recommendation(
    status: PaperSimulationResultStatus,
    errors: list[str],
    warnings: list[str],
    checks: dict[str, bool],
) -> PaperSimulationRecommendation:
    if status in {
        "blocked_invalid_plan",
        "blocked_unsafe_plan",
        "blocked_external_evidence_mismatch",
        "failed",
    } or errors:
        return "reject"
    if status == "blocked_missing_external_evidence":
        return "needs_more_data"
    if warnings or not all(checks.values()) or status == "completed_with_warnings":
        return "rerun_paper_simulation"
    return "ready_for_review"


def _plan_summary(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("paper_simulation_plan") if isinstance(payload.get("paper_simulation_plan"), dict) else {}
    return {
        "mode": payload.get("mode"),
        "plan_id": plan.get("plan_id"),
        "status": plan.get("status"),
        "source_receipt_id": plan.get("source_receipt_id"),
        "source_decision": plan.get("source_decision"),
        "source_artifact_decision": plan.get("source_artifact_decision"),
        "source_receipt_sha256": plan.get("source_receipt_sha256"),
        "lineage_verified": plan.get("lineage_verified"),
        "target": plan.get("target"),
        "step_count": len(plan.get("steps") or []),
    }


def _plan_lineage_issues(
    *,
    plan_payload: dict[str, Any],
    plan_artifact_path: str | None,
) -> list[str]:
    issues: list[str] = []
    if plan_payload.get("mode") != "paper_simulation_plan":
        issues.append("plan_mode_invalid")
    if (
        plan_payload.get("schema_version")
        != PAPER_LIFECYCLE_SCHEMA_VERSION
    ):
        issues.append("plan_schema_version_invalid")
    if not fingerprint_matches(
        plan_payload,
        object_key="paper_simulation_plan",
        fingerprint_key="plan_fingerprint",
    ):
        issues.append("plan_fingerprint_invalid")
    if not valid_sha256(file_sha256(plan_artifact_path)):
        issues.append("plan_file_sha256_unavailable")
    plan = plan_payload.get("paper_simulation_plan") or {}
    if plan.get("lineage_verified") is not True:
        issues.append("plan_lineage_not_verified")
    if not valid_sha256(plan.get("source_receipt_sha256")):
        issues.append("plan_source_receipt_sha256_missing")
    elif (
        file_sha256(plan.get("source_receipt_path"))
        != plan.get("source_receipt_sha256")
    ):
        issues.append("plan_source_receipt_sha256_mismatch")
    if not valid_sha256(plan.get("source_artifact_sha256")):
        issues.append("plan_source_artifact_sha256_missing")
    elif (
        file_sha256(plan.get("source_artifact_path"))
        != plan.get("source_artifact_sha256")
    ):
        issues.append("plan_source_artifact_sha256_mismatch")
    receipt_payload = load_json_object(plan.get("source_receipt_path"))
    if receipt_payload is None:
        issues.append("plan_source_receipt_unavailable")
    else:
        if (
            receipt_payload.get("receipt_fingerprint")
            != plan.get("source_receipt_fingerprint")
        ):
            issues.append("plan_source_receipt_fingerprint_mismatch")
        issues.extend(
            f"plan_{issue}"
            for issue in receipt_lineage_issues(
                receipt_payload,
                receipt_path=plan.get("source_receipt_path"),
            )
        )
    return sorted(set(issues))


def _external_evidence_issues(
    *,
    payload: dict[str, Any] | None,
    external_result_path: str | None,
    executor: str,
    source_plan_id: str | None,
    source_plan_sha256: str | None,
    requested_status: str,
    summary: str,
    metrics: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    artifacts: list[str],
    guardrail_checks: dict[str, bool],
) -> list[str]:
    if payload is None:
        return ["external_result_manifest_missing_or_invalid"]
    issues: list[str] = []
    if payload.get("mode") != "isolated_paper_simulation_output":
        issues.append("external_result_mode_invalid")
    if payload.get("schema_version") != PAPER_LIFECYCLE_SCHEMA_VERSION:
        issues.append("external_result_schema_version_invalid")
    content = payload.get("output")
    if not isinstance(content, dict):
        return [*issues, "external_result_output_missing"]
    if not fingerprint_matches(
        payload,
        object_key="output",
        fingerprint_key="output_fingerprint",
    ):
        issues.append("external_result_fingerprint_invalid")
    expected = {
        "executor": executor,
        "source_plan_id": source_plan_id,
        "source_plan_sha256": source_plan_sha256,
        "status": requested_status,
        "summary": summary,
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
        "artifacts": artifacts,
        "guardrail_checks": guardrail_checks,
    }
    for key, value in expected.items():
        if content.get(key) != value:
            issues.append(f"external_result_{key}_mismatch")
    safety = content.get("safety")
    if not isinstance(safety, dict):
        issues.append("external_result_safety_missing")
    else:
        for key in (
            "live_execution_performed",
            "broker_access_performed",
            "production_config_write_performed",
            "learning_write_performed",
            "model_promotion_performed",
        ):
            if safety.get(key) is not False:
                issues.append(f"external_result_{key}_not_false")
    if not valid_sha256(file_sha256(external_result_path)):
        issues.append("external_result_file_sha256_unavailable")
    return sorted(set(issues))
