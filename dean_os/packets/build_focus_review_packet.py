from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_ALIGNMENT_REVIEW_JSON = "reports/dean_os/current_system_alignment_review_two_branch_current/latest.json"
DEFAULT_TEMPLATE_STANDARDIZATION_JSON = "reports/dean_os/domain_analyst_template_standardization_packet_current/latest.json"
DEFAULT_CASE_REGISTRY_JSON = "reports/dean_os/domain_analyst_case_registry_packet_current/latest.json"
DEFAULT_PIPELINE_CONTROL_INSTANCE_JSON = "reports/dean_os/pipeline_control_instance_contract_current/latest.json"


class BuildFocusReviewPacket:
    """Review-only guardrail against unproductive implementation deepening."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/build_focus_review_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        alignment_review_json: str | Path = DEFAULT_ALIGNMENT_REVIEW_JSON,
        template_standardization_json: str | Path | None = DEFAULT_TEMPLATE_STANDARDIZATION_JSON,
        case_registry_json: str | Path | None = DEFAULT_CASE_REGISTRY_JSON,
        pipeline_control_instance_json: str | Path | None = DEFAULT_PIPELINE_CONTROL_INSTANCE_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        alignment = _load_json(alignment_review_json)
        template = _load_optional_json(template_standardization_json)
        case_registry = _load_optional_json(case_registry_json)
        pipeline = _load_optional_json(pipeline_control_instance_json)
        assessment = _branch_assessment(alignment, template, case_registry, pipeline)
        checks = _review_checks(alignment, template, case_registry, pipeline, assessment)
        decision = _decision(assessment, checks)
        payload = {
            "run_id": _run_id("build_focus_review_packet"),
            "created_at": utc_now_iso(),
            "mode": "build_focus_review_packet",
            "inputs": {
                "alignment_review_json": str(alignment_review_json),
                "template_standardization_json": str(template_standardization_json) if template_standardization_json else None,
                "case_registry_json": str(case_registry_json) if case_registry_json else None,
                "pipeline_control_instance_json": str(pipeline_control_instance_json) if pipeline_control_instance_json else None,
            },
            "summary": _summary(decision, checks),
            "decision_rubric": _decision_rubric(),
            "branch_assessment": assessment,
            "review_checks": checks,
            "decision_guidance": _decision_guidance(decision, checks, assessment),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(decision, assessment),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_build_focus_review_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_build_focus_review_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Build Focus Review Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Focus status: `{summary.get('focus_status')}`",
        f"- Recommended next operation: `{summary.get('recommended_next_operation')}`",
        f"- Current deepening assessment: `{summary.get('current_deepening_assessment')}`",
        f"- Should stop adding domain template gates: {summary.get('should_stop_adding_domain_template_gates')}",
        f"- Manual review useful now: {summary.get('manual_review_useful_now')}",
        f"- Should switch to pipeline-control blockers: {summary.get('should_switch_to_pipeline_control_blockers')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Decision Rubric",
        "",
    ]
    for section, items in payload.get("decision_rubric", {}).items():
        lines.append(f"### {section.replace('_', ' ').title()}")
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    lines.extend(["## Branch Assessment", ""])
    for branch_id, branch in payload.get("branch_assessment", {}).items():
        lines.append(f"- `{branch_id}`: {branch.get('status')} - {branch.get('recommendation')}")
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Decision Guidance", ""])
    lines.extend(f"- {item}" for item in guidance.get("reasons", []))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _summary(decision: dict[str, Any], checks: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "focus_status": decision["focus_status"],
        "recommended_next_operation": decision["recommended_next_operation"],
        "current_deepening_assessment": decision["current_deepening_assessment"],
        "manual_review_useful_now": decision["manual_review_useful_now"],
        "should_stop_adding_domain_template_gates": decision["should_stop_adding_domain_template_gates"],
        "should_switch_to_pipeline_control_blockers": decision["should_switch_to_pipeline_control_blockers"],
        "can_continue_domain_branch_only_for_outcome_lane": decision["can_continue_domain_branch_only_for_outcome_lane"],
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
        "pass_count": sum(1 for check in checks if check["status"] == "pass"),
        "warning_count": sum(1 for check in checks if check["status"] == "warn"),
        "fail_count": sum(1 for check in checks if check["status"] == "fail"),
    }


def _decision_rubric() -> dict[str, list[str]]:
    return {
        "productive_deepening": [
            "It closes a named blocker or removes a real ambiguity in the next operation.",
            "It creates a reusable boundary, contract, or test that prevents a likely future mistake.",
            "It changes the next decision, not only the amount of wording around the same decision.",
            "It is tied to a known artifact gap: missing outcome buckets, missing metric inputs, missing direct ticker evidence, or missing manual acceptance.",
        ],
        "unproductive_deepening": [
            "It adds another report while the recommended next operation stays identical.",
            "It re-checks safety fields that are already covered without adding a new boundary.",
            "It deepens a branch that is already waiting for human review or external outcome data.",
            "It avoids switching to a known blocker in another branch.",
        ],
        "switch_or_pause_triggers": [
            "The current branch has a review-ready artifact and the next action is manual acceptance.",
            "The current branch needs future outcome data before it can learn more.",
            "Another branch has concrete blockers that block system progress.",
            "Full tests pass and the remaining work is a decision, not code.",
        ],
    }


def _branch_assessment(
    alignment: dict[str, Any],
    template: dict[str, Any] | None,
    case_registry: dict[str, Any] | None,
    pipeline: dict[str, Any] | None,
) -> dict[str, Any]:
    alignment_summary = alignment.get("summary", {})
    template_summary = template.get("summary", {}) if template else {}
    case_summary = case_registry.get("summary", {}) if case_registry else {}
    pipeline_summary = pipeline.get("summary", {}) if pipeline else {}
    return {
        "domain_analyst_branch": {
            "status": _domain_branch_status(template_summary, case_summary),
            "template_candidate_status": template_summary.get("candidate_status"),
            "case_registry_status": case_summary.get("registry_status"),
            "recommendation": _domain_branch_recommendation(template_summary, case_summary),
            "productive_deepening_left": _domain_productive_deepening_left(template_summary, case_summary),
        },
        "pipeline_control_branch": {
            "status": pipeline_summary.get("instance_status") or "unknown",
            "blocked_metric_planes": pipeline_summary.get("blocked_metric_planes", []),
            "caution_metric_planes": pipeline_summary.get("caution_metric_planes", []),
            "recommendation": _pipeline_recommendation(pipeline_summary),
            "productive_deepening_left": bool(pipeline_summary.get("blocked_metric_planes") or pipeline_summary.get("caution_metric_planes")),
        },
        "orchestrator_branch": {
            "status": "waiting_for_branch_decisions",
            "alignment_status": alignment_summary.get("alignment_status"),
            "recommended_action": alignment_summary.get("recommended_action"),
            "recommendation": "Keep orchestration review-only until analyst template acceptance and pipeline-control blockers are clearer.",
            "productive_deepening_left": False,
        },
    }


def _domain_branch_status(template_summary: dict[str, Any], case_summary: dict[str, Any]) -> str:
    if template_summary.get("candidate_status") in {
        "ready_for_manual_template_acceptance",
        "ready_for_manual_template_acceptance_with_cautions",
    } and str(case_summary.get("registry_status", "")).startswith("case_registry_ready"):
        return "review_ready_do_not_add_more_template_gates"
    if not case_summary:
        return "needs_case_registry_before_learning"
    return "needs_review"


def _domain_branch_recommendation(template_summary: dict[str, Any], case_summary: dict[str, Any]) -> str:
    if not case_summary:
        return "Build case registry before learning promotion."
    if template_summary.get("can_mark_template_accepted_now") is False and str(case_summary.get("registry_status", "")).startswith("case_registry_ready"):
        return "Stop adding domain-template gates; either record manual acceptance or wait for outcome data."
    return "Review domain branch artifacts before additional implementation."


def _domain_productive_deepening_left(template_summary: dict[str, Any], case_summary: dict[str, Any]) -> list[str]:
    remaining: list[str] = []
    if not case_summary:
        remaining.append("case_registry")
    if str(case_summary.get("registry_status", "")).endswith("pending_outcomes"):
        remaining.append("outcome_evaluation_attachment")
    if template_summary.get("candidate_status") in {
        "ready_for_manual_template_acceptance",
        "ready_for_manual_template_acceptance_with_cautions",
    }:
        remaining.append("manual_acceptance_decision")
    return remaining


def _pipeline_recommendation(summary: dict[str, Any]) -> str:
    blocked = summary.get("blocked_metric_planes", [])
    if blocked:
        return "Switch focus to concrete metric blockers: " + ", ".join(blocked) + "."
    if summary.get("caution_metric_planes"):
        return "Review caution metric planes before tuning proposals."
    if summary.get("instance_status") in {"pipeline_control_instance_review_ready", "pipeline_control_instance_review_ready_with_cautions"}:
        return "Pipeline-control can proceed by reviewed proposal only."
    return "Attach pipeline-control instance artifact before deciding."


def _review_checks(
    alignment: dict[str, Any],
    template: dict[str, Any] | None,
    case_registry: dict[str, Any] | None,
    pipeline: dict[str, Any] | None,
    assessment: dict[str, Any],
) -> list[dict[str, str]]:
    alignment_summary = alignment.get("summary", {})
    template_summary = template.get("summary", {}) if template else {}
    case_summary = case_registry.get("summary", {}) if case_registry else {}
    pipeline_summary = pipeline.get("summary", {}) if pipeline else {}
    checks = [
        _check("pass" if alignment.get("mode") == "current_system_alignment_review" else "fail", "alignment_artifact_type", str(alignment.get("mode"))),
        _check("pass" if int(alignment_summary.get("blocker_count") or 0) == 0 else "fail", "alignment_has_no_blockers", str(alignment_summary.get("blocker_count"))),
        _check(
            "pass"
            if template_summary.get("candidate_status")
            in {"ready_for_manual_template_acceptance", "ready_for_manual_template_acceptance_with_cautions"}
            else "warn",
            "domain_template_waits_for_manual_review",
            str(template_summary.get("candidate_status")),
        ),
        _check(
            "pass" if str(case_summary.get("registry_status", "")).startswith("case_registry_ready") else "warn",
            "case_registry_prevents_hits_only_learning",
            str(case_summary.get("registry_status")),
        ),
        _check(
            "pass" if assessment["domain_analyst_branch"]["status"] == "review_ready_do_not_add_more_template_gates" else "warn",
            "domain_branch_should_not_deepen_template_gates",
            assessment["domain_analyst_branch"]["status"],
        ),
        _pipeline_focus_check(pipeline_summary),
        _must_be_false(template_summary, "can_trade", "template_no_trading"),
    ]
    if case_summary:
        checks.extend(
            [
                _must_be_false(case_summary, "can_write_learning_memory", "case_registry_no_learning_write"),
                _must_be_false(case_summary, "can_train_from_hits_only", "case_registry_no_hits_only_training"),
                _must_be_false(case_summary, "can_trade", "case_registry_no_trading"),
            ]
        )
    return checks


def _decision(assessment: dict[str, Any], checks: list[dict[str, str]]) -> dict[str, Any]:
    if any(check["status"] == "fail" for check in checks):
        return {
            "focus_status": "focus_blocked",
            "recommended_next_operation": "fix_boundary_or_alignment_failures",
            "current_deepening_assessment": "unsafe_to_continue_until_failures_are_fixed",
            "manual_review_useful_now": False,
            "should_stop_adding_domain_template_gates": True,
            "should_switch_to_pipeline_control_blockers": False,
            "can_continue_domain_branch_only_for_outcome_lane": False,
        }
    domain = assessment["domain_analyst_branch"]
    pipeline = assessment["pipeline_control_branch"]
    if domain["status"] == "review_ready_do_not_add_more_template_gates":
        recommended_next_operation = "manual_template_acceptance_or_orchestrator_review"
        if pipeline.get("blocked_metric_planes"):
            recommended_next_operation = "manual_template_acceptance_or_switch_to_pipeline_control_blockers"
        elif pipeline.get("caution_metric_planes"):
            recommended_next_operation = "manual_template_acceptance_or_review_pipeline_cautions"
        return {
            "focus_status": "focus_review_ready",
            "recommended_next_operation": recommended_next_operation,
            "current_deepening_assessment": "more_domain_template_gates_have_diminishing_returns",
            "manual_review_useful_now": True,
            "should_stop_adding_domain_template_gates": True,
            "should_switch_to_pipeline_control_blockers": bool(pipeline.get("blocked_metric_planes")),
            "can_continue_domain_branch_only_for_outcome_lane": "outcome_evaluation_attachment" in domain.get("productive_deepening_left", []),
        }
    return {
        "focus_status": "focus_needs_more_review",
        "recommended_next_operation": "complete_missing_focus_prerequisite",
        "current_deepening_assessment": "limited_deepening_allowed_only_for_named_missing_prerequisite",
        "manual_review_useful_now": False,
        "should_stop_adding_domain_template_gates": False,
        "should_switch_to_pipeline_control_blockers": bool(pipeline.get("blocked_metric_planes")),
        "can_continue_domain_branch_only_for_outcome_lane": False,
    }


def _decision_guidance(
    decision: dict[str, Any],
    checks: list[dict[str, str]],
    assessment: dict[str, Any],
) -> dict[str, Any]:
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    failures = [check["code"] for check in checks if check["status"] == "fail"]
    reasons = [
        f"Focus status is {decision['focus_status']}.",
        f"Domain branch: {assessment['domain_analyst_branch']['recommendation']}",
        f"Pipeline branch: {assessment['pipeline_control_branch']['recommendation']}",
        "A useful deepening step must change a next decision or close a named blocker.",
    ]
    if warnings:
        reasons.append("Warnings: " + ", ".join(warnings) + ".")
    if failures:
        reasons.append("Failures: " + ", ".join(failures) + ".")
    return {
        "recommended_next_operation": decision["recommended_next_operation"],
        "pass_count": sum(1 for check in checks if check["status"] == "pass"),
        "warning_count": len(warnings),
        "fail_count": len(failures),
        "reasons": reasons,
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No domain template is accepted.",
        "No sector-to-ticker bridge is executed.",
        "No learning memory, analyst-weight, model, or config write is performed.",
        "No recommendation, allocation, paper order, broker call, or live trade is generated.",
        "No new domain analyst is cloned.",
    ]


def _operator_next_steps(decision: dict[str, Any], assessment: dict[str, Any]) -> list[str]:
    if decision["focus_status"] == "focus_blocked":
        return ["Fix failed focus checks before adding more implementation layers."]
    steps = []
    if decision["manual_review_useful_now"]:
        steps.append("Stop adding domain-template gates and manually review the template standardization packet.")
    if decision["should_switch_to_pipeline_control_blockers"]:
        steps.append("Switch implementation focus to pipeline-control blockers once manual analyst review is parked or accepted.")
    elif assessment["pipeline_control_branch"].get("caution_metric_planes"):
        steps.append("Review pipeline-control caution planes before proposing bounded tuning experiments.")
    if decision["can_continue_domain_branch_only_for_outcome_lane"]:
        steps.append("Domain-branch coding is still useful only if it attaches real outcome evaluation to the case registry.")
    steps.append("Before each new implementation step, require a named blocker, missing artifact, or changed downstream decision.")
    return steps


def _pipeline_focus_check(summary: dict[str, Any]) -> dict[str, str]:
    blocked = summary.get("blocked_metric_planes", [])
    cautions = summary.get("caution_metric_planes", [])
    status = summary.get("instance_status")
    if blocked:
        return _check("pass", "pipeline_branch_has_concrete_blockers", ", ".join(blocked))
    if status in {"pipeline_control_instance_review_ready", "pipeline_control_instance_review_ready_with_cautions"}:
        if cautions:
            return _check("pass", "pipeline_branch_review_ready_with_cautions", ", ".join(cautions))
        return _check("pass", "pipeline_branch_review_ready", str(status))
    return _check("warn", "pipeline_branch_needs_instance_or_metric_inputs", str(status))


def _must_be_false(summary: dict[str, Any], field: str, code: str) -> dict[str, str]:
    if summary.get(field) is False:
        return _check("pass", code, f"{field}=False.")
    return _check("fail", code, f"{field} must stay False, got {summary.get(field)!r}.")


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return _load_json(resolved)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
