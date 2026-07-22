from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

TemplateDecision = Literal["pending_review", "accept_template", "reject_template", "needs_revision"]

DEFAULT_VERTICAL_SLICE_JSON = "reports/dean_os/domain_analyst_vertical_slice_current/latest.json"
DEFAULT_TEMPLATE_STANDARDIZATION_JSON = "reports/dean_os/domain_analyst_template_standardization_packet_current/latest.json"
DEFAULT_FORECAST_REVIEW_JSON = "reports/dean_os/domain_analyst_forecast_review_packet_current/latest.json"
DEFAULT_CASE_REGISTRY_JSON = "reports/dean_os/domain_analyst_case_registry_packet_current/latest.json"
DEFAULT_PORTABILITY_REVIEW_JSON = "reports/dean_os/domain_analyst_portability_review_current/latest.json"
DEFAULT_ARCHITECTURE_MAP_JSON = "reports/dean_os/current_architecture_map_current/latest.json"


class DomainAnalystTemplateDecisionPacket:
    """Human decision receipt for accepting or rejecting one analyst template.

    This packet records a process/template decision only. It does not declare the
    thesis true, score the forecast, write learning memory, clone profiles,
    change config, recommend trades, or execute anything.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_template_decision_packet_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        vertical_slice_json: str | Path = DEFAULT_VERTICAL_SLICE_JSON,
        template_standardization_json: str | Path = DEFAULT_TEMPLATE_STANDARDIZATION_JSON,
        forecast_review_json: str | Path = DEFAULT_FORECAST_REVIEW_JSON,
        case_registry_json: str | Path = DEFAULT_CASE_REGISTRY_JSON,
        portability_review_json: str | Path = DEFAULT_PORTABILITY_REVIEW_JSON,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        decision: TemplateDecision = "pending_review",
        reviewer: str = "human",
        rationale: str = "",
        required_followups: list[str] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        vertical = _load_artifact(vertical_slice_json)
        template = _load_artifact(template_standardization_json)
        forecast = _load_artifact(forecast_review_json)
        registry = _load_artifact(case_registry_json)
        portability = _load_artifact(portability_review_json)
        architecture = _load_artifact(architecture_map_json) if architecture_map_json else {}
        checks = _review_checks(
            vertical=vertical,
            template=template,
            forecast=forecast,
            registry=registry,
            portability=portability,
            architecture=architecture,
            decision=decision,
            rationale=rationale,
        )
        status = _decision_status(decision, checks)
        payload = {
            "run_id": _run_id("domain_analyst_template_decision_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_template_decision_packet",
            "inputs": {
                "vertical_slice_json": str(vertical_slice_json),
                "template_standardization_json": str(template_standardization_json),
                "forecast_review_json": str(forecast_review_json),
                "case_registry_json": str(case_registry_json),
                "portability_review_json": str(portability_review_json),
                "architecture_map_json": str(architecture_map_json) if architecture_map_json else None,
                "decision": decision,
                "reviewer": reviewer,
            },
            "summary": _summary(status, decision, vertical, template, forecast, registry, portability),
            "manual_decision": {
                "reviewer": reviewer,
                "decision": decision,
                "rationale": rationale,
                "required_followups": required_followups or [],
                "scope": "reusable_domain_analyst_process_only",
                "does_not_assert_thesis_truth": True,
                "does_not_score_forecast_outcome": True,
            },
            "decision_interpretation": _decision_interpretation(decision, status),
            "review_inputs": _review_inputs(vertical, template, forecast, registry, portability),
            "allowed_after_decision": _allowed_after_decision(decision, status),
            "blocked_after_decision": _blocked_after_decision(),
            "review_checks": checks,
            "operator_next_steps": _operator_next_steps(decision, status, checks),
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_template_decision_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_template_decision_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    decision = payload.get("manual_decision", {})
    interpretation = payload.get("decision_interpretation", {})
    lines = [
        "# DEAN-OS Domain Analyst Template Decision Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Decision status: `{summary.get('decision_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Reviewer: `{decision.get('reviewer')}`",
        f"- Decision: `{decision.get('decision')}`",
        f"- Scope: `{decision.get('scope')}`",
        f"- Template accepted: {summary.get('template_accepted')}",
        f"- Template rejected: {summary.get('template_rejected')}",
        f"- Needs revision: {summary.get('needs_revision')}",
        f"- Can clone one next domain profile candidate: {summary.get('can_clone_one_next_domain_profile_candidate')}",
        f"- Can scale all domains now: {summary.get('can_scale_all_domains_now')}",
        f"- Can create analyst research recommendation: {summary.get('can_create_analyst_research_recommendation')}",
        f"- Can create execution recommendation: {summary.get('can_create_execution_recommendation')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Meaning",
        "",
        f"- {interpretation.get('meaning')}",
        f"- {interpretation.get('not_meaning')}",
        "",
        "## Rationale",
        "",
        decision.get("rationale") or "No rationale supplied.",
        "",
        "## Review Inputs",
        "",
    ]
    for item in payload.get("review_inputs", []):
        lines.append(f"- `{item.get('artifact')}`: status=`{item.get('status')}`, path=`{item.get('path')}`")
    lines.extend(["", "## Allowed After Decision", ""])
    lines.extend(f"- {item}" for item in payload.get("allowed_after_decision", []))
    lines.extend(["", "## Blocked After Decision", ""])
    lines.extend(f"- {item}" for item in payload.get("blocked_after_decision", []))
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _summary(
    status: str,
    decision: TemplateDecision,
    vertical: dict[str, Any],
    template: dict[str, Any],
    forecast: dict[str, Any],
    registry: dict[str, Any],
    portability: dict[str, Any],
) -> dict[str, Any]:
    vertical_summary = vertical.get("summary", {})
    template_summary = template.get("summary", {})
    forecast_summary = forecast.get("summary", {})
    registry_summary = registry.get("summary", {})
    portability_summary = portability.get("summary", {})
    accepted = status == "manual_template_accepted_review_only"
    return {
        "decision_status": status,
        "decision": decision,
        "domain_id": (
            vertical_summary.get("domain_id")
            or template_summary.get("domain_id")
            or forecast_summary.get("domain_id")
            or portability_summary.get("source_domain_id")
        ),
        "source_vertical_status": vertical_summary.get("run_status"),
        "template_candidate_status": template_summary.get("candidate_status"),
        "forecast_review_status": forecast_summary.get("packet_status"),
        "case_registry_status": registry_summary.get("registry_status"),
        "portability_review_status": portability_summary.get("review_status"),
        "manual_decision_recorded": decision != "pending_review",
        "template_accepted": accepted,
        "template_rejected": decision == "reject_template" and status == "manual_template_rejected",
        "needs_revision": decision == "needs_revision" and status == "manual_template_needs_revision",
        "can_clone_one_next_domain_profile_candidate": accepted,
        "can_scale_all_domains_now": False,
        "can_run_sector_to_ticker_bridge_now": False,
        "can_create_analyst_research_recommendation": True,
        "can_create_analyst_self_improvement_proposal": True,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _review_inputs(
    vertical: dict[str, Any],
    template: dict[str, Any],
    forecast: dict[str, Any],
    registry: dict[str, Any],
    portability: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _input_item("domain_analyst_vertical_slice", vertical, vertical.get("summary", {}).get("run_status")),
        _input_item("template_standardization", template, template.get("summary", {}).get("candidate_status")),
        _input_item("forecast_review", forecast, forecast.get("summary", {}).get("packet_status")),
        _input_item("case_registry", registry, registry.get("summary", {}).get("registry_status")),
        _input_item("portability_review", portability, portability.get("summary", {}).get("review_status")),
    ]


def _input_item(artifact: str, payload: dict[str, Any], status: Any) -> dict[str, Any]:
    saved = payload.get("saved_paths", {}) if isinstance(payload.get("saved_paths"), dict) else {}
    inputs = payload.get("_load_source", {})
    return {
        "artifact": artifact,
        "mode": payload.get("mode"),
        "run_id": payload.get("run_id"),
        "status": status,
        "path": saved.get("latest_json") or inputs.get("path"),
    }


def _review_checks(
    *,
    vertical: dict[str, Any],
    template: dict[str, Any],
    forecast: dict[str, Any],
    registry: dict[str, Any],
    portability: dict[str, Any],
    architecture: dict[str, Any],
    decision: TemplateDecision,
    rationale: str,
) -> list[dict[str, str]]:
    vertical_summary = vertical.get("summary", {})
    template_summary = template.get("summary", {})
    forecast_summary = forecast.get("summary", {})
    registry_summary = registry.get("summary", {})
    portability_summary = portability.get("summary", {})
    architecture_summary = architecture.get("summary", {})
    audit = vertical.get("synthetic_fixture_audit", {})

    checks = [
        _check(
            "pass" if vertical.get("mode") == "domain_analyst_vertical_slice_run" else "fail",
            "vertical_slice_artifact_type",
            str(vertical.get("mode")),
        ),
        _check(
            "pass" if vertical_summary.get("run_status") == "domain_analyst_candidate_complete_pending_manual_acceptance" else "warn",
            "vertical_candidate_complete",
            str(vertical_summary.get("run_status")),
        ),
        _check("pass" if audit.get("has_synthetic_marker") is False else "fail", "vertical_no_synthetic_marker", str(audit.get("has_synthetic_marker"))),
        _check("pass" if audit.get("has_fixture_marker") is False else "fail", "vertical_no_fixture_marker", str(audit.get("has_fixture_marker"))),
        _check(
            "pass" if template.get("mode") == "domain_analyst_template_standardization_packet" else "fail",
            "template_standardization_artifact_type",
            str(template.get("mode")),
        ),
        _check(
            "pass" if template_summary.get("candidate_status") == "ready_for_manual_template_acceptance" else "fail",
            "template_ready_for_manual_decision",
            str(template_summary.get("candidate_status")),
        ),
        _must_not_be_true(template_summary, "can_mark_template_accepted_now", "template_not_auto_accepted"),
        _check(
            "pass" if str(forecast_summary.get("packet_status", "")).startswith("forecast_review_ready") else "fail",
            "forecast_review_ready",
            str(forecast_summary.get("packet_status")),
        ),
        _check(
            "pass" if int(forecast_summary.get("forecast_candidate_count") or 0) > 0 else "fail",
            "frozen_forecast_expectation_present",
            str(forecast_summary.get("forecast_candidate_count")),
        ),
        _check(
            "pass" if forecast_summary.get("can_create_analyst_research_recommendation") is not False else "fail",
            "forecast_allows_review_only_analyst_recommendations",
            "Review-only analyst recommendations are allowed.",
        ),
        _must_not_be_true(forecast_summary, "can_create_execution_recommendation", "forecast_no_execution_recommendation"),
        _check(
            "pass" if registry_summary.get("registry_status") == "case_registry_ready_pending_outcomes" else "warn",
            "case_registry_ready_or_pending",
            str(registry_summary.get("registry_status")),
        ),
        _check(
            "pass" if int(registry_summary.get("expectation_case_count") or registry_summary.get("cases", 0) or 0) > 0 else "warn",
            "pending_expectation_case_visible",
            str(registry_summary.get("expectation_case_count") or registry_summary.get("cases")),
        ),
        _check(
            "pass" if portability_summary.get("review_status") == "domain_analyst_portability_review_ready" else "fail",
            "portability_review_ready",
            str(portability_summary.get("review_status")),
        ),
        _must_not_be_true(portability_summary, "can_clone_domain_profiles_now", "portability_did_not_clone"),
        _must_not_be_true(architecture_summary, "can_clone_domain_profiles_now", "architecture_no_auto_clone"),
        _must_not_be_true(architecture_summary, "can_trade", "architecture_no_trading"),
        _check("pass" if decision in _VALID_DECISIONS else "fail", "decision_value_valid", decision),
    ]
    if decision == "pending_review":
        checks.append(_check("warn", "manual_decision_pending", "No accept/reject decision has been recorded yet."))
    elif not rationale.strip():
        checks.append(_check("fail", "decision_rationale_required", "Accept, reject, and needs-revision decisions require rationale."))
    else:
        checks.append(_check("pass", "decision_rationale_supplied", "Decision rationale is present."))
    if decision == "accept_template":
        checks.append(
            _check(
                "pass" if template_summary.get("can_standardize_domain_template_after_manual_acceptance") is True else "fail",
                "accepted_template_was_standardizable",
                str(template_summary.get("can_standardize_domain_template_after_manual_acceptance")),
            )
        )
    return checks


def _decision_status(decision: TemplateDecision, checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "manual_template_decision_blocked"
    if decision == "pending_review":
        return "manual_template_decision_pending"
    if decision == "accept_template":
        return "manual_template_accepted_review_only"
    if decision == "reject_template":
        return "manual_template_rejected"
    if decision == "needs_revision":
        return "manual_template_needs_revision"
    return "manual_template_decision_blocked"


def _decision_interpretation(decision: TemplateDecision, status: str) -> dict[str, str]:
    if status == "manual_template_accepted_review_only":
        return {
            "meaning": "The reusable analyst process/template is accepted as a source pattern for one next domain clone candidate.",
            "not_meaning": "This does not mean the semiconductor thesis is true, the forecast is scored, or any trade/action is approved.",
        }
    if status == "manual_template_rejected":
        return {
            "meaning": "The reusable analyst process/template is rejected; keep the artifacts for audit and do not clone from it.",
            "not_meaning": "This does not delete evidence, rewrite history, or score the thesis outcome.",
        }
    if status == "manual_template_needs_revision":
        return {
            "meaning": "The reusable analyst process/template needs changes before clone review can continue.",
            "not_meaning": "This does not authorize automatic repair, learning writes, or profile cloning.",
        }
    return {
        "meaning": f"Decision is {decision}; the template gate is still not accepted.",
        "not_meaning": "No downstream clone, ticker bridge, learning write, config change, or trade is authorized.",
    }


def _allowed_after_decision(decision: TemplateDecision, status: str) -> list[str]:
    base = [
        "Continue producing review-only analyst summaries, research recommendations, evidence requests, and self-improvement proposals.",
        "Keep the frozen expectation in the case registry for future outcome review.",
    ]
    if status == "manual_template_accepted_review_only":
        return [
            *base,
            "Prepare exactly one next-domain clone candidate by changing profile slots and local source paths only.",
            "Run the cloned domain through the same source gate, intake, thesis review, forecast review, case registry, and portability checks.",
        ]
    if status == "manual_template_rejected":
        return [
            *base,
            "Use the rejected packet as audit history and design feedback only.",
        ]
    if status == "manual_template_needs_revision":
        return [
            *base,
            "Create a revision plan before any domain cloning attempt.",
        ]
    return base


def _blocked_after_decision() -> list[str]:
    return [
        "No thesis truth assertion or outcome scoring is performed by this packet.",
        "No additional domain analyst profile is cloned or enabled automatically.",
        "No sector-to-ticker bridge is executed.",
        "No learning memory, analyst weight, model, prompt, or production config is changed.",
        "No execution, buy/sell/hold, sizing, allocation, order, paper-trade, broker, or live-trade recommendation is generated.",
    ]


def _operator_next_steps(decision: TemplateDecision, status: str, checks: list[dict[str, str]]) -> list[str]:
    failures = [check["code"] for check in checks if check["status"] == "fail"]
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    if failures:
        return ["Fix failed decision checks before recording a template decision: " + ", ".join(failures) + "."]
    if status == "manual_template_accepted_review_only":
        return [
            "Start one next-domain clone candidate only through portable profile slots and local source paths.",
            "Keep all analyst recommendations review-only until later outcome and learning gates approve changes.",
        ]
    if status == "manual_template_rejected":
        return ["Do not clone from this template. Preserve the packet as audit feedback and decide whether to revise or abandon the pattern."]
    if status == "manual_template_needs_revision":
        return ["Revise the analyst template issues captured in the rationale, then rerun the vertical slice and decision packet."]
    steps = ["Review the template artifacts and record accept_template, reject_template, or needs_revision when ready."]
    if warnings:
        steps.append("Current non-blocking warnings: " + ", ".join(warnings) + ".")
    return steps


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No source, evidence, forecast, or case artifact is rewritten.",
        "No analyst learning memory is written.",
        "No production config or profile file is modified.",
        "No GPT or FinBERT call is made.",
        "No recommendation to buy, sell, hold, size, allocate, route, paper trade, or live trade is generated.",
    ]


def _must_not_be_true(summary: dict[str, Any], key: str, code: str) -> dict[str, str]:
    return _check(
        "pass" if summary.get(key) is not True else "fail",
        code,
        f"{key} must not be True; got {summary.get(key)!r}.",
    )


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _load_artifact(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists():
        return {"mode": "missing_artifact", "_load_source": {"path": str(source_path)}, "errors": [f"Missing artifact: {source_path}"]}
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"mode": "invalid_artifact", "_load_source": {"path": str(source_path)}, "errors": [repr(exc)]}
    if not isinstance(payload, dict):
        return {"mode": "invalid_artifact", "_load_source": {"path": str(source_path)}, "errors": ["JSON is not an object"]}
    payload.setdefault("_load_source", {"path": str(source_path)})
    return payload


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"


_VALID_DECISIONS = {"pending_review", "accept_template", "reject_template", "needs_revision"}
