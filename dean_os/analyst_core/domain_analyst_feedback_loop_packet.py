from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.review_feedback_taxonomy import (
    build_review_feedback_taxonomy,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_CASE_REGISTRY_JSON = "reports/dean_os/domain_analyst_case_registry_packet_current/latest.json"
DEFAULT_FORECAST_REVIEW_JSON = "reports/dean_os/domain_analyst_forecast_review_packet_current/latest.json"
DEFAULT_PROFILE_POLICY_JSON = "reports/dean_os/domain_analyst_profile_policy_packet_current/latest.json"
DEFAULT_TEMPLATE_DECISION_JSON = "reports/dean_os/domain_analyst_template_decision_packet_current/latest.json"


class DomainAnalystFeedbackLoopPacket:
    """Review-only feedback loop contract for domain analyst self-improvement.

    This packet converts human review labels into learning-candidate proposals.
    It never treats feedback as automatic truth and never applies learning,
    changes prompts/config, routes orders, or trades.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_feedback_loop_packet_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        case_registry_json: str | Path = DEFAULT_CASE_REGISTRY_JSON,
        forecast_review_json: str | Path = DEFAULT_FORECAST_REVIEW_JSON,
        profile_policy_json: str | Path = DEFAULT_PROFILE_POLICY_JSON,
        template_decision_json: str | Path = DEFAULT_TEMPLATE_DECISION_JSON,
        manual_feedback_json: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        case_registry = _load_json(case_registry_json)
        forecast_review = _load_json(forecast_review_json)
        profile_policy = _load_json(profile_policy_json)
        template_decision = _load_json(template_decision_json)
        feedback_records = _load_feedback_records(manual_feedback_json)
        label_taxonomy = _review_label_taxonomy(forecast_review, profile_policy)
        feedback_targets = _feedback_targets(case_registry, forecast_review, profile_policy, template_decision)
        normalized_feedback = _normalize_feedback_records(feedback_records, label_taxonomy, feedback_targets)
        learning_candidates = _learning_candidates(normalized_feedback)
        checks = _review_checks(
            case_registry=case_registry,
            forecast_review=forecast_review,
            profile_policy=profile_policy,
            template_decision=template_decision,
            feedback_records=normalized_feedback,
            label_taxonomy=label_taxonomy,
        )
        status = _packet_status(checks, normalized_feedback)
        payload = {
            "run_id": _run_id("domain_analyst_feedback_loop_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_feedback_loop_packet",
            "inputs": {
                "case_registry_json": str(case_registry_json),
                "forecast_review_json": str(forecast_review_json),
                "profile_policy_json": str(profile_policy_json),
                "template_decision_json": str(template_decision_json),
                "manual_feedback_json": str(manual_feedback_json) if manual_feedback_json else None,
            },
            "summary": _summary(status, feedback_targets, normalized_feedback, learning_candidates),
            "review_label_taxonomy": label_taxonomy,
            "feedback_to_learning_contract": _feedback_to_learning_contract(),
            "feedback_targets": feedback_targets,
            "manual_feedback_records": normalized_feedback,
            "learning_candidate_proposals": learning_candidates,
            "after_385_harvest_decisions": _after_385_harvest_decisions(),
            "review_checks": checks,
            "operator_next_steps": _operator_next_steps(status, checks),
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_feedback_loop_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_feedback_loop_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Domain Analyst Feedback Loop Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Feedback targets: {summary.get('feedback_target_count')}",
        f"- Manual feedback records: {summary.get('manual_feedback_record_count')}",
        f"- Learning candidate proposals: {summary.get('learning_candidate_proposal_count')}",
        f"- Can capture manual feedback: {summary.get('can_capture_manual_feedback')}",
        f"- Can create learning candidates: {summary.get('can_create_learning_candidate_proposals')}",
        f"- Can apply learning: {summary.get('can_apply_learning')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can create analyst research recommendation: {summary.get('can_create_analyst_research_recommendation')}",
        f"- Can create execution recommendation: {summary.get('can_create_execution_recommendation')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Feedback Targets",
        "",
    ]
    for target in payload.get("feedback_targets", []):
        lines.append(
            f"- `{target.get('target_id')}` type=`{target.get('target_type')}` "
            f"priority=`{target.get('priority')}` reason={target.get('reason')}"
        )
    lines.extend(["", "## Label Groups", ""])
    for group, labels in payload.get("review_label_taxonomy", {}).get("labels", {}).items():
        lines.append(f"- `{group}`: {', '.join(labels)}")
    lines.extend(["", "## Learning Candidate Proposals", ""])
    if not payload.get("learning_candidate_proposals"):
        lines.append("- none")
    for candidate in payload.get("learning_candidate_proposals", []):
        lines.append(
            f"- `{candidate.get('candidate_id')}` target=`{candidate.get('target_id')}` "
            f"actions={', '.join(candidate.get('proposed_actions', []))}"
        )
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _review_label_taxonomy(forecast_review: dict[str, Any], profile_policy: dict[str, Any]) -> dict[str, Any]:
    outcome_labels = [item.get("bucket_id") for item in forecast_review.get("outcome_taxonomy", []) if item.get("bucket_id")]
    feedback_issue_types = sorted(
        {
            issue
            for profile in profile_policy.get("profile_policy_reviews", [])
            for issue in _feedback_issue_types(profile)
        }
    )
    taxonomy = build_review_feedback_taxonomy(
        case_family="domain_analyst",
        outcome_labels=outcome_labels,
        profile_feedback_issue_types=feedback_issue_types,
    )
    # Preserve the historical key while the shared taxonomy distinguishes the
    # domain-outcome family explicitly.
    taxonomy["labels"]["outcome_review"] = list(outcome_labels)
    taxonomy["applicable_label_groups"].append("outcome_review")
    return taxonomy


def _feedback_issue_types(profile_review: dict[str, Any]) -> list[str]:
    checks = profile_review.get("checks", [])
    labels = []
    for check in checks:
        code = str(check.get("code", ""))
        if code.startswith("feedback_"):
            labels.append(code.removeprefix("feedback_"))
    labels.extend(["forbidden_execution_recommendation", "time_leakage", "unsupported_inference"])
    return labels


def _feedback_targets(
    case_registry: dict[str, Any],
    forecast_review: dict[str, Any],
    profile_policy: dict[str, Any],
    template_decision: dict[str, Any],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in case_registry.get("case_entries", []):
        targets.append(
            {
                "target_id": item.get("case_id"),
                "target_type": item.get("case_type"),
                "domain_id": item.get("domain_id"),
                "priority": "high" if item.get("outcome_bucket") == "pending_expectation_outcome" else "medium",
                "reason": "Outcome and causal reasoning must be reviewed before learning promotion.",
                "allowed_review_labels": item.get("allowed_future_labels", []),
                "allowed_review_outputs": item.get("allowed_review_outputs", []),
            }
        )
    for item in forecast_review.get("forecast_candidates", []):
        targets.append(
            {
                "target_id": item.get("expectation_id"),
                "target_type": "forecast_expectation_candidate",
                "domain_id": item.get("domain_id"),
                "priority": "medium",
                "reason": "Forecast expectation can receive feedback before or after horizon maturity.",
                "allowed_review_labels": item.get("allowed_future_labels", []),
                "allowed_review_outputs": item.get("allowed_review_outputs", []),
            }
        )
    decision_summary = template_decision.get("summary", {})
    targets.append(
        {
            "target_id": f"template_decision:{decision_summary.get('domain_id')}",
            "target_type": "template_process_decision",
            "domain_id": decision_summary.get("domain_id"),
            "priority": "high" if decision_summary.get("decision_status") == "manual_template_decision_pending" else "medium",
            "reason": "Manual template decision is a process decision, not thesis truth.",
            "allowed_review_labels": ["approved", "corrected", "rejected", "needs_more_evidence"],
            "allowed_review_outputs": ["operator_rationale", "required_followup", "self_improvement_proposal"],
        }
    )
    targets.append(
        {
            "target_id": "profile_policy:all_domains",
            "target_type": "profile_policy_review",
            "domain_id": "all_configured_domains",
            "priority": "medium" if profile_policy.get("summary", {}).get("packet_status") != "domain_profile_policy_packet_ready" else "low",
            "reason": "Profile policies control clone readiness and feedback labels across domains.",
            "allowed_review_labels": ["correct", "partially_correct", "missing_source", "unsupported_inference"],
            "allowed_review_outputs": ["policy_note", "source_registry_update_candidate", "feedback_label_candidate"],
        }
    )
    return [target for target in targets if target.get("target_id")]


def _normalize_feedback_records(
    records: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    known_labels = _known_labels(taxonomy)
    target_ids = {target["target_id"] for target in targets}
    normalized = []
    for index, record in enumerate(records):
        labels = [str(label) for label in record.get("labels", [])]
        proposed_actions = [str(action) for action in record.get("proposed_learning_actions", [])]
        blockers: list[str] = []
        target_id = str(record.get("target_id") or "")
        if not target_id:
            blockers.append("missing_target_id")
        elif target_id not in target_ids:
            blockers.append("unknown_target_id")
        if not labels:
            blockers.append("missing_labels")
        unknown = sorted({label for label in labels if label not in known_labels})
        if unknown:
            blockers.append("unknown_labels:" + ",".join(unknown))
        unknown_actions = sorted({action for action in proposed_actions if action not in known_labels})
        if unknown_actions:
            blockers.append("unknown_learning_actions:" + ",".join(unknown_actions))
        if record.get("requests_execution") is True:
            blockers.append("requests_execution")
        if record.get("apply_learning") is True:
            blockers.append("requests_learning_apply")
        if record.get("write_config") is True:
            blockers.append("requests_config_write")
        normalized.append(
            {
                "feedback_id": str(record.get("feedback_id") or f"manual_feedback_{index + 1}"),
                "reviewer": str(record.get("reviewer") or "human"),
                "target_id": target_id,
                "review_type": str(record.get("review_type") or "reasoning"),
                "severity": str(record.get("severity") or "medium"),
                "labels": labels,
                "proposed_learning_actions": proposed_actions,
                "notes": str(record.get("notes") or ""),
                "can_be_learning_candidate": not blockers and bool(proposed_actions) and "no_learning_update" not in proposed_actions,
                "blockers": blockers,
            }
        )
    return normalized


def _learning_candidates(feedback_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for record in feedback_records:
        if not record.get("can_be_learning_candidate"):
            continue
        candidates.append(
            {
                "candidate_id": f"learning_candidate:{record.get('feedback_id')}",
                "source_feedback_id": record.get("feedback_id"),
                "target_id": record.get("target_id"),
                "review_type": record.get("review_type"),
                "severity": record.get("severity"),
                "labels": record.get("labels", []),
                "proposed_actions": record.get("proposed_learning_actions", []),
                "promotion_status": "proposal_only_pending_human_approval",
                "can_apply_now": False,
            }
        )
    return candidates


def _review_checks(
    *,
    case_registry: dict[str, Any],
    forecast_review: dict[str, Any],
    profile_policy: dict[str, Any],
    template_decision: dict[str, Any],
    feedback_records: list[dict[str, Any]],
    label_taxonomy: dict[str, Any],
) -> list[dict[str, str]]:
    case_summary = case_registry.get("summary", {})
    forecast_summary = forecast_review.get("summary", {})
    profile_summary = profile_policy.get("summary", {})
    decision_summary = template_decision.get("summary", {})
    checks = [
        _check("pass" if case_registry.get("mode") == "domain_analyst_case_registry_packet" else "fail", "case_registry_artifact_type", str(case_registry.get("mode"))),
        _check("pass" if int(case_summary.get("expectation_case_count") or 0) > 0 else "fail", "expectation_case_present", str(case_summary.get("expectation_case_count"))),
        _check("pass" if str(forecast_summary.get("packet_status", "")).startswith("forecast_review_ready") else "fail", "forecast_review_ready", str(forecast_summary.get("packet_status"))),
        _check("pass" if profile_summary.get("packet_status") == "domain_profile_policy_packet_ready" else "fail", "profile_policy_ready", str(profile_summary.get("packet_status"))),
        _check("pass" if decision_summary.get("can_create_analyst_research_recommendation") is True else "fail", "analyst_research_recommendations_allowed", str(decision_summary.get("can_create_analyst_research_recommendation"))),
        _check("pass" if decision_summary.get("can_create_execution_recommendation") is False else "fail", "execution_recommendations_blocked", str(decision_summary.get("can_create_execution_recommendation"))),
        _check("pass" if decision_summary.get("can_trade") is False else "fail", "trading_blocked", str(decision_summary.get("can_trade"))),
        _check("pass" if "correct_but_lucky_or_wrong_reason" in _known_labels(label_taxonomy) else "fail", "lucky_hit_label_present", "Lucky/wrong-reason hit label is available."),
        _check("pass" if "request_more_evidence" in _known_labels(label_taxonomy) else "fail", "request_more_evidence_label_present", "Evidence request label is available."),
    ]
    if not feedback_records:
        checks.append(_check("warn", "manual_feedback_not_supplied", "Packet is ready to receive human feedback labels."))
    else:
        invalid = [record["feedback_id"] for record in feedback_records if record.get("blockers")]
        checks.append(_check("pass" if not invalid else "fail", "manual_feedback_records_valid", ", ".join(invalid) or "All supplied feedback records are valid."))
    return checks


def _summary(
    status: str,
    targets: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    learning_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    domain_ids = sorted({target.get("domain_id") for target in targets if target.get("domain_id") and target.get("domain_id") != "all_configured_domains"})
    return {
        "packet_status": status,
        "domain_id": domain_ids[0] if len(domain_ids) == 1 else "multiple_or_unknown",
        "feedback_target_count": len(targets),
        "manual_feedback_record_count": len(feedback_records),
        "learning_candidate_proposal_count": len(learning_candidates),
        "can_capture_manual_feedback": True,
        "can_create_learning_candidate_proposals": True,
        "can_apply_learning": False,
        "can_write_learning_memory": False,
        "can_update_source_registry": False,
        "can_update_prompt": False,
        "can_update_pattern_memory": False,
        "can_create_analyst_research_recommendation": True,
        "can_create_analyst_self_improvement_proposal": True,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _feedback_to_learning_contract() -> dict[str, Any]:
    return {
        "contract_id": "domain_analyst_feedback_to_learning_contract_v1",
        "steps": [
            {"step_id": "capture_review", "output": "manual_feedback_record"},
            {"step_id": "classify_feedback", "output": "classified_feedback_labels"},
            {"step_id": "create_learning_candidate", "output": "proposal_only_learning_candidate"},
            {"step_id": "require_approval", "output": "approved_or_rejected_learning_update"},
            {"step_id": "apply_safe_updates", "output": "blocked_in_this_packet"},
        ],
        "allowed_candidate_updates": [
            "eval_test_backlog",
            "source_registry_note_candidate",
            "prompt_update_candidate",
            "causal_pattern_candidate",
            "human_feedback_dataset_candidate",
        ],
        "forbidden_without_gate": [
            "production_prompt",
            "production_model_weights",
            "trading_config",
            "broker_permissions",
            "live_risk_limits",
        ],
        "rule": "Operator feedback is a training/eval candidate, not automatic truth.",
    }


def _after_385_harvest_decisions() -> list[dict[str, str]]:
    return [
        _harvest("REVIEW_LABEL_TAXONOMY.json", "adapted_to_feedback_taxonomy", "Used labels for analysis, data, causal, outcome, and learning-action review."),
        _harvest("FEEDBACK_TO_LEARNING_PIPELINE_TEMPLATE.yaml", "adapted_to_contract", "Used flow as proposal-only feedback-to-learning contract."),
        _harvest("HUMAN_AGENT_PARALLEL_ANALYSIS_SCHEMA.json", "adapted_partially", "Used side-by-side human correction idea; did not require event graph fields yet."),
        _harvest("REPORT_TO_TRAINING_EXAMPLE_SCHEMA.json", "deferred", "Training/eval examples remain blocked until reviewed feedback candidates exist."),
        _harvest("REVIEW_QUEUE_PRIORITIZATION_TEMPLATE.yaml", "adapted_partially", "Used critical/high/medium/low priority vocabulary for feedback targets."),
        _harvest("REVIEW_SESSION_STATE_SCHEMA.json", "deferred", "Full review-session state is later orchestration work."),
    ]


def _operator_next_steps(status: str, checks: list[dict[str, str]]) -> list[str]:
    failures = [check["code"] for check in checks if check["status"] == "fail"]
    if failures:
        return ["Fix feedback-loop blockers before using feedback for learning candidates: " + ", ".join(failures) + "."]
    if status == "domain_analyst_feedback_loop_ready_with_feedback_candidates":
        return [
            "Review learning candidate proposals manually before any apply ceremony.",
            "Keep correct-for-reasons and lucky/wrong-reason labels separate.",
            "Do not update prompts, source registry, pattern memory, or learning store from this packet.",
        ]
    return [
        "Use this packet as the domain analyst feedback queue contract.",
        "Attach manual feedback records when reviewing the analyst thesis, forecast expectation, or case registry.",
        "Do not promote learning until outcome maturity and human causal review are available.",
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No feedback record is treated as automatic truth.",
        "No learning memory, source registry, prompt, pattern memory, model, profile, or production config is written.",
        "No news/event extraction, daily automation, GPT, or FinBERT call is made.",
        "No sector-to-ticker bridge is executed.",
        "No buy/sell/hold, sizing, allocation, order, broker, paper-trade, or live-trade recommendation is generated.",
    ]


def _packet_status(checks: list[dict[str, str]], feedback_records: list[dict[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "domain_analyst_feedback_loop_blocked"
    if feedback_records:
        return "domain_analyst_feedback_loop_ready_with_feedback_candidates"
    return "domain_analyst_feedback_loop_ready_pending_manual_feedback"


def _known_labels(taxonomy: dict[str, Any]) -> set[str]:
    return {
        label
        for labels in taxonomy.get("labels", {}).values()
        for label in labels
        if label
    }


def _load_feedback_records(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("feedback_records"), list):
        return [record for record in payload["feedback_records"] if isinstance(record, dict)]
    if isinstance(payload.get("records"), list):
        return [record for record in payload["records"] if isinstance(record, dict)]
    return []


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _harvest(source_file: str, classification: str, decision: str) -> dict[str, str]:
    return {"source_file": source_file, "classification": classification, "decision": decision}


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
