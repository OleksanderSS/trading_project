from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.analysts.profiles import get_domain_profile, list_domain_profiles
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class DomainAnalystProfilePolicyPacket:
    """Review-only packet for modular domain analyst profile policies.

    This packet checks whether configured domain profiles carry the reusable
    source, ingestion, evidence-scoring, review-output, and feedback policies
    needed for safe profile cloning. It does not clone profiles, run extraction,
    call LLMs, write learning memory, change config, recommend execution, or
    trade.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_profile_policy_packet_current"):
        self.output_dir = Path(output_dir)

    def build(self, *, save: bool = True) -> dict[str, Any]:
        profile_reviews = [_profile_policy_review(domain_id) for domain_id in list_domain_profiles()]
        checks = _review_checks(profile_reviews)
        status = _packet_status(checks)
        payload = {
            "run_id": _run_id("domain_analyst_profile_policy_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_profile_policy_packet",
            "summary": _summary(status, profile_reviews),
            "policy_contract": _policy_contract(),
            "profile_policy_reviews": profile_reviews,
            "after_385_harvest_decisions": _after_385_harvest_decisions(),
            "review_checks": checks,
            "operator_next_steps": _operator_next_steps(status, checks),
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_profile_policy_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_profile_policy_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Domain Analyst Profile Policy Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Profiles reviewed: {summary.get('profile_count')}",
        f"- Profiles policy-ready: {summary.get('profiles_policy_ready_count')}",
        f"- Can support one next-domain clone candidate after manual template acceptance: {summary.get('can_support_one_next_domain_clone_candidate_after_manual_acceptance')}",
        f"- Can clone domain profiles now: {summary.get('can_clone_domain_profiles_now')}",
        f"- Can create analyst research recommendation: {summary.get('can_create_analyst_research_recommendation')}",
        f"- Can create execution recommendation: {summary.get('can_create_execution_recommendation')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Policy Contract",
        "",
    ]
    for slot in payload.get("policy_contract", {}).get("required_policy_slots", []):
        lines.append(f"- `{slot.get('slot_id')}`: {slot.get('purpose')}")
    lines.extend(["", "## Profiles", ""])
    for item in payload.get("profile_policy_reviews", []):
        lines.append(
            f"- `{item.get('domain_id')}`: status=`{item.get('policy_status')}`, "
            f"source=`{item.get('source_registry_policy_id')}`, "
            f"scoring=`{item.get('evidence_scoring_policy_id')}`, "
            f"review=`{item.get('review_output_policy_id')}`"
        )
    lines.extend(["", "## After-385 Harvest Decisions", ""])
    for item in payload.get("after_385_harvest_decisions", []):
        lines.append(f"- `{item.get('source_file')}`: `{item.get('classification')}` - {item.get('decision')}")
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _profile_policy_review(domain_id: str) -> dict[str, Any]:
    profile = get_domain_profile(domain_id)
    checks = [
        _check("pass" if profile.domain_id else "fail", "domain_id_present", profile.domain_id),
        _check("pass" if profile.required_evidence_types else "fail", "required_evidence_types_present", f"{len(profile.required_evidence_types)} required evidence types."),
        _check("pass" if profile.direct_ticker_evidence_rules else "fail", "direct_ticker_rules_present", f"{len(profile.direct_ticker_evidence_rules)} direct ticker rules."),
        *_source_registry_checks(profile.source_registry_policy),
        *_ingestion_policy_checks(profile.ingestion_filter_policy),
        *_evidence_scoring_checks(profile.evidence_scoring_policy),
        *_review_output_checks(profile.review_output_policy),
        *_feedback_policy_checks(profile.feedback_label_policy),
    ]
    status = "profile_policy_ready"
    if any(check["status"] == "fail" for check in checks):
        status = "profile_policy_blocked"
    elif any(check["status"] == "warn" for check in checks):
        status = "profile_policy_ready_with_cautions"
    return {
        "domain_id": profile.domain_id,
        "display_name": profile.display_name,
        "policy_status": status,
        "required_evidence_type_count": len(profile.required_evidence_types),
        "useful_evidence_type_count": len(profile.useful_evidence_types),
        "source_registry_policy_id": profile.source_registry_policy.get("policy_id"),
        "ingestion_filter_policy_id": profile.ingestion_filter_policy.get("policy_id"),
        "evidence_scoring_policy_id": profile.evidence_scoring_policy.get("policy_id"),
        "review_output_policy_id": profile.review_output_policy.get("policy_id"),
        "feedback_label_policy_id": profile.feedback_label_policy.get("policy_id"),
        "allowed_review_outputs": list(profile.review_output_policy.get("allowed_review_outputs", [])),
        "blocked_outputs": list(profile.review_output_policy.get("blocked_outputs", [])),
        "checks": checks,
    }


def _source_registry_checks(policy: dict[str, Any]) -> list[dict[str, str]]:
    trust_tiers = policy.get("trust_tiers", {})
    minimum_rules = policy.get("minimum_source_rules", {})
    return [
        _check("pass" if policy.get("policy_id") else "fail", "source_registry_policy_id_present", str(policy.get("policy_id"))),
        _check("pass" if {"tier_1_core_evidence", "tier_4_weak_or_unverified"}.issubset(trust_tiers) else "fail", "source_registry_trust_tiers_present", ", ".join(sorted(trust_tiers))),
        _check("pass" if minimum_rules.get("numeric_claim", {}).get("require_unit") is True else "fail", "numeric_claim_requires_unit", str(minimum_rules.get("numeric_claim"))),
        _check("pass" if minimum_rules.get("numeric_claim", {}).get("require_period") is True else "fail", "numeric_claim_requires_period", str(minimum_rules.get("numeric_claim"))),
        _check("pass" if minimum_rules.get("final_domain_conclusion", {}).get("weak_source_allowed") is False else "fail", "weak_source_not_allowed_for_final_conclusion", str(minimum_rules.get("final_domain_conclusion"))),
    ]


def _ingestion_policy_checks(policy: dict[str, Any]) -> list[dict[str, str]]:
    required_metadata = set(policy.get("required_metadata", []))
    fail_closed = set(policy.get("fail_closed_rules", []))
    table_rules = policy.get("table_numeric_rules", {})
    return [
        _check("pass" if policy.get("policy_id") else "fail", "ingestion_policy_id_present", str(policy.get("policy_id"))),
        _check("pass" if {"source_id", "source_type", "as_of"}.issubset(required_metadata) else "fail", "ingestion_required_metadata_present", ", ".join(sorted(required_metadata))),
        _check("pass" if "use_live_fetch_without_explicit_permission" in fail_closed else "fail", "live_fetch_fail_closed", ", ".join(sorted(fail_closed))),
        _check("pass" if "future_data_detected_in_as_of_analysis" in fail_closed else "fail", "future_data_fail_closed", ", ".join(sorted(fail_closed))),
        _check("pass" if table_rules.get("require_unit_detection_for_numeric_tables") is True else "warn", "numeric_table_unit_detection_required", str(table_rules)),
    ]


def _evidence_scoring_checks(policy: dict[str, Any]) -> list[dict[str, str]]:
    weights = policy.get("weights", {})
    thresholds = policy.get("minimum_use_thresholds", {})
    fail_closed = set(policy.get("fail_closed_rules", []))
    weight_sum = sum(float(value) for value in weights.values()) if weights else 0.0
    return [
        _check("pass" if policy.get("policy_id") else "fail", "evidence_scoring_policy_id_present", str(policy.get("policy_id"))),
        _check("pass" if {"source_trust", "directness", "contradiction_status"}.issubset(weights) else "fail", "evidence_scoring_weights_present", ", ".join(sorted(weights))),
        _check("pass" if 0.95 <= weight_sum <= 1.05 else "warn", "evidence_scoring_weights_sum_near_one", f"{weight_sum:.3f}"),
        _check("pass" if float(thresholds.get("final_numeric_claim", 0.0) or 0.0) >= 0.70 else "fail", "final_numeric_claim_threshold_present", str(thresholds.get("final_numeric_claim"))),
        _check("pass" if "source_conflict_unresolved" in fail_closed else "fail", "unresolved_conflict_fail_closed", ", ".join(sorted(fail_closed))),
    ]


def _review_output_checks(policy: dict[str, Any]) -> list[dict[str, str]]:
    allowed = set(policy.get("allowed_review_outputs", []))
    blocked = set(policy.get("blocked_outputs", []))
    return [
        _check("pass" if policy.get("policy_id") else "fail", "review_output_policy_id_present", str(policy.get("policy_id"))),
        _check("pass" if {"research_recommendation", "evidence_request", "self_improvement_proposal"}.issubset(allowed) else "fail", "review_only_outputs_allowed", ", ".join(sorted(allowed))),
        _check("pass" if {"buy_sell_hold", "position_sizing", "order_creation", "live_trade"}.issubset(blocked) else "fail", "execution_outputs_blocked", ", ".join(sorted(blocked))),
        _check("pass" if "execution" in str(policy.get("recommendation_boundary", "")).lower() else "fail", "recommendation_boundary_explicit", str(policy.get("recommendation_boundary"))),
    ]


def _feedback_policy_checks(policy: dict[str, Any]) -> list[dict[str, str]]:
    issue_types = set(policy.get("issue_types", []))
    severities = set(policy.get("severity_labels", []))
    return [
        _check("pass" if policy.get("policy_id") else "fail", "feedback_policy_id_present", str(policy.get("policy_id"))),
        _check("pass" if {"low", "medium", "high", "blocker"}.issubset(severities) else "fail", "feedback_severity_labels_present", ", ".join(sorted(severities))),
        _check("pass" if "time_leakage" in issue_types else "fail", "feedback_time_leakage_label_present", ", ".join(sorted(issue_types))),
        _check("pass" if "forbidden_execution_recommendation" in issue_types else "fail", "feedback_forbidden_execution_label_present", ", ".join(sorted(issue_types))),
        _check("pass" if str(policy.get("promotion_rule", "")).lower().find("human review") >= 0 else "fail", "feedback_promotion_requires_review", str(policy.get("promotion_rule"))),
    ]


def _review_checks(profile_reviews: list[dict[str, Any]]) -> list[dict[str, str]]:
    blocked = [item["domain_id"] for item in profile_reviews if item["policy_status"] == "profile_policy_blocked"]
    cautions = [item["domain_id"] for item in profile_reviews if item["policy_status"] == "profile_policy_ready_with_cautions"]
    checks = [
        _check("pass" if profile_reviews else "fail", "profiles_present", f"{len(profile_reviews)} profiles reviewed."),
        _check("pass" if not blocked else "fail", "all_profiles_policy_ready", ", ".join(blocked) or "All profiles have required policies."),
        _check("pass" if all("research_recommendation" in item.get("allowed_review_outputs", []) for item in profile_reviews) else "fail", "all_profiles_allow_research_recommendations", "Review-only research recommendations are allowed."),
        _check("pass" if all("buy_sell_hold" in item.get("blocked_outputs", []) for item in profile_reviews) else "fail", "all_profiles_block_buy_sell_hold", "Execution recommendations remain blocked."),
        _check("pass", "no_clone_authority", "This packet has no authority to clone or enable a domain profile."),
        _check("pass", "no_execution_authority", "This packet has no execution, broker, paper-trade, or live-trade authority."),
    ]
    if cautions:
        checks.append(_check("warn", "profile_policy_cautions_present", ", ".join(cautions)))
    return checks


def _summary(status: str, profile_reviews: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = [item["domain_id"] for item in profile_reviews if item["policy_status"] == "profile_policy_blocked"]
    cautions = [item["domain_id"] for item in profile_reviews if item["policy_status"] == "profile_policy_ready_with_cautions"]
    return {
        "packet_status": status,
        "profile_count": len(profile_reviews),
        "profiles_policy_ready_count": len(profile_reviews) - len(blocked),
        "blocked_profile_ids": blocked,
        "caution_profile_ids": cautions,
        "can_support_one_next_domain_clone_candidate_after_manual_acceptance": not blocked,
        "can_clone_domain_profiles_now": False,
        "can_create_analyst_research_recommendation": True,
        "can_create_analyst_self_improvement_proposal": True,
        "can_write_learning_memory": False,
        "can_write_config": False,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _policy_contract() -> dict[str, Any]:
    return {
        "contract_id": "domain_analyst_profile_policy_contract_v1",
        "purpose": "Make analyst profiles cloneable by changing data/profile policies, not core code.",
        "required_policy_slots": [
            _slot("source_registry_policy", "Source trust tiers, minimum source rules, and weak-source behavior."),
            _slot("ingestion_filter_policy", "Fail-closed metadata, timestamp, table, dedupe, and no-live-fetch rules."),
            _slot("evidence_scoring_policy", "Evidence-quality weights, thresholds, and fail-closed scoring rules."),
            _slot("review_output_policy", "Allowed review-only outputs and blocked execution outputs."),
            _slot("feedback_label_policy", "Reviewer correction labels for future learning candidates."),
        ],
        "clone_rule": "A new domain should change profile slots and local sources first; core analyst gates remain shared.",
        "safety_rule": "Policy packets may support review and clone planning only; they cannot enable trading or unreviewed learning.",
    }


def _slot(slot_id: str, purpose: str) -> dict[str, str]:
    return {"slot_id": slot_id, "purpose": purpose}


def _after_385_harvest_decisions() -> list[dict[str, str]]:
    return [
        _harvest("SOURCE_REGISTRY_TEMPLATE_HEAVY_INDUSTRY.yaml", "adapted_to_default_policy_slot", "Generalized trust tiers and minimum source rules; did not copy heavy-industry sources."),
        _harvest("INGESTION_FILTERS_TEMPLATE.yaml", "adapted_to_default_policy_slot", "Generalized fail-closed metadata, timestamp, table, and no-live-fetch rules."),
        _harvest("EVIDENCE_SCORING_TEMPLATE.yaml", "adapted_to_default_policy_slot", "Generalized source/directness/period/entity/scoring thresholds."),
        _harvest("HUMAN_FEEDBACK_SCHEMA_TEMPLATE.json", "adapted_to_default_policy_slot", "Generalized reviewer correction labels and promotion boundary."),
        _harvest("SAFE_AUTOMATION_BOUNDARY_TEMPLATE.yaml", "policy_reference_only", "Kept as safety guidance; this packet does not implement automation."),
        _harvest("NEWS_EVENT_INTERPRETATION_SCHEMA_TEMPLATE.json", "deferred", "News event extraction is a later reviewed contract, not part of profile policy validation."),
        _harvest("FINE_TUNING_DATASET_SCHEMA_TEMPLATE.json", "deferred", "Fine-tuning remains blocked until reviewed examples and outcome data exist."),
    ]


def _harvest(source_file: str, classification: str, decision: str) -> dict[str, str]:
    return {"source_file": source_file, "classification": classification, "decision": decision}


def _operator_next_steps(status: str, checks: list[dict[str, str]]) -> list[str]:
    failures = [check["code"] for check in checks if check["status"] == "fail"]
    if failures:
        return ["Fix failed profile policy checks before using profiles as clone candidates: " + ", ".join(failures) + "."]
    steps = [
        "Use this packet as the policy readiness layer for modular domain analyst profiles.",
        "Keep the semiconductor template decision pending until the user explicitly accepts or rejects the reusable process.",
        "If the template is accepted later, clone exactly one next-domain candidate by changing profile slots and local source paths only.",
    ]
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    if warnings:
        steps.append("Review non-blocking policy cautions before clone planning: " + ", ".join(warnings) + ".")
    return steps


def _explicit_non_actions() -> list[str]:
    return [
        "No domain profile is cloned, enabled, or written to disk.",
        "No live collector, live fetch, external API, GPT, or FinBERT call is made.",
        "No source extraction, event extraction, evidence promotion, or daily automation is executed.",
        "No learning memory, model, prompt, analyst weight, or production config is changed.",
        "No buy/sell/hold, sizing, allocation, order, broker, paper-trade, or live-trade recommendation is generated.",
    ]


def _packet_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "domain_profile_policy_packet_blocked"
    if any(check["status"] == "warn" for check in checks):
        return "domain_profile_policy_packet_ready_with_cautions"
    return "domain_profile_policy_packet_ready"


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
