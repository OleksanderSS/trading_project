from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_DOMAIN_INSTANCE_CONTRACT_JSON = "reports/dean_os/domain_analyst_instance_contract_current/latest.json"
DEFAULT_DOMAIN_THESIS_REVIEW_JSON = "reports/dean_os/domain_analyst_thesis_review_packet_current/latest.json"
DEFAULT_REGIME_SCENARIO_JSON = None
DEFAULT_ARCHITECTURE_MAP_JSON = "reports/dean_os/current_architecture_map_current/latest.json"


class DomainAnalystTemplateStandardizationPacket:
    """Review-only candidate gate before reusing one domain analyst as a template.

    This packet does not accept the template. It packages the instance contract
    and domain thesis review into one human-review surface, while keeping sector
    scaling, ticker bridging, learning writes, config writes, recommendations,
    and trading disabled.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_template_standardization_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_instance_contract_json: str | Path = DEFAULT_DOMAIN_INSTANCE_CONTRACT_JSON,
        domain_thesis_review_json: str | Path = DEFAULT_DOMAIN_THESIS_REVIEW_JSON,
        regime_scenario_json: str | Path | None = DEFAULT_REGIME_SCENARIO_JSON,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        domain_instance = _load_json(domain_instance_contract_json)
        thesis_review = _load_json(domain_thesis_review_json)
        regime_scenario = _load_optional_json(regime_scenario_json)
        architecture_map = _load_optional_json(architecture_map_json)
        regime_context = _template_regime_context(thesis_review, regime_scenario)
        checks = _review_checks(
            domain_instance=domain_instance,
            thesis_review=thesis_review,
            regime_scenario=regime_scenario,
            regime_context=regime_context,
            architecture_map=architecture_map,
        )
        status = _candidate_status(domain_instance, thesis_review, checks)
        payload = {
            "run_id": _run_id("domain_analyst_template_standardization_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_template_standardization_packet",
            "inputs": {
                "domain_instance_contract_json": str(domain_instance_contract_json),
                "domain_instance_contract_run_id": domain_instance.get("run_id"),
                "domain_thesis_review_json": str(domain_thesis_review_json),
                "domain_thesis_review_run_id": thesis_review.get("run_id"),
                "regime_scenario_json": str(regime_scenario_json) if regime_scenario_json else None,
                "regime_scenario_run_id": regime_scenario.get("run_id") if regime_scenario else None,
                "regime_scenario_context_source": regime_context.get("source"),
                "architecture_map_json": str(architecture_map_json) if architecture_map_json else None,
            },
            "summary": _summary(status, domain_instance, thesis_review, regime_context),
            "template_scope": _template_scope(domain_instance, thesis_review, regime_context),
            "fixed_standardization_sequence": _fixed_standardization_sequence(),
            "manual_acceptance_checklist": _manual_acceptance_checklist(),
            "review_checks": checks,
            "decision_guidance": _decision_guidance(status, checks),
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(domain_instance_contract_json, domain_thesis_review_json, regime_scenario_json),
            "operator_next_steps": _operator_next_steps(status, checks),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_template_standardization_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_template_standardization_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    scope = payload.get("template_scope", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Domain Analyst Template Standardization Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Candidate status: `{summary.get('candidate_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Instance status: `{summary.get('instance_status')}`",
        f"- Thesis review status: `{summary.get('thesis_review_status')}`",
        f"- Regime/scenario context: {summary.get('regime_scenario_context_available')} "
        f"source=`{summary.get('regime_scenario_context_source')}` "
        f"horizons={summary.get('self_check_horizon_count')}",
        f"- Manual acceptance required: {summary.get('manual_acceptance_required')}",
        f"- Can mark template accepted now: {summary.get('can_mark_template_accepted_now')}",
        f"- Can standardize after manual acceptance: {summary.get('can_standardize_domain_template_after_manual_acceptance')}",
        f"- Can scale to other domains now: {summary.get('can_scale_to_other_domains_now')}",
        f"- Can run sector-to-ticker bridge now: {summary.get('can_run_sector_to_ticker_bridge_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Template Scope",
        "",
        f"- Domain ID: `{scope.get('domain_id')}`",
        f"- Sectors: {', '.join(scope.get('sectors', [])) or 'none'}",
        f"- Sector keywords: {', '.join(scope.get('sector_keywords', [])) or 'none'}",
        f"- Required evidence types: {', '.join(scope.get('required_evidence_types', [])) or 'none'}",
        f"- Ticker universe hint: {', '.join(scope.get('ticker_universe_hint', [])) or 'none'}",
        f"- Source policy: `{scope.get('source_registry_policy', {}).get('policy_id')}`",
        f"- Evidence scoring policy: `{scope.get('evidence_scoring_policy', {}).get('policy_id')}`",
        f"- Review output policy: `{scope.get('review_output_policy', {}).get('policy_id')}`",
        f"- Portable rule: {scope.get('portable_rule')}",
        f"- Context analysis rule: {scope.get('regime_scenario_context', {}).get('portable_rule')}",
        "",
        "## Portable Context Analysis Slots",
        "",
    ]
    for item in scope.get("portable_context_analysis_slots", []):
        lines.append(f"- `{item.get('slot_id')}`: {item.get('description')}")
    lines.extend(
        [
            "",
        "## Fixed Standardization Sequence",
        "",
        ]
    )
    lines.extend(f"- {item}" for item in payload.get("fixed_standardization_sequence", []))
    lines.extend(["", "## Manual Acceptance Checklist", ""])
    lines.extend(f"- {item}" for item in payload.get("manual_acceptance_checklist", []))
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


def _summary(
    status: str,
    domain_instance: dict[str, Any],
    thesis_review: dict[str, Any],
    regime_context: dict[str, Any],
) -> dict[str, Any]:
    instance_summary = domain_instance.get("summary", {})
    thesis_summary = thesis_review.get("summary", {})
    ready_for_manual_acceptance = status in {
        "ready_for_manual_template_acceptance",
        "ready_for_manual_template_acceptance_with_cautions",
    }
    return {
        "candidate_status": status,
        "domain_id": instance_summary.get("domain_id") or thesis_summary.get("domain_id"),
        "instance_status": instance_summary.get("instance_status"),
        "thesis_review_status": thesis_summary.get("packet_status"),
        "thesis_stance": thesis_summary.get("thesis_stance"),
        "thesis_expected_direction": thesis_summary.get("expected_direction"),
        "thesis_confidence": thesis_summary.get("confidence"),
        "evidence_item_count": thesis_summary.get("evidence_item_count"),
        "ticker_direct_count": thesis_summary.get("ticker_direct_count"),
        "regime_scenario_context_available": regime_context.get("available"),
        "regime_scenario_context_source": regime_context.get("source"),
        "regime_scenario_status": regime_context.get("packet_status"),
        "active_regime_field_count": len(regime_context.get("active_regime_fields") or []),
        "scenario_probability_count": len(regime_context.get("scenario_probabilities") or {}),
        "scenario_evidence_gap_count": len(regime_context.get("top_evidence_gaps") or []),
        "scenario_probability_mass_valid": regime_context.get("probability_mass_valid"),
        "self_check_horizon_count": len(regime_context.get("self_check_horizons") or []),
        "manual_acceptance_required": True,
        "can_enter_manual_template_review": status != "blocked_template_standardization",
        "can_mark_template_accepted_now": False,
        "can_standardize_domain_template_after_manual_acceptance": ready_for_manual_acceptance,
        "can_prepare_sector_to_ticker_bridge_after_manual_acceptance": ready_for_manual_acceptance
        and thesis_summary.get("can_prepare_separate_ticker_bridge_after_manual_review") is True,
        "can_run_sector_to_ticker_bridge_now": False,
        "can_scale_to_other_domains_now": False,
        "can_create_direct_ticker_thesis_without_bridge": False,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _template_scope(
    domain_instance: dict[str, Any],
    thesis_review: dict[str, Any],
    regime_context: dict[str, Any],
) -> dict[str, Any]:
    slots = domain_instance.get("portable_template_slots", {})
    thesis_summary = thesis_review.get("summary", {})
    return {
        "domain_id": slots.get("domain_id") or thesis_summary.get("domain_id"),
        "sectors": slots.get("sectors", []) or thesis_summary.get("sectors", []),
        "sector_keywords": slots.get("sector_keywords", []),
        "required_evidence_types": slots.get("required_evidence_types", []),
        "useful_evidence_types": slots.get("useful_evidence_types", []),
        "ticker_universe_hint": slots.get("ticker_universe_hint", []),
        "source_registry_policy": slots.get("source_registry_policy", {}),
        "ingestion_filter_policy": slots.get("ingestion_filter_policy", {}),
        "evidence_scoring_policy": slots.get("evidence_scoring_policy", {}),
        "review_output_policy": slots.get("review_output_policy", {}),
        "feedback_label_policy": slots.get("feedback_label_policy", {}),
        "regime_scenario_context": _regime_scenario_template_scope(regime_context),
        "portable_context_analysis_slots": _portable_context_analysis_slots(regime_context),
        "fixed_contract_sequence": domain_instance.get("fixed_contract_sequence", []),
        "portable_rule": slots.get("portable_rule")
        or (
            "For another domain, replace domain profile fields and source keywords; keep source gate, "
            "domain intake, thesis review, manual acceptance, ticker bridge, learning, config, and trading boundaries unchanged."
        ),
    }


def _fixed_standardization_sequence() -> list[str]:
    return [
        "AnalystEvidencePack -> SourceEvidenceValidationGate",
        "SourceEvidenceValidationGate -> DomainAnalystIntakePacket",
        "DomainAnalystIntakePacket -> DomainAnalystInstanceContract",
        "DomainAnalystInstanceContract -> DomainAnalystThesisReviewPacket",
        "DomainAnalystRegimeScenarioPacket context is frozen into thesis/forecast/template review surfaces when available",
        "DomainAnalystThesisReviewPacket -> this manual standardization candidate",
        "Human review decision records acceptance or rejection separately",
        "Only after acceptance: prepare separate SectorThesisToTickerBasketBridge or clone another domain template",
    ]


def _manual_acceptance_checklist() -> list[str]:
    return [
        "Read the DomainAnalystInstanceContract latest markdown.",
        "Read the DomainAnalystThesisReviewPacket latest markdown.",
        "Confirm regime/scenario context slots, scenario probabilities, evidence gaps, and self-check horizons are useful for this domain.",
        "Confirm required evidence lanes, risks, blind spots, and contradicting evidence are reviewable.",
        "Confirm sector/domain thesis remains separate from ticker thesis.",
        "Confirm no learning, config, recommendation, paper trading, or live trading action is authorized.",
        "Record an explicit human acceptance decision in a separate review decision artifact before scaling or bridge work.",
    ]


def _review_checks(
    *,
    domain_instance: dict[str, Any],
    thesis_review: dict[str, Any],
    regime_scenario: dict[str, Any] | None,
    regime_context: dict[str, Any],
    architecture_map: dict[str, Any] | None,
) -> list[dict[str, str]]:
    instance_summary = domain_instance.get("summary", {})
    thesis_summary = thesis_review.get("summary", {})
    checks = [
        _check(
            "pass" if domain_instance.get("mode") == "domain_analyst_instance_contract" else "fail",
            "domain_instance_artifact_type",
            str(domain_instance.get("mode")),
        ),
        _check(
            "pass" if thesis_review.get("mode") == "domain_analyst_thesis_review_packet" else "fail",
            "domain_thesis_review_artifact_type",
            str(thesis_review.get("mode")),
        ),
        _check(
            "pass" if _same_domain(instance_summary, thesis_summary) else "fail",
            "domain_ids_match",
            f"instance={instance_summary.get('domain_id')}, thesis={thesis_summary.get('domain_id')}.",
        ),
        _instance_ready_check(instance_summary),
        _thesis_ready_check(thesis_summary),
        _check(
            "pass" if instance_summary.get("manual_acceptance_required") is True else "fail",
            "instance_manual_acceptance_required",
            f"manual_acceptance_required={instance_summary.get('manual_acceptance_required')!r}.",
        ),
        _check(
            "pass" if thesis_summary.get("manual_review_required") is True else "fail",
            "thesis_manual_review_required",
            f"manual_review_required={thesis_summary.get('manual_review_required')!r}.",
        ),
        _check(
            "pass" if instance_summary.get("can_reuse_as_template_after_manual_review") is True else "warn",
            "instance_reusable_after_manual_review",
            f"reuse_after_manual_review={instance_summary.get('can_reuse_as_template_after_manual_review')!r}.",
        ),
        _check(
            "pass" if thesis_summary.get("can_standardize_domain_template_after_manual_review") is True else "warn",
            "thesis_standardizable_after_manual_review",
            f"standardize_after_review={thesis_summary.get('can_standardize_domain_template_after_manual_review')!r}.",
        ),
        _check("pass", "no_auto_template_acceptance", "This packet never marks the template accepted."),
        _check(
            "pass" if regime_context.get("available") is True else "warn",
            "regime_scenario_context_available_for_template",
            f"source={regime_context.get('source')!r}.",
        ),
        _check(
            "pass"
            if regime_context.get("probability_mass_valid") is True
            else "warn"
            if (
                not regime_context.get("available")
                or regime_context.get("scenario_graph_status")
                == "not_generated"
            )
            else "fail",
            "regime_scenario_probability_mass_valid_for_template",
            str(regime_context.get("probability_mass_valid")),
        ),
        _check("pass", "regime_scenario_template_slots_review_only", "Context slots are reusable review structure, not execution authority."),
        _must_be_false(instance_summary, "can_scale_to_other_domains_now", "instance_no_domain_scaling"),
        _must_be_false(instance_summary, "can_write_learning_memory", "instance_no_learning_write"),
        _must_be_false(instance_summary, "can_create_recommendation", "instance_no_recommendation"),
        _must_be_false(instance_summary, "can_trade", "instance_no_trading"),
        _must_be_false(thesis_summary, "can_create_direct_ticker_thesis_without_bridge", "thesis_no_direct_ticker_thesis"),
        _must_be_false(thesis_summary, "can_write_learning_memory", "thesis_no_learning_write"),
        _must_be_false(thesis_summary, "can_change_analyst_weights", "thesis_no_weight_change"),
        _must_be_false(thesis_summary, "can_write_config", "thesis_no_config_write"),
        _must_be_false(thesis_summary, "can_create_recommendation", "thesis_no_recommendation"),
        _must_be_false(thesis_summary, "can_trade", "thesis_no_trading"),
    ]
    if regime_scenario:
        regime_summary = regime_scenario.get("summary", {})
        checks.extend(
            [
                _check(
                    "pass" if regime_scenario.get("mode") == "domain_analyst_regime_scenario_packet" else "fail",
                    "regime_scenario_artifact_type",
                    str(regime_scenario.get("mode")),
                ),
                _must_be_false(
                    regime_summary,
                    "can_create_execution_recommendation",
                    "regime_scenario_no_execution_recommendation",
                ),
                _must_be_false(regime_summary, "can_trade", "regime_scenario_no_trading"),
            ]
        )
    if architecture_map:
        arch_summary = architecture_map.get("summary", {})
        checks.extend(
            [
                _must_be_false(arch_summary, "can_clone_domain_profiles_now", "architecture_no_domain_cloning"),
                _must_be_false(arch_summary, "can_write_production_config_now", "architecture_no_config_write"),
                _must_be_false(arch_summary, "can_trade", "architecture_no_trading"),
            ]
        )
    return checks


def _same_domain(instance_summary: dict[str, Any], thesis_summary: dict[str, Any]) -> bool:
    instance_domain = instance_summary.get("domain_id")
    thesis_domain = thesis_summary.get("domain_id")
    return bool(instance_domain and thesis_domain and instance_domain == thesis_domain)


def _template_regime_context(
    thesis_review: dict[str, Any],
    regime_scenario: dict[str, Any] | None,
) -> dict[str, Any]:
    if regime_scenario:
        summary = regime_scenario.get("summary", {})
        graph = regime_scenario.get("scenario_outcome_graph", {})
        return {
            "available": True,
            "source": "regime_scenario_json",
            "packet_status": summary.get("packet_status"),
            "source_run_id": regime_scenario.get("run_id"),
            "active_regime_fields": _active_regime_fields_from_packet(regime_scenario),
            "scenario_probabilities": graph.get("scenario_probabilities", {}),
            "probability_mass_valid": graph.get("probability_mass_check", {}).get("valid"),
            "top_evidence_gaps": regime_scenario.get("evidence_gap_priorities", [])[:8],
            "self_check_horizons": graph.get("horizons", []),
            "review_note": "Regime/scenario context is part of the reusable analyst review template, not an execution signal.",
        }
    embedded = thesis_review.get("regime_scenario_context")
    if isinstance(embedded, dict) and embedded.get("available"):
        context = dict(embedded)
        context["source"] = "domain_thesis_review_embedded_context"
        return context
    reasoning = thesis_review.get("reasoning_snapshot_context")
    if (
        isinstance(reasoning, dict)
        and reasoning.get("available")
        and reasoning.get("hash_bound") is True
    ):
        regime = reasoning.get("regime_context") or {}
        dimensions = regime.get("dimensions", {})
        active_dimensions = []
        for name, item in dimensions.items():
            if not isinstance(item, dict) or not item.get("evidence_ids"):
                continue
            active_dimensions.append(
                {
                    "field": name,
                    "state": item.get("state"),
                    "intensity": item.get("intensity"),
                    "trend": item.get("trend"),
                    "confidence": item.get("confidence"),
                    "evidence_ids": item.get("evidence_ids", []),
                }
            )
        horizons = sorted(
            {
                int(value)
                for hypothesis in reasoning.get("hypothesis_ledger", [])
                if isinstance(hypothesis, dict)
                for value in hypothesis.get("horizons_to_check", [])
            }
        )
        return {
            "available": True,
            "source": "verified_reasoning_snapshot",
            "packet_status": reasoning.get("status"),
            "source_run_id": reasoning.get("run_id"),
            "active_regime_fields": active_dimensions,
            "scenario_probabilities": {},
            "probability_mass_valid": None,
            "scenario_graph_status": reasoning.get(
                "scenario_graph_status"
            ),
            "top_evidence_gaps": reasoning.get("evidence_gaps", [])[:8],
            "self_check_horizons": horizons,
            "review_note": (
                "Verified reasoning context is reusable for causal review. "
                "Scenario probabilities remain absent until a calibrated "
                "scenario generator exists."
            ),
        }
    return {
        "available": False,
        "source": None,
        "packet_status": None,
        "source_run_id": None,
        "active_regime_fields": [],
        "scenario_probabilities": {},
        "probability_mass_valid": None,
        "scenario_graph_status": "not_supplied",
        "top_evidence_gaps": [],
        "self_check_horizons": [],
        "review_note": "No regime/scenario context supplied; template remains usable but less complete for causal review.",
    }


def _active_regime_fields_from_packet(regime_scenario: dict[str, Any]) -> list[dict[str, Any]]:
    fields = regime_scenario.get("regime_context_vector", {}).get("fields", {})
    active = []
    for field, item in fields.items():
        if not isinstance(item, dict):
            continue
        if float(item.get("intensity") or 0.0) <= 0:
            continue
        active.append(
            {
                "field": field,
                "state": item.get("state"),
                "intensity": item.get("intensity"),
                "trend": item.get("trend"),
                "confidence": item.get("confidence"),
                "evidence_ids": item.get("evidence_ids", []),
            }
        )
    return active


def _regime_scenario_template_scope(regime_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": regime_context.get("available"),
        "source": regime_context.get("source"),
        "packet_status": regime_context.get("packet_status"),
        "active_regime_field_count": len(regime_context.get("active_regime_fields") or []),
        "scenario_probability_count": len(regime_context.get("scenario_probabilities") or {}),
        "scenario_evidence_gap_count": len(regime_context.get("top_evidence_gaps") or []),
        "self_check_horizons": regime_context.get("self_check_horizons") or [],
        "probability_mass_valid": regime_context.get("probability_mass_valid"),
        "scenario_graph_status": regime_context.get("scenario_graph_status"),
        "portable_rule": (
            "For another domain, keep the regime/scenario review contract, evidence-gap discipline, "
            "and self-check horizons; replace domain-specific regime taxonomy, transmission channels, "
            "and source keywords. Historical analogs and probabilities remain disabled until "
            "backed by verified empirical inputs."
        ),
    }


def _portable_context_analysis_slots(regime_context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        _slot(
            "regime_context_vector",
            "Multi-field domain context vector; clone the shape, then replace domain-specific states and source keywords.",
        ),
        _slot(
            "news_vs_regime_assessments",
            "News/event interpretation against current context, including first-, second-, and third-order channels.",
        ),
        _slot(
            "scenario_outcome_graph",
            "Review-only scenario graph with sibling probability mass checks and explicit uncertainty notes.",
        ),
        _slot(
            "evidence_gap_priorities",
            "Prioritized missing evidence list carried into human review and future source collection.",
        ),
        _slot(
            "self_check_horizons",
            "Future outcome anchors used for case review after horizons mature; never a learning write by itself.",
        ),
        _slot(
            "optional_gpt_finbert_evidence_inputs",
            "GPT or FinBERT can later enrich saved review evidence, but they are not required for MVP and cannot approve actions.",
        ),
        _slot(
            "context_source_status",
            f"Current source={regime_context.get('source') or 'none'}; availability={regime_context.get('available')}.",
        ),
    ]


def _slot(slot_id: str, description: str) -> dict[str, str]:
    return {"slot_id": slot_id, "description": description}


def _instance_ready_check(summary: dict[str, Any]) -> dict[str, str]:
    status = summary.get("instance_status")
    if status == "domain_analyst_instance_review_ready":
        return _check("pass", "domain_instance_ready", status)
    if status == "domain_analyst_instance_review_ready_with_cautions":
        return _check("warn", "domain_instance_ready_with_cautions", status)
    return _check("fail" if status == "blocked_domain_analyst_instance" else "warn", "domain_instance_not_ready", str(status))


def _thesis_ready_check(summary: dict[str, Any]) -> dict[str, str]:
    status = summary.get("packet_status")
    if status == "domain_thesis_review_ready":
        return _check("pass", "domain_thesis_review_ready", status)
    if status == "domain_thesis_review_ready_with_cautions":
        return _check("warn", "domain_thesis_review_ready_with_cautions", status)
    if status == "blocked_domain_thesis_review":
        return _check("fail", "domain_thesis_review_blocked", status)
    return _check("warn", "domain_thesis_review_needs_more_review", str(status))


def _candidate_status(
    domain_instance: dict[str, Any],
    thesis_review: dict[str, Any],
    checks: list[dict[str, str]],
) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "blocked_template_standardization"
    instance_summary = domain_instance.get("summary", {})
    thesis_summary = thesis_review.get("summary", {})
    instance_ready = (
        instance_summary.get("instance_status")
        in {"domain_analyst_instance_review_ready", "domain_analyst_instance_review_ready_with_cautions"}
        and instance_summary.get("can_reuse_as_template_after_manual_review") is True
    )
    thesis_ready = (
        thesis_summary.get("packet_status")
        in {"domain_thesis_review_ready", "domain_thesis_review_ready_with_cautions"}
        and thesis_summary.get("can_standardize_domain_template_after_manual_review") is True
    )
    if not instance_ready or not thesis_ready:
        return "needs_more_template_review"
    if any(check["status"] == "warn" for check in checks):
        return "ready_for_manual_template_acceptance_with_cautions"
    return "ready_for_manual_template_acceptance"


def _decision_guidance(status: str, checks: list[dict[str, str]]) -> dict[str, Any]:
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    failures = [check["code"] for check in checks if check["status"] == "fail"]
    if failures:
        action = "fix_failed_template_checks_before_acceptance"
    elif status == "needs_more_template_review":
        action = "complete_instance_or_thesis_review_before_acceptance"
    elif warnings:
        action = "manual_acceptance_possible_after_caution_review"
    else:
        action = "manual_acceptance_can_review_candidate"
    reasons = [
        f"Candidate status is {status}.",
        "This packet can prepare a human acceptance decision but cannot record acceptance by itself.",
        "Sector-to-ticker bridge, learning promotion, execution recommendations, and trading remain separate.",
    ]
    if warnings:
        reasons.append("Warnings: " + ", ".join(warnings) + ".")
    if failures:
        reasons.append("Failures: " + ", ".join(failures) + ".")
    return {
        "recommended_review_action": action,
        "pass_count": sum(1 for check in checks if check["status"] == "pass"),
        "warning_count": len(warnings),
        "fail_count": len(failures),
        "reasons": reasons,
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No template acceptance decision is recorded.",
        "No sector-to-ticker bridge is executed.",
        "No direct ticker thesis is created.",
        "No new domain analyst profile is cloned or enabled.",
        "No live collector is started.",
        "No learning memory, analyst-weight update, model training, tuning, or production config write is performed.",
        "No execution, buy/sell/hold, allocation, price target, paper order, broker call, or live trade recommendation is generated.",
    ]


def _commands(
    domain_instance_contract_json: str | Path,
    domain_thesis_review_json: str | Path,
    regime_scenario_json: str | Path | None,
) -> dict[str, str]:
    regime_arg = f"--regime-scenario-json {regime_scenario_json} " if regime_scenario_json else ""
    return {
        "rerun_template_standardization_candidate": (
            "python run_agent_domain_analyst_template_standardization_packet.py "
            f"--domain-instance-contract-json {domain_instance_contract_json} "
            f"--domain-thesis-review-json {domain_thesis_review_json} "
            f"{regime_arg}"
            "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
            "--output-dir reports\\dean_os\\domain_analyst_template_standardization_packet_current"
        ),
        "manual_acceptance_required": (
            "Review the latest markdown and record a separate human review decision before any bridge or domain scaling."
        ),
    }


def _operator_next_steps(status: str, checks: list[dict[str, str]]) -> list[str]:
    if status == "blocked_template_standardization":
        return ["Fix failed checks before treating this analyst as a reusable template candidate."]
    if status == "needs_more_template_review":
        return ["Complete the instance contract or thesis review before manual template acceptance."]
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    steps = ["Manually review this packet plus the instance and thesis markdown artifacts."]
    if warnings:
        steps.append("Resolve or explicitly accept warning checks before recording acceptance: " + ", ".join(warnings) + ".")
    steps.append("If accepted, record acceptance separately; only then prepare the sector-to-ticker bridge or another domain clone.")
    return steps


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
