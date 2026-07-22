from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.analysts import get_domain_profile, list_domain_profiles
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_intake_packet import EVIDENCE_TYPE_ALIASES
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_VERTICAL_SLICE_JSON = "reports/dean_os/domain_analyst_vertical_slice_current/latest.json"
DEFAULT_ARCHITECTURE_MAP_JSON = "reports/dean_os/current_architecture_map_current/latest.json"


class DomainAnalystPortabilityReview:
    """Review whether one accepted domain analyst can be cloned safely.

    This packet is deliberately pre-clone. It checks that domain-specific parts
    live in profile slots and that optional GPT/FinBERT enrichers remain
    adapters, not decision authorities.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_portability_review_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        vertical_slice_json: str | Path = DEFAULT_VERTICAL_SLICE_JSON,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        vertical = _load_json(vertical_slice_json)
        architecture = _load_optional_json(architecture_map_json)
        profile_reviews = [_profile_review(domain_id) for domain_id in list_domain_profiles()]
        reusable_slots = _reusable_slots(vertical)
        adapter_contract = _optional_enrichment_adapter_contract()
        checks = _review_checks(
            vertical=vertical,
            architecture=architecture,
            profile_reviews=profile_reviews,
            reusable_slots=reusable_slots,
            adapter_contract=adapter_contract,
        )
        payload = {
            "run_id": _run_id("domain_analyst_portability_review"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_portability_review",
            "inputs": {
                "vertical_slice_json": str(vertical_slice_json),
                "architecture_map_json": str(architecture_map_json) if architecture_map_json else None,
            },
            "summary": _summary(vertical, architecture, profile_reviews, reusable_slots, checks),
            "source_template_status": _source_template_status(vertical),
            "reusable_template_slots": reusable_slots,
            "profile_reviews": profile_reviews,
            "optional_enrichment_adapter_contract": adapter_contract,
            "review_checks": checks,
            "manual_gate": _manual_gate(vertical),
            "recommended_next_steps": _recommended_next_steps(vertical, checks),
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_portability_review_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_portability_review_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    source = payload.get("source_template_status", {})
    adapter = payload.get("optional_enrichment_adapter_contract", {})
    lines = [
        "# DEAN-OS Domain Analyst Portability Review",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Review status: `{summary.get('review_status')}`",
        f"- Source domain: `{summary.get('source_domain_id')}`",
        f"- Source template status: `{source.get('template_candidate_status')}`",
        f"- Profile count: {summary.get('profile_count')}",
        f"- Profiles structurally portable: {summary.get('profiles_structurally_portable_count')}",
        f"- Can clone domains now: {summary.get('can_clone_domain_profiles_now')}",
        f"- GPT required now: {adapter.get('gpt_required_for_mvp')}",
        f"- FinBERT required now: {adapter.get('finbert_required_for_mvp')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Reusable Slots",
        "",
    ]
    for item in payload.get("reusable_template_slots", {}).get("slots", []):
        lines.append(f"- `{item.get('slot_id')}`: {item.get('description')}")
    lines.extend(["", "## Profiles", ""])
    for item in payload.get("profile_reviews", []):
        lines.append(
            f"- `{item.get('domain_id')}`: status=`{item.get('portability_status')}`, "
            f"required={len(item.get('required_evidence_types', []))}, "
            f"missing_aliases={', '.join(item.get('missing_evidence_aliases', [])) or 'none'}"
        )
    lines.extend(["", "## Optional GPT / FinBERT Boundary", ""])
    for item in adapter.get("rules", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Recommended Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("recommended_next_steps", []))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _profile_review(domain_id: str) -> dict[str, Any]:
    profile = get_domain_profile(domain_id)
    required = list(profile.required_evidence_types)
    useful = list(profile.useful_evidence_types)
    missing_aliases = [item for item in required if item not in EVIDENCE_TYPE_ALIASES]
    checks = [
        _check("pass" if profile.domain_id else "fail", "domain_id_present", profile.domain_id),
        _check("pass" if profile.display_name else "fail", "display_name_present", profile.display_name),
        _check("pass" if profile.core_questions else "fail", "core_questions_present", f"{len(profile.core_questions)} questions."),
        _check("pass" if required else "fail", "required_evidence_types_present", ", ".join(required)),
        _check("pass" if not missing_aliases else "fail", "required_evidence_aliases_present", ", ".join(missing_aliases) or "All required aliases present."),
        _check("pass" if profile.sector_keywords else "warn", "sector_keywords_present", f"{len(profile.sector_keywords)} keywords."),
        _check("pass" if profile.ticker_universe_hint else "warn", "ticker_universe_hint_present", f"{len(profile.ticker_universe_hint)} ticker hints."),
        _check("pass" if profile.direct_ticker_evidence_rules else "fail", "direct_ticker_rules_present", f"{len(profile.direct_ticker_evidence_rules)} rules."),
        _check("pass" if profile.blocked_if_missing else "warn", "blocked_if_missing_present", f"{len(profile.blocked_if_missing)} blockers."),
        _check("pass" if profile.source_registry_policy.get("trust_tiers") else "fail", "source_registry_policy_present", str(profile.source_registry_policy.get("policy_id"))),
        _check("pass" if profile.ingestion_filter_policy.get("fail_closed_rules") else "fail", "ingestion_filter_policy_present", str(profile.ingestion_filter_policy.get("policy_id"))),
        _check("pass" if profile.evidence_scoring_policy.get("weights") else "fail", "evidence_scoring_policy_present", str(profile.evidence_scoring_policy.get("policy_id"))),
        _check("pass" if profile.review_output_policy.get("allowed_review_outputs") else "fail", "review_output_policy_present", str(profile.review_output_policy.get("policy_id"))),
        _check("pass" if profile.feedback_label_policy.get("issue_types") else "fail", "feedback_label_policy_present", str(profile.feedback_label_policy.get("policy_id"))),
    ]
    status = "profile_structurally_portable"
    if any(check["status"] == "fail" for check in checks):
        status = "profile_portability_blocked"
    elif any(check["status"] == "warn" for check in checks):
        status = "profile_portable_with_cautions"
    return {
        "domain_id": profile.domain_id,
        "display_name": profile.display_name,
        "portability_status": status,
        "required_evidence_types": required,
        "useful_evidence_types": useful,
        "sector_keyword_count": len(profile.sector_keywords),
        "ticker_universe_hint": list(profile.ticker_universe_hint),
        "direct_ticker_rule_count": len(profile.direct_ticker_evidence_rules),
        "source_registry_policy_id": profile.source_registry_policy.get("policy_id"),
        "ingestion_filter_policy_id": profile.ingestion_filter_policy.get("policy_id"),
        "evidence_scoring_policy_id": profile.evidence_scoring_policy.get("policy_id"),
        "review_output_policy_id": profile.review_output_policy.get("policy_id"),
        "feedback_label_policy_id": profile.feedback_label_policy.get("policy_id"),
        "missing_evidence_aliases": missing_aliases,
        "checks": checks,
    }


def _reusable_slots(vertical: dict[str, Any]) -> dict[str, Any]:
    inputs = vertical.get("inputs", {})
    return {
        "source_domain_id": vertical.get("summary", {}).get("domain_id"),
        "source_vertical_status": vertical.get("summary", {}).get("run_status"),
        "slots": [
            _slot("domain_id", "Profile id and domain display name."),
            _slot("core_questions", "Domain-specific questions the analyst should answer."),
            _slot("required_evidence_types", "Evidence lanes required before a domain thesis is reviewable."),
            _slot("useful_evidence_types", "Optional evidence lanes that improve analyst confidence."),
            _slot("sector_keywords", "Source filtering and retrieval keywords for this domain."),
            _slot("ticker_universe_hint", "Exposure candidates only; not direct ticker thesis."),
            _slot("contradiction_rules", "Domain-specific conflict rules for confidence reduction."),
            _slot("direct_ticker_evidence_rules", "Separate bridge requirements before ticker thesis."),
            _slot("blocked_if_missing", "Domain-specific blockers that keep review honest."),
            _slot("source_registry_policy", "Source trust tiers, minimum source rules, and weak-source behavior."),
            _slot("ingestion_filter_policy", "Fail-closed metadata, timestamp, table, and no-live-fetch rules."),
            _slot("evidence_scoring_policy", "Evidence-quality weights, thresholds, and fail-closed scoring rules."),
            _slot("review_output_policy", "Allowed review-only outputs and blocked execution outputs."),
            _slot("feedback_label_policy", "Reviewer correction labels for future learning candidates."),
            _slot("regime_context_vector", "Portable context field shape; domain-specific states and source keywords must be replaced per domain."),
            _slot("news_vs_regime_assessments", "Portable event-against-context analysis questions and transmission-channel review."),
            _slot("scenario_outcome_graph", "Portable scenario graph shape with probability-mass checks and uncertainty notes."),
            _slot("evidence_gap_priorities", "Portable evidence-gap discipline for source collection and human review."),
            _slot("self_check_horizons", "Portable future-outcome anchors for later case review; not a learning write by itself."),
            _slot("local_source_paths", "News, macro, materials, and optional enriched local tables."),
        ],
        "context_analysis_source_status": {
            "regime_scenario_status": vertical.get("summary", {}).get("regime_scenario_status"),
            "scenario_node_count": vertical.get("summary", {}).get("scenario_node_count"),
            "scenario_probability_mass_valid": vertical.get("summary", {}).get("scenario_probability_mass_valid"),
            "scenario_evidence_gap_count": vertical.get("summary", {}).get("scenario_evidence_gap_count"),
            "regime_scenario_json": vertical.get("artifact_paths", {}).get("regime_scenario_json"),
        },
        "current_local_paths": {
            "news_data_paths": inputs.get("news_data_paths", []),
            "macro_data_paths": inputs.get("macro_data_paths", []),
            "materials_paths": inputs.get("materials_paths", []),
        },
        "fixed_non_portable_contract": [
            "SourceEvidenceValidationGate",
            "DomainAnalystIntakePacket",
            "DomainAnalystInstanceContract",
            "DomainAnalystRegimeScenarioPacket",
            "DomainAnalystThesisReviewPacket",
            "DomainAnalystForecastReviewPacket",
            "DomainAnalystTemplateStandardizationPacket",
            "Manual accept/reject before clone",
            "Sector-to-ticker bridge remains separate",
        ],
    }


def _slot(slot_id: str, description: str) -> dict[str, str]:
    return {"slot_id": slot_id, "description": description}


def _optional_enrichment_adapter_contract() -> dict[str, Any]:
    return {
        "status": "optional_enrichers_supported_by_contract_but_not_required",
        "gpt_required_for_mvp": False,
        "finbert_required_for_mvp": False,
        "current_baseline": "deterministic_rule_based_profile_and_evidence_aliases",
        "finbert_current_path": "RuleBasedFinancialNLP with OptionalLocalFinBERT(local_files_only=True) fallback",
        "gpt_current_path": "not wired into this domain analyst vertical slice",
        "rules": [
            "GPT may draft or summarize only from cited evidence; it must not create uncited claims.",
            "FinBERT may add sentiment/tone features only when the local model is available; no downloads are allowed inside the review run.",
            "Neither GPT nor FinBERT can mark a template accepted, clone a domain, create ticker thesis, recommend, allocate, or trade.",
            "Deterministic gates remain authoritative for source shape, evidence lane coverage, ticker bridge boundary, and safety flags.",
            "Adapter outputs must be stored as supporting annotations with method/version/provenance metadata.",
        ],
    }


def _review_checks(
    *,
    vertical: dict[str, Any],
    architecture: dict[str, Any] | None,
    profile_reviews: list[dict[str, Any]],
    reusable_slots: dict[str, Any],
    adapter_contract: dict[str, Any],
) -> list[dict[str, str]]:
    summary = vertical.get("summary", {})
    checks = [
        _check(
            "pass" if vertical.get("mode") == "domain_analyst_vertical_slice_run" else "fail",
            "vertical_slice_artifact_type",
            str(vertical.get("mode")),
        ),
        _check(
            "pass" if summary.get("run_status") == "domain_analyst_candidate_complete_pending_manual_acceptance" else "warn",
            "source_template_candidate_complete",
            str(summary.get("run_status")),
        ),
        _check(
            "pass" if summary.get("can_mark_template_accepted_now") is False else "fail",
            "manual_acceptance_not_auto_recorded",
            "Template acceptance remains separate.",
        ),
        _check(
            "pass" if summary.get("can_scale_to_other_domains_now") is False else "fail",
            "domain_cloning_disabled_until_manual_acceptance",
            "No domain cloning is enabled by the source slice.",
        ),
        _check(
            "pass" if vertical.get("artifact_paths", {}).get("forecast_review_json") else "warn",
            "forecast_review_available_for_learning_trace",
            str(vertical.get("artifact_paths", {}).get("forecast_review_json")),
        ),
        _check("pass" if reusable_slots.get("slots") else "fail", "portable_slots_present", f"{len(reusable_slots.get('slots', []))} slots."),
        _check(
            "pass" if _context_analysis_slots_present(reusable_slots) else "fail",
            "portable_context_analysis_slots_present",
            "Regime/scenario context-analysis slots are part of the reusable template contract.",
        ),
        _check(
            "pass"
            if reusable_slots.get("context_analysis_source_status", {}).get("scenario_probability_mass_valid") is True
            else "warn",
            "source_regime_scenario_context_reviewable",
            str(reusable_slots.get("context_analysis_source_status", {}).get("regime_scenario_status")),
        ),
        _check(
            "pass" if all(item["portability_status"] != "profile_portability_blocked" for item in profile_reviews) else "fail",
            "all_profiles_have_required_contract_fields",
            f"{len(profile_reviews)} profiles reviewed.",
        ),
        _check(
            "pass" if adapter_contract.get("gpt_required_for_mvp") is False else "fail",
            "gpt_not_required_for_mvp",
            "GPT remains optional.",
        ),
        _check(
            "pass" if adapter_contract.get("finbert_required_for_mvp") is False else "fail",
            "finbert_not_required_for_mvp",
            "FinBERT remains optional.",
        ),
        _check("pass" if summary.get("can_create_recommendation") is False else "fail", "no_recommendation", "No execution recommendation authority."),
        _check("pass" if summary.get("can_trade") is False else "fail", "no_trading", "No trading authority."),
    ]
    if architecture:
        arch_summary = architecture.get("summary", {})
        checks.extend(
            [
                _check(
                    "pass" if arch_summary.get("can_clone_domain_profiles_now") is False else "fail",
                    "architecture_cloning_disabled",
                    "Architecture keeps cloning disabled.",
                ),
                _check(
                    "pass" if arch_summary.get("can_trade") is False else "fail",
                    "architecture_trading_disabled",
                    "Architecture keeps trading disabled.",
                ),
            ]
        )
    return checks


def _summary(
    vertical: dict[str, Any],
    architecture: dict[str, Any] | None,
    profile_reviews: list[dict[str, Any]],
    reusable_slots: dict[str, Any],
    checks: list[dict[str, str]],
) -> dict[str, Any]:
    source_summary = vertical.get("summary", {})
    blocked_profiles = [item for item in profile_reviews if item["portability_status"] == "profile_portability_blocked"]
    return {
        "review_status": "domain_analyst_portability_review_ready" if not any(check["status"] == "fail" for check in checks) else "domain_analyst_portability_blocked",
        "source_domain_id": source_summary.get("domain_id"),
        "source_template_candidate_status": source_summary.get("template_candidate_status"),
        "profile_count": len(profile_reviews),
        "profiles_structurally_portable_count": len(profile_reviews) - len(blocked_profiles),
        "blocked_profile_ids": [item["domain_id"] for item in blocked_profiles],
        "context_analysis_slot_count": sum(
            1
            for item in reusable_slots.get("slots", [])
            if item["slot_id"]
            in {
                "regime_context_vector",
                "news_vs_regime_assessments",
                "scenario_outcome_graph",
                "evidence_gap_priorities",
                "self_check_horizons",
            }
        ),
        "regime_scenario_context_portable": _context_analysis_slots_present(reusable_slots),
        "source_regime_scenario_status": reusable_slots.get("context_analysis_source_status", {}).get("regime_scenario_status"),
        "manual_acceptance_required_before_clone": True,
        "can_clone_domain_profiles_now": False,
        "can_wire_gpt_as_optional_adapter_later": True,
        "can_wire_local_finbert_as_optional_adapter_later": True,
        "can_create_recommendation": False,
        "can_trade": False,
        "architecture_version": architecture.get("architecture_version") if architecture else None,
    }


def _context_analysis_slots_present(reusable_slots: dict[str, Any]) -> bool:
    required = {
        "regime_context_vector",
        "news_vs_regime_assessments",
        "scenario_outcome_graph",
        "evidence_gap_priorities",
        "self_check_horizons",
    }
    present = {item.get("slot_id") for item in reusable_slots.get("slots", [])}
    return required.issubset(present)


def _source_template_status(vertical: dict[str, Any]) -> dict[str, Any]:
    summary = vertical.get("summary", {})
    audit = vertical.get("synthetic_fixture_audit", {})
    return {
        "run_status": summary.get("run_status"),
        "domain_id": summary.get("domain_id"),
        "template_candidate_status": summary.get("template_candidate_status"),
        "forecast_review_status": summary.get("forecast_review_status"),
        "forecast_candidate_count": summary.get("forecast_candidate_count"),
        "forecast_review_json": vertical.get("artifact_paths", {}).get("forecast_review_json"),
        "document_count": summary.get("document_count"),
        "evidence_item_count": summary.get("evidence_item_count"),
        "synthetic_marker": audit.get("has_synthetic_marker"),
        "fixture_marker": audit.get("has_fixture_marker"),
        "smoke_label": audit.get("has_smoke_label"),
        "manual_acceptance_required": summary.get("manual_acceptance_required"),
    }


def _manual_gate(vertical: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "manual_acceptance_required_before_clone",
        "source_template_candidate_status": vertical.get("summary", {}).get("template_candidate_status"),
        "can_record_acceptance_here": False,
    }


def _recommended_next_steps(vertical: dict[str, Any], checks: list[dict[str, str]]) -> list[str]:
    if any(check["status"] == "fail" for check in checks):
        return ["Fix failed portability checks before manual acceptance or domain cloning."]
    return [
        "Manually accept or reject the source semiconductor analyst template.",
        "If accepted, clone one next domain by changing only profile slots and local source paths.",
        "Keep the forecast-review packet in the cloned flow so every domain thesis becomes a future-evaluable expectation.",
        "Keep GPT and FinBERT optional: add them as annotation adapters only after the deterministic slice stays green.",
        "Do not run sector-to-ticker bridge until direct ticker evidence is supplied.",
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No new domain profile is cloned or enabled.",
        "No template acceptance decision is recorded.",
        "No GPT or external API call is made.",
        "No FinBERT model is downloaded or executed.",
        "No source evidence is promoted into learning memory.",
        "No sector-to-ticker bridge, execution recommendation, allocation, paper order, broker route, or live trade is generated.",
    ]


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
