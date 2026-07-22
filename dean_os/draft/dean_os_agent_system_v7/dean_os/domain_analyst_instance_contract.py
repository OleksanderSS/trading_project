from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.analysts import get_domain_profile
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_EVIDENCE_PACK_JSON = "reports/dean_os/analyst_evidence_pack_semiconductor_sector_only_strict_current/latest.json"
DEFAULT_SOURCE_GATE_JSON = "reports/dean_os/source_evidence_validation_gate_semiconductor_sector_only_strict_current/latest.json"
DEFAULT_DOMAIN_INTAKE_JSON = "reports/dean_os/domain_analyst_intake_packet_semiconductor_sector_only_strict_current/latest.json"
DEFAULT_ARCHITECTURE_MAP_JSON = "reports/dean_os/current_architecture_map_current/latest.json"


class DomainAnalystInstanceContract:
    """Review-only passport for one reusable domain analyst instance.

    This contract does not create a new analyst. It checks whether one existing
    domain analyst path is coherent enough to use as the pattern for future
    domains after manual acceptance.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_instance_contract"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        evidence_pack_json: str | Path = DEFAULT_EVIDENCE_PACK_JSON,
        source_gate_json: str | Path = DEFAULT_SOURCE_GATE_JSON,
        domain_intake_json: str | Path = DEFAULT_DOMAIN_INTAKE_JSON,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        evidence_pack = _load_json(evidence_pack_json)
        source_gate = _load_json(source_gate_json)
        domain_intake = _load_json(domain_intake_json)
        architecture_map = _load_optional_json(architecture_map_json)
        domain_id = str(domain_intake.get("summary", {}).get("domain_id") or domain_intake.get("inputs", {}).get("domain_id") or "")
        profile = get_domain_profile(domain_id) if domain_id else None
        checks = _review_checks(
            evidence_pack=evidence_pack,
            source_gate=source_gate,
            domain_intake=domain_intake,
            architecture_map=architecture_map,
        )
        status = _instance_status(checks)
        payload = {
            "run_id": _run_id("domain_analyst_instance_contract"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_instance_contract",
            "inputs": {
                "evidence_pack_json": str(evidence_pack_json),
                "source_gate_json": str(source_gate_json),
                "domain_intake_json": str(domain_intake_json),
                "architecture_map_json": str(architecture_map_json) if architecture_map_json else None,
            },
            "summary": _summary(status, evidence_pack, source_gate, domain_intake),
            "portable_template_slots": _portable_template_slots(evidence_pack, domain_intake, profile),
            "fixed_contract_sequence": _fixed_contract_sequence(),
            "review_checks": checks,
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(status, checks),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_instance_contract_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_instance_contract_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    slots = payload.get("portable_template_slots", {})
    lines = [
        "# DEAN-OS Domain Analyst Instance Contract",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Instance status: `{summary.get('instance_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Sectors: {', '.join(summary.get('sectors', [])) or 'none'}",
        f"- Documents: {summary.get('document_count')}",
        f"- Evidence items: {summary.get('evidence_item_count')}",
        f"- Analyst recommendation: `{summary.get('analyst_recommendation')}`",
        f"- Required evidence missing: `{', '.join(summary.get('required_evidence_missing') or []) or 'none'}`",
        f"- Ticker-direct evidence: {summary.get('ticker_direct_count')}",
        f"- Manual acceptance required: {summary.get('manual_acceptance_required')}",
        f"- Can reuse after manual review: {summary.get('can_reuse_as_template_after_manual_review')}",
        f"- Can scale to other domains now: {summary.get('can_scale_to_other_domains_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Portable Slots",
        "",
        f"- Domain ID: `{slots.get('domain_id')}`",
        f"- Sectors: {', '.join(slots.get('sectors', [])) or 'none'}",
        f"- Sector keywords: {', '.join(slots.get('sector_keywords', [])) or 'none'}",
        f"- Required evidence types: {', '.join(slots.get('required_evidence_types', [])) or 'none'}",
        f"- Useful evidence types: {', '.join(slots.get('useful_evidence_types', [])) or 'none'}",
        f"- Ticker universe hint: {', '.join(slots.get('ticker_universe_hint', [])) or 'none'}",
        f"- Source policy: `{slots.get('source_registry_policy', {}).get('policy_id')}`",
        f"- Ingestion filter policy: `{slots.get('ingestion_filter_policy', {}).get('policy_id')}`",
        f"- Evidence scoring policy: `{slots.get('evidence_scoring_policy', {}).get('policy_id')}`",
        f"- Review output policy: `{slots.get('review_output_policy', {}).get('policy_id')}`",
        f"- Feedback label policy: `{slots.get('feedback_label_policy', {}).get('policy_id')}`",
        "",
        "## Fixed Contract Sequence",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("fixed_contract_sequence", []))
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _summary(status: str, evidence_pack: dict[str, Any], source_gate: dict[str, Any], domain_intake: dict[str, Any]) -> dict[str, Any]:
    coverage = evidence_pack.get("coverage", {})
    intake_summary = domain_intake.get("summary", {})
    report = domain_intake.get("analyst_report") or {}
    return {
        "instance_status": status,
        "domain_id": intake_summary.get("domain_id"),
        "sectors": coverage.get("sectors", []) or domain_intake.get("inputs", {}).get("sectors", []),
        "document_count": coverage.get("document_count", intake_summary.get("document_count")),
        "source_types": coverage.get("by_source_type", {}),
        "source_gate_status": source_gate.get("summary", {}).get("gate_status"),
        "domain_intake_status": intake_summary.get("intake_status"),
        "evidence_item_count": intake_summary.get("evidence_item_count"),
        "evidence_type_summary": domain_intake.get("evidence_type_summary", {}),
        "directness_summary": domain_intake.get("directness_summary", {}),
        "ticker_direct_count": intake_summary.get("ticker_direct_count"),
        "analyst_recommendation": report.get("recommendation"),
        "required_evidence_missing": intake_summary.get("required_evidence_missing") or [],
        "manual_acceptance_required": True,
        "can_reuse_as_template_after_manual_review": status == "domain_analyst_instance_review_ready",
        "can_scale_to_other_domains_now": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _portable_template_slots(
    evidence_pack: dict[str, Any],
    domain_intake: dict[str, Any],
    profile: Any | None,
) -> dict[str, Any]:
    coverage = evidence_pack.get("coverage", {})
    inputs = evidence_pack.get("inputs", {})
    intake_inputs = domain_intake.get("inputs", {})
    profile_snapshot = domain_intake.get("domain_profile_snapshot", {})
    return {
        "domain_id": intake_inputs.get("domain_id") or profile_snapshot.get("domain_id"),
        "sectors": coverage.get("sectors", []) or intake_inputs.get("sectors", []),
        "sector_keywords": inputs.get("sector_keywords", []),
        "required_evidence_types": profile_snapshot.get("required_evidence_types") or (profile.required_evidence_types if profile else []),
        "useful_evidence_types": profile_snapshot.get("useful_evidence_types") or (profile.useful_evidence_types if profile else []),
        "ticker_universe_hint": profile_snapshot.get("ticker_universe_hint") or (profile.ticker_universe_hint if profile else []),
        "source_registry_policy": profile_snapshot.get("source_registry_policy") or (profile.source_registry_policy if profile else {}),
        "ingestion_filter_policy": profile_snapshot.get("ingestion_filter_policy") or (profile.ingestion_filter_policy if profile else {}),
        "evidence_scoring_policy": profile_snapshot.get("evidence_scoring_policy") or (profile.evidence_scoring_policy if profile else {}),
        "review_output_policy": profile_snapshot.get("review_output_policy") or (profile.review_output_policy if profile else {}),
        "feedback_label_policy": profile_snapshot.get("feedback_label_policy") or (profile.feedback_label_policy if profile else {}),
        "news_data_paths": inputs.get("news_data_paths", []),
        "macro_data_paths": inputs.get("macro_data_paths", []),
        "portable_rule": (
            "For another domain, change domain_id, sectors, sector_keywords, required/useful evidence types, "
            "ticker universe hints, and profile policies; keep source gate, intake checks, bridge boundary, and non-actions unchanged."
        ),
    }


def _review_checks(
    *,
    evidence_pack: dict[str, Any],
    source_gate: dict[str, Any],
    domain_intake: dict[str, Any],
    architecture_map: dict[str, Any] | None,
) -> list[dict[str, str]]:
    coverage = evidence_pack.get("coverage", {})
    inputs = evidence_pack.get("inputs", {})
    gate_summary = source_gate.get("summary", {})
    intake_summary = domain_intake.get("summary", {})
    missing = intake_summary.get("required_evidence_missing") or []
    checks = [
        _check("pass" if int(coverage.get("document_count") or 0) > 0 else "fail", "evidence_pack_documents_present", f"{coverage.get('document_count', 0)} documents."),
        _check("pass" if len(coverage.get("by_source_type", {})) >= 2 else "warn", "evidence_pack_multi_source", str(coverage.get("by_source_type", {}))),
        _check("pass" if coverage.get("data_quality") in {"strong", "partial"} else "fail", "evidence_pack_quality_reviewable", str(coverage.get("data_quality"))),
    ]
    if not coverage.get("tickers") and coverage.get("sectors"):
        checks.append(
            _check(
                "pass" if inputs.get("sector_keywords") else "warn",
                "sector_only_keywords_recorded",
                f"{len(inputs.get('sector_keywords') or [])} sector keywords recorded.",
            )
        )
    checks.extend(
        [
            _check("pass" if gate_summary.get("can_enter_domain_research") is True else "fail", "source_gate_allows_domain_research", str(gate_summary.get("gate_status"))),
            _check("pass" if _source_gate_downstream_disabled(gate_summary) else "fail", "source_gate_downstream_actions_disabled", "extraction, promotion, learning, recommendation, and trading disabled."),
            _check("pass" if intake_summary.get("analyst_report_created") is True else "fail", "domain_analyst_report_created", str(intake_summary.get("intake_status"))),
            _check("pass" if not missing else "warn", "required_evidence_lanes_covered", ", ".join(missing) if missing else "All required lanes covered."),
            _check("pass" if intake_summary.get("can_create_direct_ticker_thesis_without_bridge") is False else "fail", "ticker_thesis_requires_bridge", "Direct ticker thesis remains bridge-gated."),
            _check("pass" if intake_summary.get("can_write_learning_memory") is False else "fail", "no_learning_write", "No learning write."),
            _check("pass" if intake_summary.get("can_create_recommendation") is False else "fail", "no_recommendation", "No execution recommendation."),
            _check("pass" if intake_summary.get("can_trade") is False else "fail", "no_trading", "No trading."),
        ]
    )
    if int(intake_summary.get("ticker_direct_count") or 0) == 0:
        checks.append(_check("pass", "sector_thesis_before_ticker_thesis", "0 direct ticker evidence items; ticker bridge remains required."))
    else:
        checks.append(_check("pass", "direct_ticker_evidence_partitioned", f"{intake_summary.get('ticker_direct_count')} direct ticker evidence items partitioned."))
    if architecture_map:
        arch_summary = architecture_map.get("summary", {})
        checks.append(_check("pass" if arch_summary.get("can_clone_domain_profiles_now") is False else "fail", "architecture_defer_profile_cloning", "Profile cloning remains disabled by architecture map."))
        checks.append(_check("pass" if arch_summary.get("can_trade") is False else "fail", "architecture_no_trading", "Architecture map keeps trading disabled."))
    return checks


def _source_gate_downstream_disabled(summary: dict[str, Any]) -> bool:
    return (
        summary.get("can_promote_to_evidence") is False
        and summary.get("can_extract_claims_events_entities") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )


def _instance_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "blocked_domain_analyst_instance"
    if any(check["status"] == "warn" for check in checks):
        return "domain_analyst_instance_review_ready_with_cautions"
    return "domain_analyst_instance_review_ready"


def _fixed_contract_sequence() -> list[str]:
    return [
        "local/cached sources -> AnalystEvidencePackRunner",
        "AnalystEvidencePack -> SourceEvidenceValidationGate",
        "validated source pack -> DomainAnalystIntakePacket",
        "domain thesis remains sector/domain-first",
        "ticker thesis requires SectorThesisToTickerBasketBridge and review packet",
        "learning promotion requires separate reviewed outcome/calibration gates",
        "execution recommendation, allocation, paper trading, and live trading remain outside this contract",
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No claim/event/entity extraction is executed.",
        "No evidence promotion is performed.",
        "No learning memory or analyst-weight update is written.",
        "No execution, buy/sell/hold, allocation, price target, paper order, broker call, or live trade recommendation is generated.",
        "No new domain profile is cloned or enabled.",
    ]


def _operator_next_steps(status: str, checks: list[dict[str, str]]) -> list[str]:
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    if status == "blocked_domain_analyst_instance":
        return ["Fix failing checks before treating this analyst instance as reusable."]
    steps = ["Manually review the domain thesis and evidence lane coverage before accepting this as the first reusable analyst instance."]
    if warnings:
        steps.append("Review caution checks before standardization: " + ", ".join(warnings) + ".")
    steps.append("After manual acceptance, implement the pipeline-control agent against saved metric artifacts; do not clone new domains yet.")
    return steps


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
