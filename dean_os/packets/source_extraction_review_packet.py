from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.source_evidence_validation_gate import DEFAULT_SOURCE_EVIDENCE_JSON
from dean_os.utils import json_ready

DEFAULT_SOURCE_EVIDENCE_GATE_JSON = "reports/dean_os/source_evidence_validation_gate_current/latest.json"
DEFAULT_DOMAIN_SPECIALIST_PACKET_JSON = "reports/dean_os/domain_specialist_review_packet_current/latest.json"
CONTRACT_ID = "247_review_only_real_source_claim_event_entity_extraction_contract_v1"


class SourceExtractionReviewPacket:
    """Review-only extraction contract after source validation and before extraction execution."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/source_extraction_review_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        source_json: str | Path = DEFAULT_SOURCE_EVIDENCE_JSON,
        source_gate_json: str | Path | None = DEFAULT_SOURCE_EVIDENCE_GATE_JSON,
        domain_packet_json: str | Path | None = DEFAULT_DOMAIN_SPECIALIST_PACKET_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        source_artifact = _load_json(source_json)
        source_gate = _load_json(source_gate_json) if source_gate_json else None
        domain_packet = _load_json(domain_packet_json) if domain_packet_json else None
        artifact_type = _artifact_type(source_artifact)
        source_units = _source_units(source_artifact, artifact_type)
        routing = _candidate_routing_indexes(source_units)
        checks = _review_checks(
            source_json=source_json,
            source_gate=source_gate,
            domain_packet=domain_packet,
            artifact_type=artifact_type,
            source_units=source_units,
        )
        guidance = _decision_guidance(checks, source_gate)
        payload = {
            "run_id": _run_id("source_extraction_review_packet"),
            "created_at": utc_now_iso(),
            "mode": "source_extraction_review_packet",
            "inputs": {
                "source_json": str(source_json),
                "source_gate_json": str(source_gate_json) if source_gate_json else None,
                "source_gate_run_id": source_gate.get("run_id") if source_gate else None,
                "domain_packet_json": str(domain_packet_json) if domain_packet_json else None,
                "domain_packet_run_id": domain_packet.get("run_id") if domain_packet else None,
            },
            "summary": {
                "packet_status": guidance["status"],
                "recommended_review_action": guidance["recommended_review_action"],
                "contract_id": CONTRACT_ID,
                "artifact_type": artifact_type,
                "source_gate_status": source_gate.get("summary", {}).get("gate_status") if source_gate else None,
                "domain_packet_status": domain_packet.get("summary", {}).get("packet_status") if domain_packet else None,
                "source_unit_count": len(source_units),
                "document_count": sum(1 for unit in source_units if unit.get("unit_kind") == "document"),
                "content_unit_count": len(source_units),
                "candidate_entity_count": len(routing["entities"]),
                "candidate_topic_count": len(routing["topics"]),
                "candidate_sector_count": len(routing["sectors"]),
                "claim_review_slot_count": len(source_units),
                "event_review_slot_count": len(source_units),
                "entity_review_slot_count": len(source_units),
                "timestamp_missing_count": sum(1 for unit in source_units if unit.get("timestamp_status") == "missing"),
                "can_enter_manual_extraction_contract_review": guidance["can_enter_manual_extraction_contract_review"],
                "can_execute_extraction_now": False,
                "can_emit_claims_events_entities": False,
                "can_promote_to_evidence": False,
                "can_write_learning_memory": False,
                "can_change_analyst_weights": False,
                "can_create_recommendation": False,
                "can_trade": False,
                "can_standardize_contract": guidance["can_standardize_contract"],
            },
            "extraction_contract": _extraction_contract(artifact_type),
            "candidate_routing_indexes": routing,
            "source_anchor_plan": {
                "anchor_count": len(source_units),
                "timestamp_status_counts": dict(sorted(Counter(unit.get("timestamp_status") for unit in source_units).items())),
                "source_type_counts": routing["source_types"],
                "anchors": source_units,
                "boundary": "Anchors identify source locations for later extraction review; they are not extracted claims, events, or resolved entities.",
            },
            "extraction_work_queue": _extraction_work_queue(source_units),
            "review_checks": checks,
            "decision_guidance": guidance,
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(source_json, source_gate_json, domain_packet_json),
            "recommendations": _recommendations(guidance),
        }
        if save:
            self.save(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_source_extraction_review_packet_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_source_extraction_review_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    routing = payload.get("candidate_routing_indexes", {})
    guidance = payload.get("decision_guidance", {})
    anchor_plan = payload.get("source_anchor_plan", {})
    lines = [
        "# DEAN-OS Source Extraction Review Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Contract ID: `{summary.get('contract_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Recommended action: `{summary.get('recommended_review_action')}`",
        f"- Artifact type: `{summary.get('artifact_type')}`",
        f"- Source gate status: `{summary.get('source_gate_status')}`",
        f"- Domain packet status: `{summary.get('domain_packet_status')}`",
        f"- Source units: {summary.get('source_unit_count')}",
        f"- Candidate entities: {', '.join(routing.get('entities', [])) or 'none'}",
        f"- Candidate sectors: {', '.join(routing.get('sectors', [])) or 'none'}",
        f"- Timestamp missing count: {summary.get('timestamp_missing_count')}",
        f"- Can enter manual extraction contract review: {summary.get('can_enter_manual_extraction_contract_review')}",
        f"- Can execute extraction now: {summary.get('can_execute_extraction_now')}",
        f"- Can emit claims/events/entities: {summary.get('can_emit_claims_events_entities')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Contract Boundary",
        "",
    ]
    contract = payload.get("extraction_contract", {})
    lines.extend(f"- {item}" for item in contract.get("boundary_rules", []))
    lines.extend(["", "## Source Anchor Plan", ""])
    lines.extend(
        [
            f"- Anchor count: {anchor_plan.get('anchor_count')}",
            f"- Timestamp status counts: `{anchor_plan.get('timestamp_status_counts')}`",
            f"- Source type counts: `{anchor_plan.get('source_type_counts')}`",
        ]
    )
    lines.extend(["", "## Anchor Samples", ""])
    lines.extend(_render_anchor_samples(anchor_plan.get("anchors", [])))
    lines.extend(["", "## Extraction Work Queue Samples", ""])
    lines.extend(_render_work_samples(payload.get("extraction_work_queue", [])))
    lines.extend(["", "## Review Checks", ""])
    lines.extend(_render_check_samples(payload.get("review_checks", [])))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    lines.extend(["", "## Decision Rationale", ""])
    lines.extend(_render_reason_samples(guidance.get("reasons", [])))
    return "\n".join(lines).strip() + "\n"


def _artifact_type(artifact: dict[str, Any]) -> str:
    if isinstance(artifact.get("normalized_packet_rows"), list):
        return "real_source_normalized_packet"
    if isinstance(artifact.get("documents"), list) and isinstance(artifact.get("coverage"), dict):
        return "analyst_evidence_pack"
    if isinstance(artifact.get("normalized_packet_fixture_rows"), list):
        return "normalized_packet_fixture"
    return "unknown_source_artifact"


def _source_units(artifact: dict[str, Any], artifact_type: str) -> list[dict[str, Any]]:
    if artifact_type == "analyst_evidence_pack":
        return _evidence_pack_units(artifact)
    if artifact_type == "real_source_normalized_packet":
        return _real_normalized_packet_units(artifact)
    if artifact_type == "normalized_packet_fixture":
        return _fixture_units(artifact)
    return []


def _evidence_pack_units(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for index, document in enumerate(artifact.get("documents", []), start=1):
        document_id = str(document.get("document_id") or document.get("title") or f"document_{index}")
        published_at = document.get("published_at")
        text = str(document.get("text") or "").strip()
        units.append(
            {
                "source_unit_id": document_id,
                "unit_kind": "document",
                "anchor_id": f"document:{document_id}",
                "title": document.get("title"),
                "source_type": document.get("source_type"),
                "uri": document.get("uri"),
                "published_at": published_at,
                "timestamp_status": "present" if published_at else "missing",
                "text_present": bool(text),
                "text_preview": _compact_text(text),
                "candidate_entities": sorted({str(item).upper() for item in document.get("tickers", []) if item}),
                "candidate_sectors": sorted({str(item) for item in document.get("sectors", []) if item}),
                "candidate_topics": sorted({str(item) for item in document.get("tags", []) if item}),
                "metadata_boundary": "Candidate routing is metadata-derived; no claim, event, or entity resolution has been executed.",
            }
        )
    return units


def _fixture_units(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for packet in artifact.get("normalized_packet_fixture_rows", []):
        routing = packet.get("routing_prefilter", {})
        packet_id = str(packet.get("packet_id") or "unknown_packet")
        for index, content_unit in enumerate(packet.get("content_units", []), start=1):
            content_unit_id = str(content_unit.get("content_unit_id") or f"{packet_id}_unit_{index}")
            text = str(content_unit.get("normalized_text") or "").strip()
            anchor_id = str(content_unit.get("anchor_id") or f"{packet_id}:{content_unit_id}")
            units.append(
                {
                    "source_unit_id": content_unit_id,
                    "unit_kind": "normalized_fixture_content_unit",
                    "anchor_id": anchor_id,
                    "parent_packet_id": packet_id,
                    "title": packet.get("source_type_id"),
                    "source_type": packet.get("source_type_id"),
                    "uri": None,
                    "published_at": None,
                    "timestamp_status": "fixture_not_real_source",
                    "text_present": bool(text),
                    "text_preview": _compact_text(text),
                    "candidate_entities": sorted({str(item).upper() for item in routing.get("candidate_assets_or_entities", []) if item}),
                    "candidate_sectors": sorted({str(item) for item in routing.get("candidate_sectors", []) if item}),
                    "candidate_topics": sorted({str(item) for item in routing.get("candidate_topics", []) if item}),
                    "metadata_boundary": "Fixture routing is review-only and must not be promoted to evidence.",
                }
            )
    return units


def _real_normalized_packet_units(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for packet in artifact.get("normalized_packet_rows", []):
        routing = packet.get("routing_prefilter", {})
        provenance = packet.get("provenance", {})
        packet_id = str(packet.get("packet_id") or "unknown_packet")
        published_at = provenance.get("published_at")
        for index, content_unit in enumerate(packet.get("content_units", []), start=1):
            content_unit_id = str(content_unit.get("content_unit_id") or f"{packet_id}_unit_{index}")
            text = str(content_unit.get("normalized_text") or "").strip()
            anchor_id = str(content_unit.get("anchor_id") or f"{packet_id}:{content_unit_id}")
            units.append(
                {
                    "source_unit_id": content_unit_id,
                    "unit_kind": "real_normalized_content_unit",
                    "anchor_id": anchor_id,
                    "parent_packet_id": packet_id,
                    "title": packet.get("source_name"),
                    "source_type": packet.get("source_type_id"),
                    "uri": provenance.get("document_uri") or provenance.get("original_reference_or_file_id"),
                    "published_at": published_at,
                    "timestamp_status": "present" if published_at else "missing",
                    "text_present": bool(text),
                    "text_preview": _compact_text(text),
                    "candidate_entities": sorted({str(item).upper() for item in routing.get("candidate_assets_or_entities", []) if item}),
                    "candidate_sectors": sorted({str(item) for item in routing.get("candidate_sectors", []) if item}),
                    "candidate_topics": sorted({str(item) for item in routing.get("candidate_topics", []) if item}),
                    "quarantine_flags": list(content_unit.get("quarantine_flags", [])),
                    "extraction_eligible": content_unit.get("extraction_eligible") is True,
                    "metadata_boundary": "Operator-supplied normalized packet unit; still review-only until human evidence promotion.",
                }
            )
    return units


def _candidate_routing_indexes(source_units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "entities": sorted({item for unit in source_units for item in unit.get("candidate_entities", [])}),
        "topics": sorted({item for unit in source_units for item in unit.get("candidate_topics", [])}),
        "sectors": sorted({item for unit in source_units for item in unit.get("candidate_sectors", [])}),
        "source_types": dict(sorted(Counter(str(unit.get("source_type") or "unknown") for unit in source_units).items())),
    }


def _extraction_contract(artifact_type: str) -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "stage": "contract_definition_only",
        "artifact_type": artifact_type,
        "allowed_candidate_outputs_for_future_stage": [
            "candidate_claims",
            "candidate_events",
            "candidate_entity_mentions",
            "candidate_topics",
            "candidate_sectors",
            "candidate_assets",
            "candidate_financial_implications",
            "source_anchor_links",
            "extraction_limitations",
        ],
        "required_claim_candidate_fields": [
            "claim_id",
            "claim_text",
            "claim_type",
            "polarity",
            "source_anchor_id",
            "entity_refs",
            "topic_refs",
            "confidence",
            "limitations",
        ],
        "required_event_candidate_fields": [
            "event_id",
            "event_type",
            "event_time",
            "source_anchor_id",
            "entity_refs",
            "time_confidence",
            "limitations",
        ],
        "required_entity_candidate_fields": [
            "entity_ref_id",
            "surface_form",
            "entity_type",
            "symbol_or_identifier",
            "resolution_status",
            "source_anchor_id",
            "limitations",
        ],
        "required_financial_implication_candidate_fields": [
            "implication_id",
            "scope",
            "direction",
            "horizon",
            "source_anchor_id",
            "reasoning_boundary",
            "limitations",
        ],
        "output_boundary": {
            "claims_emitted_now": False,
            "events_emitted_now": False,
            "entities_resolved_now": False,
            "financial_implications_emitted_now": False,
            "event_propagation_executed_now": False,
            "company_thesis_generated_now": False,
            "valuation_generated_now": False,
            "recommendation_output_now": False,
            "trade_signal_output_now": False,
        },
        "boundary_rules": [
            "This packet defines the extraction contract only; it does not execute extraction.",
            "Candidate outputs from a future extraction stage must remain review-only until separately approved.",
            "Financial implication candidates are not recommendations, ratings, price targets, allocation advice, or trade signals.",
            "Event chronology requires source timestamps or explicit timestamp limitations.",
        ],
    }


def _extraction_work_queue(source_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    work_items = []
    for unit in source_units:
        blockers = []
        if not unit.get("text_present"):
            blockers.append("missing_normalized_text")
        if unit.get("timestamp_status") == "missing":
            blockers.append("missing_source_timestamp_for_event_chronology")
        if unit.get("timestamp_status") == "fixture_not_real_source":
            blockers.append("fixture_not_real_evidence")
        if unit.get("extraction_eligible") is False:
            blockers.append("source_unit_not_extraction_eligible")
        if unit.get("quarantine_flags"):
            blockers.append("quarantined_source_unit")
        work_items.append(
            {
                "work_item_id": f"extract_contract::{unit.get('anchor_id')}",
                "source_unit_id": unit.get("source_unit_id"),
                "anchor_id": unit.get("anchor_id"),
                "status": "contract_defined_not_executed",
                "candidate_entities": unit.get("candidate_entities", []),
                "candidate_topics": unit.get("candidate_topics", []),
                "candidate_sectors": unit.get("candidate_sectors", []),
                "review_tasks": [
                    "review_candidate_entity_mentions",
                    "review_candidate_claims",
                    "review_candidate_events",
                    "review_candidate_financial_implications",
                ],
                "blockers_or_limitations": blockers,
                "allowed_use": "manual_extraction_contract_review_only",
            }
        )
    return work_items


def _review_checks(
    source_json: str | Path,
    source_gate: dict[str, Any] | None,
    domain_packet: dict[str, Any] | None,
    artifact_type: str,
    source_units: list[dict[str, Any]],
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    if artifact_type in {"analyst_evidence_pack", "normalized_packet_fixture", "real_source_normalized_packet"}:
        checks.append(_check("pass", "source_artifact_type_supported", f"Source artifact type is `{artifact_type}`."))
    else:
        checks.append(_check("fail", "source_artifact_type_unknown", "Source artifact is not a supported evidence pack or normalized source packet."))

    if source_units:
        checks.append(_check("pass", "source_units_present", f"{len(source_units)} source unit(s) are available for contract review."))
    else:
        checks.append(_check("fail", "source_units_missing", "No source units are available for extraction contract review."))

    missing_text = [unit for unit in source_units if not unit.get("text_present")]
    if missing_text:
        checks.append(_check("fail", "source_units_missing_text", f"{len(missing_text)} source unit(s) have no normalized text."))

    missing_timestamp = [unit for unit in source_units if unit.get("timestamp_status") == "missing"]
    if missing_timestamp:
        checks.append(_check("warn", "source_timestamps_missing", f"{len(missing_timestamp)} source unit(s) lack published_at timestamps."))

    if artifact_type == "normalized_packet_fixture":
        checks.append(_check("warn", "fixture_contract_only", "Normalized packet fixtures are contract material only, not production evidence."))

    if artifact_type == "real_source_normalized_packet":
        checks.append(_check("pass", "real_source_packet_review_only", "Real-source normalized packet is accepted for review-only contract planning."))

    quarantined_units = [unit for unit in source_units if unit.get("quarantine_flags")]
    if quarantined_units:
        checks.append(_check("warn", "quarantined_source_units_present", f"{len(quarantined_units)} source unit(s) carry quarantine flags and must remain excluded from extraction."))

    _extend_source_gate_checks(checks, source_json, source_gate)
    _extend_domain_packet_checks(checks, domain_packet)
    checks.append(_check("pass", "extraction_not_executed", "This packet does not execute extraction or emit claims/events/entities."))
    checks.append(_check("pass", "downstream_actions_disabled", "Learning writes, recommendations, and trading remain disabled."))
    return checks


def _extend_source_gate_checks(checks: list[dict[str, str]], source_json: str | Path, source_gate: dict[str, Any] | None) -> None:
    if not source_gate:
        checks.append(_check("fail", "source_gate_missing", "Source evidence validation gate is required before extraction contract review."))
        return
    if source_gate.get("mode") == "source_evidence_validation_gate":
        checks.append(_check("pass", "source_gate_mode_valid", "Source gate artifact is attached."))
    else:
        checks.append(_check("fail", "source_gate_mode_invalid", f"Unexpected source gate mode: {source_gate.get('mode')}."))

    gate_summary = source_gate.get("summary", {})
    gate_guidance = source_gate.get("decision_guidance", {})
    gate_status = gate_summary.get("gate_status")
    if gate_guidance.get("fail_count"):
        checks.append(_check("fail", "source_gate_has_failures", "Source gate has failures and blocks extraction contract review."))
    elif gate_status == "fixture_validated_not_evidence":
        checks.append(_check("pass", "source_gate_fixture_contract_review_allowed", "Source gate allows fixture contract review only."))
    elif gate_summary.get("can_enter_domain_research") is True:
        checks.append(_check("pass", "source_gate_allows_domain_research", "Source gate allows manual domain research."))
    else:
        checks.append(_check("fail", "source_gate_blocks_domain_research", "Source gate does not allow manual domain research or contract review."))

    if gate_guidance.get("warning_count"):
        checks.append(_check("warn", "source_gate_warnings_present", "Source gate warnings must be reviewed before standardization."))

    gate_source = str(source_gate.get("inputs", {}).get("source_json") or "")
    if gate_source and _normalize_path_string(gate_source) == _normalize_path_string(source_json):
        checks.append(_check("pass", "source_gate_matches_source_artifact", "Source gate was built from the same source artifact."))
    elif gate_source:
        checks.append(_check("warn", "source_gate_source_mismatch", "Source gate input path differs from this packet source path."))

    if (
        gate_summary.get("can_extract_claims_events_entities") is False
        and gate_summary.get("can_promote_to_evidence") is False
        and gate_summary.get("can_trade") is False
    ):
        checks.append(_check("pass", "source_gate_downstream_actions_disabled", "Source gate did not already extract, promote, or trade."))
    else:
        checks.append(_check("fail", "source_gate_downstream_action_enabled", "Source gate enables a downstream action that must remain disabled."))


def _extend_domain_packet_checks(checks: list[dict[str, str]], domain_packet: dict[str, Any] | None) -> None:
    if not domain_packet:
        checks.append(_check("warn", "domain_packet_not_attached", "Domain specialist packet was not attached to this extraction contract."))
        return
    if domain_packet.get("mode") == "domain_specialist_review_packet":
        checks.append(_check("pass", "domain_packet_mode_valid", "Domain specialist packet is attached."))
    else:
        checks.append(_check("fail", "domain_packet_mode_invalid", f"Unexpected domain packet mode: {domain_packet.get('mode')}."))

    summary = domain_packet.get("summary", {})
    if summary.get("packet_status") == "blocked" or summary.get("can_enter_manual_domain_review") is False:
        checks.append(_check("fail", "domain_packet_blocks_review", "Domain specialist packet blocks manual domain review."))
    elif summary.get("packet_status") == "domain_review_ready_with_limitations":
        checks.append(_check("warn", "domain_packet_has_limitations", "Domain specialist packet has limitations that must be reviewed."))
    else:
        checks.append(_check("pass", "domain_packet_allows_review", "Domain specialist packet allows manual review."))

    if (
        summary.get("can_write_learning_memory") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    ):
        checks.append(_check("pass", "domain_packet_downstream_actions_disabled", "Domain packet disables learning, recommendations, and trading."))
    else:
        checks.append(_check("fail", "domain_packet_downstream_action_enabled", "Domain packet enables a downstream action that must remain disabled."))


def _decision_guidance(checks: list[dict[str, str]], source_gate: dict[str, Any] | None) -> dict[str, Any]:
    fails = [check for check in checks if check.get("status") == "fail"]
    warnings = [check for check in checks if check.get("status") == "warn"]
    gate_status = source_gate.get("summary", {}).get("gate_status") if source_gate else None
    if fails:
        status = "blocked_extraction_contract"
        action = "fix_source_or_domain_gate_before_contract_review"
        can_review = False
        can_standardize = False
    elif gate_status == "fixture_validated_not_evidence":
        status = "fixture_extraction_contract_ready_not_evidence"
        action = "review_contract_only_do_not_use_as_evidence"
        can_review = True
        can_standardize = False
    elif warnings:
        status = "extraction_contract_ready_with_warnings"
        action = "manual_extraction_contract_review_with_limitations"
        can_review = True
        can_standardize = False
    else:
        status = "extraction_contract_ready_for_manual_review"
        action = "manual_extraction_contract_review"
        can_review = True
        can_standardize = True
    return {
        "status": status,
        "recommended_review_action": action,
        "can_enter_manual_extraction_contract_review": can_review,
        "can_standardize_contract": can_standardize,
        "can_execute_extraction_now": False,
        "can_emit_claims_events_entities": False,
        "fail_count": len(fails),
        "warning_count": len(warnings),
        "pass_count": sum(1 for check in checks if check.get("status") == "pass"),
        "reasons": [check["message"] for check in [*fails, *warnings] if check.get("message")],
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No live fetch, connector fetch, or external API call is performed.",
        "No claim, event, entity, or financial implication extraction is executed.",
        "No extracted claim/event/entity object is emitted.",
        "No event propagation is executed.",
        "No company thesis, valuation, recommendation, price target, or position size is created.",
        "No learning memory, analyst weight, production config, operation queue, pipeline, broker, or trading action is written.",
    ]


def _commands(
    source_json: str | Path,
    source_gate_json: str | Path | None,
    domain_packet_json: str | Path | None,
) -> dict[str, str]:
    source_gate_arg = f"--source-gate-json {source_gate_json} " if source_gate_json else ""
    domain_arg = f"--domain-packet-json {domain_packet_json} " if domain_packet_json else ""
    return {
        "rerun_packet": (
            "python run_agent_source_extraction_review_packet.py "
            f"--source-json {source_json} "
            f"{source_gate_arg}"
            f"{domain_arg}"
            "--output-dir reports\\dean_os\\source_extraction_review_packet_current"
        )
    }


def _recommendations(guidance: dict[str, Any]) -> list[str]:
    if guidance["status"] == "blocked_extraction_contract":
        return ["Fix source gate, domain packet, timestamps/text, or source artifact shape before reviewing this extraction contract."]
    recommendations = [
        "Review the contract fields before implementing any extractor.",
        "Use source anchors for every future candidate claim, event, entity, topic, sector, and financial implication.",
        "Keep future extraction output review-only until a separate promotion gate is accepted.",
        "Do not generate company thesis, valuation, recommendation, price target, allocation, or trading output from this contract.",
    ]
    if guidance["status"] == "extraction_contract_ready_with_warnings":
        recommendations.append("Resolve or explicitly accept warnings before standardizing the contract.")
    if guidance["status"] == "fixture_extraction_contract_ready_not_evidence":
        recommendations.append("Use this only to test contract shape; do not treat fixture content as evidence.")
    return recommendations


def _render_anchor_samples(items: list[dict[str, Any]], max_items: int = 12) -> list[str]:
    if not items:
        return ["- None."]
    lines = []
    for item in items[:max_items]:
        entities = ", ".join(item.get("candidate_entities", [])) or "none"
        lines.append(
            f"- `{item.get('anchor_id')}` source_type=`{item.get('source_type')}` "
            f"timestamp=`{item.get('timestamp_status')}` entities={entities}"
        )
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional anchor(s) in JSON.")
    return lines


def _render_work_samples(items: list[dict[str, Any]], max_items: int = 12) -> list[str]:
    if not items:
        return ["- None."]
    lines = []
    for item in items[:max_items]:
        blockers = ", ".join(item.get("blockers_or_limitations", [])) or "none"
        lines.append(f"- `{item.get('work_item_id')}` status=`{item.get('status')}` blockers=`{blockers}`")
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional work item(s) in JSON.")
    return lines


def _render_check_samples(items: list[dict[str, Any]], max_items: int = 14) -> list[str]:
    if not items:
        return ["- None."]
    failed = [item for item in items if item.get("status") == "fail"]
    warned = [item for item in items if item.get("status") == "warn"]
    passed = [item for item in items if item.get("status") == "pass"]
    selected = [*failed, *warned, *passed]
    lines = [f"- `{item.get('status')}` {item.get('code')}: {item.get('message')}" for item in selected[:max_items]]
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional check(s) in JSON.")
    return lines


def _render_reason_samples(items: list[str], max_items: int = 8) -> list[str]:
    if not items:
        return ["- No blockers."]
    lines = [f"- {item}" for item in items[:max_items]]
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional reason(s) in JSON.")
    return lines


def _compact_text(value: str, limit: int = 180) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _normalize_path_string(value: str | Path) -> str:
    return str(value).replace("/", "\\").lower()


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
