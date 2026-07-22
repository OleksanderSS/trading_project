from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_SOURCE_EVIDENCE_JSON = "reports/dean_os/analyst_evidence_pack_refreshed_gap_check/latest.json"
ALLOWED_RESEARCH_SOURCE_TYPES = {"news", "article", "book", "report", "filing", "transcript"}
FORBIDDEN_OUTPUT_FLAGS = {
    "claims_emitted_now",
    "events_emitted_now",
    "entities_resolved_now",
    "event_propagation_executed_now",
    "company_thesis_generated_now",
    "recommendation_output_now",
    "trade_signal_output_now",
}


class SourceEvidenceValidationGate:
    """Read-only gate between source intake/evidence packs and domain specialists."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/source_evidence_validation_gate"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        source_json: str | Path = DEFAULT_SOURCE_EVIDENCE_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        artifact = _load_json(source_json)
        artifact_type = _artifact_type(artifact)
        if artifact_type == "real_source_normalized_packet":
            validation = _validate_real_source_normalized_packet(artifact)
        elif artifact_type == "normalized_packet_fixture":
            validation = _validate_normalized_packet_fixture(artifact)
        elif artifact_type == "analyst_evidence_pack":
            validation = _validate_analyst_evidence_pack(artifact)
        else:
            validation = _validate_unknown_artifact(artifact)
        guidance = _decision_guidance(validation)
        payload = {
            "run_id": _run_id("source_evidence_validation_gate"),
            "created_at": utc_now_iso(),
            "mode": "source_evidence_validation_gate",
            "inputs": {
                "source_json": str(source_json),
                "artifact_type": artifact_type,
            },
            "summary": {
                "gate_status": guidance["status"],
                "recommended_action": guidance["recommended_action"],
                "artifact_type": artifact_type,
                "source_count": validation.get("source_count", 0),
                "document_count": validation.get("document_count", 0),
                "content_unit_count": validation.get("content_unit_count", 0),
                "candidate_entity_count": len(validation.get("candidate_entities", [])),
                "candidate_topic_count": len(validation.get("candidate_topics", [])),
                "candidate_sector_count": len(validation.get("candidate_sectors", [])),
                "can_enter_domain_research": guidance["can_enter_domain_research"],
                "can_enter_domain_contract_review": guidance["can_enter_domain_contract_review"],
                "can_promote_to_evidence": guidance["can_promote_to_evidence"],
                "can_extract_claims_events_entities": False,
                "can_write_learning_memory": False,
                "can_change_analyst_weights": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
            "source_artifact_summary": validation.get("artifact_summary", {}),
            "candidate_routing_indexes": {
                "entities": validation.get("candidate_entities", []),
                "topics": validation.get("candidate_topics", []),
                "sectors": validation.get("candidate_sectors", []),
                "source_types": validation.get("source_types", {}),
            },
            "validation_checks": validation.get("checks", []),
            "safety_assertions": validation.get("safety_assertions", []),
            "decision_guidance": guidance,
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(source_json),
            "recommendations": _recommendations(guidance, artifact_type),
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
        rendered_md = render_source_evidence_validation_gate_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_source_evidence_validation_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    routing = payload.get("candidate_routing_indexes", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Source Evidence Validation Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Artifact type: `{summary.get('artifact_type')}`",
        f"- Gate status: `{summary.get('gate_status')}`",
        f"- Recommended action: `{summary.get('recommended_action')}`",
        f"- Documents: {summary.get('document_count')}",
        f"- Content units: {summary.get('content_unit_count')}",
        f"- Candidate entities: {', '.join(routing.get('entities', [])) or 'none'}",
        f"- Candidate sectors: {', '.join(routing.get('sectors', [])) or 'none'}",
        f"- Can enter domain research: {summary.get('can_enter_domain_research')}",
        f"- Can promote to evidence: {summary.get('can_promote_to_evidence')}",
        f"- Can extract claims/events/entities: {summary.get('can_extract_claims_events_entities')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Validation Checks",
        "",
        f"- Pass: {guidance.get('pass_count')}",
        f"- Warn: {guidance.get('warning_count')}",
        f"- Fail: {guidance.get('fail_count')}",
        "",
    ]
    lines.extend(_render_check_samples(payload.get("validation_checks", [])))
    lines.extend(["", "## Safety Assertions", ""])
    lines.extend(_render_check_samples(payload.get("safety_assertions", [])))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    lines.extend(["", "## Decision Rationale", ""])
    lines.extend(_render_reason_samples(guidance.get("reasons", [])))
    return "\n".join(lines).strip() + "\n"


def _render_check_samples(items: list[dict[str, Any]], max_items: int = 12) -> list[str]:
    if not items:
        return ["- None."]
    failed = [item for item in items if item.get("status") == "fail"]
    warned = [item for item in items if item.get("status") == "warn"]
    passed = [item for item in items if item.get("status") == "pass"]
    selected = [*failed, *warned[:max_items], *passed[: max(max_items - len(failed) - min(len(warned), max_items), 0)]]
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


def _artifact_type(artifact: dict[str, Any]) -> str:
    if isinstance(artifact.get("normalized_packet_rows"), list):
        return "real_source_normalized_packet"
    if isinstance(artifact.get("normalized_packet_fixture_rows"), list):
        return "normalized_packet_fixture"
    if isinstance(artifact.get("documents"), list) and isinstance(artifact.get("coverage"), dict):
        return "analyst_evidence_pack"
    return "unknown_source_artifact"


def _validate_real_source_normalized_packet(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = artifact.get("normalized_packet_rows", [])
    checks: list[dict[str, str]] = []
    assertions: list[dict[str, str]] = []
    _add_check(checks, bool(rows), "normalized_packets_present", "Real-source normalized packet rows are present.")
    _add_check(
        checks,
        artifact.get("summary", {}).get("real_source_content_supplied") is True,
        "real_source_content_supplied",
        "Artifact declares operator-supplied real source content.",
    )
    _add_check(
        checks,
        artifact.get("summary", {}).get("fixtures_are_production_evidence") is False,
        "not_fixture_or_production_evidence",
        "Artifact is not a production evidence artifact.",
    )
    for packet in rows:
        packet_id = packet.get("packet_id", "unknown_packet")
        content_units = packet.get("content_units", [])
        anchors = packet.get("anchors", [])
        provenance = packet.get("provenance", {})
        _add_check(
            checks,
            bool(content_units) and len(anchors) == len(content_units),
            f"{packet_id}_content_units_have_anchors",
            f"{packet_id} has content units with matching anchors.",
        )
        _add_check(
            checks,
            bool(packet.get("hashes", {}).get("source_content_hash")) and bool(packet.get("hashes", {}).get("normalized_text_hash")),
            f"{packet_id}_hashes_present",
            f"{packet_id} preserves source and normalized text hashes.",
        )
        _add_check(
            checks,
            bool(provenance.get("original_reference_or_file_id")),
            f"{packet_id}_provenance_reference_present",
            f"{packet_id} has a local source reference.",
        )
        _add_check(
            checks,
            packet.get("routing_prefilter", {}).get("candidate_links_are_final") is False,
            f"{packet_id}_candidate_links_not_final",
            f"{packet_id} routing links are candidate-only.",
        )
        if not provenance.get("published_at"):
            _add_check(
                checks,
                False,
                f"{packet_id}_timestamp_missing",
                f"{packet_id} has no published_at timestamp.",
                status_if_false="warn",
            )
        quarantined_units = [unit for unit in content_units if unit.get("quarantine_flags")]
        if quarantined_units:
            _add_check(
                checks,
                False,
                f"{packet_id}_quarantine_present",
                f"{packet_id} has quarantine-marked content units that must remain excluded from sentiment and extraction.",
                status_if_false="warn",
            )
            _add_check(
                checks,
                all(unit.get("extraction_eligible") is False for unit in quarantined_units),
                f"{packet_id}_quarantined_units_not_extraction_eligible",
                f"{packet_id} keeps quarantine-marked content units ineligible for extraction.",
            )
        for unit in content_units:
            content_unit_id = unit.get("content_unit_id", "unknown_content_unit")
            _add_assertion(
                assertions,
                unit.get("claim_extraction_performed") is False,
                f"{packet_id}_{content_unit_id}_claim_extraction_not_performed",
                f"{packet_id}/{content_unit_id} has not performed claim extraction.",
            )
            _add_assertion(
                assertions,
                unit.get("event_extraction_performed") is False,
                f"{packet_id}_{content_unit_id}_event_extraction_not_performed",
                f"{packet_id}/{content_unit_id} has not performed event extraction.",
            )
        _add_assertion(
            assertions,
            packet.get("downstream_extraction_outputs") == [],
            f"{packet_id}_no_downstream_outputs",
            f"{packet_id} has no downstream extraction outputs.",
        )
        output_boundary = packet.get("output_boundary", {})
        for flag in sorted(FORBIDDEN_OUTPUT_FLAGS):
            _add_assertion(
                assertions,
                output_boundary.get(flag) is False,
                f"{packet_id}_{flag}_false",
                f"{packet_id} keeps `{flag}` false.",
            )
    return {
        "source_count": len(rows),
        "document_count": len(rows),
        "content_unit_count": sum(len(packet.get("content_units", [])) for packet in rows),
        "candidate_entities": sorted({item for packet in rows for item in packet.get("routing_prefilter", {}).get("candidate_assets_or_entities", [])}),
        "candidate_topics": sorted({item for packet in rows for item in packet.get("routing_prefilter", {}).get("candidate_topics", [])}),
        "candidate_sectors": sorted({item for packet in rows for item in packet.get("routing_prefilter", {}).get("candidate_sectors", [])}),
        "source_types": dict(sorted(Counter(str(packet.get("source_type_id") or "unknown") for packet in rows).items())),
        "artifact_summary": {
            "block_id": artifact.get("block_id"),
            "schema_version": artifact.get("schema_version"),
            "mode": artifact.get("mode"),
            "input": artifact.get("input", {}),
            "summary": artifact.get("summary", {}),
        },
        "checks": checks,
        "safety_assertions": assertions,
        "evidence_kind": "real_source_normalized_packet",
    }


def _validate_normalized_packet_fixture(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = artifact.get("normalized_packet_fixture_rows", [])
    checks: list[dict[str, str]] = []
    assertions: list[dict[str, str]] = []
    _add_check(checks, bool(rows), "normalized_packets_present", "Normalized packet fixture rows are present.")
    _add_check(
        checks,
        artifact.get("summary", {}).get("real_source_content_supplied_in_245") is False,
        "fixture_real_source_not_supplied",
        "Fixture declares that no real source content was supplied.",
    )
    _add_check(
        checks,
        artifact.get("summary", {}).get("fixtures_are_production_evidence") is False,
        "fixture_not_production_evidence",
        "Fixture declares that it is not production evidence.",
    )
    for packet in rows:
        packet_id = packet.get("packet_id", "unknown_packet")
        _add_check(
            checks,
            bool(packet.get("content_units")) and len(packet.get("anchors", [])) == len(packet.get("content_units", [])),
            f"{packet_id}_content_units_have_anchors",
            f"{packet_id} has content units with matching anchors.",
        )
        _add_check(
            checks,
            packet.get("routing_prefilter", {}).get("candidate_links_are_final") is False,
            f"{packet_id}_candidate_links_not_final",
            f"{packet_id} routing links are candidate-only.",
        )
        _add_assertion(
            assertions,
            packet.get("downstream_extraction_outputs") == [],
            f"{packet_id}_no_downstream_outputs",
            f"{packet_id} has no downstream extraction outputs.",
        )
        output_boundary = packet.get("output_boundary", {})
        for flag in sorted(FORBIDDEN_OUTPUT_FLAGS):
            _add_assertion(
                assertions,
                output_boundary.get(flag) is False,
                f"{packet_id}_{flag}_false",
                f"{packet_id} keeps `{flag}` false.",
            )
    return {
        "source_count": len(rows),
        "content_unit_count": sum(len(packet.get("content_units", [])) for packet in rows),
        "document_count": 0,
        "candidate_entities": sorted({item for packet in rows for item in packet.get("routing_prefilter", {}).get("candidate_assets_or_entities", [])}),
        "candidate_topics": sorted({item for packet in rows for item in packet.get("routing_prefilter", {}).get("candidate_topics", [])}),
        "candidate_sectors": sorted({item for packet in rows for item in packet.get("routing_prefilter", {}).get("candidate_sectors", [])}),
        "source_types": dict(sorted(Counter(str(packet.get("source_type_id") or "unknown") for packet in rows).items())),
        "artifact_summary": {
            "block_id": artifact.get("block_id"),
            "schema_version": artifact.get("schema_version"),
            "fixture_status": artifact.get("fixture_status"),
            "real_data_usage_note": artifact.get("real_data_usage_note", {}),
        },
        "checks": checks,
        "safety_assertions": assertions,
        "evidence_kind": "fixture_not_evidence",
    }


def _validate_analyst_evidence_pack(artifact: dict[str, Any]) -> dict[str, Any]:
    documents = artifact.get("documents", [])
    coverage = artifact.get("coverage", {})
    checks: list[dict[str, str]] = []
    assertions: list[dict[str, str]] = []
    _add_check(checks, bool(documents), "documents_present", "Evidence pack contains research documents.")
    for document in documents:
        document_id = str(document.get("document_id") or document.get("title") or "unknown_document")
        source_type = str(document.get("source_type") or "")
        text = str(document.get("text") or "").strip()
        _add_check(
            checks,
            source_type in ALLOWED_RESEARCH_SOURCE_TYPES,
            f"{document_id}_source_type_supported",
            f"{document_id} uses a supported research source type.",
        )
        _add_check(checks, bool(text), f"{document_id}_text_present", f"{document_id} has normalized text.")
        if not document.get("published_at"):
            _add_check(checks, False, f"{document_id}_timestamp_missing", f"{document_id} has no published_at timestamp.", status_if_false="warn")
        elif _parse_optional_datetime(document.get("published_at")) is None:
            _add_check(checks, False, f"{document_id}_timestamp_unparseable", f"{document_id} has an unparseable published_at timestamp.")
    if coverage.get("missing_requested_tickers"):
        missing = ", ".join(coverage.get("missing_requested_tickers", []))
        _add_check(checks, False, "missing_requested_tickers", f"Evidence pack is missing requested tickers: {missing}.", status_if_false="warn")
    if coverage.get("warning_count"):
        _add_check(checks, False, "coverage_warnings_present", "Evidence pack has coverage warnings.", status_if_false="warn")
    if coverage.get("dropped_count"):
        _add_check(checks, False, "dropped_rows_present", "Evidence pack dropped source rows during normalization.", status_if_false="warn")
    for code, message in [
        ("claim_extraction_not_performed", "This gate does not execute claim extraction."),
        ("event_extraction_not_performed", "This gate does not execute event extraction."),
        ("recommendation_not_created", "This gate does not create recommendations."),
        ("trading_not_allowed", "This gate does not allow trading."),
    ]:
        _add_assertion(assertions, True, code, message)
    tickers = sorted({ticker for document in documents for ticker in document.get("tickers", [])})
    sectors = sorted({sector for document in documents for sector in document.get("sectors", [])})
    tags = sorted({tag for document in documents for tag in document.get("tags", [])})
    return {
        "source_count": len(documents),
        "document_count": len(documents),
        "content_unit_count": len(documents),
        "candidate_entities": tickers,
        "candidate_topics": tags,
        "candidate_sectors": sectors,
        "source_types": coverage.get("by_source_type") or dict(sorted(Counter(str(document.get("source_type") or "unknown") for document in documents).items())),
        "artifact_summary": {
            "run_id": artifact.get("run_id"),
            "coverage": coverage,
            "analyst_inputs": artifact.get("analyst_inputs", {}),
            "warning_count": int(coverage.get("warning_count", 0) or 0),
            "dropped_count": int(coverage.get("dropped_count", 0) or 0),
        },
        "checks": checks,
        "safety_assertions": assertions,
        "evidence_kind": "local_evidence_pack",
    }


def _validate_unknown_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_count": 0,
        "document_count": 0,
        "content_unit_count": 0,
        "candidate_entities": [],
        "candidate_topics": [],
        "candidate_sectors": [],
        "source_types": {},
        "artifact_summary": {"top_level_keys": sorted(artifact.keys())},
        "checks": [_check("fail", "unknown_source_artifact_shape", "Artifact is neither a real-source normalized packet, normalized packet fixture, nor analyst evidence pack.")],
        "safety_assertions": [],
        "evidence_kind": "unknown",
    }


def _decision_guidance(validation: dict[str, Any]) -> dict[str, Any]:
    checks = validation.get("checks", [])
    assertions = validation.get("safety_assertions", [])
    fails = [item for item in [*checks, *assertions] if item.get("status") == "fail"]
    warnings = [item for item in [*checks, *assertions] if item.get("status") == "warn"]
    evidence_kind = validation.get("evidence_kind")
    if fails:
        status = "blocked_source_evidence"
        action = "fix_source_artifact_before_domain_use"
        can_domain = False
        can_contract_review = False
        can_promote = False
    elif evidence_kind == "fixture_not_evidence":
        status = "fixture_validated_not_evidence"
        action = "use_for_contract_review_only"
        can_domain = False
        can_contract_review = True
        can_promote = False
    elif warnings:
        status = "source_evidence_ready_with_warnings"
        action = "manual_domain_review_with_source_warnings"
        can_domain = True
        can_contract_review = True
        can_promote = False
    else:
        status = "source_evidence_ready_for_domain_research"
        action = "manual_domain_research_allowed"
        can_domain = True
        can_contract_review = True
        can_promote = False
    return {
        "status": status,
        "recommended_action": action,
        "can_enter_domain_research": can_domain,
        "can_enter_domain_contract_review": can_contract_review,
        "can_promote_to_evidence": can_promote,
        "fail_count": len(fails),
        "warning_count": len(warnings),
        "pass_count": sum(1 for item in [*checks, *assertions] if item.get("status") == "pass"),
        "reasons": [item["message"] for item in [*fails, *warnings] if item.get("message")],
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No live fetch, connector fetch, or external API call is performed.",
        "No claim/event/entity extraction is executed.",
        "No event propagation is executed.",
        "No company thesis, valuation, recommendation, price target, or position size is created.",
        "No learning memory, analyst weight, production config, operation queue, pipeline, broker, or trading action is written.",
    ]


def _recommendations(guidance: dict[str, Any], artifact_type: str) -> list[str]:
    if guidance["status"] == "blocked_source_evidence":
        return ["Fix source artifact shape, timestamps, source types, or safety flags before domain specialist use."]
    if guidance["status"] == "fixture_validated_not_evidence":
        return [
            "Keep this artifact in staged contract review only.",
            "Do not pass fixture content into domain specialists as evidence.",
            "Use the schema boundary ideas, not the fixture facts.",
        ]
    recommendations = [
        "Use this source artifact as input to domain specialists only after manual review of warnings.",
        "Keep ticker mapping behind the sector-to-ticker bridge.",
        "Run claim/event/entity extraction only after its separate staged contract is accepted.",
    ]
    if artifact_type == "analyst_evidence_pack":
        recommendations.append("If this becomes the standard source gate, wire it before DomainSpecialistReviewPacket in the command checklist.")
    if artifact_type == "real_source_normalized_packet":
        recommendations.append("Route this packet through SourceExtractionReviewPacket only as review-only contract input; quarantine units stay non-extraction.")
    return recommendations


def _commands(source_json: str | Path) -> dict[str, str]:
    return {
        "rerun_gate": (
            "python run_agent_source_evidence_validation_gate.py "
            f"--source-json {source_json} "
            "--output-dir reports\\dean_os\\source_evidence_validation_gate_current"
        )
    }


def _add_check(
    checks: list[dict[str, str]],
    passed: bool,
    code: str,
    message: str,
    status_if_false: str = "fail",
) -> None:
    checks.append(_check("pass" if passed else status_if_false, code, message))


def _add_assertion(assertions: list[dict[str, str]], passed: bool, code: str, message: str) -> None:
    assertions.append(_check("pass" if passed else "fail", code, message))


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _parse_optional_datetime(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
