from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_SECTOR_TO_TICKER_BRIDGE_JSON = "reports/dean_os/sector_thesis_to_ticker_basket_current/latest.json"
DEFAULT_SOURCE_EVIDENCE_GATE_JSON = "reports/dean_os/source_evidence_validation_gate_current/latest.json"


class SectorToTickerReviewPacket:
    """Read-only human review packet for a sector-to-ticker bridge output."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/sector_to_ticker_review_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        bridge_path: str | Path = DEFAULT_SECTOR_TO_TICKER_BRIDGE_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        bridge = _load_json(bridge_path)
        ticker_map = [_ticker_review_item(candidate, bridge.get("mapping_runs", [])) for candidate in bridge.get("ticker_candidates", [])]
        blocked_or_limited = [
            item
            for item in ticker_map
            if item.get("review_status") != "review_ready" or item.get("blocked_evidence", {}).get("blocked_runs", 0) > 0
        ]
        checks = _review_checks(bridge=bridge, ticker_map=ticker_map)
        guidance = _decision_guidance(checks, ticker_map)
        payload = {
            "run_id": _run_id("sector_to_ticker_review_packet"),
            "created_at": utc_now_iso(),
            "mode": "sector_to_ticker_review_packet",
            "inputs": {
                "bridge_path": str(bridge_path),
                "bridge_run_id": bridge.get("run_id"),
            },
            "summary": {
                "packet_status": guidance["status"],
                "recommended_review_action": guidance["recommended_review_action"],
                "manual_review_required": True,
                "sector": bridge.get("sector_thesis", {}).get("sector"),
                "domain_profile": bridge.get("sector_thesis", {}).get("domain_profile"),
                "sector_stance": bridge.get("sector_thesis", {}).get("sector_stance"),
                "ticker_count": len(ticker_map),
                "review_ready_count": sum(1 for item in ticker_map if item.get("review_status") == "review_ready"),
                "review_ready_with_limits_count": sum(1 for item in ticker_map if item.get("review_status") == "review_ready_with_evidence_limits"),
                "ticker_evidence_ready_pipeline_blocked_count": sum(
                    1
                    for item in ticker_map
                    if item.get("review_status")
                    == "ticker_evidence_ready_pipeline_blocked"
                ),
                "blocked_or_context_count": sum(1 for item in ticker_map if item.get("review_status") not in {"review_ready", "review_ready_with_evidence_limits"}),
                "can_enter_manual_sector_to_ticker_review": guidance["can_enter_manual_sector_to_ticker_review"],
                "can_create_ticker_forecast": False,
                "can_change_analyst_weights": False,
                "can_write_learning_memory": False,
                "can_write_config": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
            "domain_specialist_review_contract": _domain_specialist_review_contract(bridge),
            "sector_thesis": _sector_thesis_section(bridge),
            "ticker_review_map": ticker_map,
            "blocked_or_limited_candidates": blocked_or_limited,
            "direct_ticker_evidence": _direct_ticker_evidence(bridge.get("mapping_runs", [])),
            "blocked_or_context_windows": _blocked_or_context_windows(bridge.get("mapping_runs", [])),
            "review_checks": checks,
            "decision_guidance": guidance,
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(bridge_path),
            "operator_next_steps": _operator_next_steps(guidance, blocked_or_limited),
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
        rendered_md = render_sector_to_ticker_review_packet_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


class DomainSpecialistReviewPacket:
    """Domain-first review packet that keeps ticker mapping as a gated bridge."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_specialist_review_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        bridge_path: str | Path = DEFAULT_SECTOR_TO_TICKER_BRIDGE_JSON,
        source_gate_path: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        bridge = _load_json(bridge_path)
        source_gate = _load_json(source_gate_path) if source_gate_path else None
        ticker_map = [_ticker_review_item(candidate, bridge.get("mapping_runs", [])) for candidate in bridge.get("ticker_candidates", [])]
        blocked_or_limited = [
            item
            for item in ticker_map
            if item.get("review_status") != "review_ready" or item.get("blocked_evidence", {}).get("blocked_runs", 0) > 0
        ]
        checks = _domain_review_checks(bridge=bridge, ticker_map=ticker_map, source_gate=source_gate)
        guidance = _domain_decision_guidance(checks, ticker_map)
        exposure_map = _sector_exposure_map(bridge, ticker_map)
        source_gate_summary = source_gate.get("summary", {}) if source_gate else {}
        payload = {
            "run_id": _run_id("domain_specialist_review_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_specialist_review_packet",
            "inputs": {
                "bridge_path": str(bridge_path),
                "bridge_run_id": bridge.get("run_id"),
                "source_gate_path": str(source_gate_path) if source_gate_path else None,
                "source_gate_run_id": source_gate.get("run_id") if source_gate else None,
                "source_contract": "sector_thesis_to_ticker_basket_bridge",
            },
            "summary": {
                "packet_status": guidance["status"],
                "recommended_review_action": guidance["recommended_review_action"],
                "manual_review_required": True,
                "domain_profile": bridge.get("sector_thesis", {}).get("domain_profile"),
                "sector": bridge.get("sector_thesis", {}).get("sector"),
                "sector_stance": bridge.get("sector_thesis", {}).get("sector_stance"),
                "domain_entity_count": len(exposure_map.get("entities", [])),
                "direct_ticker_candidate_count": sum(
                    1 for item in ticker_map if item.get("review_status") in {"review_ready", "review_ready_with_evidence_limits"}
                ),
                "ticker_bridge_limited_count": len(blocked_or_limited),
                "source_gate_status": source_gate_summary.get("gate_status"),
                "source_gate_warning_count": source_gate.get("decision_guidance", {}).get("warning_count") if source_gate else None,
                "source_gate_fail_count": source_gate.get("decision_guidance", {}).get("fail_count") if source_gate else None,
                "can_enter_manual_domain_review": guidance["can_enter_manual_domain_review"],
                "can_enter_ticker_candidate_review": guidance["can_enter_ticker_candidate_review"],
                "can_standardize_domain_template": guidance["can_standardize_domain_template"],
                "can_change_analyst_weights": False,
                "can_write_learning_memory": False,
                "can_write_config": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
            "domain_first_contract": _domain_first_contract(bridge),
            "domain_thesis": _domain_thesis_section(bridge),
            "source_evidence_context": _source_evidence_context(bridge, source_gate),
            "claims_events_entities": _claims_events_entities_contract_stub(bridge, exposure_map),
            "sector_exposure_map": exposure_map,
            "sector_to_ticker_bridge_review": {
                "bridge_run_id": bridge.get("run_id"),
                "bridge_status": bridge.get("summary", {}).get("bridge_status"),
                "bridge_next_action": bridge.get("summary", {}).get("next_action"),
                "ticker_review_map": ticker_map,
                "blocked_or_limited_candidates": blocked_or_limited,
                "direct_ticker_evidence": _direct_ticker_evidence(bridge.get("mapping_runs", [])),
                "blocked_or_context_windows": _blocked_or_context_windows(bridge.get("mapping_runs", [])),
                "boundary": "Ticker candidates remain review candidates only; they are not recommendations or learning promotions.",
            },
            "review_checks": checks,
            "decision_guidance": guidance,
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _domain_commands(bridge_path, source_gate_path),
            "operator_next_steps": _domain_operator_next_steps(guidance, blocked_or_limited, source_gate),
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
        rendered_md = render_domain_specialist_review_packet_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_sector_to_ticker_review_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    sector = payload.get("sector_thesis", {})
    lines = [
        "# DEAN-OS Sector To Ticker Review Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Bridge run ID: `{payload.get('inputs', {}).get('bridge_run_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Recommended review action: `{summary.get('recommended_review_action')}`",
        f"- Domain profile: `{summary.get('domain_profile')}`",
        f"- Sector: `{summary.get('sector')}`",
        f"- Sector stance: `{summary.get('sector_stance')}`",
        f"- Manual review required: {summary.get('manual_review_required')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can change analyst weights: {summary.get('can_change_analyst_weights')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Sector Thesis",
        "",
        f"- Thesis level: `{sector.get('thesis_level')}`",
        f"- Thesis: {sector.get('thesis')}",
        "",
        "## Ticker Review Map",
        "",
    ]
    for item in payload.get("ticker_review_map", []):
        direct = item.get("direct_evidence", {})
        blocked = item.get("blocked_evidence", {})
        lines.append(
            f"- `{item.get('ticker')}` review_status=`{item.get('review_status')}` "
            f"candidate_status=`{item.get('candidate_status')}` "
            f"direct_windows={direct.get('directional_ready_runs')} "
            f"blocked_windows={blocked.get('blocked_runs')} "
            f"allowed_use=`{item.get('allowed_use')}`"
        )
        pipeline_contexts = item.get("exact_pipeline_contexts", [])
        if pipeline_contexts:
            for context in pipeline_contexts:
                lines.append(
                    "  - pipeline "
                    f"{context.get('model')}/"
                    f"{context.get('target_name')}/"
                    f"{context.get('timeframe')} "
                    f"status=`{context.get('case_classification')}` "
                    f"blocked={context.get('blocked_metric_planes')}"
                )
        prediction_review = item.get(
            "stage5_prediction_review", {}
        )
        if prediction_review.get("context_count"):
            lines.append(
                "  - Stage 5 review: "
                f"status=`{prediction_review.get('status')}` "
                f"complete="
                f"{prediction_review.get('complete_context_count', 0)}/"
                f"{prediction_review.get('context_count', 0)} "
                f"quarantined="
                f"{prediction_review.get('quarantined_context_count', 0)}"
            )
        timeframe_audit = item.get(
            "feature_timeframe_audit", {}
        )
        if timeframe_audit.get("status") not in {
            None,
            "not_supplied",
        }:
            lines.append(
                "  - Feature timeframe audit: "
                f"status=`{timeframe_audit.get('status')}` "
                f"declared=`{timeframe_audit.get('declared_timeframe')}` "
                f"observed=`{timeframe_audit.get('observed_timeframe')}` "
                f"timezone_aware="
                f"{timeframe_audit.get('datetime_timezone_aware')}"
            )
        if item.get("required_next_inputs"):
            lines.append(
                "  - required next inputs: "
                + ", ".join(item.get("required_next_inputs", []))
            )
        ticker_evidence = item.get(
            "ticker_specific_evidence", {}
        )
        if ticker_evidence.get("eligible_record_count"):
            lines.append(
                "  - ticker evidence: "
                f"eligible_records="
                f"{ticker_evidence.get('eligible_record_count')} "
                f"corroborated_lanes="
                f"{ticker_evidence.get('corroborated_lane_count')}"
            )
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- `{check.get('status')}` {check.get('code')}: {check.get('message')}")
    lines.extend(["", "## Explicit Non-Actions", ""])
    for item in payload.get("explicit_non_actions", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    lines.extend(["", "## Decision Rationale", ""])
    lines.extend(f"- {item}" for item in guidance.get("reasons", [])) if guidance.get("reasons") else lines.append("- No blockers.")
    return "\n".join(lines).strip() + "\n"


def render_domain_specialist_review_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    thesis = payload.get("domain_thesis", {})
    extraction = payload.get("claims_events_entities", {})
    bridge = payload.get("sector_to_ticker_bridge_review", {})
    lines = [
        "# DEAN-OS Domain Specialist Review Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Bridge run ID: `{payload.get('inputs', {}).get('bridge_run_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Recommended review action: `{summary.get('recommended_review_action')}`",
        f"- Domain profile: `{summary.get('domain_profile')}`",
        f"- Sector: `{summary.get('sector')}`",
        f"- Manual domain review: {summary.get('can_enter_manual_domain_review')}",
        f"- Ticker candidate review: {summary.get('can_enter_ticker_candidate_review')}",
        f"- Can standardize domain template: {summary.get('can_standardize_domain_template')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can change analyst weights: {summary.get('can_change_analyst_weights')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Domain Thesis",
        "",
        f"- Thesis level: `{thesis.get('thesis_level')}`",
        f"- Allowed use: `{thesis.get('allowed_use')}`",
        f"- Thesis: {thesis.get('thesis')}",
        "",
        "## Source Evidence Context",
        "",
    ]
    evidence = payload.get("source_evidence_context", {})
    lines.extend(
        [
            f"- Source run count: {evidence.get('source_run_count')}",
            f"- Evidence quality counts: `{evidence.get('evidence_quality_counts')}`",
            f"- Source gate status: `{evidence.get('source_gate_status')}`",
            f"- Source artifact type: `{evidence.get('source_artifact_type')}`",
            f"- Source documents: {evidence.get('source_document_count')}",
            f"- Source validation counts: `{evidence.get('source_validation_counts')}`",
            f"- Evidence boundary: {evidence.get('evidence_boundary')}",
            "",
            "## Claims, Events, Entities",
            "",
            f"- Extraction status: `{extraction.get('extraction_status')}`",
            f"- Candidate entities: {', '.join(extraction.get('candidate_entities', [])) or 'none'}",
            f"- Boundary: {extraction.get('boundary')}",
            "",
            "## Sector Exposure Map",
            "",
        ]
    )
    for entity in payload.get("sector_exposure_map", {}).get("entities", []):
        lines.append(
            f"- `{entity.get('entity_id')}` relation=`{entity.get('evidence_relation')}` "
            f"allowed_use=`{entity.get('allowed_use')}`"
        )
    lines.extend(["", "## Sector To Ticker Bridge", ""])
    for item in bridge.get("ticker_review_map", []):
        direct = item.get("direct_evidence", {})
        blocked = item.get("blocked_evidence", {})
        lines.append(
            f"- `{item.get('ticker')}` review_status=`{item.get('review_status')}` "
            f"direct_windows={direct.get('directional_ready_runs')} blocked_windows={blocked.get('blocked_runs')}"
        )
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- `{check.get('status')}` {check.get('code')}: {check.get('message')}")
    lines.extend(["", "## Explicit Non-Actions", ""])
    for item in payload.get("explicit_non_actions", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    lines.extend(["", "## Decision Rationale", ""])
    lines.extend(f"- {item}" for item in guidance.get("reasons", [])) if guidance.get("reasons") else lines.append("- No blockers.")
    return "\n".join(lines).strip() + "\n"


def _domain_first_contract(bridge: dict[str, Any]) -> dict[str, Any]:
    contract = bridge.get("domain_analyst_contract", {})
    return {
        "profile_id": contract.get("profile_id") or bridge.get("sector_thesis", {}).get("domain_profile"),
        "sector": contract.get("sector") or bridge.get("sector_thesis", {}).get("sector"),
        "packet_type": "DomainSpecialistReviewPacket",
        "primary_axis": "domain_or_sector_research",
        "ticker_axis": "derived_bridge_only",
        "required_review_sections": [
            "domain_thesis",
            "source_evidence_context",
            "claims_events_entities",
            "sector_exposure_map",
            "sector_to_ticker_bridge_review",
            "explicit_non_actions",
        ],
        "boundary_rule": "Domain specialists analyze sectors, themes, sources, and economic context first; ticker candidates are derived only through a separate evidence bridge.",
        "allowed_output": "human_review_packet_json_and_markdown",
        "disallowed_outputs": [
            "claim_extraction_execution",
            "event_propagation_execution",
            "learning_write",
            "analyst_weight_change",
            "production_config_write",
            "recommendation_or_rating",
            "price_target",
            "position_size",
            "broker_or_order_action",
        ],
    }


def _domain_thesis_section(bridge: dict[str, Any]) -> dict[str, Any]:
    sector = _sector_thesis_section(bridge)
    return {
        **sector,
        "thesis_level": "domain_or_sector_thesis",
        "allowed_use": "manual_domain_review_and_context_mapping",
        "disallowed_use": [
            "ticker_thesis_without_direct_evidence",
            "recommendation",
            "learning_promotion",
            "analyst_weight_change",
            "trading",
        ],
        "review_boundary": "This is the domain specialist output. Ticker use requires the sector-to-ticker bridge.",
    }


def _source_evidence_context(bridge: dict[str, Any], source_gate: dict[str, Any] | None = None) -> dict[str, Any]:
    sector = bridge.get("sector_thesis", {})
    summary = bridge.get("summary", {})
    context = {
        "source_run_count": sector.get("source_run_count") or summary.get("run_count"),
        "research_stance_counts": sector.get("research_stance_counts", {}),
        "exam_verdict_counts": sector.get("exam_verdict_counts", {}),
        "evidence_quality_counts": sector.get("evidence_quality_counts", {}),
        "warnings": sector.get("warnings", []),
        "evidence_boundary": "This packet summarizes existing bridge/replay evidence only; it does not retrieve sources or execute extraction.",
    }
    if not source_gate:
        context.update(
            {
                "source_gate_attached": False,
                "source_gate_status": None,
                "source_artifact_type": None,
                "source_document_count": None,
                "source_content_unit_count": None,
                "source_candidate_entities": [],
                "source_candidate_sectors": [],
                "source_validation_counts": None,
                "source_gate_warning_reasons_sample": [],
                "source_gate_boundary": "No source gate artifact was attached to this packet.",
            }
        )
        return context

    gate_summary = source_gate.get("summary", {})
    gate_guidance = source_gate.get("decision_guidance", {})
    routing = source_gate.get("candidate_routing_indexes", {})
    reasons = gate_guidance.get("reasons", [])
    context.update(
        {
            "source_gate_attached": True,
            "source_gate_run_id": source_gate.get("run_id"),
            "source_gate_status": gate_summary.get("gate_status"),
            "source_gate_recommended_action": gate_summary.get("recommended_action"),
            "source_artifact_type": gate_summary.get("artifact_type"),
            "source_document_count": gate_summary.get("document_count"),
            "source_content_unit_count": gate_summary.get("content_unit_count"),
            "source_candidate_entities": routing.get("entities", []),
            "source_candidate_sectors": routing.get("sectors", []),
            "source_validation_counts": {
                "pass": gate_guidance.get("pass_count"),
                "warn": gate_guidance.get("warning_count"),
                "fail": gate_guidance.get("fail_count"),
            },
            "can_enter_domain_research": gate_summary.get("can_enter_domain_research"),
            "can_promote_to_evidence": gate_summary.get("can_promote_to_evidence"),
            "can_extract_claims_events_entities": gate_summary.get("can_extract_claims_events_entities"),
            "can_trade": gate_summary.get("can_trade"),
            "source_gate_warning_reasons_sample": reasons[:8],
            "source_gate_warning_reason_count": len(reasons),
            "source_gate_boundary": "Source gate validates artifact shape and safety boundaries only; it does not execute extraction, promotion, recommendations, or trading.",
        }
    )
    return context


def _claims_events_entities_contract_stub(bridge: dict[str, Any], exposure_map: dict[str, Any]) -> dict[str, Any]:
    sector = bridge.get("sector_thesis", {})
    entities = [entity.get("entity_id") for entity in exposure_map.get("entities", []) if entity.get("entity_id")]
    topics = sorted(
        {
            str(value)
            for value in [
                sector.get("sector"),
                sector.get("domain_profile"),
                sector.get("sector_stance"),
                "sector_thesis",
                "ticker_bridge_candidate",
            ]
            if value
        }
    )
    return {
        "extraction_status": "not_extracted_in_this_packet",
        "candidate_entities": entities,
        "candidate_topics": topics,
        "candidate_claims": [],
        "candidate_events": [],
        "boundary": "Claim/event/entity extraction belongs to the staged extraction contract; this packet only exposes candidate entities from reviewed bridge metadata.",
    }


def _sector_exposure_map(bridge: dict[str, Any], ticker_map: list[dict[str, Any]]) -> dict[str, Any]:
    sector = bridge.get("sector_thesis", {})
    entities = []
    for item in ticker_map:
        review_status = item.get("review_status")
        if review_status in {"review_ready", "review_ready_with_evidence_limits"}:
            relation = "direct_ticker_evidence_candidate"
        elif review_status == "context_only_not_directional":
            relation = "ticker_context_only"
        elif review_status == "blocked_missing_direct_evidence":
            relation = "blocked_missing_direct_evidence"
        else:
            relation = "sector_context_only"
        entities.append(
            {
                "entity_id": item.get("ticker"),
                "entity_type": "ticker",
                "domain_profile": sector.get("domain_profile"),
                "sector": sector.get("sector"),
                "evidence_relation": relation,
                "allowed_use": item.get("allowed_use"),
                "review_status": review_status,
                "risk_flags": item.get("risk_and_counter_thesis_flags", []),
            }
        )
    return {
        "domain_profile": sector.get("domain_profile"),
        "sector": sector.get("sector"),
        "entities": entities,
        "boundary": "Exposure mapping is not allocation, recommendation, or position sizing.",
    }


def _domain_specialist_review_contract(bridge: dict[str, Any]) -> dict[str, Any]:
    contract = bridge.get("domain_analyst_contract", {})
    return {
        "profile_id": contract.get("profile_id") or bridge.get("sector_thesis", {}).get("domain_profile"),
        "sector": contract.get("sector") or bridge.get("sector_thesis", {}).get("sector"),
        "packet_type": "DomainSpecialistReviewPacket",
        "required_review_sections": [
            "sector_thesis",
            "ticker_review_map",
            "direct_ticker_evidence",
            "blocked_or_context_windows",
            "risks_and_counter_thesis",
            "explicit_non_actions",
        ],
        "boundary_rule": contract.get(
            "rule",
            "A sector thesis may propose ticker candidates, but direct ticker evidence is required before a ticker thesis candidate can be reviewed.",
        ),
        "allowed_output": "human_review_packet_json_and_markdown",
        "disallowed_outputs": [
            "learning_write",
            "analyst_weight_change",
            "production_config_write",
            "recommendation_or_rating",
            "price_target",
            "position_size",
            "broker_or_order_action",
        ],
    }


def _sector_thesis_section(bridge: dict[str, Any]) -> dict[str, Any]:
    sector = bridge.get("sector_thesis", {})
    return {
        "domain_profile": sector.get("domain_profile"),
        "sector": sector.get("sector"),
        "thesis_level": sector.get("thesis_level"),
        "sector_stance": sector.get("sector_stance"),
        "thesis": sector.get("thesis"),
        "source_run_count": sector.get("source_run_count"),
        "research_stance_counts": sector.get("research_stance_counts", {}),
        "exam_verdict_counts": sector.get("exam_verdict_counts", {}),
        "evidence_quality_counts": sector.get("evidence_quality_counts", {}),
        "warnings": sector.get("warnings", []),
        "review_boundary": "This is a sector thesis; it is not a ticker thesis or recommendation.",
    }


def _ticker_review_item(candidate: dict[str, Any], mapping_runs: list[dict[str, Any]]) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").upper()
    candidate_runs = [run for run in mapping_runs if str(run.get("price_ticker") or "").upper() == ticker]
    review_status = _ticker_review_status(candidate)
    return {
        "ticker": ticker,
        "candidate_status": candidate.get("candidate_status"),
        "review_status": review_status,
        "thesis_level": _ticker_thesis_level(candidate),
        "allowed_use": _allowed_use(review_status),
        "disallowed_use": [
            "recommendation",
            "rating",
            "price_target",
            "position_sizing",
            "learning_promotion",
            "analyst_weight_change",
            "trading",
        ],
        "direct_evidence": {
            "overlay_ready_runs": int(candidate.get("overlay_ready_runs", 0) or 0),
            "directional_ready_runs": int(candidate.get("directional_ready_runs", 0) or 0),
            "neutral_ready_runs": int(candidate.get("neutral_ready_runs", 0) or 0),
            "supporting_as_of": candidate.get("supporting_as_of", []),
            "focused_direction_counts": candidate.get("focused_direction_counts", {}),
            "dominant_focused_stance": candidate.get("dominant_focused_stance"),
        },
        "blocked_evidence": {
            "blocked_runs": int(candidate.get("blocked_runs", 0) or 0),
            "blocked_as_of": candidate.get("blocked_as_of", []),
            "limitations": candidate.get("limitations", []),
        },
        "context_metrics": {
            "runs": int(candidate.get("runs", 0) or 0),
            "exam_verdict_counts": candidate.get("exam_verdict_counts", {}),
            "outcome_counts": candidate.get("outcome_counts", {}),
            "hit_rate_context": candidate.get("hit_rate_context"),
            "average_realized_return_context": candidate.get("average_realized_return_context"),
        },
        "sector_context": candidate.get("sector_context", {}),
        "ticker_specific_evidence": candidate.get(
            "ticker_specific_evidence", {}
        ),
        "stage5_prediction_review": candidate.get(
            "stage5_prediction_review", {}
        ),
        "feature_timeframe_audit": candidate.get(
            "feature_timeframe_audit", {}
        ),
        "exact_pipeline_contexts": candidate.get(
            "exact_pipeline_contexts", []
        ),
        "required_next_inputs": candidate.get(
            "required_next_inputs", []
        ),
        "can_create_ticker_forecast": candidate.get(
            "can_create_ticker_forecast", False
        ),
        "risk_and_counter_thesis_flags": _risk_flags(candidate, candidate_runs),
        "review_questions": _review_questions(candidate, review_status),
    }


def _ticker_review_status(candidate: dict[str, Any]) -> str:
    status = candidate.get("candidate_status")
    blocked_runs = int(candidate.get("blocked_runs", 0) or 0)
    directional = int(candidate.get("directional_ready_runs", 0) or 0)
    if status == "direct_ticker_thesis_ready" and blocked_runs:
        return "review_ready_with_evidence_limits"
    if status == "direct_ticker_thesis_ready" and directional:
        return "review_ready"
    if status == "ticker_context_ready":
        return "context_only_not_directional"
    if status == "ticker_evidence_ready_pipeline_blocked":
        return "ticker_evidence_ready_pipeline_blocked"
    if status == "blocked_missing_ticker_evidence":
        return "blocked_missing_direct_evidence"
    return "sector_context_only"


def _ticker_thesis_level(candidate: dict[str, Any]) -> str:
    status = candidate.get("candidate_status")
    if status == "direct_ticker_thesis_ready":
        return "direct_ticker_thesis_candidate_for_manual_review"
    if status == "ticker_context_ready":
        return "ticker_context_only"
    if status == "ticker_evidence_ready_pipeline_blocked":
        return "direct_ticker_evidence_context_pipeline_blocked"
    if status == "blocked_missing_ticker_evidence":
        return "blocked_until_direct_ticker_evidence"
    return "sector_context_only"


def _allowed_use(review_status: str) -> str:
    if review_status == "review_ready":
        return "manual_review_of_ticker_candidate_only"
    if review_status == "review_ready_with_evidence_limits":
        return "manual_review_with_blocked_window_callouts_only"
    if review_status == "ticker_evidence_ready_pipeline_blocked":
        return "manual_review_of_ticker_evidence_not_forecast"
    return "review_context_only"


def _risk_flags(candidate: dict[str, Any], runs: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    if int(candidate.get("blocked_runs", 0) or 0) > 0:
        flags.append("some_windows_blocked_by_weak_direct_evidence")
    if int(candidate.get("neutral_ready_runs", 0) or 0) > 0:
        flags.append("some_ready_windows_are_neutral_context_not_directional")
    outcome_counts = candidate.get("outcome_counts", {})
    if int(outcome_counts.get("miss", 0) or 0) > 0:
        flags.append("historical_context_contains_miss_outcomes")
    if any(run.get("research_stance") in {"mixed", "insufficient_data"} for run in runs):
        flags.append("research_stance_is_not_uniformly_constructive")
    if candidate.get("candidate_status") == (
        "ticker_evidence_ready_pipeline_blocked"
    ):
        flags.append(
            "company_mechanism_corroborated_but_pipeline_not_ready"
        )
    prediction_status = (
        candidate.get("stage5_prediction_review") or {}
    ).get("status")
    if prediction_status == "prediction_review_quarantined":
        flags.append(
            "real_stage5_output_exists_but_is_quarantined"
        )
    elif prediction_status == "prediction_review_partial":
        flags.append(
            "real_stage5_output_contains_quarantined_contexts"
        )
    timeframe_status = (
        candidate.get("feature_timeframe_audit") or {}
    ).get("status")
    if timeframe_status in {
        "timeframe_cadence_mismatch",
        "timeframe_cadence_ambiguous",
    }:
        flags.append(
            "feature_timeframe_conflicts_with_observed_cadence"
        )
    if not flags:
        flags.append("no_packet_level_counter_flag_detected")
    return flags


def _review_questions(candidate: dict[str, Any], review_status: str) -> list[str]:
    ticker = candidate.get("ticker")
    questions = [
        f"Does the direct evidence actually refer to {ticker}, not only the broader sector?",
        "Are neutral or mixed windows being kept out of directional thesis language?",
    ]
    if review_status == "review_ready_with_evidence_limits":
        questions.append("Do blocked windows materially weaken the ticker candidate, or should they trigger evidence backfill first?")
    elif review_status.startswith("blocked") or review_status.endswith("context_only"):
        questions.append("What direct ticker evidence is missing before this can become a reviewable ticker thesis candidate?")
    elif review_status == "ticker_evidence_ready_pipeline_blocked":
        questions.append(
            "Which exact prediction, evaluation, or outcome-calibration "
            "input still blocks forecast use?"
        )
    else:
        questions.append("Are risks and counter-thesis evidence visible enough for manual review?")
    prediction = candidate.get("stage5_prediction_review") or {}
    if prediction.get("status") in {
        "prediction_review_quarantined",
        "prediction_review_partial",
    }:
        questions.append(
            "Has Stage 5 been regenerated through the repaired lineage "
            "path instead of mutating the quarantined output?"
        )
    timeframe_audit = candidate.get(
        "feature_timeframe_audit"
    ) or {}
    if timeframe_audit.get("status") in {
        "timeframe_cadence_mismatch",
        "timeframe_cadence_ambiguous",
    }:
        questions.append(
            "Has Stage 2/3 been regenerated with cadence-validated "
            "timeframe before any model or prediction reuse?"
        )
    return questions


def _direct_ticker_evidence(mapping_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_run_review_slice(run) for run in mapping_runs if run.get("ticker_signal_level") == "direct_ticker_thesis"]


def _blocked_or_context_windows(mapping_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_run_review_slice(run) for run in mapping_runs if run.get("ticker_signal_level") != "direct_ticker_thesis"]


def _run_review_slice(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "as_of": run.get("as_of"),
        "horizon_days": run.get("horizon_days"),
        "ticker": run.get("price_ticker"),
        "sector_signal_level": run.get("sector_signal_level"),
        "ticker_signal_level": run.get("ticker_signal_level"),
        "research_stance": run.get("research_stance"),
        "research_expected_direction": run.get("research_expected_direction"),
        "ticker_specificity": run.get("ticker_specificity"),
        "exam_verdict": run.get("exam_verdict"),
        "focused_overlay_status": run.get("focused_overlay_status"),
        "focused_overlay_applied": run.get("focused_overlay_applied"),
        "outcome_label": run.get("outcome_label"),
        "realized_return": run.get("realized_return"),
    }


def _review_checks(bridge: dict[str, Any], ticker_map: list[dict[str, Any]]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    if bridge.get("mode") == "sector_thesis_to_ticker_basket_bridge":
        checks.append(_check("pass", "bridge_mode_valid", "Input is a sector-to-ticker bridge artifact."))
    else:
        checks.append(_check("fail", "bridge_mode_invalid", f"Unexpected bridge mode: {bridge.get('mode')}."))

    sector = bridge.get("sector_thesis", {})
    if sector.get("thesis_level") == "sector_thesis":
        checks.append(_check("pass", "sector_thesis_present", "Sector thesis is explicitly marked as sector_thesis."))
    else:
        checks.append(_check("fail", "sector_thesis_missing", "Input does not include a valid sector thesis."))

    direct_items = [item for item in ticker_map if item.get("review_status") in {"review_ready", "review_ready_with_evidence_limits"}]
    if direct_items:
        checks.append(_check("pass", "direct_ticker_candidates_present", f"{len(direct_items)} ticker candidate(s) have direct ticker evidence for manual review."))
    else:
        checks.append(
            _check(
                "warn",
                "no_direct_ticker_candidates",
                (
                    "No ticker candidate has direct ticker evidence; the "
                    "packet is reviewable as a readiness-gap map only."
                ),
            )
        )

    limited_items = [item for item in ticker_map if item.get("review_status") == "review_ready_with_evidence_limits"]
    if limited_items:
        tickers = ", ".join(item.get("ticker", "") for item in limited_items)
        checks.append(_check("warn", "limited_candidates_present", f"Some review-ready candidates still have blocked windows: {tickers}."))

    context_only = [item for item in ticker_map if item.get("review_status") not in {"review_ready", "review_ready_with_evidence_limits"}]
    if context_only:
        tickers = ", ".join(item.get("ticker", "") for item in context_only)
        checks.append(_check("warn", "context_or_blocked_candidates_present", f"Some candidates are context-only or blocked: {tickers}."))

    safety = bridge.get("safety", {})
    if safety.get("read_only") is True and not any(
        safety.get(flag)
        for flag in [
            "data_mutation_performed",
            "collector_run_performed",
            "network_access_performed",
            "pipeline_run_performed",
            "learning_write_performed",
            "operation_proposal_created",
            "config_write_performed",
            "broker_access_performed",
        ]
    ):
        checks.append(_check("pass", "bridge_safety_read_only", "Bridge artifact reports read-only execution with no side effects."))
    else:
        checks.append(_check("fail", "bridge_safety_violation", "Bridge artifact safety flags are not read-only."))

    summary = bridge.get("summary", {})
    if summary.get("can_write_learning_memory") is False and summary.get("can_change_analyst_weights") is False:
        checks.append(_check("pass", "learning_and_weight_changes_disabled", "Bridge summary disables learning writes and analyst weight changes."))
    else:
        checks.append(_check("fail", "learning_or_weight_change_enabled", "Bridge summary allows learning writes or analyst weight changes."))
    return checks


def _domain_review_checks(
    bridge: dict[str, Any],
    ticker_map: list[dict[str, Any]],
    source_gate: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    if bridge.get("mode") == "sector_thesis_to_ticker_basket_bridge":
        checks.append(_check("pass", "bridge_mode_valid", "Input is a sector-to-ticker bridge artifact."))
    else:
        checks.append(_check("fail", "bridge_mode_invalid", f"Unexpected bridge mode: {bridge.get('mode')}."))

    sector = bridge.get("sector_thesis", {})
    if sector.get("thesis_level") == "sector_thesis" and sector.get("thesis"):
        checks.append(_check("pass", "domain_thesis_present", "Domain/sector thesis is available for manual review."))
    else:
        checks.append(_check("fail", "domain_thesis_missing", "Input does not include a reviewable domain/sector thesis."))

    if sector.get("warnings"):
        checks.append(_check("warn", "domain_evidence_warnings_present", "Domain thesis has evidence warnings that must be reviewed."))

    if source_gate:
        gate_summary = source_gate.get("summary", {})
        gate_guidance = source_gate.get("decision_guidance", {})
        if source_gate.get("mode") == "source_evidence_validation_gate":
            checks.append(_check("pass", "source_gate_mode_valid", "Source gate artifact is attached."))
        else:
            checks.append(_check("fail", "source_gate_mode_invalid", f"Unexpected source gate mode: {source_gate.get('mode')}."))

        if gate_summary.get("can_enter_domain_research") is True and not gate_guidance.get("fail_count"):
            checks.append(_check("pass", "source_gate_allows_domain_research", "Source gate allows manual domain research."))
        else:
            checks.append(_check("fail", "source_gate_blocks_domain_research", "Source gate does not allow manual domain research."))

        if gate_guidance.get("warning_count"):
            checks.append(_check("warn", "source_gate_warnings_present", "Source gate has warnings that must be reviewed before standardization."))

        if (
            gate_summary.get("can_promote_to_evidence") is False
            and gate_summary.get("can_extract_claims_events_entities") is False
            and gate_summary.get("can_trade") is False
        ):
            checks.append(_check("pass", "source_gate_downstream_actions_disabled", "Source gate disables evidence promotion, extraction, and trading."))
        else:
            checks.append(_check("fail", "source_gate_downstream_action_enabled", "Source gate allows a downstream action that must stay disabled."))
    else:
        checks.append(_check("warn", "source_gate_not_attached", "No source evidence validation gate was attached to this domain packet."))

    direct_items = [item for item in ticker_map if item.get("review_status") in {"review_ready", "review_ready_with_evidence_limits"}]
    if direct_items:
        checks.append(_check("pass", "ticker_bridge_has_direct_candidates", f"{len(direct_items)} ticker candidate(s) have direct ticker evidence."))
    else:
        checks.append(_check("warn", "ticker_bridge_has_no_direct_candidates", "Domain review can proceed, but ticker candidate review is blocked until direct ticker evidence exists."))

    limited_items = [item for item in ticker_map if item.get("review_status") == "review_ready_with_evidence_limits"]
    if limited_items:
        tickers = ", ".join(item.get("ticker", "") for item in limited_items)
        checks.append(_check("warn", "ticker_bridge_limited_candidates_present", f"Some ticker bridge candidates still have blocked windows: {tickers}."))

    safety = bridge.get("safety", {})
    if safety.get("read_only") is True and not any(
        safety.get(flag)
        for flag in [
            "data_mutation_performed",
            "collector_run_performed",
            "network_access_performed",
            "pipeline_run_performed",
            "learning_write_performed",
            "operation_proposal_created",
            "config_write_performed",
            "broker_access_performed",
        ]
    ):
        checks.append(_check("pass", "bridge_safety_read_only", "Bridge artifact reports read-only execution with no side effects."))
    else:
        checks.append(_check("fail", "bridge_safety_violation", "Bridge artifact safety flags are not read-only."))

    summary = bridge.get("summary", {})
    if summary.get("can_write_learning_memory") is False and summary.get("can_change_analyst_weights") is False:
        checks.append(_check("pass", "learning_and_weight_changes_disabled", "Bridge summary disables learning writes and analyst weight changes."))
    else:
        checks.append(_check("fail", "learning_or_weight_change_enabled", "Bridge summary allows learning writes or analyst weight changes."))

    checks.append(_check("pass", "claim_extraction_not_executed", "This packet does not execute claim/event/entity extraction."))
    return checks


def _decision_guidance(checks: list[dict[str, str]], ticker_map: list[dict[str, Any]]) -> dict[str, Any]:
    fails = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    if fails:
        status = "blocked"
        action = "do_not_review_until_input_is_fixed"
        can_review = False
    elif warnings:
        status = "review_ready_with_limitations"
        action = "manual_review_with_evidence_limitations"
        can_review = True
    else:
        status = "review_ready"
        action = "manual_review"
        can_review = bool(ticker_map)
    return {
        "status": status,
        "recommended_review_action": action,
        "can_enter_manual_sector_to_ticker_review": can_review,
        "fail_count": len(fails),
        "warning_count": len(warnings),
        "pass_count": sum(1 for check in checks if check["status"] == "pass"),
        "reasons": [check["message"] for check in [*fails, *warnings] if check.get("message")],
    }


def _domain_decision_guidance(checks: list[dict[str, str]], ticker_map: list[dict[str, Any]]) -> dict[str, Any]:
    fails = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    direct_items = [item for item in ticker_map if item.get("review_status") in {"review_ready", "review_ready_with_evidence_limits"}]
    source_warnings = [check for check in warnings if check.get("code", "").startswith("source_gate_")]
    bridge_warnings = [check for check in warnings if check.get("code", "").startswith("ticker_bridge_")]
    if fails:
        status = "blocked"
        action = "do_not_review_until_input_is_fixed"
        can_domain_review = False
        can_ticker_review = False
        can_standardize = False
    elif warnings:
        status = "domain_review_ready_with_limitations"
        if source_warnings and bridge_warnings:
            action = "manual_domain_review_with_source_and_bridge_limitations"
        elif source_warnings:
            action = "manual_domain_review_with_source_limitations"
        else:
            action = "manual_domain_review_with_bridge_limitations"
        can_domain_review = True
        can_ticker_review = bool(direct_items)
        can_standardize = False
    else:
        status = "domain_review_ready"
        action = "manual_domain_review"
        can_domain_review = True
        can_ticker_review = bool(direct_items)
        can_standardize = bool(direct_items)
    return {
        "status": status,
        "recommended_review_action": action,
        "can_enter_manual_domain_review": can_domain_review,
        "can_enter_ticker_candidate_review": can_ticker_review,
        "can_standardize_domain_template": can_standardize,
        "fail_count": len(fails),
        "warning_count": len(warnings),
        "pass_count": sum(1 for check in checks if check["status"] == "pass"),
        "reasons": [check["message"] for check in [*fails, *warnings] if check.get("message")],
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No learning memory write is performed.",
        "No analyst profile weight is changed.",
        "No production config is written.",
        "No operation proposal is queued.",
        "No collector, source retrieval, or network call is performed.",
        "No pipeline run is triggered.",
        "No recommendation, rating, price target, or position size is created.",
        "No broker, order, paper-trading, or live-trading action is performed.",
    ]


def _commands(bridge_path: str | Path) -> dict[str, str]:
    return {
        "rerun_packet": (
            "python run_agent_sector_to_ticker_review_packet.py "
            f"--bridge-json {bridge_path} "
            "--output-dir reports\\dean_os\\sector_to_ticker_review_packet_current"
        )
    }


def _domain_commands(bridge_path: str | Path, source_gate_path: str | Path | None = None) -> dict[str, str]:
    source_arg = f"--source-gate-json {source_gate_path} " if source_gate_path else ""
    return {
        "rerun_domain_packet": (
            "python run_agent_domain_specialist_review_packet.py "
            f"--bridge-json {bridge_path} "
            f"{source_arg}"
            "--output-dir reports\\dean_os\\domain_specialist_review_packet_current"
        ),
        "rerun_sector_to_ticker_packet": (
            "python run_agent_sector_to_ticker_review_packet.py "
            f"--bridge-json {bridge_path} "
            "--output-dir reports\\dean_os\\sector_to_ticker_review_packet_current"
        ),
    }


def _operator_next_steps(guidance: dict[str, Any], blocked_or_limited: list[dict[str, Any]]) -> list[str]:
    if guidance["status"] == "blocked":
        return [
            "Do not approve a sector-to-ticker review packet from this input.",
            "Fix bridge safety, input shape, or direct ticker evidence first.",
            "Rerun the bridge and rebuild this packet.",
        ]
    steps = [
        "Inspect sector thesis language separately from ticker candidate language.",
        "Verify each direct ticker evidence window before treating a ticker as manually reviewable.",
        "Keep this packet out of learning promotion, analyst weighting, recommendations, and trading.",
    ]
    if blocked_or_limited:
        tickers = ", ".join(sorted({item.get("ticker", "") for item in blocked_or_limited if item.get("ticker")}))
        steps.append(f"Resolve or explicitly accept blocked/context-only limitations before expanding the pattern: {tickers}.")
    return steps


def _domain_operator_next_steps(
    guidance: dict[str, Any],
    blocked_or_limited: list[dict[str, Any]],
    source_gate: dict[str, Any] | None = None,
) -> list[str]:
    if guidance["status"] == "blocked":
        return [
            "Do not approve a domain specialist review packet from this input.",
            "Fix bridge safety, input shape, or missing domain thesis first.",
            "Rebuild the bridge and domain packet after the input is fixed.",
        ]
    steps = [
        "Review the domain thesis as sector/theme context first, independent of ticker candidates.",
        "Review the source gate status and warning sample before accepting the domain packet.",
        "Review candidate entities as an exposure map, not as allocation or recommendation output.",
        "Use the sector-to-ticker bridge only to decide which entities have direct ticker evidence for a separate ticker review.",
        "Keep this packet out of learning promotion, analyst weighting, recommendations, and trading.",
    ]
    if not source_gate:
        steps.append("Attach the source evidence validation gate before treating this as the standard domain template.")
    elif source_gate.get("summary", {}).get("gate_status") == "source_evidence_ready_with_warnings":
        steps.append("Resolve or explicitly accept source gate warnings before standardizing this domain template.")
    if not guidance.get("can_enter_ticker_candidate_review"):
        steps.append("Keep ticker candidate review blocked until direct ticker evidence is available.")
    if blocked_or_limited:
        tickers = ", ".join(sorted({item.get("ticker", "") for item in blocked_or_limited if item.get("ticker")}))
        steps.append(f"Resolve or explicitly accept ticker bridge limitations before standardizing the template: {tickers}.")
    return steps


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
