from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_SOURCE_EXTRACTION_CONTRACT_JSON = "reports/dean_os/source_extraction_review_packet_current/latest.json"
FIXTURE_CONTRACT_ID = "248_review_only_real_source_claim_event_entity_extraction_fixture_v1"
UPSTREAM_CONTRACT_ID = "247_review_only_real_source_claim_event_entity_extraction_contract_v1"


class SourceExtractionFixturePacket:
    """Fixture-only candidate extraction packet over a reviewed extraction contract."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/source_extraction_fixture_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        contract_json: str | Path = DEFAULT_SOURCE_EXTRACTION_CONTRACT_JSON,
        max_items: int = 12,
        prefer_timestamped: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        contract_packet = _load_json(contract_json)
        anchors = contract_packet.get("source_anchor_plan", {}).get("anchors", [])
        selected_anchors = _select_anchors(anchors, max_items=max_items, prefer_timestamped=prefer_timestamped)
        candidate_claims = [_claim_fixture(anchor, index) for index, anchor in enumerate(selected_anchors, start=1)]
        candidate_events = [_event_fixture(anchor, index) for index, anchor in enumerate(selected_anchors, start=1)]
        candidate_entities = [
            entity
            for index, anchor in enumerate(selected_anchors, start=1)
            for entity in _entity_fixtures(anchor, index)
        ]
        candidate_implications = [_financial_implication_fixture(anchor, index) for index, anchor in enumerate(selected_anchors, start=1)]
        checks = _review_checks(contract_packet, anchors, selected_anchors, candidate_claims, candidate_events, candidate_entities, candidate_implications)
        guidance = _decision_guidance(checks)
        payload = {
            "run_id": _run_id("source_extraction_fixture_packet"),
            "created_at": utc_now_iso(),
            "mode": "source_extraction_fixture_packet",
            "inputs": {
                "contract_json": str(contract_json),
                "contract_run_id": contract_packet.get("run_id"),
                "max_items": max_items,
                "prefer_timestamped": prefer_timestamped,
            },
            "summary": {
                "packet_status": guidance["status"],
                "recommended_review_action": guidance["recommended_review_action"],
                "fixture_contract_id": FIXTURE_CONTRACT_ID,
                "upstream_contract_id": contract_packet.get("summary", {}).get("contract_id"),
                "upstream_packet_status": contract_packet.get("summary", {}).get("packet_status"),
                "available_anchor_count": len(anchors),
                "selected_anchor_count": len(selected_anchors),
                "selected_missing_timestamp_count": sum(1 for anchor in selected_anchors if anchor.get("timestamp_status") != "present"),
                "candidate_claim_fixture_count": len(candidate_claims),
                "candidate_event_fixture_count": len(candidate_events),
                "candidate_entity_fixture_count": len(candidate_entities),
                "candidate_financial_implication_fixture_count": len(candidate_implications),
                "can_enter_manual_fixture_review": guidance["can_enter_manual_fixture_review"],
                "can_execute_real_extraction": False,
                "can_emit_claims_events_entities_as_evidence": False,
                "can_promote_to_evidence": False,
                "can_write_learning_memory": False,
                "can_change_analyst_weights": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
            "fixture_boundary": _fixture_boundary(contract_packet),
            "selected_source_anchors": selected_anchors,
            "candidate_claim_fixtures": candidate_claims,
            "candidate_event_fixtures": candidate_events,
            "candidate_entity_fixtures": candidate_entities,
            "candidate_financial_implication_fixtures": candidate_implications,
            "review_checks": checks,
            "decision_guidance": guidance,
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(contract_json, max_items, prefer_timestamped),
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
        rendered_md = render_source_extraction_fixture_packet_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_source_extraction_fixture_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Source Extraction Fixture Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Fixture contract ID: `{summary.get('fixture_contract_id')}`",
        f"- Upstream contract ID: `{summary.get('upstream_contract_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Recommended action: `{summary.get('recommended_review_action')}`",
        f"- Upstream packet status: `{summary.get('upstream_packet_status')}`",
        f"- Available anchors: {summary.get('available_anchor_count')}",
        f"- Selected anchors: {summary.get('selected_anchor_count')}",
        f"- Selected missing timestamps: {summary.get('selected_missing_timestamp_count')}",
        f"- Candidate claim fixtures: {summary.get('candidate_claim_fixture_count')}",
        f"- Candidate event fixtures: {summary.get('candidate_event_fixture_count')}",
        f"- Candidate entity fixtures: {summary.get('candidate_entity_fixture_count')}",
        f"- Candidate financial implication fixtures: {summary.get('candidate_financial_implication_fixture_count')}",
        f"- Can execute real extraction: {summary.get('can_execute_real_extraction')}",
        f"- Can emit claims/events/entities as evidence: {summary.get('can_emit_claims_events_entities_as_evidence')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Fixture Boundary",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("fixture_boundary", {}).get("boundary_rules", []))
    lines.extend(["", "## Selected Anchor Samples", ""])
    lines.extend(_render_anchor_samples(payload.get("selected_source_anchors", [])))
    lines.extend(["", "## Candidate Claim Fixture Samples", ""])
    lines.extend(_render_claim_samples(payload.get("candidate_claim_fixtures", [])))
    lines.extend(["", "## Candidate Event Fixture Samples", ""])
    lines.extend(_render_event_samples(payload.get("candidate_event_fixtures", [])))
    lines.extend(["", "## Review Checks", ""])
    lines.extend(_render_check_samples(payload.get("review_checks", [])))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    lines.extend(["", "## Decision Rationale", ""])
    lines.extend(_render_reason_samples(guidance.get("reasons", [])))
    return "\n".join(lines).strip() + "\n"


def _select_anchors(anchors: list[dict[str, Any]], max_items: int, prefer_timestamped: bool) -> list[dict[str, Any]]:
    limit = max(0, int(max_items or 0))
    if limit == 0:
        return []
    if prefer_timestamped:
        buckets = [
            [anchor for anchor in anchors if anchor.get("timestamp_status") == "present" and anchor.get("candidate_entities")],
            [anchor for anchor in anchors if anchor.get("timestamp_status") != "present" and anchor.get("candidate_entities")],
            [anchor for anchor in anchors if anchor.get("timestamp_status") == "present" and not anchor.get("candidate_entities")],
            [anchor for anchor in anchors if anchor.get("timestamp_status") != "present" and not anchor.get("candidate_entities")],
        ]
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for bucket in buckets:
            for anchor in bucket:
                anchor_id = str(anchor.get("anchor_id"))
                if anchor_id in seen:
                    continue
                selected.append(anchor)
                seen.add(anchor_id)
                if len(selected) == limit:
                    return selected
        return selected
    return anchors[:limit]


def _claim_fixture(anchor: dict[str, Any], index: int) -> dict[str, Any]:
    anchor_id = anchor.get("anchor_id")
    return {
        "claim_id": f"claim_fixture_{index:04d}",
        "claim_text": anchor.get("text_preview") or anchor.get("title") or "Fixture source text unavailable.",
        "claim_type": "source_statement_candidate_fixture",
        "polarity": "unknown_not_interpreted",
        "source_anchor_id": anchor_id,
        "entity_refs": _entity_refs(anchor),
        "topic_refs": anchor.get("candidate_topics", []),
        "confidence": "fixture_only_no_confidence_score",
        "limitations": _common_limitations(anchor),
        "fixture_status": "candidate_shape_only_not_evidence",
    }


def _event_fixture(anchor: dict[str, Any], index: int) -> dict[str, Any]:
    timestamp_present = anchor.get("timestamp_status") == "present"
    return {
        "event_id": f"event_fixture_{index:04d}",
        "event_type": "unspecified_source_event_candidate_fixture",
        "event_time": anchor.get("published_at") if timestamp_present else None,
        "source_anchor_id": anchor.get("anchor_id"),
        "entity_refs": _entity_refs(anchor),
        "time_confidence": "source_timestamp_present" if timestamp_present else "missing_source_timestamp",
        "limitations": _common_limitations(anchor),
        "fixture_status": "candidate_shape_only_not_evidence",
    }


def _entity_fixtures(anchor: dict[str, Any], index: int) -> list[dict[str, Any]]:
    fixtures = []
    for entity_index, entity in enumerate(anchor.get("candidate_entities", []), start=1):
        fixtures.append(
            {
                "entity_ref_id": f"entity_fixture_{index:04d}_{entity_index:02d}",
                "surface_form": entity,
                "entity_type": "ticker_candidate",
                "symbol_or_identifier": entity,
                "resolution_status": "metadata_candidate_not_resolved",
                "source_anchor_id": anchor.get("anchor_id"),
                "limitations": _common_limitations(anchor),
                "fixture_status": "candidate_shape_only_not_evidence",
            }
        )
    return fixtures


def _financial_implication_fixture(anchor: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "implication_id": f"implication_fixture_{index:04d}",
        "scope": "source_context_candidate_fixture",
        "direction": "unknown_not_interpreted",
        "horizon": "unknown_not_interpreted",
        "source_anchor_id": anchor.get("anchor_id"),
        "reasoning_boundary": "Fixture only; no financial implication, recommendation, valuation, price target, allocation, or trade signal is inferred.",
        "limitations": _common_limitations(anchor),
        "fixture_status": "candidate_shape_only_not_evidence",
    }


def _entity_refs(anchor: dict[str, Any]) -> list[str]:
    return [f"entity::{entity}" for entity in anchor.get("candidate_entities", [])]


def _common_limitations(anchor: dict[str, Any]) -> list[str]:
    limitations = [
        "fixture_only_not_production_evidence",
        "semantic_extraction_not_performed",
        "manual_review_required",
    ]
    if anchor.get("timestamp_status") != "present":
        limitations.append("source_timestamp_missing_or_not_real")
    if not anchor.get("text_present"):
        limitations.append("source_text_missing")
    return limitations


def _fixture_boundary(contract_packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_contract_id": FIXTURE_CONTRACT_ID,
        "upstream_contract_id": contract_packet.get("summary", {}).get("contract_id"),
        "fixture_status": "review_only_extraction_candidate_fixture_not_evidence",
        "real_extraction_performed": False,
        "fixtures_are_production_evidence": False,
        "claims_emitted_as_evidence": False,
        "events_emitted_as_evidence": False,
        "entities_resolved_as_evidence": False,
        "financial_implications_emitted_as_evidence": False,
        "event_propagation_executed": False,
        "company_thesis_generated": False,
        "valuation_generated": False,
        "recommendation_generated": False,
        "trade_signal_generated": False,
        "boundary_rules": [
            "This packet materializes candidate output shapes only; it is not production extraction.",
            "Candidate claim/event/entity/financial implication fixtures are not evidence.",
            "Fixture text may mirror source previews only to test anchoring and required fields.",
            "No semantic interpretation, event propagation, thesis, valuation, recommendation, allocation, or trading action is performed.",
        ],
    }


def _review_checks(
    contract_packet: dict[str, Any],
    anchors: list[dict[str, Any]],
    selected_anchors: list[dict[str, Any]],
    candidate_claims: list[dict[str, Any]],
    candidate_events: list[dict[str, Any]],
    candidate_entities: list[dict[str, Any]],
    candidate_implications: list[dict[str, Any]],
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    summary = contract_packet.get("summary", {})
    if contract_packet.get("mode") == "source_extraction_review_packet":
        checks.append(_check("pass", "contract_packet_mode_valid", "Input is a source extraction review packet."))
    else:
        checks.append(_check("fail", "contract_packet_mode_invalid", f"Unexpected contract packet mode: {contract_packet.get('mode')}."))

    if summary.get("contract_id") == UPSTREAM_CONTRACT_ID:
        checks.append(_check("pass", "upstream_contract_id_valid", "Upstream extraction contract id is valid."))
    else:
        checks.append(_check("fail", "upstream_contract_id_invalid", f"Unexpected upstream contract id: {summary.get('contract_id')}."))

    if summary.get("packet_status") == "blocked_extraction_contract":
        checks.append(_check("fail", "upstream_contract_blocked", "Upstream extraction contract is blocked."))
    elif summary.get("packet_status") == "extraction_contract_ready_with_warnings":
        checks.append(_check("warn", "upstream_contract_has_warnings", "Upstream extraction contract has warnings that must remain visible."))
    else:
        checks.append(_check("pass", "upstream_contract_reviewable", "Upstream extraction contract is reviewable."))

    if anchors:
        checks.append(_check("pass", "anchors_available", f"{len(anchors)} source anchor(s) are available."))
    else:
        checks.append(_check("fail", "anchors_missing", "No source anchors are available."))

    if selected_anchors:
        checks.append(_check("pass", "fixture_subset_selected", f"{len(selected_anchors)} anchor(s) selected for fixture materialization."))
    else:
        checks.append(_check("fail", "fixture_subset_empty", "No anchors were selected for fixture materialization."))

    missing_timestamps = [anchor for anchor in selected_anchors if anchor.get("timestamp_status") != "present"]
    if missing_timestamps:
        checks.append(_check("warn", "selected_timestamps_missing", f"{len(missing_timestamps)} selected anchor(s) lack real timestamps."))

    missing_text = [anchor for anchor in selected_anchors if not anchor.get("text_present")]
    if missing_text:
        checks.append(_check("fail", "selected_text_missing", f"{len(missing_text)} selected anchor(s) lack source text."))

    _check_candidate_counts(checks, selected_anchors, candidate_claims, candidate_events, candidate_entities, candidate_implications)
    _check_anchor_links(checks, selected_anchors, candidate_claims, candidate_events, candidate_entities, candidate_implications)
    checks.append(_check("pass", "real_extraction_not_performed", "Fixture packet does not perform real extraction."))
    checks.append(_check("pass", "downstream_actions_disabled", "Evidence promotion, learning writes, recommendations, and trading remain disabled."))
    return checks


def _check_candidate_counts(
    checks: list[dict[str, str]],
    selected_anchors: list[dict[str, Any]],
    candidate_claims: list[dict[str, Any]],
    candidate_events: list[dict[str, Any]],
    candidate_entities: list[dict[str, Any]],
    candidate_implications: list[dict[str, Any]],
) -> None:
    if len(candidate_claims) == len(selected_anchors):
        checks.append(_check("pass", "claim_fixture_count_matches_subset", "One claim fixture exists per selected anchor."))
    else:
        checks.append(_check("fail", "claim_fixture_count_mismatch", "Claim fixture count does not match selected anchors."))
    if len(candidate_events) == len(selected_anchors):
        checks.append(_check("pass", "event_fixture_count_matches_subset", "One event fixture exists per selected anchor."))
    else:
        checks.append(_check("fail", "event_fixture_count_mismatch", "Event fixture count does not match selected anchors."))
    if len(candidate_implications) == len(selected_anchors):
        checks.append(_check("pass", "financial_implication_fixture_count_matches_subset", "One financial implication fixture exists per selected anchor."))
    else:
        checks.append(_check("fail", "financial_implication_fixture_count_mismatch", "Financial implication fixture count does not match selected anchors."))
    expected_entities = sum(len(anchor.get("candidate_entities", [])) for anchor in selected_anchors)
    if len(candidate_entities) == expected_entities:
        checks.append(_check("pass", "entity_fixture_count_matches_metadata", "Entity fixture count matches candidate entity metadata."))
    else:
        checks.append(_check("fail", "entity_fixture_count_mismatch", "Entity fixture count does not match candidate entity metadata."))


def _check_anchor_links(
    checks: list[dict[str, str]],
    selected_anchors: list[dict[str, Any]],
    *candidate_groups: list[dict[str, Any]],
) -> None:
    anchor_ids = {anchor.get("anchor_id") for anchor in selected_anchors}
    candidates = [candidate for group in candidate_groups for candidate in group]
    invalid = [candidate for candidate in candidates if candidate.get("source_anchor_id") not in anchor_ids]
    if invalid:
        checks.append(_check("fail", "candidate_anchor_links_invalid", f"{len(invalid)} candidate fixture(s) have invalid source anchors."))
    else:
        checks.append(_check("pass", "candidate_anchor_links_valid", "All candidate fixtures link to selected source anchors."))


def _decision_guidance(checks: list[dict[str, str]]) -> dict[str, Any]:
    fails = [check for check in checks if check.get("status") == "fail"]
    warnings = [check for check in checks if check.get("status") == "warn"]
    if fails:
        status = "blocked_extraction_fixture"
        action = "fix_contract_or_fixture_shape_before_review"
        can_review = False
    elif warnings:
        status = "extraction_fixture_ready_with_warnings"
        action = "manual_fixture_review_with_limitations"
        can_review = True
    else:
        status = "extraction_fixture_ready_for_manual_review"
        action = "manual_fixture_review"
        can_review = True
    return {
        "status": status,
        "recommended_review_action": action,
        "can_enter_manual_fixture_review": can_review,
        "can_execute_real_extraction": False,
        "can_emit_claims_events_entities_as_evidence": False,
        "fail_count": len(fails),
        "warning_count": len(warnings),
        "pass_count": sum(1 for check in checks if check.get("status") == "pass"),
        "reasons": [check["message"] for check in [*fails, *warnings] if check.get("message")],
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No live fetch, connector fetch, or external API call is performed.",
        "No real claim/event/entity extraction is executed.",
        "No candidate fixture is promoted to production evidence.",
        "No event propagation is executed.",
        "No company thesis, valuation, recommendation, price target, allocation, or position size is created.",
        "No learning memory, analyst weight, production config, operation queue, pipeline, broker, or trading action is written.",
    ]


def _commands(contract_json: str | Path, max_items: int, prefer_timestamped: bool) -> dict[str, str]:
    include_missing = " --include-missing-timestamps" if not prefer_timestamped else ""
    return {
        "rerun_packet": (
            "python run_agent_source_extraction_fixture_packet.py "
            f"--contract-json {contract_json} "
            f"--max-items {max_items}"
            f"{include_missing} "
            "--output-dir reports\\dean_os\\source_extraction_fixture_packet_current"
        )
    }


def _recommendations(guidance: dict[str, Any]) -> list[str]:
    if guidance["status"] == "blocked_extraction_fixture":
        return ["Fix upstream contract status, selected anchors, or candidate fixture shape before manual review."]
    recommendations = [
        "Review whether the fixture candidate shapes are sufficient before implementing real extraction.",
        "Keep all candidate fixtures out of evidence promotion, learning, recommendations, and trading.",
        "Use timestamped anchors for event fixtures whenever possible.",
    ]
    if guidance["status"] == "extraction_fixture_ready_with_warnings":
        recommendations.append("Resolve or explicitly accept warnings before using this as the standard fixture pattern.")
    return recommendations


def _render_anchor_samples(items: list[dict[str, Any]], max_items: int = 8) -> list[str]:
    if not items:
        return ["- None."]
    lines = []
    for item in items[:max_items]:
        entities = ", ".join(item.get("candidate_entities", [])) or "none"
        lines.append(f"- `{item.get('anchor_id')}` timestamp=`{item.get('timestamp_status')}` entities={entities}")
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional anchor(s) in JSON.")
    return lines


def _render_claim_samples(items: list[dict[str, Any]], max_items: int = 6) -> list[str]:
    if not items:
        return ["- None."]
    lines = []
    for item in items[:max_items]:
        lines.append(
            f"- `{item.get('claim_id')}` anchor=`{item.get('source_anchor_id')}` "
            f"polarity=`{item.get('polarity')}` status=`{item.get('fixture_status')}`"
        )
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional claim fixture(s) in JSON.")
    return lines


def _render_event_samples(items: list[dict[str, Any]], max_items: int = 6) -> list[str]:
    if not items:
        return ["- None."]
    lines = []
    for item in items[:max_items]:
        lines.append(
            f"- `{item.get('event_id')}` anchor=`{item.get('source_anchor_id')}` "
            f"time=`{item.get('event_time')}` time_confidence=`{item.get('time_confidence')}`"
        )
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional event fixture(s) in JSON.")
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


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
