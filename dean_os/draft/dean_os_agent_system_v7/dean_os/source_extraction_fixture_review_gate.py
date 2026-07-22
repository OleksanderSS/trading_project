from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.packets.source_extraction_fixture_packet import FIXTURE_CONTRACT_ID, UPSTREAM_CONTRACT_ID
from dean_os.utils import json_ready

DEFAULT_SOURCE_EXTRACTION_FIXTURE_JSON = "reports/dean_os/source_extraction_fixture_packet_current/latest.json"


class SourceExtractionFixtureReviewGate:
    """Review gate for fixture-only extraction candidate shapes."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/source_extraction_fixture_review_gate"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        fixture_json: str | Path = DEFAULT_SOURCE_EXTRACTION_FIXTURE_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        fixture = _load_json(fixture_json)
        checks = _review_checks(fixture)
        guidance = _decision_guidance(checks)
        payload = {
            "run_id": _run_id("source_extraction_fixture_review_gate"),
            "created_at": utc_now_iso(),
            "mode": "source_extraction_fixture_review_gate",
            "inputs": {
                "fixture_json": str(fixture_json),
                "fixture_run_id": fixture.get("run_id"),
            },
            "summary": {
                "gate_status": guidance["status"],
                "recommended_review_action": guidance["recommended_review_action"],
                "fixture_contract_id": fixture.get("summary", {}).get("fixture_contract_id"),
                "upstream_contract_id": fixture.get("summary", {}).get("upstream_contract_id"),
                "fixture_packet_status": fixture.get("summary", {}).get("packet_status"),
                "selected_anchor_count": fixture.get("summary", {}).get("selected_anchor_count", 0),
                "selected_missing_timestamp_count": fixture.get("summary", {}).get("selected_missing_timestamp_count", 0),
                "candidate_claim_fixture_count": fixture.get("summary", {}).get("candidate_claim_fixture_count", 0),
                "candidate_event_fixture_count": fixture.get("summary", {}).get("candidate_event_fixture_count", 0),
                "candidate_entity_fixture_count": fixture.get("summary", {}).get("candidate_entity_fixture_count", 0),
                "candidate_financial_implication_fixture_count": fixture.get("summary", {}).get("candidate_financial_implication_fixture_count", 0),
                "can_enter_manual_fixture_shape_review": guidance["can_enter_manual_fixture_shape_review"],
                "can_standardize_fixture_shape": guidance["can_standardize_fixture_shape"],
                "requires_timestamp_strategy": guidance["requires_timestamp_strategy"],
                "can_execute_real_extraction": False,
                "can_emit_claims_events_entities_as_evidence": False,
                "can_promote_to_evidence": False,
                "can_write_learning_memory": False,
                "can_change_analyst_weights": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
            "fixture_shape_review": _fixture_shape_review(fixture),
            "timestamp_review": _timestamp_review(fixture),
            "review_checks": checks,
            "decision_guidance": guidance,
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(fixture_json),
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
        rendered_md = render_source_extraction_fixture_review_gate_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_source_extraction_fixture_review_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    shape = payload.get("fixture_shape_review", {})
    timestamp = payload.get("timestamp_review", {})
    lines = [
        "# DEAN-OS Source Extraction Fixture Review Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Gate status: `{summary.get('gate_status')}`",
        f"- Recommended action: `{summary.get('recommended_review_action')}`",
        f"- Fixture contract ID: `{summary.get('fixture_contract_id')}`",
        f"- Upstream contract ID: `{summary.get('upstream_contract_id')}`",
        f"- Fixture packet status: `{summary.get('fixture_packet_status')}`",
        f"- Selected anchors: {summary.get('selected_anchor_count')}",
        f"- Selected missing timestamps: {summary.get('selected_missing_timestamp_count')}",
        f"- Can enter manual fixture shape review: {summary.get('can_enter_manual_fixture_shape_review')}",
        f"- Can standardize fixture shape: {summary.get('can_standardize_fixture_shape')}",
        f"- Requires timestamp strategy: {summary.get('requires_timestamp_strategy')}",
        f"- Can execute real extraction: {summary.get('can_execute_real_extraction')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Fixture Shape Review",
        "",
        f"- Shape status: `{shape.get('shape_status')}`",
        f"- Candidate groups present: {', '.join(shape.get('candidate_groups_present', [])) or 'none'}",
        f"- Anchor link status: `{shape.get('anchor_link_status')}`",
        f"- Evidence boundary status: `{shape.get('evidence_boundary_status')}`",
        "",
        "## Timestamp Review",
        "",
        f"- Timestamp status: `{timestamp.get('timestamp_status')}`",
        f"- Missing selected timestamps: {timestamp.get('missing_selected_timestamps')}",
        f"- Suggested action: {timestamp.get('suggested_action')}",
        "",
        "## Review Checks",
        "",
    ]
    lines.extend(_render_check_samples(payload.get("review_checks", [])))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    lines.extend(["", "## Decision Rationale", ""])
    lines.extend(_render_reason_samples(guidance.get("reasons", [])))
    return "\n".join(lines).strip() + "\n"


def _review_checks(fixture: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    summary = fixture.get("summary", {})
    boundary = fixture.get("fixture_boundary", {})
    anchors = fixture.get("selected_source_anchors", [])
    claims = fixture.get("candidate_claim_fixtures", [])
    events = fixture.get("candidate_event_fixtures", [])
    entities = fixture.get("candidate_entity_fixtures", [])
    implications = fixture.get("candidate_financial_implication_fixtures", [])

    if fixture.get("mode") == "source_extraction_fixture_packet":
        checks.append(_check("pass", "fixture_mode_valid", "Input is a source extraction fixture packet."))
    else:
        checks.append(_check("fail", "fixture_mode_invalid", f"Unexpected fixture mode: {fixture.get('mode')}."))

    if summary.get("fixture_contract_id") == FIXTURE_CONTRACT_ID:
        checks.append(_check("pass", "fixture_contract_id_valid", "Fixture contract id is valid."))
    else:
        checks.append(_check("fail", "fixture_contract_id_invalid", f"Unexpected fixture contract id: {summary.get('fixture_contract_id')}."))

    if summary.get("upstream_contract_id") == UPSTREAM_CONTRACT_ID:
        checks.append(_check("pass", "upstream_contract_id_valid", "Upstream extraction contract id is valid."))
    else:
        checks.append(_check("fail", "upstream_contract_id_invalid", f"Unexpected upstream contract id: {summary.get('upstream_contract_id')}."))

    if summary.get("packet_status") == "blocked_extraction_fixture":
        checks.append(_check("fail", "fixture_packet_blocked", "Fixture packet is blocked."))
    elif summary.get("packet_status") == "extraction_fixture_ready_with_warnings":
        checks.append(_check("warn", "fixture_packet_has_warnings", "Fixture packet has warnings that must remain visible."))
    else:
        checks.append(_check("pass", "fixture_packet_reviewable", "Fixture packet is reviewable."))

    if anchors:
        checks.append(_check("pass", "selected_anchors_present", f"{len(anchors)} selected anchor(s) are present."))
    else:
        checks.append(_check("fail", "selected_anchors_missing", "No selected anchors are present."))

    _extend_candidate_shape_checks(checks, anchors, claims, events, entities, implications)
    _extend_anchor_link_checks(checks, anchors, claims, events, entities, implications)
    _extend_boundary_checks(checks, summary, boundary)
    _extend_timestamp_checks(checks, summary)
    return checks


def _extend_candidate_shape_checks(
    checks: list[dict[str, str]],
    anchors: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    events: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    implications: list[dict[str, Any]],
) -> None:
    selected_count = len(anchors)
    expected_entities = sum(len(anchor.get("candidate_entities", [])) for anchor in anchors)
    if len(claims) == selected_count:
        checks.append(_check("pass", "claim_fixture_shape_count_valid", "Claim fixture count matches selected anchors."))
    else:
        checks.append(_check("fail", "claim_fixture_shape_count_invalid", "Claim fixture count does not match selected anchors."))
    if len(events) == selected_count:
        checks.append(_check("pass", "event_fixture_shape_count_valid", "Event fixture count matches selected anchors."))
    else:
        checks.append(_check("fail", "event_fixture_shape_count_invalid", "Event fixture count does not match selected anchors."))
    if len(implications) == selected_count:
        checks.append(_check("pass", "financial_implication_fixture_shape_count_valid", "Financial implication fixture count matches selected anchors."))
    else:
        checks.append(_check("fail", "financial_implication_fixture_shape_count_invalid", "Financial implication fixture count does not match selected anchors."))
    if len(entities) == expected_entities:
        checks.append(_check("pass", "entity_fixture_shape_count_valid", "Entity fixture count matches candidate entity metadata."))
    else:
        checks.append(_check("fail", "entity_fixture_shape_count_invalid", "Entity fixture count does not match candidate entity metadata."))

    invalid_status = [
        item
        for item in [*claims, *events, *entities, *implications]
        if item.get("fixture_status") != "candidate_shape_only_not_evidence"
    ]
    if invalid_status:
        checks.append(_check("fail", "candidate_fixture_status_invalid", "Some candidate fixtures are not marked as shape-only non-evidence."))
    else:
        checks.append(_check("pass", "candidate_fixture_status_valid", "All candidate fixtures are marked as shape-only non-evidence."))


def _extend_anchor_link_checks(
    checks: list[dict[str, str]],
    anchors: list[dict[str, Any]],
    *candidate_groups: list[dict[str, Any]],
) -> None:
    anchor_ids = {anchor.get("anchor_id") for anchor in anchors}
    candidates = [candidate for group in candidate_groups for candidate in group]
    invalid = [candidate for candidate in candidates if candidate.get("source_anchor_id") not in anchor_ids]
    if invalid:
        checks.append(_check("fail", "candidate_anchor_links_invalid", f"{len(invalid)} candidate fixture(s) have invalid source anchors."))
    else:
        checks.append(_check("pass", "candidate_anchor_links_valid", "All candidate fixtures link to selected source anchors."))


def _extend_boundary_checks(checks: list[dict[str, str]], summary: dict[str, Any], boundary: dict[str, Any]) -> None:
    boundary_flags = [
        "real_extraction_performed",
        "fixtures_are_production_evidence",
        "claims_emitted_as_evidence",
        "events_emitted_as_evidence",
        "entities_resolved_as_evidence",
        "financial_implications_emitted_as_evidence",
        "event_propagation_executed",
        "company_thesis_generated",
        "valuation_generated",
        "recommendation_generated",
        "trade_signal_generated",
    ]
    enabled_boundary = [flag for flag in boundary_flags if boundary.get(flag) is not False]
    if enabled_boundary:
        checks.append(_check("fail", "fixture_boundary_flags_invalid", f"Fixture boundary has enabled or missing false flags: {', '.join(enabled_boundary)}."))
    else:
        checks.append(_check("pass", "fixture_boundary_flags_valid", "Fixture boundary keeps extraction/evidence/downstream outputs disabled."))

    summary_flags = [
        "can_execute_real_extraction",
        "can_emit_claims_events_entities_as_evidence",
        "can_promote_to_evidence",
        "can_write_learning_memory",
        "can_change_analyst_weights",
        "can_create_recommendation",
        "can_trade",
    ]
    enabled_summary = [flag for flag in summary_flags if summary.get(flag) is not False]
    if enabled_summary:
        checks.append(_check("fail", "fixture_summary_flags_invalid", f"Fixture summary enables downstream actions: {', '.join(enabled_summary)}."))
    else:
        checks.append(_check("pass", "fixture_summary_flags_valid", "Fixture summary keeps extraction/evidence/downstream actions disabled."))


def _extend_timestamp_checks(checks: list[dict[str, str]], summary: dict[str, Any]) -> None:
    missing = int(summary.get("selected_missing_timestamp_count", 0) or 0)
    selected = int(summary.get("selected_anchor_count", 0) or 0)
    if missing:
        checks.append(_check("warn", "selected_timestamps_missing", f"{missing} of {selected} selected anchor(s) lack timestamps."))
    else:
        checks.append(_check("pass", "selected_timestamps_present", "Selected anchors have timestamps."))


def _fixture_shape_review(fixture: dict[str, Any]) -> dict[str, Any]:
    summary = fixture.get("summary", {})
    checks = _review_checks(fixture)
    fails = [check for check in checks if check.get("status") == "fail"]
    shape_fail_codes = {
        "claim_fixture_shape_count_invalid",
        "event_fixture_shape_count_invalid",
        "entity_fixture_shape_count_invalid",
        "financial_implication_fixture_shape_count_invalid",
        "candidate_fixture_status_invalid",
        "candidate_anchor_links_invalid",
    }
    shape_fails = [check for check in fails if check.get("code") in shape_fail_codes]
    return {
        "shape_status": "blocked" if shape_fails else "reviewable",
        "candidate_groups_present": [
            name
            for name, count in [
                ("claims", summary.get("candidate_claim_fixture_count", 0)),
                ("events", summary.get("candidate_event_fixture_count", 0)),
                ("entities", summary.get("candidate_entity_fixture_count", 0)),
                ("financial_implications", summary.get("candidate_financial_implication_fixture_count", 0)),
            ]
            if int(count or 0) > 0
        ],
        "anchor_link_status": "blocked" if any(check.get("code") == "candidate_anchor_links_invalid" for check in fails) else "valid",
        "evidence_boundary_status": "blocked" if any(check.get("code") in {"fixture_boundary_flags_invalid", "fixture_summary_flags_invalid"} for check in fails) else "disabled",
    }


def _timestamp_review(fixture: dict[str, Any]) -> dict[str, Any]:
    summary = fixture.get("summary", {})
    missing = int(summary.get("selected_missing_timestamp_count", 0) or 0)
    selected = int(summary.get("selected_anchor_count", 0) or 0)
    if missing:
        status = "timestamp_strategy_required"
        action = "Repair/import source timestamps or explicitly mark event chronology as limited before real extraction."
    else:
        status = "timestamp_ready_for_fixture_review"
        action = "Timestamp shape is ready for manual fixture review."
    return {
        "timestamp_status": status,
        "missing_selected_timestamps": missing,
        "selected_anchor_count": selected,
        "suggested_action": action,
    }


def _decision_guidance(checks: list[dict[str, str]]) -> dict[str, Any]:
    fails = [check for check in checks if check.get("status") == "fail"]
    warnings = [check for check in checks if check.get("status") == "warn"]
    timestamp_warnings = [check for check in warnings if check.get("code") == "selected_timestamps_missing"]
    if fails:
        status = "blocked_fixture_review"
        action = "fix_fixture_shape_or_boundaries_before_review"
        can_review = False
        can_standardize = False
    elif warnings:
        status = "fixture_review_ready_with_warnings"
        action = "manual_fixture_shape_review_with_timestamp_limitations"
        can_review = True
        can_standardize = False
    else:
        status = "fixture_review_ready"
        action = "manual_fixture_shape_review"
        can_review = True
        can_standardize = True
    return {
        "status": status,
        "recommended_review_action": action,
        "can_enter_manual_fixture_shape_review": can_review,
        "can_standardize_fixture_shape": can_standardize,
        "requires_timestamp_strategy": bool(timestamp_warnings),
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
        "No real extraction is executed.",
        "No fixture candidate is promoted to production evidence.",
        "No event propagation is executed.",
        "No company thesis, valuation, recommendation, price target, allocation, or position size is created.",
        "No learning memory, analyst weight, production config, operation queue, pipeline, broker, or trading action is written.",
    ]


def _commands(fixture_json: str | Path) -> dict[str, str]:
    return {
        "rerun_gate": (
            "python run_agent_source_extraction_fixture_review_gate.py "
            f"--fixture-json {fixture_json} "
            "--output-dir reports\\dean_os\\source_extraction_fixture_review_gate_current"
        )
    }


def _recommendations(guidance: dict[str, Any]) -> list[str]:
    if guidance["status"] == "blocked_fixture_review":
        return ["Fix fixture candidate counts, source anchors, status fields, or disabled boundary flags before manual review."]
    recommendations = [
        "Review the fixture candidate shape before implementing real extraction.",
        "Keep fixture candidates out of evidence promotion, learning, recommendations, and trading.",
        "Do not standardize this pattern until timestamp limitations are resolved or explicitly accepted.",
    ]
    if guidance["requires_timestamp_strategy"]:
        recommendations.append("Create or run a timestamp repair strategy for entity-bearing news anchors before trusting event chronology.")
    if guidance["can_standardize_fixture_shape"]:
        recommendations.append("After manual approval, the next build can be a real-extraction design stub, still review-only.")
    return recommendations


def _render_check_samples(items: list[dict[str, Any]], max_items: int = 16) -> list[str]:
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
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
