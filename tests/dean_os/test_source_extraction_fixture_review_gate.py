from __future__ import annotations

import copy
import json

from dean_os.source_extraction_fixture_review_gate import SourceExtractionFixtureReviewGate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _anchor(anchor_id="document:doc1", timestamp_status="present", entities=None):
    entities = entities if entities is not None else ["AMD"]
    return {
        "anchor_id": anchor_id,
        "timestamp_status": timestamp_status,
        "published_at": "2026-01-05T00:00:00+00:00" if timestamp_status == "present" else None,
        "text_present": True,
        "candidate_entities": entities,
        "candidate_topics": ["ai_capex_cycle"],
        "candidate_sectors": ["semiconductor"],
    }


def _fixture_packet(timestamp_status="present", boundary_override=None, summary_override=None):
    anchor = _anchor(timestamp_status=timestamp_status)
    limitations = ["fixture_only_not_production_evidence", "semantic_extraction_not_performed", "manual_review_required"]
    if timestamp_status != "present":
        limitations.append("source_timestamp_missing_or_not_real")
    payload = {
        "run_id": "source_extraction_fixture_packet_test",
        "mode": "source_extraction_fixture_packet",
        "summary": {
            "packet_status": "extraction_fixture_ready_for_manual_review" if timestamp_status == "present" else "extraction_fixture_ready_with_warnings",
            "fixture_contract_id": "248_review_only_real_source_claim_event_entity_extraction_fixture_v1",
            "upstream_contract_id": "247_review_only_real_source_claim_event_entity_extraction_contract_v1",
            "selected_anchor_count": 1,
            "selected_missing_timestamp_count": 0 if timestamp_status == "present" else 1,
            "candidate_claim_fixture_count": 1,
            "candidate_event_fixture_count": 1,
            "candidate_entity_fixture_count": 1,
            "candidate_financial_implication_fixture_count": 1,
            "can_execute_real_extraction": False,
            "can_emit_claims_events_entities_as_evidence": False,
            "can_promote_to_evidence": False,
            "can_write_learning_memory": False,
            "can_change_analyst_weights": False,
            "can_create_recommendation": False,
            "can_trade": False,
            **(summary_override or {}),
        },
        "fixture_boundary": {
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
            **(boundary_override or {}),
        },
        "selected_source_anchors": [anchor],
        "candidate_claim_fixtures": [
            {
                "claim_id": "claim_fixture_0001",
                "source_anchor_id": anchor["anchor_id"],
                "fixture_status": "candidate_shape_only_not_evidence",
            }
        ],
        "candidate_event_fixtures": [
            {
                "event_id": "event_fixture_0001",
                "source_anchor_id": anchor["anchor_id"],
                "fixture_status": "candidate_shape_only_not_evidence",
            }
        ],
        "candidate_entity_fixtures": [
            {
                "entity_ref_id": "entity_fixture_0001_01",
                "source_anchor_id": anchor["anchor_id"],
                "fixture_status": "candidate_shape_only_not_evidence",
                "limitations": limitations,
            }
        ],
        "candidate_financial_implication_fixtures": [
            {
                "implication_id": "implication_fixture_0001",
                "source_anchor_id": anchor["anchor_id"],
                "fixture_status": "candidate_shape_only_not_evidence",
            }
        ],
    }
    return payload


def test_fixture_review_gate_allows_clean_fixture_shape_review(tmp_path):
    fixture_path = _write_json(tmp_path / "fixture.json", _fixture_packet())

    payload = SourceExtractionFixtureReviewGate(tmp_path / "reports").build(fixture_json=fixture_path, save=False)

    assert payload["summary"]["gate_status"] == "fixture_review_ready"
    assert payload["summary"]["can_enter_manual_fixture_shape_review"] is True
    assert payload["summary"]["can_standardize_fixture_shape"] is True
    assert payload["summary"]["can_execute_real_extraction"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["fixture_shape_review"]["shape_status"] == "reviewable"
    assert payload["timestamp_review"]["timestamp_status"] == "timestamp_ready_for_fixture_review"


def test_fixture_review_gate_warns_for_timestamp_limited_fixture(tmp_path):
    fixture_path = _write_json(tmp_path / "fixture.json", _fixture_packet(timestamp_status="missing"))

    payload = SourceExtractionFixtureReviewGate(tmp_path / "reports").build(fixture_json=fixture_path, save=False)

    assert payload["summary"]["gate_status"] == "fixture_review_ready_with_warnings"
    assert payload["summary"]["can_enter_manual_fixture_shape_review"] is True
    assert payload["summary"]["can_standardize_fixture_shape"] is False
    assert payload["summary"]["requires_timestamp_strategy"] is True
    assert payload["timestamp_review"]["timestamp_status"] == "timestamp_strategy_required"
    assert any(check["code"] == "selected_timestamps_missing" for check in payload["review_checks"])


def test_fixture_review_gate_blocks_enabled_downstream_boundary(tmp_path):
    fixture_path = _write_json(
        tmp_path / "fixture.json",
        _fixture_packet(boundary_override={"fixtures_are_production_evidence": True}),
    )

    payload = SourceExtractionFixtureReviewGate(tmp_path / "reports").build(fixture_json=fixture_path, save=False)

    assert payload["summary"]["gate_status"] == "blocked_fixture_review"
    failed_codes = {check["code"] for check in payload["review_checks"] if check["status"] == "fail"}
    assert "fixture_boundary_flags_invalid" in failed_codes
    assert payload["fixture_shape_review"]["evidence_boundary_status"] == "blocked"


def test_fixture_review_gate_blocks_invalid_anchor_links(tmp_path):
    fixture = _fixture_packet()
    broken = copy.deepcopy(fixture)
    broken["candidate_claim_fixtures"][0]["source_anchor_id"] = "document:missing"
    fixture_path = _write_json(tmp_path / "fixture.json", broken)

    payload = SourceExtractionFixtureReviewGate(tmp_path / "reports").build(fixture_json=fixture_path, save=False)

    assert payload["summary"]["gate_status"] == "blocked_fixture_review"
    failed_codes = {check["code"] for check in payload["review_checks"] if check["status"] == "fail"}
    assert "candidate_anchor_links_invalid" in failed_codes
    assert payload["fixture_shape_review"]["anchor_link_status"] == "blocked"


def test_fixture_review_gate_saves_markdown_with_non_actions(tmp_path):
    fixture_path = _write_json(tmp_path / "fixture.json", _fixture_packet(timestamp_status="missing"))

    payload = SourceExtractionFixtureReviewGate(tmp_path / "reports").build(fixture_json=fixture_path)

    latest_md = tmp_path / "reports" / "latest.md"
    assert latest_md.exists()
    markdown = latest_md.read_text(encoding="utf-8")
    assert "Can execute real extraction: False" in markdown
    assert "Can trade: False" in markdown
    assert "No real extraction is executed." in markdown
    assert "No fixture candidate is promoted to production evidence." in markdown
    assert payload["saved_paths"]["latest_markdown"].endswith("latest.md")
