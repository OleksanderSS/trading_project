from __future__ import annotations

import json

from dean_os.packets.source_extraction_fixture_packet import SourceExtractionFixturePacket


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _anchor(anchor_id, *, timestamp_status="present", published_at="2026-01-05T00:00:00+00:00", entities=None):
    entities = entities if entities is not None else ["AMD"]
    return {
        "source_unit_id": anchor_id.replace("document:", ""),
        "unit_kind": "document",
        "anchor_id": anchor_id,
        "title": f"{anchor_id} title",
        "source_type": "news",
        "uri": "data/source.parquet",
        "published_at": published_at if timestamp_status == "present" else None,
        "timestamp_status": timestamp_status,
        "text_present": True,
        "text_preview": f"{anchor_id} source preview.",
        "candidate_entities": entities,
        "candidate_sectors": ["semiconductor"],
        "candidate_topics": ["ai_capex_cycle"],
    }


def _contract_packet(anchors=None, status="extraction_contract_ready_for_manual_review"):
    anchors = anchors if anchors is not None else [_anchor("document:doc1"), _anchor("document:doc2", entities=["TSM"])]
    return {
        "run_id": "source_extraction_review_packet_test",
        "mode": "source_extraction_review_packet",
        "summary": {
            "packet_status": status,
            "contract_id": "247_review_only_real_source_claim_event_entity_extraction_contract_v1",
            "can_execute_extraction_now": False,
            "can_emit_claims_events_entities": False,
            "can_trade": False,
        },
        "source_anchor_plan": {
            "anchor_count": len(anchors),
            "anchors": anchors,
        },
    }


def test_source_extraction_fixture_materializes_candidate_shapes_without_evidence_promotion(tmp_path):
    contract_path = _write_json(tmp_path / "contract.json", _contract_packet())

    payload = SourceExtractionFixturePacket(tmp_path / "reports").build(contract_json=contract_path, max_items=2, save=False)

    assert payload["summary"]["packet_status"] == "extraction_fixture_ready_for_manual_review"
    assert payload["summary"]["selected_anchor_count"] == 2
    assert payload["summary"]["candidate_claim_fixture_count"] == 2
    assert payload["summary"]["candidate_event_fixture_count"] == 2
    assert payload["summary"]["candidate_entity_fixture_count"] == 2
    assert payload["summary"]["candidate_financial_implication_fixture_count"] == 2
    assert payload["summary"]["can_execute_real_extraction"] is False
    assert payload["summary"]["can_emit_claims_events_entities_as_evidence"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["fixture_boundary"]["fixtures_are_production_evidence"] is False
    assert payload["candidate_claim_fixtures"][0]["fixture_status"] == "candidate_shape_only_not_evidence"
    assert payload["candidate_claim_fixtures"][0]["source_anchor_id"] == "document:doc1"


def test_source_extraction_fixture_prefers_timestamped_anchors(tmp_path):
    anchors = [
        _anchor("document:missing1", timestamp_status="missing"),
        _anchor("document:present1", timestamp_status="present", entities=["NVDA"]),
        _anchor("document:present2", timestamp_status="present", entities=["TSM"]),
    ]
    contract_path = _write_json(tmp_path / "contract.json", _contract_packet(anchors=anchors, status="extraction_contract_ready_with_warnings"))

    payload = SourceExtractionFixturePacket(tmp_path / "reports").build(contract_json=contract_path, max_items=2, save=False)

    assert payload["summary"]["packet_status"] == "extraction_fixture_ready_with_warnings"
    assert payload["summary"]["selected_missing_timestamp_count"] == 0
    selected_ids = [anchor["anchor_id"] for anchor in payload["selected_source_anchors"]]
    assert selected_ids == ["document:present1", "document:present2"]
    assert all(event["time_confidence"] == "source_timestamp_present" for event in payload["candidate_event_fixtures"])
    assert any(check["code"] == "upstream_contract_has_warnings" for check in payload["review_checks"])


def test_source_extraction_fixture_can_include_missing_timestamps_with_warning(tmp_path):
    anchors = [
        _anchor("document:missing1", timestamp_status="missing"),
        _anchor("document:present1", timestamp_status="present"),
    ]
    contract_path = _write_json(tmp_path / "contract.json", _contract_packet(anchors=anchors))

    payload = SourceExtractionFixturePacket(tmp_path / "reports").build(
        contract_json=contract_path,
        max_items=1,
        prefer_timestamped=False,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "extraction_fixture_ready_with_warnings"
    assert payload["summary"]["selected_missing_timestamp_count"] == 1
    assert payload["candidate_event_fixtures"][0]["event_time"] is None
    assert payload["candidate_event_fixtures"][0]["time_confidence"] == "missing_source_timestamp"
    assert any(check["code"] == "selected_timestamps_missing" for check in payload["review_checks"])


def test_source_extraction_fixture_blocks_when_upstream_contract_is_blocked(tmp_path):
    contract_path = _write_json(tmp_path / "contract.json", _contract_packet(status="blocked_extraction_contract"))

    payload = SourceExtractionFixturePacket(tmp_path / "reports").build(contract_json=contract_path, max_items=1, save=False)

    assert payload["summary"]["packet_status"] == "blocked_extraction_fixture"
    assert payload["summary"]["can_enter_manual_fixture_review"] is False
    failed_codes = {check["code"] for check in payload["review_checks"] if check["status"] == "fail"}
    assert "upstream_contract_blocked" in failed_codes


def test_source_extraction_fixture_saves_markdown_with_non_actions(tmp_path):
    contract_path = _write_json(tmp_path / "contract.json", _contract_packet())

    payload = SourceExtractionFixturePacket(tmp_path / "reports").build(contract_json=contract_path, max_items=1)

    latest_md = tmp_path / "reports" / "latest.md"
    assert latest_md.exists()
    markdown = latest_md.read_text(encoding="utf-8")
    assert "Can execute real extraction: False" in markdown
    assert "Can emit claims/events/entities as evidence: False" in markdown
    assert "No real claim/event/entity extraction is executed." in markdown
    assert "No candidate fixture is promoted to production evidence." in markdown
    assert payload["saved_paths"]["latest_markdown"].endswith("latest.md")
