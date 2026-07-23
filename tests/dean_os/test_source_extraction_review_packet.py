from __future__ import annotations

import json

from dean_os.packets.real_source_normalized_packet import RealSourceNormalizedPacketBuilder
from dean_os.source_evidence_validation_gate import SourceEvidenceValidationGate
from dean_os.packets.source_extraction_review_packet import SourceExtractionReviewPacket


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _document(document_id="doc1", published_at="2026-01-05T00:00:00+00:00", text="AMD AI accelerator demand is discussed."):
    return {
        "document_id": document_id,
        "title": f"{document_id} title",
        "source_type": "news",
        "text": text,
        "published_at": published_at,
        "uri": "data/source.parquet",
        "tickers": ["AMD"],
        "sectors": ["semiconductor"],
        "tags": ["ai_capex_cycle"],
    }


def _evidence_pack(documents=None):
    documents = documents if documents is not None else [_document()]
    return {
        "run_id": "analyst_evidence_pack_test",
        "documents": documents,
        "coverage": {
            "document_count": len(documents),
            "data_quality": "strong",
            "tickers": ["AMD"],
            "sectors": ["semiconductor"],
            "warning_count": 0,
            "dropped_count": 0,
        },
    }


def _source_gate(
    status="source_evidence_ready_for_domain_research",
    can_domain=True,
    warning_count=0,
    fail_count=0,
    source_json="source.json",
):
    reasons = []
    if warning_count:
        reasons = ["doc1 has no published_at timestamp."]
    if fail_count:
        reasons = ["Source artifact is blocked."]
    return {
        "run_id": "source_gate_test",
        "mode": "source_evidence_validation_gate",
        "inputs": {"source_json": str(source_json), "artifact_type": "analyst_evidence_pack"},
        "summary": {
            "gate_status": status,
            "artifact_type": "analyst_evidence_pack",
            "document_count": 1,
            "content_unit_count": 1,
            "can_enter_domain_research": can_domain,
            "can_enter_domain_contract_review": can_domain or status == "fixture_validated_not_evidence",
            "can_promote_to_evidence": False,
            "can_extract_claims_events_entities": False,
            "can_trade": False,
        },
        "decision_guidance": {
            "pass_count": 5,
            "warning_count": warning_count,
            "fail_count": fail_count,
            "reasons": reasons,
        },
        "candidate_routing_indexes": {"entities": ["AMD"], "topics": ["ai_capex_cycle"], "sectors": ["semiconductor"]},
    }


def _domain_packet(status="domain_review_ready"):
    return {
        "run_id": "domain_packet_test",
        "mode": "domain_specialist_review_packet",
        "summary": {
            "packet_status": status,
            "can_enter_manual_domain_review": status != "blocked",
            "can_write_learning_memory": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _fixture_artifact():
    return {
        "block_id": "245_review_only_real_source_normalized_packet_fixture_v1",
        "summary": {
            "real_source_content_supplied_in_245": False,
            "fixtures_are_production_evidence": False,
        },
        "normalized_packet_fixture_rows": [
            {
                "packet_id": "packet1",
                "source_type_id": "news_articles_general_business",
                "content_units": [
                    {
                        "content_unit_id": "unit1",
                        "normalized_text": "Fixture text only.",
                        "anchor_id": "anchor1",
                    }
                ],
                "routing_prefilter": {
                    "candidate_assets_or_entities": ["AMD"],
                    "candidate_topics": ["ai_capex_cycle"],
                    "candidate_sectors": ["semiconductors"],
                },
            }
        ],
    }


def _real_source_normalized_packet(monkeypatch, tmp_path):
    monkeypatch.setenv("DEAN_PROJECT_ROOT", str(tmp_path))
    source_path = tmp_path / "semiconductor_note.md"
    source_path.write_text(
        "\n".join(
            [
                "# Semiconductor Supply Note",
                "",
                "Demand growth and backlog improved for AMD.",
                "",
                "Forward-looking statements involve risks and uncertainties.",
            ]
        ),
        encoding="utf-8",
    )
    return RealSourceNormalizedPacketBuilder(output_dir=tmp_path / "real_source_reports").build_from_path(
        source_path,
        source_type="report",
        tickers=["AMD"],
        sectors=["semiconductors"],
        tags=["semiconductor_supply_chain"],
        save=False,
    )


def test_source_extraction_contract_ready_for_clean_evidence_pack(tmp_path):
    source_path = _write_json(tmp_path / "source.json", _evidence_pack())
    gate_path = _write_json(tmp_path / "gate.json", _source_gate(source_json=source_path))
    domain_path = _write_json(tmp_path / "domain.json", _domain_packet())

    payload = SourceExtractionReviewPacket(tmp_path / "reports").build(
        source_json=source_path,
        source_gate_json=gate_path,
        domain_packet_json=domain_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "extraction_contract_ready_for_manual_review"
    assert payload["summary"]["can_enter_manual_extraction_contract_review"] is True
    assert payload["summary"]["can_execute_extraction_now"] is False
    assert payload["summary"]["can_emit_claims_events_entities"] is False
    assert payload["source_anchor_plan"]["anchors"][0]["anchor_id"] == "document:doc1"
    assert payload["extraction_work_queue"][0]["status"] == "contract_defined_not_executed"
    assert payload["extraction_contract"]["output_boundary"]["claims_emitted_now"] is False


def test_source_extraction_contract_warns_for_missing_timestamps(tmp_path):
    source_path = _write_json(tmp_path / "source.json", _evidence_pack(documents=[_document(published_at=None)]))
    gate_path = _write_json(tmp_path / "gate.json", _source_gate(warning_count=1, source_json=source_path))
    domain_path = _write_json(tmp_path / "domain.json", _domain_packet(status="domain_review_ready_with_limitations"))

    payload = SourceExtractionReviewPacket(tmp_path / "reports").build(
        source_json=source_path,
        source_gate_json=gate_path,
        domain_packet_json=domain_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "extraction_contract_ready_with_warnings"
    assert payload["summary"]["timestamp_missing_count"] == 1
    assert "missing_source_timestamp_for_event_chronology" in payload["extraction_work_queue"][0]["blockers_or_limitations"]
    assert any(check["code"] == "source_timestamps_missing" for check in payload["review_checks"])


def test_source_extraction_contract_blocks_when_source_gate_blocks(tmp_path):
    source_path = _write_json(tmp_path / "source.json", _evidence_pack())
    gate_path = _write_json(
        tmp_path / "gate.json",
        _source_gate(status="blocked_source_evidence", can_domain=False, fail_count=1, source_json=source_path),
    )
    domain_path = _write_json(tmp_path / "domain.json", _domain_packet())

    payload = SourceExtractionReviewPacket(tmp_path / "reports").build(
        source_json=source_path,
        source_gate_json=gate_path,
        domain_packet_json=domain_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "blocked_extraction_contract"
    assert payload["summary"]["can_enter_manual_extraction_contract_review"] is False
    failed_codes = {check["code"] for check in payload["review_checks"] if check["status"] == "fail"}
    assert "source_gate_has_failures" in failed_codes


def test_source_extraction_contract_keeps_fixtures_review_only(tmp_path):
    fixture_path = _write_json(tmp_path / "fixture.json", _fixture_artifact())
    gate_path = _write_json(
        tmp_path / "gate.json",
        _source_gate(status="fixture_validated_not_evidence", can_domain=False, source_json=fixture_path),
    )
    domain_path = _write_json(tmp_path / "domain.json", _domain_packet())

    payload = SourceExtractionReviewPacket(tmp_path / "reports").build(
        source_json=fixture_path,
        source_gate_json=gate_path,
        domain_packet_json=domain_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "fixture_extraction_contract_ready_not_evidence"
    assert payload["summary"]["artifact_type"] == "normalized_packet_fixture"
    assert payload["source_anchor_plan"]["anchors"][0]["timestamp_status"] == "fixture_not_real_source"
    assert "fixture_not_real_evidence" in payload["extraction_work_queue"][0]["blockers_or_limitations"]
    assert payload["summary"]["can_execute_extraction_now"] is False


def test_source_extraction_contract_accepts_real_source_normalized_packet_review_only(monkeypatch, tmp_path):
    source_path = _write_json(tmp_path / "real_source_packet.json", _real_source_normalized_packet(monkeypatch, tmp_path))
    gate_payload = SourceEvidenceValidationGate(tmp_path / "gate_reports").build(source_json=source_path, save=False)
    gate_path = _write_json(tmp_path / "gate.json", gate_payload)
    domain_path = _write_json(tmp_path / "domain.json", _domain_packet(status="domain_review_ready_with_limitations"))

    payload = SourceExtractionReviewPacket(tmp_path / "reports").build(
        source_json=source_path,
        source_gate_json=gate_path,
        domain_packet_json=domain_path,
        save=False,
    )

    assert payload["summary"]["artifact_type"] == "real_source_normalized_packet"
    assert payload["summary"]["packet_status"] == "extraction_contract_ready_with_warnings"
    assert payload["summary"]["can_enter_manual_extraction_contract_review"] is True
    assert payload["summary"]["can_execute_extraction_now"] is False
    assert payload["summary"]["can_emit_claims_events_entities"] is False
    assert payload["candidate_routing_indexes"]["entities"] == ["AMD"]
    assert any(anchor["unit_kind"] == "real_normalized_content_unit" for anchor in payload["source_anchor_plan"]["anchors"])
    blockers = {blocker for item in payload["extraction_work_queue"] for blocker in item["blockers_or_limitations"]}
    assert "missing_source_timestamp_for_event_chronology" in blockers
    assert "source_unit_not_extraction_eligible" in blockers
    assert "quarantined_source_unit" in blockers
    check_codes = {check["code"] for check in payload["review_checks"]}
    assert "real_source_packet_review_only" in check_codes
    assert "quarantined_source_units_present" in check_codes


def test_source_extraction_contract_saves_markdown_with_boundaries(tmp_path):
    source_path = _write_json(tmp_path / "source.json", _evidence_pack())
    gate_path = _write_json(tmp_path / "gate.json", _source_gate(source_json=source_path))
    domain_path = _write_json(tmp_path / "domain.json", _domain_packet())

    payload = SourceExtractionReviewPacket(tmp_path / "reports").build(
        source_json=source_path,
        source_gate_json=gate_path,
        domain_packet_json=domain_path,
    )

    latest_md = tmp_path / "reports" / "latest.md"
    assert latest_md.exists()
    markdown = latest_md.read_text(encoding="utf-8")
    assert "Can execute extraction now: False" in markdown
    assert "Can emit claims/events/entities: False" in markdown
    assert "No claim, event, entity, or financial implication extraction is executed." in markdown
    assert "No company thesis, valuation, recommendation, price target, or position size is created." in markdown
    assert payload["saved_paths"]["latest_markdown"].endswith("latest.md")
