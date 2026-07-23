from __future__ import annotations

import json

from dean_os.packets.real_source_normalized_packet import RealSourceNormalizedPacketBuilder
from dean_os.source_evidence_validation_gate import SourceEvidenceValidationGate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture_packet(output_boundary=None):
    return {
        "packet_id": "norm245_01_news",
        "source_type_id": "news_articles_general_business",
        "source_fixture_status": "offline_normalized_packet_fixture_not_real_external_source",
        "real_source_content_supplied_in_245": False,
        "content_units": [
            {
                "content_unit_id": "unit_1",
                "content_unit_type_id": "paragraph",
                "normalized_text": "Fixture text only.",
                "anchor_id": "anchor_1",
                "extraction_eligible": True,
            }
        ],
        "anchors": [{"anchor_id": "anchor_1", "content_unit_id": "unit_1"}],
        "routing_prefilter": {
            "candidate_links_are_final": False,
            "candidate_assets_or_entities": ["AMD", "SPY"],
            "candidate_topics": ["interest_rates"],
            "candidate_sectors": ["semiconductors", "market_etfs"],
        },
        "downstream_extraction_outputs": [],
        "output_boundary": {
            "claims_emitted_now": False,
            "events_emitted_now": False,
            "entities_resolved_now": False,
            "event_propagation_executed_now": False,
            "company_thesis_generated_now": False,
            "recommendation_output_now": False,
            "trade_signal_output_now": False,
            **(output_boundary or {}),
        },
    }


def _normalized_fixture(output_boundary=None):
    return {
        "block_id": "245_review_only_real_source_normalized_packet_fixture_v1",
        "schema_version": "real_source_normalized_packet_fixture_v1_review_only",
        "fixture_status": "review_only_normalized_packet_fixture_materialized_for_validation_not_evidence",
        "summary": {
            "real_source_content_supplied_in_245": False,
            "fixtures_are_production_evidence": False,
        },
        "normalized_packet_fixture_rows": [_fixture_packet(output_boundary=output_boundary)],
    }


def _evidence_pack(documents=None, coverage=None):
    documents = documents if documents is not None else [
        {
            "document_id": "doc1",
            "title": "AMD AI article",
            "source_type": "news",
            "text": "AMD AI accelerator demand and semiconductor context.",
            "published_at": "2026-01-05T00:00:00+00:00",
            "tickers": ["AMD"],
            "sectors": ["semiconductors"],
            "tags": ["ai_capex_cycle"],
        },
        {
            "document_id": "doc2",
            "title": "Semiconductor industry report",
            "source_type": "report",
            "text": "Semiconductor supply chain and AI capex context.",
            "published_at": "2026-01-04T00:00:00+00:00",
            "tickers": ["TSM"],
            "sectors": ["semiconductors"],
            "tags": ["semiconductor_supply_chain"],
        },
    ]
    coverage = coverage if coverage is not None else {
        "document_count": len(documents),
        "data_quality": "strong",
        "by_source_type": {"news": 1, "report": 1},
        "tickers": ["AMD", "TSM"],
        "missing_requested_tickers": [],
        "warning_count": 0,
        "dropped_count": 0,
    }
    return {"run_id": "evidence_pack_test", "documents": documents, "coverage": coverage}


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


def test_source_gate_validates_fixture_but_blocks_evidence_promotion(tmp_path):
    source_path = _write_json(tmp_path / "fixture.json", _normalized_fixture())

    payload = SourceEvidenceValidationGate(tmp_path / "reports").build(source_json=source_path, save=False)

    assert payload["summary"]["artifact_type"] == "normalized_packet_fixture"
    assert payload["summary"]["gate_status"] == "fixture_validated_not_evidence"
    assert payload["summary"]["can_enter_domain_research"] is False
    assert payload["summary"]["can_enter_domain_contract_review"] is True
    assert payload["summary"]["can_promote_to_evidence"] is False
    assert "AMD" in payload["candidate_routing_indexes"]["entities"]
    assert all(assertion["status"] == "pass" for assertion in payload["safety_assertions"])


def test_source_gate_accepts_real_source_normalized_packet_with_review_warnings(monkeypatch, tmp_path):
    source_path = _write_json(tmp_path / "real_source_packet.json", _real_source_normalized_packet(monkeypatch, tmp_path))

    payload = SourceEvidenceValidationGate(tmp_path / "reports").build(source_json=source_path, save=False)

    assert payload["summary"]["artifact_type"] == "real_source_normalized_packet"
    assert payload["summary"]["gate_status"] == "source_evidence_ready_with_warnings"
    assert payload["summary"]["can_enter_domain_research"] is True
    assert payload["summary"]["can_promote_to_evidence"] is False
    assert payload["summary"]["can_extract_claims_events_entities"] is False
    assert payload["candidate_routing_indexes"]["entities"] == ["AMD"]
    assert any(check["code"].endswith("_timestamp_missing") for check in payload["validation_checks"])
    assert any(check["code"].endswith("_quarantine_present") for check in payload["validation_checks"])
    assert any(check["code"].endswith("_quarantined_units_not_extraction_eligible") and check["status"] == "pass" for check in payload["validation_checks"])
    assert all(assertion["status"] == "pass" for assertion in payload["safety_assertions"])


def test_source_gate_allows_clean_evidence_pack_for_domain_research(tmp_path):
    source_path = _write_json(tmp_path / "evidence_pack.json", _evidence_pack())

    payload = SourceEvidenceValidationGate(tmp_path / "reports").build(source_json=source_path, save=False)

    assert payload["summary"]["artifact_type"] == "analyst_evidence_pack"
    assert payload["summary"]["gate_status"] == "source_evidence_ready_for_domain_research"
    assert payload["summary"]["can_enter_domain_research"] is True
    assert payload["summary"]["can_extract_claims_events_entities"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["candidate_routing_indexes"]["entities"] == ["AMD", "TSM"]


def test_source_gate_warns_but_keeps_domain_review_possible_for_partial_evidence(tmp_path):
    source_path = _write_json(
        tmp_path / "evidence_pack.json",
        _evidence_pack(
            coverage={
                "document_count": 1,
                "data_quality": "partial",
                "by_source_type": {"news": 1},
                "tickers": ["AMD"],
                "missing_requested_tickers": ["TSM"],
                "warning_count": 1,
                "dropped_count": 0,
            }
        ),
    )

    payload = SourceEvidenceValidationGate(tmp_path / "reports").build(source_json=source_path, save=False)

    assert payload["summary"]["gate_status"] == "source_evidence_ready_with_warnings"
    assert payload["summary"]["can_enter_domain_research"] is True
    assert any(check["code"] == "missing_requested_tickers" for check in payload["validation_checks"])


def test_source_gate_blocks_unsafe_fixture_boundary(tmp_path):
    source_path = _write_json(tmp_path / "fixture.json", _normalized_fixture(output_boundary={"recommendation_output_now": True}))

    payload = SourceEvidenceValidationGate(tmp_path / "reports").build(source_json=source_path, save=False)

    assert payload["summary"]["gate_status"] == "blocked_source_evidence"
    assert payload["summary"]["can_enter_domain_research"] is False
    assert any(assertion["code"].endswith("recommendation_output_now_false") for assertion in payload["safety_assertions"])


def test_source_gate_saves_markdown_with_non_actions(tmp_path):
    source_path = _write_json(tmp_path / "evidence_pack.json", _evidence_pack())

    payload = SourceEvidenceValidationGate(tmp_path / "reports").build(source_json=source_path)

    latest_md = tmp_path / "reports" / "latest.md"
    assert latest_md.exists()
    markdown = latest_md.read_text(encoding="utf-8")
    assert "Can extract claims/events/entities: False" in markdown
    assert "No claim/event/entity extraction is executed." in markdown
    assert "No company thesis, valuation, recommendation, price target, or position size is created." in markdown
    assert payload["saved_paths"]["latest_markdown"].endswith("latest.md")


def test_source_gate_markdown_keeps_repeated_reasons_concise(tmp_path):
    documents = []
    for index in range(20):
        documents.append(
            {
                "document_id": f"doc{index}",
                "title": f"Document {index}",
                "source_type": "news",
                "text": "Semiconductor source text.",
                "tickers": ["AMD"],
                "sectors": ["semiconductors"],
                "tags": ["ai_capex_cycle"],
            }
        )
    source_path = _write_json(tmp_path / "evidence_pack.json", _evidence_pack(documents=documents))

    SourceEvidenceValidationGate(tmp_path / "reports").build(source_json=source_path)

    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")
    rationale = markdown.split("## Decision Rationale", 1)[1]
    assert rationale.count("has no published_at timestamp.") == 8
    assert "... 12 additional reason(s) in JSON." in rationale
