from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.packets.real_source_normalized_packet import RealSourceNormalizedPacketBuilder
from dean_os.review_only_real_source_normalized_packet_validation_gate import build_validation_gate


def test_real_source_normalized_packet_builds_review_only_packet(monkeypatch, tmp_path):
    monkeypatch.setenv("DEAN_PROJECT_ROOT", str(tmp_path))
    source_path = tmp_path / "semiconductor_note.md"
    source_path.write_text(
        "\n".join(
            [
                "# Semiconductor Supply Note",
                "",
                "Demand growth and backlog improved for AMD and NVDA.",
                "",
                "Forward-looking statements involve risks and uncertainties.",
                "Actual results may differ materially.",
            ]
        ),
        encoding="utf-8",
    )

    payload = RealSourceNormalizedPacketBuilder(output_dir=tmp_path / "reports").build_from_path(
        source_path,
        source_type="report",
        tickers=["AMD", "NVDA"],
        sectors=["semiconductors"],
        tags=["semiconductor_supply_chain"],
        save=False,
    )

    packet = payload["normalized_packet_rows"][0]
    assert payload["summary"]["normalized_packet_count"] == 1
    assert payload["summary"]["real_source_content_supplied"] is True
    assert payload["summary"]["claim_extraction_performed"] is False
    assert packet["source_type_id"] == "industry_reports_and_whitepapers"
    assert packet["source_material_status"] == "operator_supplied_review_only_not_promoted_evidence"
    assert packet["quality_precheck"]["primary_secondary_classification"] == "operator_supplied_material_not_yet_promoted_evidence"
    assert packet["routing_prefilter"]["candidate_assets_or_entities"] == ["AMD", "NVDA"]
    assert packet["routing_prefilter"]["candidate_links_are_final"] is False
    assert packet["downstream_extraction_outputs"] == []
    assert packet["output_boundary"]["trade_signal_output_now"] is False
    assert any("legal_disclaimer" in row["quarantine_flags"] for row in packet["quarantine_partitions"])


def test_real_source_normalized_packet_passes_existing_validation_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("DEAN_PROJECT_ROOT", str(tmp_path))
    source_path = tmp_path / "market_note.txt"
    source_path.write_text("Demand growth and backlog improved for AMD.", encoding="utf-8")
    payload = RealSourceNormalizedPacketBuilder(output_dir=tmp_path / "reports").build_from_path(
        source_path,
        source_type="news",
        tickers=["AMD"],
        save=False,
    )

    validation = build_validation_gate(payload)

    assert validation["packet_collection_key"] == "normalized_packet_rows"
    assert validation["gate_status"] == "passed"
    assert validation["summary"]["total_packets_evaluated"] == 1
    assert validation["summary"]["invalid_packets"] == 0


def test_real_source_normalized_packet_save_writes_review_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("DEAN_PROJECT_ROOT", str(tmp_path))
    source_path = tmp_path / "saved_note.md"
    source_path.write_text("Demand growth and backlog improved.", encoding="utf-8")

    payload = RealSourceNormalizedPacketBuilder(output_dir=tmp_path / "reports").build_from_path(
        source_path,
        source_type="article",
        save=True,
    )

    assert payload["saved_paths"]["latest_json"].endswith("latest.json")
    assert payload["saved_paths"]["latest_markdown"].endswith("latest.md")


def test_validation_gate_cli_accepts_real_source_packet_json(monkeypatch, tmp_path):
    monkeypatch.setenv("DEAN_PROJECT_ROOT", str(tmp_path))
    source_path = tmp_path / "cli_note.md"
    source_path.write_text("Demand growth and backlog improved for AMD.", encoding="utf-8")
    payload = RealSourceNormalizedPacketBuilder(output_dir=tmp_path / "reports").build_from_path(
        source_path,
        source_type="report",
        tickers=["AMD"],
        save=False,
    )
    input_json = tmp_path / "real_source_packet.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_review_only_real_source_normalized_packet_validation_gate.py"),
            "--input-json",
            str(input_json),
            "--output-dir",
            str(tmp_path / "validation_gate"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Gate status: passed" in result.stdout
    assert (tmp_path / "validation_gate" / "latest.json").exists()
    assert (tmp_path / "validation_gate" / "latest.md").exists()
