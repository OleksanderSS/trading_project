from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dean_os.analyst_core.artifact_evidence_loader import ArtifactEvidenceLoader
from dean_os.analyst_core.lens_contract import AnalysisPacket
from dean_os.analyst_core.lenses.event_classifier_lens import EventClassifierLens
from dean_os.analyst_core.sector_analyst import _build_default_registry
from dean_os.analyst_core.analyst_core_reasoning_snapshot import AnalystCoreReasoningSnapshot


DOMAIN_ID = "semiconductor_ai_infrastructure"
AS_OF = "2026-06-30T21:00:00+00:00"


def _current_runtime_fixture(tmp_path: Path) -> tuple[Path, Path]:
    linked_path = tmp_path / "linked_source.json"
    linked_path.write_text('{"source": "fixture"}\n', encoding="utf-8")
    linked_hash = hashlib.sha256(linked_path.read_bytes()).hexdigest()
    evidence = [
        {
            "evidence_id": "evidence_fixture_1",
            "source_type": "news",
            "source": "fixture",
            "published_at": "2026-06-29T10:00:00+00:00",
            "as_of": AS_OF,
            "domain_id": DOMAIN_ID,
            "tickers": [],
            "sectors": [DOMAIN_ID],
            "evidence_type": "sector_demand",
            "summary": "AMD demand and AI infrastructure spending remain strong.",
            "stance_hint": "positive",
            "strength": 0.8,
            "freshness_score": 0.9,
            "directness": "sector",
            "reliability_score": 0.8,
            "provenance": {"required_lane_eligible": True},
            "point_in_time": {
                "status": "point_in_time_compatible",
                "future_evidence": False,
            },
        }
    ]
    runtime = {
        "run_id": "runtime_fixture",
        "created_at": "2026-07-01T00:00:00+00:00",
        "mode": "semiconductor_analyst_runtime",
        "runtime_contract": "dean_semiconductor_analyst_runtime_v1",
        "domain_id": DOMAIN_ID,
        "status": "semiconductor_analysis_partial_ready_for_review",
        "inputs": {"as_of": AS_OF},
        "source_artifacts": {
            "fixture": {"path": str(linked_path), "sha256": linked_hash}
        },
        "summary": {"evidence_count": 1},
        "adapter": {
            "as_of": AS_OF,
            "summary": {"evidence_count": 1},
        },
        "analyst_report": {
            "as_of": AS_OF,
            "domain_id": DOMAIN_ID,
            "evidence": evidence,
        },
        "safety": {
            "review_only": True,
            "pipeline_run_performed": False,
            "training_run_performed": False,
            "tuning_run_performed": False,
            "learning_write_performed": False,
            "production_config_write_performed": False,
            "broker_access_performed": False,
            "live_execution_performed": False,
        },
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    return runtime_path, linked_path


def test_current_runtime_loader_verifies_linked_hashes(tmp_path: Path) -> None:
    runtime_path, linked_path = _current_runtime_fixture(tmp_path)

    evidence = ArtifactEvidenceLoader().from_runtime_artifact(runtime_path)
    assert len(evidence) == 1

    linked_path.write_text('{"source": "tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ArtifactEvidenceLoader().from_runtime_artifact(runtime_path)


def test_verified_snapshot_runs_one_event_once(tmp_path: Path) -> None:
    runtime_path, _ = _current_runtime_fixture(tmp_path)

    snapshot = AnalystCoreReasoningSnapshot().build(
        runtime_json=runtime_path,
        save=False,
    )

    assert snapshot["summary"]["evidence_count"] == 1
    assert snapshot["summary"]["classified_event_count"] == 1
    assert snapshot["summary"]["directional_ticker_reasoning_event_count"] == 0
    assert snapshot["event_classification"]["class_counts"] == {
        "demand_driver": 1
    }
    assert snapshot["scenario_boundary"]["probabilities_generated"] is False
    receipt = snapshot["reasoning_receipt"]
    assert receipt["contract"] == "dean_analyst_reasoning_receipt_v1"
    assert receipt["analysis_output_sha256"]
    assert len(receipt["delta_sha256s"]) == 5
    assert receipt["hypothesis_status_change_performed"] is False
    assert snapshot["hypothesis_review_proposals"] == []


def test_classifier_does_not_promote_plain_text_ticker() -> None:
    packet = AnalysisPacket(
        packet_id="packet_fixture",
        event_records=[
            {
                "event_id": "evidence_fixture_1",
                "evidence_id": "evidence_fixture_1",
                "text": "AMD demand and AI infrastructure spending remain strong.",
                "event_class": "sector_demand",
                "evidence_type": "sector_demand",
                "tickers": [],
                "sectors": [DOMAIN_ID],
                "directness": "sector",
                "stance_hint": "positive",
                "strength": 0.8,
                "reliability_score": 0.8,
            }
        ],
    )

    delta = EventClassifierLens().analyze(
        packet,
        {"domain_id": DOMAIN_ID},
    )

    assert len(delta.classified_events_added) == 1
    assert delta.classified_events_added[0]["affected_tickers"] == []
    assert delta.classified_events_added[0]["directness"] == "indirect"


def test_default_registry_excludes_unverified_probability_modules() -> None:
    names = [lens.lens_name for lens in _build_default_registry().all_lenses()]
    assert names == [
        "event_classifier",
        "regime_context",
        "transmission_mapper",
        "hypothesis_ledger",
        "evidence_gap",
    ]
    assert "expectation_gap" not in names
    assert "historical_analog" not in names
