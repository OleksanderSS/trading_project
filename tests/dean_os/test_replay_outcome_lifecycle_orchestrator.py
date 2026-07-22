from __future__ import annotations

import hashlib
import json

import pandas as pd

from dean_os.replay_outcome_lifecycle_orchestrator import (
    ReplayOutcomeLifecycleOrchestrator,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path, *, horizon=20, price_metric=False):
    packet = tmp_path / "packet.json"
    gate = tmp_path / "gate.json"
    registration = tmp_path / "registration.json"
    _write(
        packet,
        {
            "run_id": "packet_1",
            "hypotheses": [
                {"hypothesis_id": "h1", "hypothesis": "bounded test claim"}
            ],
        },
    )
    measurement = {
        "primary_horizon_days": 20,
        "target_metrics": ["verified_licensing_friction_indicator_count"],
    }
    if price_metric:
        measurement = {
            "primary_horizon_days": 20,
            "target_metrics": ["capital_equipment_basket_relative_total_return"],
            "measurement_context": {
                "capital_equipment_basket": {
                    "members": ["AMAT", "LRCX", "KLAC", "ASML"],
                    "minimum_checkpoint_coverage": 3,
                    "benchmark": "SOXX",
                }
            },
        }
    _write(
        gate,
        {
            "run_id": "gate_1",
            "source_packet": {
                "run_id": "packet_1",
                "path": str(packet),
                "sha256": _sha(packet),
            },
            "hypothesis_review": [
                {
                    "hypothesis_id": "h1",
                    "hypothesis": "bounded test claim",
                    "disposition": "accept_for_replay",
                    "measurement_spec": measurement,
                }
            ],
        },
    )
    _write(
        registration,
        {
            "contract": "dean_world_model_replay_registration_bridge_v1",
            "source_gate": {"run_id": "gate_1", "sha256": _sha(gate)},
            "registration_plan": [
                {
                    "task_id": f"task_{horizon}",
                    "hypothesis_id": "h1",
                    "event_anchor_at": "2026-06-01T10:00:00+00:00",
                    "horizon_days": horizon,
                    "due_at": "2026-06-21T10:00:00+00:00",
                }
            ],
            "deferred_historical_tasks": [],
        },
    )
    return packet, gate, registration


def _builder(tmp_path):
    return ReplayOutcomeLifecycleOrchestrator(
        output_dir=tmp_path / "lifecycle",
        router_output_dir=tmp_path / "router",
        outcome_output_dir=tmp_path / "outcome",
        learning_output_dir=tmp_path / "learning",
    )


def test_primary_matured_task_builds_outcome_reverse_analysis_and_closes_route(
    tmp_path,
):
    packet, gate, registration = _artifacts(tmp_path)
    payload = _builder(tmp_path).build(
        registration_json=registration,
        review_gate_json=gate,
        packet_json=packet,
        as_of="2026-06-22T21:00:00+00:00",
        verified_price_paths=[],
        pipeline_paths=[],
        prior_outcome_json_paths=[],
        journal_path=tmp_path / "journal.jsonl",
    )

    assert payload["summary"]["status"] == (
        "primary_outcome_packet_pending_causal_review"
    )
    assert payload["summary"]["outcome_packet_created"] is True
    assert payload["summary"]["reverse_analysis_created"] is True
    assert payload["summary"]["primary_outcome_count"] == 1
    assert payload["final_due_router"]["summary"]["reviewed_checkpoint_count"] == 1
    assert len(payload["review_inbox"]["pending_decisions"]) == 1


def test_intermediate_checkpoint_is_recorded_without_final_learning_proposal(tmp_path):
    packet, gate, registration = _artifacts(tmp_path, horizon=60)
    payload = _builder(tmp_path).build(
        registration_json=registration,
        review_gate_json=gate,
        packet_json=packet,
        as_of="2026-06-22T21:00:00+00:00",
        verified_price_paths=[],
        pipeline_paths=[],
        prior_outcome_json_paths=[],
        journal_path=tmp_path / "journal.jsonl",
    )

    assert payload["summary"]["status"] == (
        "intermediate_checkpoint_packet_recorded"
    )
    assert payload["summary"]["primary_outcome_count"] == 0
    assert payload["summary"]["reverse_analysis_created"] is False
    assert payload["review_inbox"]["pending_decisions"] == []


def test_due_price_checkpoint_stops_before_outcome_when_verified_session_missing(
    tmp_path,
):
    packet, gate, registration = _artifacts(tmp_path, price_metric=True)
    prices = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-06-20T00:00:00+00:00"] * 4,
            "ticker": ["AMAT", "LRCX", "KLAC", "SOXX"],
            "close": [100.0] * 4,
        }
    ).to_csv(prices, index=False)

    payload = _builder(tmp_path).build(
        registration_json=registration,
        review_gate_json=gate,
        packet_json=packet,
        as_of="2026-06-22T21:00:00+00:00",
        verified_price_paths=[prices],
        pipeline_paths=[],
        prior_outcome_json_paths=[],
        journal_path=tmp_path / "journal.jsonl",
    )

    assert payload["summary"]["status"] == (
        "waiting_for_verified_checkpoint_data"
    )
    assert payload["summary"]["outcome_packet_created"] is False
    assert payload["summary"]["reverse_analysis_created"] is False
    assert payload["system_recommendations"][0]["action_type"] == (
        "refresh_verified_checkpoint_evidence"
    )
