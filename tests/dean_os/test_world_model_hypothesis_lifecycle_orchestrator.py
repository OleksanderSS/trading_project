from __future__ import annotations

import hashlib
import json

import pandas as pd

from dean_os.world_model_event_learning import WORLD_MODEL_EVENT_LEARNING_CONTRACT
from dean_os.world_model_hypothesis_lifecycle_orchestrator import (
    WorldModelHypothesisLifecycleOrchestrator,
)
from dean_os.world_model_replay_review_gate import (
    WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
)


NOW = "2026-01-01T15:00:00+00:00"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path):
    packet_path = tmp_path / "packet.json"
    gate_path = tmp_path / "gate.json"
    draft_path = tmp_path / "draft_v2.json"
    prices_path = tmp_path / "prices.parquet"
    packet = {
        "run_id": "packet_1",
        "created_at": NOW,
        "contract": WORLD_MODEL_EVENT_LEARNING_CONTRACT,
        "cycle_binding_contract": "dean_full_system_cycle_world_model_bridge_v1",
        "summary": {
            "domain_id": "test_domain",
            "as_of": NOW,
            "downstream_hash_binding_ready": True,
            "expectation_context_available": False,
            "can_trade": False,
        },
        "classified_events": [
            {
                "evidence_id": "e1",
                "title": "Funding warning",
                "provenance": {"published_at": NOW},
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "hypothesis": "Relative performance weakens",
                "trigger_evidence_ids": ["e1"],
                "supporting_evidence_ids": [],
                "horizons_to_check": [20],
            }
        ],
        "replay_tasks": [
            {
                "task_id": "replay_h1_20d",
                "hypothesis_id": "h1",
                "as_of": NOW,
                "packet_as_of": NOW,
                "trigger_event_at": NOW,
                "trigger_evidence_id": "e1",
                "horizon_days": 20,
                "due_at": "2026-01-21T15:00:00+00:00",
                "checkpoint_state_at_packet": "scheduled",
                "replay_scope": "event_response",
                "horizon_family": "event_response_fixed_v1",
                "manual_review_gate_required": True,
            }
        ],
        "hypothesis_alignment_review": {
            "contract": "dean_cycle_hypothesis_alignment_review_v1",
            "summary": {
                "horizon_substitution_allowed": False,
                "unaligned_upstream_hypothesis_count": 0,
            },
            "alignments": [
                {
                    "upstream_hypothesis_id": "u1",
                    "world_hypothesis_ids": ["h1"],
                    "upstream_horizons_days": [30, 90, 180],
                }
            ],
        },
        "analysis_packet": {"packet_id": "packet_1"},
        "delta_trail": [],
    }
    _write(packet_path, packet)
    gate = {
        "run_id": "gate_1",
        "created_at": NOW,
        "contract": WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
        "source_packet": {"run_id": "packet_1", "sha256": _sha(packet_path)},
        "summary": {
            "manual_hypothesis_review_complete": True,
            "pending_hypothesis_disposition_count": 0,
        },
        "hypothesis_review": [
            {
                "hypothesis_id": "h1",
                "disposition": "accept_for_replay",
                "rationale": "Directional claim is coherent",
            }
        ],
    }
    _write(gate_path, gate)
    draft = {
        "contract": "dean_world_model_hypothesis_resolution_specs_v2",
        "source_packet": {"run_id": "packet_1", "sha256": _sha(packet_path)},
        "source_review_gate": {"run_id": "gate_1", "sha256": _sha(gate_path)},
        "resolutions": {
            "h1": {
                "resolution_action": "retain_claim",
                "resolved_hypothesis": "Relative performance weakens",
                "expected_observations": ["Basket underperforms"],
                "invalidation_signals": ["Basket outperforms"],
                "measurement_spec": {
                    "primary_horizon_days": 20,
                    "target_metrics": ["basket_relative_total_return"],
                    "relative_return_expected_direction": "negative",
                    "assessment_rule": "Use the direction contract.",
                    "measurement_context": {
                        "context_contract": "dean_hypothesis_measurement_context_v1",
                        "context_as_of": NOW,
                        "trigger_event_at": NOW,
                        "capital_equipment_basket": {
                            "members": ["A", "B", "C"],
                            "minimum_checkpoint_coverage": 3,
                            "benchmark": "BM",
                        },
                        "baseline_sources": [
                            {
                                "published_at": "2025-12-31T10:00:00+00:00",
                                "source_locator": "https://example.test/baseline",
                            }
                        ],
                        "automatic_outcome_scoring_allowed": False,
                    },
                },
                "registration_blockers": [],
            }
        },
    }
    _write(draft_path, draft)
    sessions = pd.bdate_range("2024-01-02", "2025-12-31", tz="UTC")
    rows = []
    for index, session in enumerate(sessions):
        benchmark = 100 * (1.0003 ** index)
        cycle = ((index % 29) - 14) / 1000
        for offset, ticker in enumerate(("A", "B", "C")):
            rows.append(
                {
                    "datetime": session,
                    "ticker": ticker,
                    "close": benchmark * (1 + cycle * (offset + 1) / 3),
                }
            )
        rows.append({"datetime": session, "ticker": "BM", "close": benchmark})
    pd.DataFrame(rows).to_parquet(prices_path, index=False)
    return packet_path, gate_path, draft_path, prices_path


def test_lifecycle_prepares_resolves_and_stops_at_manual_gate(tmp_path):
    packet, gate, draft, prices = _fixtures(tmp_path)
    payload = WorldModelHypothesisLifecycleOrchestrator(
        tmp_path / "lifecycle"
    ).build(
        packet_json=packet,
        source_review_gate_json=gate,
        resolution_specs_v2_json=draft,
        price_paths=[prices],
        pipeline_paths=[],
        save=True,
    )

    summary = payload["summary"]
    assert summary["status"] == "prepared_resolved_pending_manual_review"
    assert summary["measurement_contract_ready_count"] == 1
    assert summary["resolved_packet_created"] is True
    assert summary["manual_review_gate_created"] is True
    assert summary["hypothesis_approval_performed"] is False
    assert summary["replay_registration_performed"] is False
    assert summary["can_trade"] is False
    inbox = payload["review_inbox"]
    assert len(inbox["proposed_contracts"]) == 1
    assert len(inbox["blockers"]) == 0
    assert len(inbox["pending_decisions"]) == 1
    assert inbox["pending_decisions"][0]["decision_type"] == (
        "hypothesis_disposition"
    )


def test_lifecycle_attaches_machine_proposal_without_changing_status(tmp_path):
    packet, gate, draft, prices = _fixtures(tmp_path)
    reasoning_path = tmp_path / "reasoning.json"
    _write(
        reasoning_path,
        {
            "run_id": "reasoning_1",
            "created_at": NOW,
            "contract": "dean_analyst_core_reasoning_snapshot_v1",
            "inputs": {"domain_id": "test_domain", "as_of": NOW},
            "reasoning_receipt": {
                "contract": "dean_analyst_reasoning_receipt_v1",
                "receipt_id": "reasoning_receipt_1",
            },
            "hypothesis_review_proposals": [
                {
                    "proposal_id": "proposal_h1",
                    "hypothesis_id": "h1",
                    "proposal_type": "candidate_contradiction",
                    "suggested_status": "weakened",
                    "requires_manual_review": True,
                    "requires_outcome_evidence": True,
                    "status_changed": False,
                }
            ],
        },
    )

    payload = WorldModelHypothesisLifecycleOrchestrator(
        tmp_path / "lifecycle_with_reasoning"
    ).build(
        packet_json=packet,
        source_review_gate_json=gate,
        resolution_specs_v2_json=draft,
        reasoning_snapshot_json=reasoning_path,
        price_paths=[prices],
        pipeline_paths=[],
        save=True,
    )

    assert payload["summary"]["machine_review_proposal_count"] == 1
    decision = payload["review_inbox"]["pending_decisions"][0]
    attached = decision["machine_review_proposals"][0]
    assert attached["proposal_id"] == "proposal_h1"
    assert attached["proposal_only"] is True
    assert attached["status_changed"] is False
    assert payload["summary"]["hypothesis_approval_performed"] is False
