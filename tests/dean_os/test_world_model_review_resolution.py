from __future__ import annotations

import hashlib
import json

import pytest

from dean_os.world_model_event_learning import WORLD_MODEL_EVENT_LEARNING_CONTRACT
from dean_os.world_model_replay_review_gate import (
    WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
    WorldModelReplayReviewGate,
)
from dean_os.world_model.world_model_resolution_journal import (
    append_world_model_resolution_journal,
)
from dean_os.world_model_review_resolution import (
    HYPOTHESIS_RESOLUTION_SPECS_CONTRACT,
    HYPOTHESIS_RESOLUTION_SPECS_CONTRACT_V2,
    WorldModelReviewResolutionBuilder,
)


NOW = "2026-07-13T10:00:00+00:00"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path):
    packet_path = tmp_path / "packet.json"
    gate_path = tmp_path / "gate.json"
    specs_path = tmp_path / "specs.json"
    packet = {
        "run_id": "packet_1",
        "created_at": NOW,
        "contract": WORLD_MODEL_EVENT_LEARNING_CONTRACT,
        "cycle_binding_contract": "dean_full_system_cycle_world_model_bridge_v1",
        "summary": {
            "domain_id": "test_domain",
            "as_of": NOW,
            "can_trade": False,
            "can_write_learning_memory": False,
            "downstream_hash_binding_ready": True,
            "expectation_context_available": False,
        },
        "classified_events": [
            {
                "evidence_id": "e1",
                "title": "Positive guidance",
                "provenance": {"published_at": NOW},
            },
            {
                "evidence_id": "e2",
                "title": "Funding warning",
                "provenance": {"published_at": NOW},
            },
        ],
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "hypothesis": "Demand accelerates",
                "trigger_evidence_ids": ["e1"],
                "supporting_evidence_ids": [],
                "horizons_to_check": [20],
            },
            {
                "hypothesis_id": "h2",
                "hypothesis": "Capex accelerates",
                "trigger_evidence_ids": ["e2"],
                "supporting_evidence_ids": [],
                "horizons_to_check": [20],
            },
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
                "due_at": "2026-08-02T10:00:00+00:00",
                "checkpoint_state_at_packet": "scheduled",
                "replay_scope": "event_response",
                "horizon_family": "event_response_fixed_v1",
                "manual_review_gate_required": True,
            },
            {
                "task_id": "replay_h2_20d",
                "hypothesis_id": "h2",
                "as_of": NOW,
                "packet_as_of": NOW,
                "trigger_event_at": NOW,
                "trigger_evidence_id": "e2",
                "horizon_days": 20,
                "due_at": "2026-08-02T10:00:00+00:00",
                "checkpoint_state_at_packet": "scheduled",
                "replay_scope": "event_response",
                "horizon_family": "event_response_fixed_v1",
                "manual_review_gate_required": True,
            },
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
                },
                {
                    "upstream_hypothesis_id": "u2",
                    "world_hypothesis_ids": ["h2"],
                    "upstream_horizons_days": [30, 90, 180],
                },
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
                "rationale": "Coherent trigger",
            },
            {
                "hypothesis_id": "h2",
                "disposition": "reformulate",
                "rationale": "Wrong polarity",
                "proposed_hypothesis": "Capex expectations weaken",
            },
        ],
    }
    _write(gate_path, gate)
    specs = {
        "contract": HYPOTHESIS_RESOLUTION_SPECS_CONTRACT,
        "reviewer": "reviewer",
        "source_packet": {"run_id": "packet_1", "sha256": _sha(packet_path)},
        "source_review_gate": {"run_id": "gate_1", "sha256": _sha(gate_path)},
        "resolutions": {
            "h1": {
                "resolution_action": "retain_claim",
                "resolved_hypothesis": "Demand accelerates",
                "expected_observations": ["Estimates rise"],
                "invalidation_signals": ["Estimates fall"],
                "measurement_spec": {
                    "primary_horizon_days": 20,
                    "target_metrics": ["estimate_revision"],
                    "assessment_rule": "Compare point-in-time baselines.",
                },
                "registration_blockers": [],
            },
            "h2": {
                "resolution_action": "replace_claim",
                "resolved_hypothesis": "Capex expectations weaken",
                "expected_observations": ["Capex estimates fall"],
                "invalidation_signals": ["Capex estimates rise"],
                "measurement_spec": {
                    "primary_horizon_days": 20,
                    "target_metrics": ["capex_revision"],
                    "assessment_rule": "Compare point-in-time baselines.",
                },
                "registration_blockers": ["baseline_missing"],
            },
        },
    }
    _write(specs_path, specs)
    return packet_path, gate_path, specs_path


def test_resolution_versions_claims_and_preserves_lineage(tmp_path):
    packet_path, gate_path, specs_path = _fixtures(tmp_path)
    result = WorldModelReviewResolutionBuilder(tmp_path / "reports").build(
        packet_path, gate_path, specs_path, save=False
    )

    assert result["summary"]["retained_hypothesis_count"] == 1
    assert result["summary"]["reformulated_hypothesis_count"] == 1
    assert result["summary"]["registration_blocked_hypothesis_count"] == 1
    retained, replaced = result["hypotheses"]
    assert retained["hypothesis_id"] == "h1"
    assert replaced["hypothesis_id"] != "h2"
    assert replaced["original_hypothesis_id"] == "h2"
    assert replaced["resolution_lineage"]["source_packet_run_id"] == "packet_1"
    replaced_task = next(
        task for task in result["replay_tasks"] if task["hypothesis_id"] == replaced["hypothesis_id"]
    )
    assert replaced_task["registration_status"] == "candidate_blocked_pending_required_context"
    assert replaced_task["scenario_graph_id"] is None


def test_review_gate_cannot_accept_resolution_with_registration_blocker(tmp_path):
    packet_path, gate_path, specs_path = _fixtures(tmp_path)
    resolved = WorldModelReviewResolutionBuilder(tmp_path / "reports").build(
        packet_path, gate_path, specs_path, save=False
    )
    resolved_path = tmp_path / "resolved.json"
    _write(resolved_path, resolved)
    decisions = {
        item["hypothesis_id"]: {
            "disposition": "accept_for_replay",
            "rationale": "test",
        }
        for item in resolved["hypotheses"]
    }

    gate = WorldModelReplayReviewGate(tmp_path / "resolved_gate").build(
        resolved_path,
        approve=True,
        reviewer="reviewer",
        review_notes="test approval",
        hypothesis_dispositions=decisions,
        save=False,
    )

    assert gate["summary"]["can_register_replay_tasks"] is False
    assert any(
        issue.startswith("cycle_bound_accepted_hypotheses_have_registration_blockers")
        for issue in gate["summary"]["issues"]
    )


def test_resolution_journal_append_is_idempotent(tmp_path):
    packet_path, gate_path, specs_path = _fixtures(tmp_path)
    resolved = WorldModelReviewResolutionBuilder(tmp_path / "reports").build(
        packet_path, gate_path, specs_path, save=False
    )
    resolved_path = tmp_path / "resolved.json"
    _write(resolved_path, resolved)
    decisions = {
        resolved["hypotheses"][0]["hypothesis_id"]: {
            "disposition": "accept_for_replay",
            "rationale": "ready",
        },
        resolved["hypotheses"][1]["hypothesis_id"]: {
            "disposition": "defer",
            "rationale": "baseline missing",
        },
    }
    gate = WorldModelReplayReviewGate(tmp_path / "resolved_gate").build(
        resolved_path,
        reviewer="reviewer",
        review_notes="content review",
        hypothesis_dispositions=decisions,
        save=True,
    )
    gate_result_path = tmp_path / "resolved_gate" / "latest.json"
    closure_path = tmp_path / "closure.json"
    _write(
        closure_path,
        {
            "run_id": "closure_1",
            "created_at": NOW,
            "inputs": {
                "world_model": {"sha256": _sha(resolved_path)},
                "replay_review_gate": {"sha256": _sha(gate_result_path)},
            },
            "summary": {
                "closure_status": "current_cycle_hypothesis_review_complete_deferred",
                "current_cycle_decision_state": "deferred_pending_evidence",
                "can_register_new_replay_tasks": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
        },
    )
    kwargs = {
        "resolution_packet_json": resolved_path,
        "review_gate_json": gate_result_path,
        "closure_json": closure_path,
        "journal_path": tmp_path / "journal.jsonl",
    }

    first = append_world_model_resolution_journal(**kwargs)
    second = append_world_model_resolution_journal(**kwargs)

    assert gate["summary"]["gate_status"] == "hypothesis_review_complete_deferred"
    assert first["write_result"]["appended_count"] == 10
    assert second["write_result"]["appended_count"] == 0
    assert second["journal_status"]["record_count"] == 10


def test_resolution_rejects_source_packet_changed_after_review(tmp_path):
    packet_path, gate_path, specs_path = _fixtures(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["hypotheses"][0]["hypothesis"] = "mutated"
    _write(packet_path, packet)

    with pytest.raises(ValueError, match="changed after manual review"):
        WorldModelReviewResolutionBuilder(tmp_path / "reports").build(
            packet_path, gate_path, specs_path, save=False
        )


def test_resolution_rejects_measurement_baseline_published_after_trigger(tmp_path):
    packet_path, gate_path, specs_path = _fixtures(tmp_path)
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    specs["resolutions"]["h2"]["measurement_spec"]["measurement_context"] = {
        "context_contract": "dean_hypothesis_measurement_context_v1",
        "context_as_of": NOW,
        "trigger_event_at": NOW,
        "buyer_basket": {
            "minimum_checkpoint_coverage": 1,
            "members": [
                {
                    "ticker": "BUY",
                    "baseline_low_usd_billions": 10,
                    "baseline_midpoint_usd_billions": 10,
                    "baseline_high_usd_billions": 10,
                    "published_at": "2026-07-14T10:00:00+00:00",
                    "source_locator": "https://example.test/official",
                }
            ],
        },
        "capital_equipment_basket": {
            "members": ["EQP"],
            "minimum_checkpoint_coverage": 1,
            "benchmark": "SECTOR",
        },
        "automatic_outcome_scoring_allowed": False,
    }
    _write(specs_path, specs)

    with pytest.raises(ValueError, match="baseline source is after trigger"):
        WorldModelReviewResolutionBuilder(tmp_path / "reports").build(
            packet_path, gate_path, specs_path, save=False
        )


def test_v2_requires_direction_contract_for_relative_return_metric(tmp_path):
    packet_path, gate_path, specs_path = _fixtures(tmp_path)
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    specs["contract"] = HYPOTHESIS_RESOLUTION_SPECS_CONTRACT_V2
    measurement = specs["resolutions"]["h1"]["measurement_spec"]
    measurement["target_metrics"] = ["basket_relative_total_return"]
    _write(specs_path, specs)

    with pytest.raises(ValueError, match="requires a calibrated direction contract"):
        WorldModelReviewResolutionBuilder(tmp_path / "reports").build(
            packet_path, gate_path, specs_path, save=False
        )

    measurement["relative_return_direction_contract"] = {
        "contract": "dean_relative_return_direction_contract_v1",
        "status": "calibrated_pre_outcome_direction_contract",
        "expected_direction": "negative",
        "horizon_days": 20,
        "neutral_band_absolute_return": 0.04,
        "blockers": [],
    }
    _write(specs_path, specs)
    result = WorldModelReviewResolutionBuilder(tmp_path / "reports").build(
        packet_path, gate_path, specs_path, save=False
    )
    assert result["hypotheses"][0]["measurement_spec"][
        "relative_return_direction_contract"
    ]["expected_direction"] == "negative"
