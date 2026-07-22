from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dean_os.full_system_cycle_closure import FullSystemCycleClosureBuilder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    cycle = tmp_path / "cycle.json"
    cycle_payload = {"contract": "dean_full_system_review_cycle_v1", "run_id": "cycle_1"}
    cycle.write_text(json.dumps(cycle_payload), encoding="utf-8")
    world = tmp_path / "world.json"
    world.write_text(
        json.dumps(
            {
                "cycle_binding_contract": "dean_full_system_cycle_world_model_bridge_v1",
                "upstream_bindings": {
                    "full_system_review_cycle": {
                        "sha256": _sha(cycle),
                        "run_id": "cycle_1",
                    }
                },
                "summary": {
                    "downstream_hash_binding_ready": True,
                    "hypothesis_count": 0,
                    "replay_task_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    monitor = tmp_path / "monitor.json"
    monitor.write_text(
        json.dumps(
            {
                "contract": "dean_replay_checkpoint_monitor_v1",
                "summary": {"task_count": 9},
            }
        ),
        encoding="utf-8",
    )
    return cycle, world, monitor


def test_closes_zero_hypothesis_cycle_without_stealing_prior_replay(tmp_path: Path) -> None:
    cycle, world, monitor = _fixtures(tmp_path)
    payload = FullSystemCycleClosureBuilder(
        tmp_path / "out", tmp_path / "ledger.jsonl"
    ).build(
        cycle_path=cycle,
        world_model_path=world,
        prior_checkpoint_monitor_path=monitor,
        save=False,
    )
    assert payload["summary"]["current_cycle_decision_state"] == "needs_more_data"
    assert payload["summary"]["current_cycle_new_replay_task_count"] == 0
    assert payload["summary"]["prior_lineage_monitoring_task_count"] == 9
    assert payload["summary"]["prior_tasks_promoted_to_current_cycle"] is False
    assert payload["summary"]["authorization_ledger_record_count"] == 0
    assert payload["safety"]["replay_registration_performed"] is False


def test_rejects_world_model_bound_to_another_cycle(tmp_path: Path) -> None:
    cycle, world, monitor = _fixtures(tmp_path)
    payload = json.loads(world.read_text(encoding="utf-8"))
    payload["upstream_bindings"]["full_system_review_cycle"]["sha256"] = "0" * 64
    world.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FullSystemCycleClosureBuilder(tmp_path / "out").build(
            cycle_path=cycle,
            world_model_path=world,
            prior_checkpoint_monitor_path=monitor,
            save=False,
        )


def test_new_replay_candidates_are_not_registration_authority(tmp_path: Path) -> None:
    cycle, world, monitor = _fixtures(tmp_path)
    world_payload = json.loads(world.read_text(encoding="utf-8"))
    world_payload["run_id"] = "world_1"
    world_payload["summary"]["hypothesis_count"] = 1
    world_payload["summary"]["replay_task_count"] = 5
    world.write_text(json.dumps(world_payload), encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "contract": "dean_world_model_replay_review_gate_v1",
                "source_packet": {
                    "run_id": "world_1",
                    "sha256": _sha(world),
                },
                "summary": {
                    "gate_status": "manual_review_required_for_replay_registration"
                },
            }
        ),
        encoding="utf-8",
    )

    payload = FullSystemCycleClosureBuilder(
        tmp_path / "out", tmp_path / "ledger.jsonl"
    ).build(
        cycle_path=cycle,
        world_model_path=world,
        prior_checkpoint_monitor_path=monitor,
        replay_review_gate_path=gate,
        save=False,
    )

    assert payload["summary"]["can_submit_new_replay_tasks_for_manual_review"] is True
    assert payload["summary"]["can_register_new_replay_tasks"] is False
    assert payload["summary"]["replay_review_gate_status"] == (
        "manual_review_required_for_replay_registration"
    )


def test_content_review_reformulation_state_is_not_reported_as_pending_review(
    tmp_path: Path,
) -> None:
    cycle, world, monitor = _fixtures(tmp_path)
    world_payload = json.loads(world.read_text(encoding="utf-8"))
    world_payload["run_id"] = "world_2"
    world_payload["summary"]["hypothesis_count"] = 4
    world_payload["summary"]["replay_task_count"] = 20
    world.write_text(json.dumps(world_payload), encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "contract": "dean_world_model_replay_review_gate_v1",
                "source_packet": {
                    "run_id": "world_2",
                    "sha256": _sha(world),
                },
                "summary": {
                    "gate_status": (
                        "hypothesis_review_complete_reformulation_required"
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    payload = FullSystemCycleClosureBuilder(
        tmp_path / "out", tmp_path / "ledger.jsonl"
    ).build(
        cycle_path=cycle,
        world_model_path=world,
        prior_checkpoint_monitor_path=monitor,
        replay_review_gate_path=gate,
        save=False,
    )

    assert payload["summary"]["closure_status"] == (
        "current_cycle_hypothesis_review_complete_reformulation_required"
    )
    assert payload["summary"]["current_cycle_decision_state"] == (
        "reformulation_required"
    )
    assert payload["summary"]["manual_hypothesis_review_complete"] is True
    assert payload["summary"]["can_submit_new_replay_tasks_for_manual_review"] is False
    assert payload["summary"]["can_register_new_replay_tasks"] is False


def test_closure_records_partial_replay_registration_and_historical_review(
    tmp_path: Path,
) -> None:
    cycle, world, monitor = _fixtures(tmp_path)
    world_payload = json.loads(world.read_text(encoding="utf-8"))
    world_payload["run_id"] = "world_registered"
    world_payload["summary"].update(
        {"hypothesis_count": 4, "replay_task_count": 20}
    )
    world.write_text(json.dumps(world_payload), encoding="utf-8")
    gate = tmp_path / "approved_gate.json"
    gate.write_text(
        json.dumps(
            {
                "run_id": "gate_registered",
                "contract": "dean_world_model_replay_review_gate_v1",
                "source_packet": {
                    "run_id": "world_registered",
                    "sha256": _sha(world),
                },
                "summary": {
                    "gate_status": "replay_tasks_approved_for_registration"
                },
            }
        ),
        encoding="utf-8",
    )
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "contract": "dean_world_model_replay_registration_bridge_v1",
                "source_gate": {
                    "run_id": "gate_registered",
                    "sha256": _sha(gate),
                },
                "summary": {
                    "bridge_status": "outcome_tracker_registration_partially_applied_historical_review_required",
                    "apply_requested": True,
                    "issue_count": 0,
                    "planned_registration_count": 10,
                    "registered_or_existing_count": 5,
                    "deferred_historical_count": 5,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = FullSystemCycleClosureBuilder(
        tmp_path / "out", tmp_path / "ledger.jsonl"
    ).build(
        cycle_path=cycle,
        world_model_path=world,
        prior_checkpoint_monitor_path=monitor,
        replay_review_gate_path=gate,
        replay_registration_path=registration,
        save=False,
    )

    summary = payload["summary"]
    assert summary["closure_status"] == (
        "current_cycle_replay_partially_registered_historical_review_required"
    )
    assert summary["registered_or_existing_replay_task_count"] == 5
    assert summary["historical_review_required_replay_task_count"] == 5
    assert summary["can_register_new_replay_tasks"] is False
    assert summary["outcome_scoring_performed"] is False
    assert payload["safety"]["replay_registration_performed"] is True
    assert payload["safety"]["can_trade"] is False
