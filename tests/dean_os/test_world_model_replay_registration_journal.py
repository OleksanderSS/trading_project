from __future__ import annotations

import hashlib
import json

from dean_os.world_model_replay_registration_journal import (
    append_world_model_replay_registration_journal,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registration_journal_is_hash_bound_and_idempotent(tmp_path):
    gate_path = tmp_path / "gate.json"
    registration_path = tmp_path / "registration.json"
    closure_path = tmp_path / "closure.json"
    journal_path = tmp_path / "journal.jsonl"
    gate = {
        "run_id": "gate_1",
        "created_at": "2026-07-13T16:00:00+00:00",
        "contract": "dean_world_model_replay_review_gate_v1",
        "source_packet": {"domain_id": "test_domain"},
        "summary": {"can_register_replay_tasks": True},
        "registration_bundle": {
            "bundle_id": "bundle_1",
            "approved_by": "operator",
            "approved_at": "2026-07-13T16:00:00+00:00",
            "task_count": 2,
            "review_notes": "observation only",
        },
    }
    _write(gate_path, gate)
    registration = {
        "run_id": "registration_1",
        "created_at": "2026-07-13T16:05:00+00:00",
        "contract": "dean_world_model_replay_registration_bridge_v1",
        "source_gate": {"run_id": "gate_1", "sha256": _sha(gate_path)},
        "summary": {"apply_requested": True, "issue_count": 0},
        "registered_events": [],
        "skipped_existing_events": [
            {
                "task_id": "task_future",
                "event_id": "event_1",
                "source": "world_model_replay|task=task_future",
                "status": "skipped_existing_outcome_tracker_event",
            }
        ],
        "deferred_historical_tasks": [
            {
                "task_id": "task_history",
                "due_at": "2026-07-01T00:00:00+00:00",
                "status": "deferred_to_historical_point_in_time_outcome_review",
            }
        ],
    }
    _write(registration_path, registration)
    closure = {
        "run_id": "closure_1",
        "created_at": "2026-07-13T16:10:00+00:00",
        "contract": "dean_full_system_cycle_closure_v1",
        "inputs": {
            "replay_review_gate": {"sha256": _sha(gate_path)},
            "replay_registration": {"sha256": _sha(registration_path)},
        },
        "summary": {
            "closure_status": "current_cycle_replay_partially_registered_historical_review_required",
            "replay_registration_observed": True,
            "registered_or_existing_replay_task_count": 1,
            "historical_review_required_replay_task_count": 1,
        },
    }
    _write(closure_path, closure)

    first = append_world_model_replay_registration_journal(
        review_gate_json=gate_path,
        registration_json=registration_path,
        closure_json=closure_path,
        journal_path=journal_path,
    )
    second = append_world_model_replay_registration_journal(
        review_gate_json=gate_path,
        registration_json=registration_path,
        closure_json=closure_path,
        journal_path=journal_path,
    )

    assert first["write_result"]["appended_count"] == 7
    assert first["journal_status"]["chain_valid"] is True
    assert first["journal_status"]["action_execution_performed"] is True
    assert second["write_result"]["appended_count"] == 0
    assert second["write_result"]["existing_count"] == 7
    assert second["can_trade"] is False
