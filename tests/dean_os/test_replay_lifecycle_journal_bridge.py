from __future__ import annotations

import json

from dean_os.replay_lifecycle_journal_bridge import ReplayLifecycleJournalBridge


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_refresh_failure_is_journaled_once_with_valid_chain(tmp_path):
    lifecycle = tmp_path / "lifecycle.json"
    registration = tmp_path / "registration.json"
    refresh = tmp_path / "refresh.json"
    journal = tmp_path / "journal.jsonl"
    _write(registration, {"run_id": "registration_1", "contract": "registration"})
    _write(
        lifecycle,
        {
            "run_id": "life_1",
            "contract": "dean_replay_outcome_lifecycle_v1",
            "created_at": "2026-07-14T01:00:00+00:00",
            "review_inbox": {
                "data_actions": [
                    {
                        "task_id": "task_60",
                        "hypothesis_id": "h1",
                        "due_at": "2026-07-13T20:13:54+00:00",
                        "route_state": "due_waiting_for_verified_checkpoint_data",
                    }
                ]
            },
            "outcome_review": None,
            "learning_review": None,
        },
    )
    _write(
        refresh,
        {
            "run_id": "refresh_1",
            "contract": "dean_replay_evidence_refresh_controller_v1",
            "created_at": "2026-07-14T01:01:00+00:00",
            "inputs": {"apply_refresh": True},
            "summary": {
                "status": "single_refresh_pass_failed",
                "refresh_job_count": 1,
            },
            "refresh_jobs": [{"task_id": "task_60"}],
            "refresh_failure": {
                "error_type": "RuntimeError",
                "error": "source returned no rows",
                "retry_automatically": False,
            },
            "snapshot": None,
            "safety": {"network_access_performed": True},
        },
    )

    bridge = ReplayLifecycleJournalBridge(tmp_path / "report")
    first = bridge.build(
        lifecycle_json=lifecycle,
        registration_json=registration,
        refresh_json=refresh,
        journal_path=journal,
        apply=True,
        save=False,
    )
    second = bridge.build(
        lifecycle_json=lifecycle,
        registration_json=registration,
        refresh_json=refresh,
        journal_path=journal,
        apply=True,
        save=False,
    )

    assert first["summary"]["new_event_count"] == 3
    assert first["event_type_counts"] == {
        "action_executed": 1,
        "action_proposed": 1,
        "incident_recorded": 1,
    }
    assert second["summary"]["new_event_count"] == 0
    assert second["summary"]["existing_event_count"] == 3
    assert second["summary"]["journal_record_count"] == 3
    assert second["summary"]["journal_chain_valid"] is True


def test_local_snapshot_ingestion_is_journaled_as_source_not_outcome(tmp_path):
    lifecycle = tmp_path / "lifecycle.json"
    registration = tmp_path / "registration.json"
    ingestion = tmp_path / "ingestion.json"
    snapshot = tmp_path / "snapshot.parquet"
    journal = tmp_path / "journal.jsonl"
    snapshot.write_bytes(b"verified snapshot")
    import hashlib

    snapshot_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    _write(registration, {"run_id": "registration_1", "contract": "registration"})
    _write(
        lifecycle,
        {
            "run_id": "life_1",
            "contract": "dean_replay_outcome_lifecycle_v1",
            "created_at": "2026-07-14T01:00:00+00:00",
            "review_inbox": {"data_actions": []},
            "outcome_review": None,
            "learning_review": None,
        },
    )
    _write(
        ingestion,
        {
            "run_id": "ingest_1",
            "contract": "dean_verified_local_market_snapshot_ingestion_v1",
            "created_at": "2026-07-14T01:01:00+00:00",
            "inputs": {"apply_ingestion": True, "candidate_path": "candidate.csv"},
            "summary": {
                "status": "snapshot_ingested_lifecycle_completed",
                "candidate_valid": True,
                "snapshot_ingested": True,
            },
            "snapshot": {
                "path": str(snapshot),
                "sha256": snapshot_sha,
                "source_candidate_sha256": "a" * 64,
                "format": "parquet",
            },
        },
    )

    payload = ReplayLifecycleJournalBridge(tmp_path / "report").build(
        lifecycle_json=lifecycle,
        registration_json=registration,
        ingestion_json=ingestion,
        journal_path=journal,
        apply=True,
        save=False,
    )

    assert payload["event_type_counts"] == {
        "action_executed": 1,
        "source_snapshot_recorded": 1,
    }
    assert "outcome_recorded" not in payload["event_type_counts"]
    assert payload["summary"]["journal_chain_valid"] is True
