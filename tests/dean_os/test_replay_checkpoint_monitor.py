import json

from dean_os.replays.replay_checkpoint_monitor import ReplayCheckpointMonitorBuilder


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _plan(tmp_path):
    return _write(tmp_path / "plan.json", {
        "contract": "dean_replay_outcome_evidence_plan_v1",
        "task_plans": [{
            "task_id": "t1", "hypothesis_id": "h1", "horizon_days": 30,
            "checkpoints": {
                "collection_start": "2026-07-01T00:00:00+00:00",
                "pre_due_source_review": "2026-07-24T00:00:00+00:00",
                "due_outcome_review": "2026-07-31T00:00:00+00:00",
            },
            "evidence_lanes": [{
                "gap_id": "g1", "description": "orders", "resolution_status": "missing",
                "collection_route": {"status": "source_missing", "next_action": "collect orders"},
            }],
            "expected_observations": ["orders rise"], "invalidation_signals": ["orders fall"],
        }],
    })


def test_collecting_status_does_not_evaluate_outcome(tmp_path):
    payload = ReplayCheckpointMonitorBuilder(tmp_path / "out").build(
        _plan(tmp_path), as_of="2026-07-12T00:00:00+00:00", save=False
    )
    task = payload["tasks"][0]
    assert task["checkpoint_status"] == "collecting"
    assert task["can_evaluate_outcome"] is False
    assert task["actions"][0]["automatic_execution_allowed"] is False


def test_pre_due_and_due_transitions(tmp_path):
    plan = _plan(tmp_path)
    pre_due = ReplayCheckpointMonitorBuilder().build(
        plan, as_of="2026-07-25T00:00:00+00:00", save=False
    )
    due = ReplayCheckpointMonitorBuilder().build(
        plan, as_of="2026-08-01T00:00:00+00:00", save=False
    )
    assert pre_due["tasks"][0]["checkpoint_status"] == "pre_due_source_review_due"
    assert due["tasks"][0]["checkpoint_status"] == "outcome_review_due"
    assert due["tasks"][0]["can_evaluate_outcome"] is True
    assert due["safety"]["outcome_evaluation_performed"] is False
