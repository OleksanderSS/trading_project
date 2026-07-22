import hashlib
import json

import pytest

from dean_os.review_decision_state import DecisionStateTransition, ReviewDecisionStateBuilder


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path, *, tasks=1, unresolved=2, can_evaluate=False, unscored=1):
    plan = _write(tmp_path / "plan.json", {
        "contract": "dean_replay_outcome_evidence_plan_v1",
        "summary": {
            "task_plan_count": tasks,
            "unresolved_lane_reference_count": unresolved,
            "outcome_evaluation_can_run": can_evaluate,
        },
    })
    voi = _write(tmp_path / "voi.json", {
        "contract": "dean_unknown_voi_review_v1",
        "inputs": {"evidence_plan": {"sha256": _sha(plan)}},
        "summary": {"validated_scored_count": 0, "unscored_count": unscored},
    })
    return plan, voi


def test_realistic_waiting_case_is_needs_more_data(tmp_path):
    plan, voi = _inputs(tmp_path)
    payload = ReviewDecisionStateBuilder(tmp_path / "out").build(plan, voi, save=False)
    assert payload["state"] == "needs_more_data"
    assert payload["safety"]["automatic_execution_allowed"] is False


def test_matured_with_open_gaps_is_partial_ready(tmp_path):
    plan, voi = _inputs(tmp_path, can_evaluate=True)
    payload = ReviewDecisionStateBuilder().build(
        plan, voi, previous_state="needs_more_data", save=False
    )
    assert payload["state"] == "partial_ready"


def test_no_tasks_is_no_action(tmp_path):
    plan, voi = _inputs(tmp_path, tasks=0, unresolved=0, unscored=0)
    payload = ReviewDecisionStateBuilder().build(plan, voi, save=False)
    assert payload["state"] == "no_action"


def test_direct_blocked_to_ready_transition_is_rejected():
    transition = DecisionStateTransition(
        previous_state="blocked", next_state="ready_for_review", reasons=["unsafe jump"],
        actor="reviewer", decided_at="2026-07-12T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="invalid review decision transition"):
        transition.validate_transition()


def test_stale_voi_binding_is_rejected(tmp_path):
    plan, voi = _inputs(tmp_path)
    payload = json.loads(voi.read_text(encoding="utf-8"))
    payload["inputs"]["evidence_plan"]["sha256"] = "0" * 64
    voi.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        ReviewDecisionStateBuilder().build(plan, voi, save=False)
