from __future__ import annotations

import json

from dean_os.agent_learning_loop_runbook import AgentLearningLoopRunbook


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _stage_paths(tmp_path):
    return {
        "evidence_pack": tmp_path / "evidence.json",
        "analyst_profiles": tmp_path / "profiles.json",
        "profile_scorecard": tmp_path / "scorecard.json",
        "learning_bridge": tmp_path / "bridge.json",
        "review_approved_learning": tmp_path / "review_learning.json",
        "outcome_evaluation": tmp_path / "outcomes.json",
        "calibration_gate": tmp_path / "gate.json",
        "calibration_proposals": tmp_path / "proposals.json",
        "calibration_review": tmp_path / "review.json",
        "manual_backlog": tmp_path / "backlog.json",
    }


def _write_ready_prefix(paths):
    _write_json(paths["evidence_pack"], {"mode": "analyst_evidence_pack", "coverage": {"agent_lab_ready": True}})
    _write_json(
        paths["analyst_profiles"],
        {"mode": "analyst_profile_orchestrator", "profile_runs": [{"status": "completed"}]},
    )
    _write_json(
        paths["profile_scorecard"],
        {"mode": "analyst_profile_scorecard", "summary": {"profile_count": 1, "activation_ready_profiles": ["base"]}},
    )
    _write_json(
        paths["learning_bridge"],
        {"mode": "analyst_learning_promotion_bridge", "promotion_gate": {"status": "dry_run_ready"}},
    )
    _write_json(
        paths["review_approved_learning"],
        {"mode": "review_approved_learning_loop", "loop_gate": {"status": "reviewed_ready_to_apply"}},
    )


def test_runbook_stops_at_missing_first_artifact(tmp_path):
    payload = AgentLearningLoopRunbook(tmp_path / "reports").build(
        stage_paths={key: value for key, value in _stage_paths(tmp_path).items()},
        save=False,
    )

    assert payload["loop_position"]["stage_id"] == "evidence_pack"
    assert payload["loop_position"]["status"] == "missing_artifact"
    assert "run_agent_analyst_evidence_pack.py" in payload["loop_position"]["next_command"]
    assert payload["summary"]["config_write_performed"] is False


def test_runbook_stops_at_outcome_newer_prices_gate(tmp_path):
    paths = _stage_paths(tmp_path)
    _write_ready_prefix(paths)
    _write_json(
        paths["outcome_evaluation"],
        {"mode": "analyst_outcome_evaluation_loop", "evaluation_gate": {"status": "blocked_need_newer_prices"}},
    )

    payload = AgentLearningLoopRunbook(tmp_path / "reports").build(
        stage_paths={key: value for key, value in paths.items()},
        save=False,
    )

    assert payload["loop_position"]["stage_id"] == "outcome_evaluation"
    assert payload["loop_position"]["status"] == "blocked_need_newer_prices"
    assert "newer prices" in payload["loop_position"]["stop_reason"]


def test_runbook_reaches_manual_implementation_boundary(tmp_path):
    paths = _stage_paths(tmp_path)
    _write_ready_prefix(paths)
    _write_json(
        paths["outcome_evaluation"],
        {"mode": "analyst_outcome_evaluation_loop", "evaluation_gate": {"status": "applied"}},
    )
    _write_json(
        paths["calibration_gate"],
        {"mode": "analyst_calibration_gate", "summary": {"ready_for_review_profiles": ["base"], "blocked_profiles": []}},
    )
    _write_json(
        paths["calibration_proposals"],
        {"mode": "calibration_proposal_agent", "proposal_gate": {"status": "enqueued"}},
    )
    _write_json(
        paths["calibration_review"],
        {"mode": "calibration_review_lifecycle", "lifecycle_gate": {"status": "approved_waiting_manual_implementation"}},
    )
    _write_json(
        paths["manual_backlog"],
        {"mode": "manual_implementation_backlog", "backlog_gate": {"status": "manual_implementation_required"}},
    )

    payload = AgentLearningLoopRunbook(tmp_path / "reports").build(
        stage_paths={key: value for key, value in paths.items()},
        save=False,
    )

    assert payload["loop_position"]["stage_id"] == "manual_backlog"
    assert payload["loop_position"]["status"] == "manual_implementation_required"
    assert "manual PR" in payload["loop_position"]["stop_reason"]
    assert payload["summary"]["pipeline_run_performed"] is False
