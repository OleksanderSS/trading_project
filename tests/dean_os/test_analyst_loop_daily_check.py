from __future__ import annotations

import json

from dean_os.analyst_core.analyst_loop_daily_check import AnalystLoopDailyCheck


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


def _write_ready_artifacts(paths, *, learning_bridge_status="dry_run_ready"):
    _write_json(
        paths["evidence_pack"],
        {
            "mode": "analyst_evidence_pack",
            "coverage": {
                "agent_lab_ready": True,
                "data_quality": "clean",
                "document_count": 2,
                "by_source_type": {"news": 1, "report": 1},
                "tickers": ["AMD"],
                "missing_requested_tickers": [],
            },
            "analyst_inputs": {"manager_plan": {"candidate_profiles": ["news_catalyst"]}},
        },
    )
    _write_json(paths["analyst_profiles"], {"mode": "analyst_profile_orchestrator", "profile_runs": [{"status": "completed"}]})
    _write_json(paths["profile_scorecard"], {"mode": "analyst_profile_scorecard", "summary": {"profile_count": 1}})
    _write_json(
        paths["learning_bridge"],
        {"mode": "analyst_learning_promotion_bridge", "promotion_gate": {"status": learning_bridge_status}},
    )
    _write_json(
        paths["review_approved_learning"],
        {"mode": "review_approved_learning_loop", "loop_gate": {"status": "reviewed_ready_to_apply"}},
    )
    _write_json(
        paths["outcome_evaluation"],
        {"mode": "analyst_outcome_evaluation_loop", "evaluation_gate": {"status": "applied"}},
    )
    _write_json(
        paths["calibration_gate"],
        {"mode": "analyst_calibration_gate", "summary": {"ready_for_review_profiles": ["generalist_base_analyst"]}},
    )
    _write_json(paths["calibration_proposals"], {"mode": "calibration_proposal_agent", "proposal_gate": {"status": "enqueued"}})
    _write_json(
        paths["calibration_review"],
        {"mode": "calibration_review_lifecycle", "lifecycle_gate": {"status": "approved_waiting_manual_implementation"}},
    )
    _write_json(paths["manual_backlog"], {"mode": "manual_implementation_backlog", "backlog_gate": {"status": "clear"}})


def _write_market_csv(path):
    path.write_text(
        "datetime,ticker,close\n2026-06-12T00:00:00+00:00,AMD,100.0\n",
        encoding="utf-8",
    )
    return str(path)


def _write_event_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"event_type":"agent_run","source":"test","run_id":"r1","timestamp":"2026-06-12T00:00:00+00:00","payload":{}}\n',
        encoding="utf-8",
    )
    return str(path)


def test_daily_check_blocks_on_learning_loop_gate(tmp_path):
    paths = _stage_paths(tmp_path)
    _write_ready_artifacts(paths, learning_bridge_status="blocked")
    market_path = _write_market_csv(tmp_path / "prices.csv")
    event_log_path = _write_event_log(tmp_path / "events.jsonl")

    payload = AnalystLoopDailyCheck(tmp_path / "reports").build(
        stage_paths=paths,
        market_data_path=market_path,
        latest_processed_prices=None,
        tickers=["AMD"],
        as_of="2026-06-12T01:00:00+00:00",
        event_log_path=event_log_path,
        save=False,
    )

    assert payload["summary"]["decision"] == "blocked"
    assert payload["summary"]["current_stage"] == "learning_bridge"
    assert any(item["code"] == "learning_loop_blocked" for item in payload["blockers"])
    assert payload["summary"]["stage_execution_performed"] is False


def test_daily_check_blocks_when_market_data_unavailable(tmp_path):
    paths = _stage_paths(tmp_path)
    _write_ready_artifacts(paths)
    event_log_path = _write_event_log(tmp_path / "events.jsonl")

    payload = AnalystLoopDailyCheck(tmp_path / "reports").build(
        stage_paths=paths,
        market_data_path=tmp_path / "missing_prices.csv",
        latest_processed_prices=None,
        tickers=["AMD"],
        as_of="2026-06-12T01:00:00+00:00",
        event_log_path=event_log_path,
        save=False,
    )

    assert payload["summary"]["decision"] == "blocked"
    assert any(item["code"] == "market_data_unavailable" for item in payload["blockers"])
    assert payload["summary"]["pipeline_run_performed"] is False


def test_daily_check_needs_operator_review_at_manual_boundary(tmp_path):
    paths = _stage_paths(tmp_path)
    _write_ready_artifacts(paths)
    market_path = _write_market_csv(tmp_path / "prices.csv")
    event_log_path = _write_event_log(tmp_path / "events.jsonl")

    payload = AnalystLoopDailyCheck(tmp_path / "reports").build(
        stage_paths=paths,
        market_data_path=market_path,
        latest_processed_prices=None,
        tickers=["AMD"],
        as_of="2026-06-12T01:00:00+00:00",
        event_log_path=event_log_path,
        save=False,
    )

    assert payload["summary"]["decision"] == "needs_operator_review"
    assert payload["summary"]["blocker_count"] == 0
    assert payload["checks"]["market_freshness"]["status"] == "fresh"
    assert payload["summary"]["broker_access_performed"] is False
