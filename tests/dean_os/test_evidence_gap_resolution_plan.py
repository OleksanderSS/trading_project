from __future__ import annotations

import json

from dean_os.evidence_gap_resolution_plan import EvidenceGapResolutionPlan


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _review_action_artifact(path, *, action_type="needs_more_data"):
    return _write_json(
        path,
        {
            "mode": "review_action_apply_ceremony",
            "recorded_action": {
                "action_id": "action_1",
                "source_type": "agent_lab_report",
                "source_id": "source_1",
                "action_type": action_type,
                "status": "recorded",
                "reviewer": "human",
                "notes": "Current source is too thin.",
                "payload": {"data_request": "Add AMD and NVDA source coverage before learning promotion."},
            },
        },
    )


def _evidence_pack(path):
    return _write_json(
        path,
        {
            "mode": "analyst_evidence_pack",
            "inputs": {
                "news_data_paths": ["data/news.parquet"],
                "macro_data_paths": ["data/macro.parquet"],
                "materials_paths": [],
                "tickers": ["AMD", "MSFT", "NVDA"],
                "sectors": ["semiconductor"],
                "tags": ["ai_cycle"],
                "max_rows_per_table": 5,
                "source_routing_path": None,
            },
            "coverage": {
                "document_count": 10,
                "data_quality": "partial",
                "by_source_type": {"news": 5, "report": 5},
                "tickers": ["MSFT"],
                "missing_requested_tickers": ["AMD", "NVDA"],
                "date_range": {"start": "2026-03-09T00:00:00+00:00", "end": "2026-03-13T00:00:00+00:00"},
                "warning_count": 0,
                "dropped_count": 2,
            },
            "dropped": [{"path": "data/news.parquet", "reason": "max_rows_per_table", "remaining_rows": 25}],
            "source_routing": {"available": False},
        },
    )


def _decision_packet(path, evidence_path):
    return _write_json(
        path,
        {
            "mode": "review_decision_packet",
            "summary": {"source_id": "source_1", "packet_status": "manual_review_with_warnings"},
            "source": {"source_id": "source_1", "source_type": "agent_lab_report", "evidence_pack_path": evidence_path},
            "evidence_pack": {
                "tickers": ["MSFT"],
                "missing_requested_tickers": ["AMD", "NVDA"],
            },
            "review_checks": [{"status": "warn", "code": "missing_requested_tickers", "message": "Missing AMD, NVDA."}],
        },
    )


def test_evidence_gap_plan_builds_tasks_from_needs_more_data(tmp_path):
    review_action = _review_action_artifact(tmp_path / "review_action.json")
    evidence = _evidence_pack(tmp_path / "evidence.json")
    decision = _decision_packet(tmp_path / "packet.json", evidence)

    payload = EvidenceGapResolutionPlan(tmp_path / "reports").build(
        review_action_path=review_action,
        decision_packet_path=decision,
        save=False,
    )

    assert payload["summary"]["plan_status"] == "ready_to_collect"
    assert payload["summary"]["missing_tickers"] == ["AMD", "NVDA"]
    task_ids = {task["task_id"] for task in payload["resolution_tasks"]}
    assert "increase_table_row_window" in task_ids
    assert "add_amd_ticker_sources" in task_ids
    assert "add_nvda_ticker_sources" in task_ids
    assert "build_source_routing_snapshot" in task_ids
    assert "run_agent_analyst_evidence_pack.py" in payload["commands"]["rebuild_evidence_pack_after_sources_added"]


def test_evidence_gap_plan_blocks_non_needs_more_data_action(tmp_path):
    review_action = _review_action_artifact(tmp_path / "review_action.json", action_type="mark_reviewed")
    evidence = _evidence_pack(tmp_path / "evidence.json")
    decision = _decision_packet(tmp_path / "packet.json", evidence)

    payload = EvidenceGapResolutionPlan(tmp_path / "reports").build(
        review_action_path=review_action,
        decision_packet_path=decision,
        save=False,
    )

    assert payload["summary"]["plan_status"] == "blocked_not_needs_more_data"
    assert payload["summary"]["task_count"] == 0
    assert payload["validation"]["can_plan"] is False


def test_evidence_gap_plan_uses_explicit_evidence_path(tmp_path):
    review_action = _review_action_artifact(tmp_path / "review_action.json")
    evidence = _evidence_pack(tmp_path / "custom_evidence.json")

    payload = EvidenceGapResolutionPlan(tmp_path / "reports").build(
        review_action_path=review_action,
        decision_packet_path=None,
        evidence_pack_path=evidence,
        save=False,
    )

    assert payload["inputs"]["evidence_pack_path"] == evidence
    assert payload["summary"]["missing_ticker_count"] == 2
