from __future__ import annotations

import json
import sqlite3

from dean_os.analyst_core.analyst_review_inbox import AnalystReviewInbox


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _write_report(path, run_id="run_1", note_count=2):
    return _write_json(
        path,
        {
            "run_id": run_id,
            "mode": "agent_lab",
            "note_count": note_count,
            "research_notes": [{"note_id": f"note_{index}"} for index in range(note_count)],
            "summary": {"context_tags": ["ai_cycle"]},
        },
    )


def _bridge_source(report_json, *, source_id="run_1", blockers=None):
    blockers = blockers or ["source_agent_lab_report_not_marked_reviewed"]
    return {
        "source_type": "agent_lab_report",
        "source_id": source_id,
        "profile": "generalist_base_analyst",
        "profile_run_id": "profile_run_1",
        "evidence_pack_run_id": "evidence_1",
        "evidence_pack_path": "reports/evidence/latest.json",
        "report_json": report_json,
        "review": {"reviewed": False, "needs_more_data": False, "action_count": 0, "review_action_ids": []},
        "note_count": 2,
        "candidate_count": 2,
        "promotable_count": 0,
        "promoted_count": 0,
        "candidates": [
            {
                "note_id": "note_1",
                "agent_name": "evidence_synthesis",
                "data_quality": "strong",
                "blockers": blockers,
            },
            {
                "note_id": "note_2",
                "agent_name": "specialist_research",
                "data_quality": "partial",
                "blockers": blockers,
            },
        ],
    }


def _write_bridge(path, source):
    return _write_json(
        path,
        {
            "run_id": "bridge_1",
            "mode": "analyst_learning_promotion_bridge",
            "promotion_gate": {"status": "blocked"},
            "sources": [source],
        },
    )


def _write_review_action_store(path, source_id, action_type="mark_reviewed"):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE review_actions (
                action_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                status TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                linked_proposal_id TEXT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        payload = {
            "action_id": "action_1",
            "source_type": "agent_lab_report",
            "source_id": source_id,
            "action_type": action_type,
            "status": "active",
            "reviewer": "human",
            "created_at": "2026-06-12T00:00:00+00:00",
        }
        conn.execute(
            """
            INSERT INTO review_actions
            (action_id, source_type, source_id, action_type, status, reviewer, linked_proposal_id, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["action_id"],
                payload["source_type"],
                payload["source_id"],
                payload["action_type"],
                payload["status"],
                payload["reviewer"],
                None,
                payload["created_at"],
                json.dumps(payload),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return str(path)


def test_review_inbox_groups_unreviewed_source_as_ready(tmp_path):
    report_json = _write_report(tmp_path / "agent_lab" / "run_1.json")
    bridge_json = _write_bridge(tmp_path / "bridge.json", _bridge_source(report_json))

    payload = AnalystReviewInbox(tmp_path / "reports").build(
        learning_bridge_path=bridge_json,
        profile_run_path=None,
        review_actions_path=tmp_path / "missing_review.sqlite",
        save=False,
    )

    assert payload["summary"]["status"] == "ready_for_manual_review"
    assert payload["summary"]["ready_for_manual_review_count"] == 1
    item = payload["groups"]["ready_for_manual_review"][0]
    assert item["source_id"] == "run_1"
    assert "mark-reviewed" in item["suggested_commands"]["mark_reviewed_preview"]
    assert payload["summary"]["review_action_write_performed"] is False


def test_review_inbox_groups_extra_blockers_as_needs_more_data(tmp_path):
    report_json = _write_report(tmp_path / "agent_lab" / "run_1.json")
    bridge_json = _write_bridge(
        tmp_path / "bridge.json",
        _bridge_source(report_json, blockers=["source_agent_lab_report_not_marked_reviewed", "weak_note_data_quality"]),
    )

    payload = AnalystReviewInbox(tmp_path / "reports").build(
        learning_bridge_path=bridge_json,
        profile_run_path=None,
        review_actions_path=tmp_path / "missing_review.sqlite",
        save=False,
    )

    assert payload["summary"]["status"] == "needs_more_data"
    assert payload["summary"]["needs_more_data_candidate_count"] == 1
    assert payload["groups"]["needs_more_data_candidate"][0]["group_reason"] == "candidate_blockers_beyond_review_gate"


def test_review_inbox_does_not_requeue_already_reviewed_source(tmp_path):
    report_json = _write_report(tmp_path / "agent_lab" / "run_1.json")
    bridge_json = _write_bridge(tmp_path / "bridge.json", _bridge_source(report_json))
    review_store = _write_review_action_store(tmp_path / "review_actions.sqlite", "run_1")

    payload = AnalystReviewInbox(tmp_path / "reports").build(
        learning_bridge_path=bridge_json,
        profile_run_path=None,
        review_actions_path=review_store,
        save=False,
    )

    assert payload["summary"]["status"] == "no_reviewable_sources"
    assert payload["summary"]["ready_for_manual_review_count"] == 0
    assert payload["groups"]["not_reviewable_yet"][0]["group_reason"] == "already_reviewed"
