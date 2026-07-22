from __future__ import annotations

import json

from dean_os.review_decision_packet import ReviewDecisionPacket


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _report(path):
    return _write_json(
        path,
        {
            "run_id": "source_1",
            "document_count": 2,
            "chunk_count": 2,
            "note_count": 1,
            "reports": [
                {
                    "agent_name": "evidence_synthesis",
                    "verdict": "bullish",
                    "confidence": 0.8,
                    "data_quality_score": 0.9,
                    "reasons": ["Evidence-bound thesis."],
                }
            ],
            "research_notes": [
                {
                    "note_id": "note_1",
                    "agent_name": "evidence_synthesis",
                    "topic": "ai_cycle",
                    "thesis": "AI demand supports the company over the next year.",
                    "patterns": ["ai_compute_cycle"],
                    "tickers": ["AMD"],
                    "sectors": ["semiconductor"],
                    "horizon_days": 365,
                    "confidence": 0.8,
                    "data_quality": "strong",
                    "citations": [
                        {
                            "source_id": "doc_1",
                            "source_type": "report",
                            "title": "Semiconductor report",
                            "uri": "docs/report.txt",
                            "excerpt": "AI accelerator demand is growing.",
                        }
                    ],
                    "risks": ["Valuation can compress."],
                    "blind_spots": ["No transcript evidence."],
                }
            ],
        },
    )


def _evidence(path, *, data_quality="clean", missing=None):
    return _write_json(
        path,
        {
            "run_id": "evidence_1",
            "coverage": {
                "document_count": 2,
                "data_quality": data_quality,
                "agent_lab_ready": True,
                "by_source_type": {"report": 2},
                "tickers": ["AMD"],
                "missing_requested_tickers": missing or [],
                "date_range": {"start": "2026-01-01", "end": "2026-02-01"},
                "warning_count": 0,
                "dropped_count": 0,
            },
        },
    )


def _inbox(path, report_json, evidence_pack_path, *, group="ready_for_manual_review"):
    reason = "requires_human_citation_and_thesis_review" if group == "ready_for_manual_review" else "candidate_blockers_beyond_review_gate"
    return _write_json(
        path,
        {
            "run_id": "inbox_1",
            "mode": "analyst_review_inbox",
            "groups": {
                "ready_for_manual_review": [],
                "needs_more_data_candidate": [],
                "not_reviewable_yet": [],
            },
            "items": [
                {
                    "group": group,
                    "group_reason": reason,
                    "source_type": "agent_lab_report",
                    "source_id": "source_1",
                    "profile": "generalist_base_analyst",
                    "profile_run_id": "profile_1",
                    "evidence_pack_run_id": "evidence_1",
                    "evidence_pack_path": evidence_pack_path,
                    "report_json": report_json,
                    "review": {"reviewed": False, "needs_more_data": False},
                    "candidate_summary": {
                        "blocker_counts": {"source_agent_lab_report_not_marked_reviewed": 1},
                        "only_missing_review": True,
                    },
                    "suggested_commands": {
                        "mark_reviewed_preview": "python run_agent_review_approved_learning.py --mark-reviewed",
                        "needs_more_data_preview": "python run_agent_review_approved_learning.py --needs-more-data",
                    },
                }
            ],
        },
    )


def test_review_decision_packet_marks_clean_source_reviewable(tmp_path):
    report_json = _report(tmp_path / "report.json")
    evidence_json = _evidence(tmp_path / "evidence.json")
    inbox_json = _inbox(tmp_path / "inbox.json", report_json, evidence_json)

    payload = ReviewDecisionPacket(tmp_path / "reports").build(inbox_path=inbox_json, save=False)

    assert payload["summary"]["packet_status"] == "reviewable"
    assert payload["summary"]["recommended_review_action"] == "mark_reviewed_candidate"
    assert payload["summary"]["review_action_write_performed"] is False
    assert payload["notes"][0]["citation_count"] == 1


def test_review_decision_packet_warns_on_partial_missing_tickers(tmp_path):
    report_json = _report(tmp_path / "report.json")
    evidence_json = _evidence(tmp_path / "evidence.json", data_quality="partial", missing=["NVDA"])
    inbox_json = _inbox(tmp_path / "inbox.json", report_json, evidence_json)

    payload = ReviewDecisionPacket(tmp_path / "reports").build(inbox_path=inbox_json, save=False)

    assert payload["summary"]["packet_status"] == "manual_review_with_warnings"
    assert payload["summary"]["recommended_review_action"] == "operator_decides"
    assert any(check["code"] == "missing_requested_tickers" for check in payload["review_checks"])


def test_review_decision_packet_accepts_strong_evidence_quality(tmp_path):
    report_json = _report(tmp_path / "report.json")
    evidence_json = _evidence(tmp_path / "evidence.json", data_quality="strong")
    inbox_json = _inbox(tmp_path / "inbox.json", report_json, evidence_json)

    payload = ReviewDecisionPacket(tmp_path / "reports").build(inbox_path=inbox_json, save=False)

    assert payload["summary"]["packet_status"] == "reviewable"
    assert any(check["code"] == "evidence_quality_strong" for check in payload["review_checks"])
    assert not any(check["code"] == "evidence_quality_not_clean" for check in payload["review_checks"])


def test_review_decision_packet_recommends_needs_more_data_for_blocked_group(tmp_path):
    report_json = _report(tmp_path / "report.json")
    evidence_json = _evidence(tmp_path / "evidence.json")
    inbox_json = _inbox(tmp_path / "inbox.json", report_json, evidence_json, group="needs_more_data_candidate")

    payload = ReviewDecisionPacket(tmp_path / "reports").build(inbox_path=inbox_json, save=False)

    assert payload["summary"]["packet_status"] == "needs_more_data_recommended"
    assert payload["summary"]["recommended_review_action"] == "needs_more_data"
    assert any(check["status"] == "fail" for check in payload["review_checks"])
