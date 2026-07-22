from __future__ import annotations

import json

from dean_os.review_action_dry_run import ReviewActionDryRun


def _write_packet(path, *, status="reviewable", check_statuses=None):
    check_statuses = check_statuses or [("pass", "inbox_ready")]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "packet_1",
        "mode": "review_decision_packet",
        "summary": {
            "source_id": "source_1",
            "profile": "generalist_base_analyst",
            "packet_status": status,
            "recommended_review_action": "operator_decides" if status == "manual_review_with_warnings" else "mark_reviewed_candidate",
        },
        "source": {
            "source_type": "agent_lab_report",
            "source_id": "source_1",
            "profile": "generalist_base_analyst",
            "suggested_commands": {
                "mark_reviewed_preview": (
                    "python run_agent_review_approved_learning.py "
                    "--profile-run-json reports/profiles/latest.json "
                    "--learning-store reports/learning.sqlite "
                    "--review-actions-store reports/review_actions.sqlite "
                    "--operations-store reports/operation_queue.sqlite "
                    '--mark-reviewed --review-notes "Reviewed"'
                ),
                "needs_more_data_preview": (
                    "python run_agent_review_approved_learning.py "
                    "--profile-run-json reports/profiles/latest.json "
                    "--learning-store reports/learning.sqlite "
                    "--review-actions-store reports/review_actions.sqlite "
                    "--operations-store reports/operation_queue.sqlite "
                    '--needs-more-data "More data" --review-notes "Thin"'
                ),
            },
        },
        "review_checks": [{"status": item[0], "code": item[1], "message": item[1]} for item in check_statuses],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_mark_reviewed_allowed_for_clean_reviewable_packet(tmp_path):
    packet = _write_packet(tmp_path / "packet.json")

    payload = ReviewActionDryRun(tmp_path / "reports").build(
        packet_path=packet,
        intent="mark_reviewed",
        review_notes="Reviewed citations.",
        save=False,
    )

    assert payload["summary"]["dry_run_status"] == "allowed"
    assert payload["summary"]["can_record_review_action"] is True
    assert payload["would_record_review_action"]["action_type"] == "mark_reviewed"
    assert "--apply" in payload["commands"]["bridge_apply_after_review_only_if_dry_run_passes"]
    assert payload["summary"]["review_action_write_performed"] is False


def test_mark_reviewed_blocked_for_warning_packet_without_ack(tmp_path):
    packet = _write_packet(
        tmp_path / "packet.json",
        status="manual_review_with_warnings",
        check_statuses=[("pass", "inbox_ready"), ("warn", "missing_requested_tickers")],
    )

    payload = ReviewActionDryRun(tmp_path / "reports").build(
        packet_path=packet,
        intent="mark_reviewed",
        review_notes="Reviewed citations.",
        save=False,
    )

    assert payload["summary"]["dry_run_status"] == "blocked_warning_ack_required"
    assert payload["summary"]["can_record_review_action"] is False


def test_mark_reviewed_allowed_for_warning_packet_with_ack(tmp_path):
    packet = _write_packet(
        tmp_path / "packet.json",
        status="manual_review_with_warnings",
        check_statuses=[("pass", "inbox_ready"), ("warn", "missing_requested_tickers")],
    )

    payload = ReviewActionDryRun(tmp_path / "reports").build(
        packet_path=packet,
        intent="mark_reviewed",
        review_notes="Warnings accepted for diagnostic learning.",
        acknowledge_warnings=True,
        save=False,
    )

    assert payload["summary"]["dry_run_status"] == "allowed_with_warning_ack"
    assert payload["summary"]["can_record_review_action"] is True


def test_needs_more_data_allowed_for_warning_packet(tmp_path):
    packet = _write_packet(
        tmp_path / "packet.json",
        status="manual_review_with_warnings",
        check_statuses=[("pass", "inbox_ready"), ("warn", "missing_requested_tickers")],
    )

    payload = ReviewActionDryRun(tmp_path / "reports").build(
        packet_path=packet,
        intent="needs_more_data",
        review_notes="Coverage is too thin.",
        data_request="Add AMD and NVDA coverage.",
        save=False,
    )

    assert payload["summary"]["dry_run_status"] == "allowed"
    assert payload["would_record_review_action"]["action_type"] == "needs_more_data"
    assert payload["would_record_review_action"]["payload"]["data_request"] == "Add AMD and NVDA coverage."
