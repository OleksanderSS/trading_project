from __future__ import annotations

import json

from dean_os.review_action_apply_ceremony import ReviewActionApplyCeremony
from dean_os.review_actions import ReviewActionStore


def _write_dry_run(path, *, action_type="mark_reviewed", can_record=True, status="allowed"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "dry_run_1",
        "mode": "review_action_dry_run",
        "summary": {
            "source_id": "source_1",
            "profile": "generalist_base_analyst",
            "intent": action_type,
            "dry_run_status": status,
            "can_record_review_action": can_record,
        },
        "validation": {"status": status, "can_record": can_record, "reasons": ["ok"]},
        "would_record_review_action": {
            "dry_run": True,
            "source_type": "agent_lab_report",
            "source_id": "source_1",
            "action_type": action_type,
            "reviewer": "human",
            "notes": "Reviewed.",
            "payload": {"data_request": "Add filings."} if action_type == "needs_more_data" else {},
        },
        "commands": {
            "bridge_dry_run_after_action": "python run_agent_analyst_learning_bridge.py --profile-run-json profile.json",
            "bridge_apply_after_review_only_if_dry_run_passes": (
                "python run_agent_analyst_learning_bridge.py --profile-run-json profile.json --apply"
            ),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_apply_ceremony_requires_explicit_apply_flag(tmp_path):
    dry_run = _write_dry_run(tmp_path / "dry_run.json")

    payload = ReviewActionApplyCeremony(tmp_path / "reports").apply(
        dry_run_path=dry_run,
        review_actions_path=tmp_path / "review.sqlite",
        event_log_path=None,
        save=False,
    )

    assert payload["summary"]["apply_status"] == "blocked_apply_flag_required"
    assert payload["summary"]["review_action_write_performed"] is False
    assert not (tmp_path / "review.sqlite").exists()


def test_apply_ceremony_records_one_mark_reviewed_action(tmp_path):
    dry_run = _write_dry_run(tmp_path / "dry_run.json")
    review_store = tmp_path / "review.sqlite"

    payload = ReviewActionApplyCeremony(tmp_path / "reports").apply(
        dry_run_path=dry_run,
        review_actions_path=review_store,
        event_log_path=None,
        apply_review_action=True,
        save=False,
    )
    actions = ReviewActionStore(review_store, event_log_path=None).list_actions()

    assert payload["summary"]["apply_status"] == "applied"
    assert payload["summary"]["review_action_write_performed"] is True
    assert len(actions) == 1
    assert actions[0].action_type == "mark_reviewed"
    assert actions[0].source_id == "source_1"


def test_apply_ceremony_records_needs_more_data_action(tmp_path):
    dry_run = _write_dry_run(tmp_path / "dry_run.json", action_type="needs_more_data")
    review_store = tmp_path / "review.sqlite"

    payload = ReviewActionApplyCeremony(tmp_path / "reports").apply(
        dry_run_path=dry_run,
        review_actions_path=review_store,
        event_log_path=None,
        apply_review_action=True,
        save=False,
    )
    action = ReviewActionStore(review_store, event_log_path=None).list_actions()[0]

    assert payload["summary"]["apply_status"] == "applied"
    assert action.action_type == "needs_more_data"
    assert action.payload["data_request"] == "Add filings."


def test_apply_ceremony_blocks_duplicate_active_action(tmp_path):
    dry_run = _write_dry_run(tmp_path / "dry_run.json")
    review_store = tmp_path / "review.sqlite"
    ReviewActionStore(review_store, event_log_path=None).mark_reviewed("agent_lab_report", "source_1")

    payload = ReviewActionApplyCeremony(tmp_path / "reports").apply(
        dry_run_path=dry_run,
        review_actions_path=review_store,
        event_log_path=None,
        apply_review_action=True,
        save=False,
    )
    actions = ReviewActionStore(review_store, event_log_path=None).list_actions()

    assert payload["summary"]["apply_status"] == "blocked_existing_review_action"
    assert payload["summary"]["review_action_write_performed"] is False
    assert len(actions) == 1


def test_apply_ceremony_blocks_non_recordable_dry_run(tmp_path):
    dry_run = _write_dry_run(
        tmp_path / "dry_run.json",
        can_record=False,
        status="blocked_warning_ack_required",
    )

    payload = ReviewActionApplyCeremony(tmp_path / "reports").apply(
        dry_run_path=dry_run,
        review_actions_path=tmp_path / "review.sqlite",
        event_log_path=None,
        apply_review_action=True,
        save=False,
    )

    assert payload["summary"]["apply_status"] == "blocked_dry_run_not_recordable"
    assert payload["summary"]["review_action_write_performed"] is False
