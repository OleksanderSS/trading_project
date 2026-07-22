from __future__ import annotations

from dean_os.calibration_review_lifecycle import CalibrationReviewLifecycle
from dean_os.operation_queue import OperationQueue
from dean_os.schemas import PipelineActionProposal


def _add_calibration_proposal(tmp_path, status: str = "proposed") -> str:
    proposal = PipelineActionProposal(
        agent_name="calibration_proposal_agent",
        action_type="report",
        target="analyst_calibration:generalist_base_analyst",
        reason="Ready for reviewed calibration.",
        status=status,
        command_preview="Manual review only; no config write.",
        expected_effect="Create review item only.",
        risks=["Do not auto-apply config."],
    )
    OperationQueue(tmp_path / "operations.sqlite", event_log_path=None).add_proposal(proposal)
    return proposal.proposal_id


def _add_non_calibration_proposal(tmp_path) -> str:
    proposal = PipelineActionProposal(
        agent_name="other_agent",
        action_type="report",
        target="not_calibration",
        reason="Not in calibration scope.",
    )
    OperationQueue(tmp_path / "operations.sqlite", event_log_path=None).add_proposal(proposal)
    return proposal.proposal_id


def test_calibration_review_lifecycle_snapshot_and_dry_run(tmp_path):
    proposal_id = _add_calibration_proposal(tmp_path)

    payload = CalibrationReviewLifecycle(tmp_path / "reports").run(
        operations_path=tmp_path / "operations.sqlite",
        log_path=None,
        dry_run_proposals=True,
        save=False,
    )
    proposal = OperationQueue(tmp_path / "operations.sqlite", event_log_path=None).get_proposal(proposal_id)

    assert payload["lifecycle_gate"]["status"] == "dry_run_reviewed"
    assert payload["lifecycle_gate"]["dry_run_count"] == 1
    assert payload["dry_run_previews"][0]["config_write_performed"] is False
    assert payload["dry_run_previews"][0]["ready_for_manual_execution"] is False
    assert proposal.status == "proposed"


def test_calibration_review_lifecycle_approve_waits_for_manual_implementation(tmp_path):
    proposal_id = _add_calibration_proposal(tmp_path)

    payload = CalibrationReviewLifecycle(tmp_path / "reports").run(
        operations_path=tmp_path / "operations.sqlite",
        log_path=None,
        approve_ids=[proposal_id],
        dry_run_proposals=True,
        save=False,
    )
    proposal = OperationQueue(tmp_path / "operations.sqlite", event_log_path=None).get_proposal(proposal_id)

    assert payload["lifecycle_gate"]["status"] == "approved_waiting_manual_implementation"
    assert payload["lifecycle_gate"]["approved_waiting_manual_implementation_count"] == 1
    assert payload["approved_waiting_manual_implementation"][0]["proposal_id"] == proposal_id
    assert payload["dry_run_previews"][0]["manual_implementation_required"] is True
    assert proposal.status == "approved"


def test_calibration_review_lifecycle_rejects_proposal(tmp_path):
    proposal_id = _add_calibration_proposal(tmp_path)

    payload = CalibrationReviewLifecycle(tmp_path / "reports").run(
        operations_path=tmp_path / "operations.sqlite",
        log_path=None,
        reject_ids=[proposal_id],
        save=False,
    )
    proposal = OperationQueue(tmp_path / "operations.sqlite", event_log_path=None).get_proposal(proposal_id)

    assert payload["lifecycle_gate"]["status"] == "actions_applied"
    assert payload["final_status_counts"] == {"rejected": 1}
    assert proposal.status == "rejected"


def test_calibration_review_lifecycle_skips_non_calibration_action(tmp_path):
    proposal_id = _add_non_calibration_proposal(tmp_path)

    payload = CalibrationReviewLifecycle(tmp_path / "reports").run(
        operations_path=tmp_path / "operations.sqlite",
        log_path=None,
        approve_ids=[proposal_id],
        save=False,
    )
    proposal = OperationQueue(tmp_path / "operations.sqlite", event_log_path=None).get_proposal(proposal_id)

    assert payload["lifecycle_gate"]["status"] == "no_calibration_proposals"
    assert payload["action_results"][0]["status"] == "skipped_non_calibration"
    assert proposal.status == "proposed"
