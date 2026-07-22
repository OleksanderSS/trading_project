from __future__ import annotations

from dean_os.manual_implementation_backlog import ManualImplementationBacklog
from dean_os.operation_queue import OperationQueue
from dean_os.schemas import EvidenceItem, PipelineActionProposal


def _add_calibration_proposal(tmp_path, status: str = "approved") -> str:
    proposal = PipelineActionProposal(
        agent_name="calibration_proposal_agent",
        action_type="report",
        target="analyst_calibration:generalist_base_analyst",
        reason="Gate approved reviewed calibration.",
        status=status,
        command_preview="Manual review only; no config write.",
        expected_effect="Create implementation task only.",
        risks=["Manual config change still needs review."],
        evidence=[
            EvidenceItem(
                source_type="metric",
                source="profile:generalist_base_analyst",
                key="suggested_weight_delta",
                value=0.05,
            )
        ],
    )
    OperationQueue(tmp_path / "operations.sqlite", event_log_path=None).add_proposal(proposal)
    return proposal.proposal_id


def _add_non_calibration_proposal(tmp_path, status: str = "approved") -> str:
    proposal = PipelineActionProposal(
        agent_name="other_agent",
        action_type="report",
        target="other_target",
        reason="Not a calibration proposal.",
        status=status,
    )
    OperationQueue(tmp_path / "operations.sqlite", event_log_path=None).add_proposal(proposal)
    return proposal.proposal_id


def test_manual_backlog_reports_approved_calibration_task(tmp_path):
    proposal_id = _add_calibration_proposal(tmp_path, status="approved")

    payload = ManualImplementationBacklog(tmp_path / "reports").build(
        operations_path=tmp_path / "operations.sqlite",
        save=False,
    )

    assert payload["backlog_gate"]["status"] == "manual_implementation_required"
    assert payload["backlog_gate"]["approved_task_count"] == 1
    assert payload["backlog_gate"]["config_write_performed"] is False
    task = payload["tasks"][0]
    assert task["proposal_id"] == proposal_id
    assert task["manual_status"] == "waiting_manual_implementation"
    assert task["profile"] == "generalist_base_analyst"
    assert task["suggested_weight_delta"] == 0.05
    assert task["requires_separate_pr"] is True
    assert any("separate branch or PR" in item for item in task["implementation_checklist"])


def test_manual_backlog_hides_proposed_by_default(tmp_path):
    _add_calibration_proposal(tmp_path, status="proposed")

    payload = ManualImplementationBacklog(tmp_path / "reports").build(
        operations_path=tmp_path / "operations.sqlite",
        save=False,
    )

    assert payload["backlog_gate"]["status"] == "no_manual_tasks_in_scope"
    assert payload["tasks"] == []


def test_manual_backlog_can_include_proposed_for_visibility(tmp_path):
    proposal_id = _add_calibration_proposal(tmp_path, status="proposed")

    payload = ManualImplementationBacklog(tmp_path / "reports").build(
        operations_path=tmp_path / "operations.sqlite",
        include_proposed=True,
        save=False,
    )

    assert payload["backlog_gate"]["status"] == "review_items_visible"
    assert payload["tasks"][0]["proposal_id"] == proposal_id
    assert payload["tasks"][0]["manual_status"] == "waiting_review"
    assert payload["tasks"][0]["requires_separate_pr"] is False


def test_manual_backlog_ignores_non_calibration_by_default(tmp_path):
    _add_non_calibration_proposal(tmp_path, status="approved")

    payload = ManualImplementationBacklog(tmp_path / "reports").build(
        operations_path=tmp_path / "operations.sqlite",
        save=False,
    )

    assert payload["backlog_gate"]["status"] == "no_manual_tasks_in_scope"
    assert payload["tasks"] == []
