from __future__ import annotations

import json

from dean_os.calibration_proposal_agent import CalibrationProposalAgent
from dean_os.operation_queue import OperationQueue


def _write_gate(tmp_path, status: str = "ready_for_review") -> str:
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "gate_1",
                "mode": "analyst_calibration_gate",
                "summary": {
                    "profile_count": 1,
                    "ready_for_review_profiles": ["generalist_base_analyst"] if status == "ready_for_review" else [],
                    "blocked_profiles": [] if status == "ready_for_review" else ["generalist_base_analyst"],
                },
                "profiles": {
                    "generalist_base_analyst": {
                        "profile": "generalist_base_analyst",
                        "calibration_status": status,
                        "suggested_weight_delta": 0.05,
                        "scorecard": {"activation_status": "ready_to_activate", "completed_count": 3},
                        "outcomes": {
                            "record_count": 3,
                            "completed_count": 3,
                            "pending_count": 0,
                            "hit_rate": 0.67,
                            "miss_rate": 0.33,
                        },
                        "blockers": [] if status == "ready_for_review" else ["Needs more completed outcomes."],
                        "cautions": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_calibration_proposal_agent_dry_run_does_not_enqueue(tmp_path):
    gate_path = _write_gate(tmp_path)

    payload = CalibrationProposalAgent(tmp_path / "reports").run(
        calibration_gate_path=gate_path,
        operations_path=tmp_path / "operations.sqlite",
        log_path=None,
        enqueue=False,
        save=False,
    )

    proposals = OperationQueue(tmp_path / "operations.sqlite").list_proposals()
    assert payload["proposal_gate"]["status"] == "dry_run_ready"
    assert payload["proposal_gate"]["proposal_count"] == 1
    assert payload["proposals"][0]["target"] == "analyst_calibration:generalist_base_analyst"
    assert payload["proposals"][0]["requires_human_approval"] is True
    assert proposals == []


def test_calibration_proposal_agent_enqueue_writes_proposed_operation(tmp_path):
    gate_path = _write_gate(tmp_path)

    payload = CalibrationProposalAgent(tmp_path / "reports").run(
        calibration_gate_path=gate_path,
        operations_path=tmp_path / "operations.sqlite",
        log_path=None,
        enqueue=True,
        save=False,
    )
    proposals = OperationQueue(tmp_path / "operations.sqlite").list_proposals()

    assert payload["proposal_gate"]["status"] == "enqueued"
    assert payload["proposal_gate"]["enqueued_count"] == 1
    assert len(proposals) == 1
    assert proposals[0].status == "proposed"
    assert proposals[0].dry_run is True
    assert proposals[0].requires_human_approval is True
    assert proposals[0].action_type == "report"


def test_calibration_proposal_agent_skips_blocked_profiles(tmp_path):
    gate_path = _write_gate(tmp_path, status="blocked")

    payload = CalibrationProposalAgent(tmp_path / "reports").run(
        calibration_gate_path=gate_path,
        operations_path=tmp_path / "operations.sqlite",
        log_path=None,
        enqueue=True,
        save=False,
    )

    assert payload["proposal_gate"]["status"] == "no_ready_profiles"
    assert payload["proposal_gate"]["proposal_count"] == 0
    assert OperationQueue(tmp_path / "operations.sqlite").list_proposals() == []
