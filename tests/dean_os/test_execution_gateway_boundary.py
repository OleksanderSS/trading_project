"""Tests for ExecutionGateway authority boundaries (fail-closed logic).

Covers the explicit status codes introduced in dean_os_system_notes.md:
- blocked
- paper_trade_preview
- paper_trade_logged
- queued_for_review
- blocked_no_adapter
- executed
"""
from __future__ import annotations

from dean_os.execution_gateway import ExecutionGateway, ExecutionPolicy
from dean_os.schemas import ApprovalReceipt, ConsensusDecision


def _decision(decision: str = "candidate_long", requires_human_approval: bool = True) -> ConsensusDecision:
    return ConsensusDecision(
        decision_id="dec_1",
        decision=decision,  # type: ignore[arg-type]
        requires_human_approval=requires_human_approval,
        final_score=0.5,
        confidence=0.6,
    )


def test_blocked_decision_never_executes():
    gw = ExecutionGateway(ExecutionPolicy(live_execution_enabled=True, execution_adapter_available=True))
    outcome = gw.process(_decision(decision="blocked", requires_human_approval=False))
    assert outcome.status == "blocked"


def test_human_approval_required_queues_for_review():
    gw = ExecutionGateway(ExecutionPolicy(paper_trading_enabled=True))
    outcome = gw.process(_decision(requires_human_approval=True))
    assert outcome.status == "queued_for_review"


def test_paper_preview_when_approval_not_required():
    gw = ExecutionGateway(ExecutionPolicy(require_human_approval=False, paper_trading_enabled=True))
    outcome = gw.process(_decision(requires_human_approval=False, decision="candidate_long"))
    assert outcome.status == "paper_trade_preview"


def test_live_without_adapter_is_blocked():
    gw = ExecutionGateway(
        ExecutionPolicy(
            require_human_approval=False,
            paper_trading_enabled=False,
            live_execution_enabled=True,
            execution_adapter_available=False,
        )
    )
    outcome = gw.process(_decision(requires_human_approval=False, decision="candidate_long"))
    assert outcome.status == "blocked_no_adapter"


def test_live_requires_receipt():
    gw = ExecutionGateway(
        ExecutionPolicy(
            require_human_approval=False,
            paper_trading_enabled=False,
            live_execution_enabled=True,
            execution_adapter_available=True,
        )
    )
    decision = _decision(requires_human_approval=False, decision="candidate_long")
    # process() must NOT return executed without explicit receipt
    outcome = gw.process(decision)
    assert outcome.status == "blocked_no_adapter"
    # execute_with_receipt requires an approved receipt
    receipt = ApprovalReceipt(
        transition_type="operation",
        source_id="dec_1",
        source_type="operation_proposal",
        reviewer="human",
        reason="approved for live test",
        approved=True,
    )
    executed = gw.execute_with_receipt(decision, receipt)
    assert executed.status == "executed"
    assert executed.details.get("receipt_id") == receipt.receipt_id


def test_execute_with_rejected_receipt_queues():
    gw = ExecutionGateway(ExecutionPolicy(live_execution_enabled=True, execution_adapter_available=True))
    decision = _decision(requires_human_approval=False, decision="candidate_long")
    receipt = ApprovalReceipt(
        transition_type="operation",
        source_id="dec_1",
        source_type="operation_proposal",
        reviewer="human",
        reason="rejected",
        approved=False,
    )
    outcome = gw.execute_with_receipt(decision, receipt)
    assert outcome.status == "queued_for_review"