from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from dean_os.schemas import ApprovalReceipt, ConsensusDecision, ExecutionOutcome


class ExecutionPolicy(BaseModel):
    live_execution_enabled: bool = False
    paper_trading_enabled: bool = True
    require_human_approval: bool = True
    price_model: Literal["last_close", "vwap", "mid_quote"] = "last_close"
    slippage_bps: float = 5.0
    commission_per_trade: float = 1.0
    execution_adapter_available: bool = False  # New flag to check if execution adapter is available


class ExecutionGateway:
    """Authority boundary for turning a ConsensusDecision into an execution outcome.

    Fail-closed rules (per dean_os_system_notes.md):
    - A blocked decision can never become an execution.
    - Human approval is checked BEFORE any paper/live path.
    - Paper preview (candidate) is distinct from paper_logged (actually recorded).
    - Live execution requires both an execution adapter AND an ApprovalReceipt.
    """

    def __init__(self, policy: ExecutionPolicy | None = None):
        self.policy = policy or ExecutionPolicy()

    def process(self, decision: ConsensusDecision) -> ExecutionOutcome:
        """Resolve the safe status for a decision without performing any write."""
        if decision.decision == "blocked":
            return ExecutionOutcome(
                status="blocked",
                decision_id=decision.decision_id,
                decision=decision.decision,
            )

        # Human approval is checked before any execution path.
        if self.policy.require_human_approval or decision.requires_human_approval:
            return ExecutionOutcome(
                status="queued_for_review",
                decision_id=decision.decision_id,
                decision=decision.decision,
            )

        # Paper trading path: a candidate that is not yet recorded is only a preview.
        if self.policy.paper_trading_enabled:
            if decision.trade_allowed:
                return ExecutionOutcome(
                    status="paper_trade_preview",
                    decision_id=decision.decision_id,
                    decision=decision.decision,
                    details={
                        "price_model": self.policy.price_model,
                        "slippage_bps": self.policy.slippage_bps,
                        "commission_per_trade": self.policy.commission_per_trade,
                    },
                )
            return ExecutionOutcome(
                status="queued_for_review",
                decision_id=decision.decision_id,
                decision=decision.decision,
            )

        # Live execution path - only if adapter is available.
        if self.policy.live_execution_enabled:
            if not self.policy.execution_adapter_available:
                return ExecutionOutcome(
                    status="blocked_no_adapter",
                    decision_id=decision.decision_id,
                    decision=decision.decision,
                    details={"reason": "Live execution requested but no execution adapter available"},
                )
            # Live execution must go through execute_with_receipt, not process().
            return ExecutionOutcome(
                status="blocked_no_adapter",
                decision_id=decision.decision_id,
                decision=decision.decision,
                details={"reason": "Live execution requires explicit ApprovalReceipt via execute_with_receipt()"},
            )

        # Default to queued for review if no execution path is configured.
        return ExecutionOutcome(
            status="queued_for_review",
            decision_id=decision.decision_id,
            decision=decision.decision,
        )

    def execute_with_receipt(
        self, decision: ConsensusDecision, receipt: ApprovalReceipt
    ) -> ExecutionOutcome:
        """Perform a live execution only when an execution adapter and a valid receipt exist."""
        if not self.policy.live_execution_enabled:
            return ExecutionOutcome(
                status="blocked_no_adapter",
                decision_id=decision.decision_id,
                decision=decision.decision,
                details={"reason": "Live execution is disabled by policy"},
            )
        if not self.policy.execution_adapter_available:
            return ExecutionOutcome(
                status="blocked_no_adapter",
                decision_id=decision.decision_id,
                decision=decision.decision,
                details={"reason": "Live execution requested but no execution adapter available"},
            )
        if not receipt.approved:
            return ExecutionOutcome(
                status="queued_for_review",
                decision_id=decision.decision_id,
                decision=decision.decision,
                details={"reason": "Approval receipt is not approved"},
            )
        return ExecutionOutcome(
            status="executed",
            decision_id=decision.decision_id,
            decision=decision.decision,
            details={"receipt_id": receipt.receipt_id},
        )
