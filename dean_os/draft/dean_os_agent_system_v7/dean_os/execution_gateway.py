from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from dean_os.schemas import ConsensusDecision, ExecutionOutcome


class ExecutionPolicy(BaseModel):
    live_execution_enabled: bool = False
    paper_trading_enabled: bool = True
    require_human_approval: bool = True
    price_model: Literal["last_close", "vwap", "mid_quote"] = "last_close"
    slippage_bps: float = 5.0
    commission_per_trade: float = 1.0
    execution_adapter_available: bool = False  # New flag to check if execution adapter is available


class ExecutionGateway:
    def __init__(self, policy: ExecutionPolicy | None = None):
        self.policy = policy or ExecutionPolicy()

    def process(self, decision: ConsensusDecision) -> ExecutionOutcome:
        if decision.decision == "blocked":
            return ExecutionOutcome(status="blocked", decision_id=decision.decision_id, decision=decision.decision)

        # Check if human approval is required (policy or decision level)
        if self.policy.require_human_approval or decision.requires_human_approval:
            return ExecutionOutcome(status="queued_for_review", decision_id=decision.decision_id, decision=decision.decision)

        # Paper trading path
        if self.policy.paper_trading_enabled:
            return ExecutionOutcome(
                status="paper_trade_logged",
                decision_id=decision.decision_id,
                decision=decision.decision,
                details={
                    "price_model": self.policy.price_model,
                    "slippage_bps": self.policy.slippage_bps,
                    "commission_per_trade": self.policy.commission_per_trade,
                },
            )

        # Live execution path - only if adapter is available
        if self.policy.live_execution_enabled:
            if not self.policy.execution_adapter_available:
                return ExecutionOutcome(
                    status="blocked_no_adapter",
                    decision_id=decision.decision_id,
                    decision=decision.decision,
                    details={"reason": "Live execution requested but no execution adapter available"}
                )
            return ExecutionOutcome(status="executed", decision_id=decision.decision_id, decision=decision.decision)

        # Default to queued for review if no execution path is configured
        return ExecutionOutcome(status="queued_for_review", decision_id=decision.decision_id, decision=decision.decision)
