"""Fail-closed simulated execution boundary for DEAN-OS.

Paper and shadow requests require an approved maturity receipt, complete
decision lineage and a fresh portfolio risk state. This component has no
broker-send path; supervised-live is disabled by architecture policy.
"""
from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from dean_os.execution.maturity_gates import verify_gate_receipt
from dean_os.risk.risk_engine import PortfolioState, RiskEngine


class OrderDecision(str, Enum):
    APPROVED_SIMULATED = "approved_simulated"
    QUEUED_FOR_REVIEW = "queued_for_review"
    REJECTED = "rejected"
    BLOCKED_HARD = "blocked_hard"


HARD_BLOCKS = frozenset({
    "llm_direct_order", "broker_bypass", "missing_lineage", "unsupported_asset",
    "kill_switch_active", "portfolio_state_missing", "maturity_receipt_missing",
    "maturity_receipt_invalid", "supervised_live_disabled", "invalid_direction",
    "invalid_position_size",
})


@dataclass
class OrderRequest:
    order_id: str = field(default_factory=lambda: f"order_{uuid.uuid4().hex[:8]}")
    strategy_id: str = "unknown"
    asset: str = "unknown"
    direction: str = "buy"
    size_pct: float = 0.01
    source: str = "model"
    decision_lineage_id: str | None = None
    mode: str = "paper"
    maturity_receipt: Mapping[str, Any] | None = None
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    @property
    def has_lineage(self) -> bool:
        return bool(self.decision_lineage_id and self.decision_lineage_id.strip())

    @property
    def is_llm_direct(self) -> bool:
        return self.source == "llm_direct"


@dataclass
class OrderResult:
    order_id: str
    decision: OrderDecision
    hard_blocks_triggered: list[str] = field(default_factory=list)
    risk_violations: list[str] = field(default_factory=list)
    notes: str = ""
    lineage_id: str | None = None
    maturity_receipt_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "decision": self.decision.value,
            "hard_blocks_triggered": self.hard_blocks_triggered,
            "risk_violations": self.risk_violations,
            "notes": self.notes,
            "lineage_id": self.lineage_id,
            "maturity_receipt_sha256": self.maturity_receipt_sha256,
        }


class ExecutionGateway:
    """The only Phase-8 order boundary; simulation only, no broker adapter."""

    LIVE_EXECUTION_ENABLED = False
    SIMULATED_MODES = {"paper", "shadow"}

    def __init__(self, risk_engine: RiskEngine | None = None, allowed_assets: set[str] | None = None):
        self.risk_engine = risk_engine or RiskEngine()
        self.allowed_assets = allowed_assets or set()
        self._order_log: list[dict[str, Any]] = []

    def submit(self, order: OrderRequest, portfolio_state: PortfolioState | None = None) -> OrderResult:
        hard_blocks: list[str] = []
        receipt_failures: list[str] = []

        if order.is_llm_direct:
            hard_blocks.append("llm_direct_order")
        if not order.has_lineage:
            hard_blocks.append("missing_lineage")
        if self.allowed_assets and order.asset not in self.allowed_assets:
            hard_blocks.append("unsupported_asset")
        if self.risk_engine.kill_switch.is_active:
            hard_blocks.append("kill_switch_active")
        if portfolio_state is None:
            hard_blocks.append("portfolio_state_missing")
        if order.direction not in {"buy", "sell", "flatten"}:
            hard_blocks.append("invalid_direction")
        if order.size_pct <= 0 or order.size_pct > self.risk_engine.limits.max_position_size_pct:
            hard_blocks.append("invalid_position_size")

        if order.mode == "supervised_live":
            hard_blocks.append("supervised_live_disabled")
        elif order.mode not in self.SIMULATED_MODES:
            hard_blocks.append("broker_bypass")

        if order.maturity_receipt is None:
            hard_blocks.append("maturity_receipt_missing")
        elif order.mode in self.SIMULATED_MODES:
            receipt_ok, receipt_failures = verify_gate_receipt(
                order.maturity_receipt,
                expected_strategy_id=order.strategy_id,
                expected_target_gate=order.mode,
            )
            if not receipt_ok:
                hard_blocks.append("maturity_receipt_invalid")

        if hard_blocks:
            result = OrderResult(
                order_id=order.order_id,
                decision=OrderDecision.BLOCKED_HARD,
                hard_blocks_triggered=list(dict.fromkeys(hard_blocks)),
                risk_violations=receipt_failures,
                notes="Fail-closed boundary rejected the request.",
                lineage_id=order.decision_lineage_id,
            )
            self._log(order, result)
            return result

        assert portfolio_state is not None
        risk_result = self.risk_engine.check(portfolio_state, strategy_id=order.strategy_id)
        receipt_sha = str(order.maturity_receipt.get("receipt_sha256"))
        if not risk_result.passed:
            result = OrderResult(
                order_id=order.order_id,
                decision=OrderDecision.REJECTED,
                risk_violations=risk_result.violations,
                notes="Risk engine check failed.",
                lineage_id=order.decision_lineage_id,
                maturity_receipt_sha256=receipt_sha,
            )
            self._log(order, result)
            return result

        result = OrderResult(
            order_id=order.order_id,
            decision=OrderDecision.APPROVED_SIMULATED,
            notes=f"Simulated approval for mode={order.mode}; broker send is disabled.",
            lineage_id=order.decision_lineage_id,
            maturity_receipt_sha256=receipt_sha,
        )
        self._log(order, result)
        return result

    def _log(self, order: OrderRequest, result: OrderResult) -> None:
        self._order_log.append({
            "logged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "order": {
                "order_id": order.order_id,
                "strategy_id": order.strategy_id,
                "asset": order.asset,
                "direction": order.direction,
                "size_pct": order.size_pct,
                "mode": order.mode,
                "decision_lineage_id": order.decision_lineage_id,
                "maturity_receipt_sha256": (
                    order.maturity_receipt.get("receipt_sha256") if order.maturity_receipt else None
                ),
            },
            "result": result.as_dict(),
        })

    def order_log(self) -> list[dict[str, Any]]:
        return list(self._order_log)
