"""
dean_os/risk/risk_engine.py

Движок управління ризиком для DEAN-OS.
Відповідає шаблону RISK_LIMITS_KILL_SWITCH_TEMPLATE з Codex Phase 8.

Правило: порушення ліміту ризику → блокування за замовчуванням.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class KillSwitchTrigger(str, Enum):
    DAILY_LOSS_LIMIT_BREACHED = "daily_loss_limit_breached"
    DRAWDOWN_LIMIT_BREACHED = "drawdown_limit_breached"
    STALE_MARKET_DATA = "stale_market_data"
    BROKER_ERROR = "broker_error"
    REPEATED_ORDER_REJECTION = "repeated_order_rejection"
    MISSING_DECISION_LINEAGE = "missing_decision_lineage"
    MODEL_STATE_UNKNOWN = "model_state_unknown"
    RISK_ENGINE_FAILURE = "risk_engine_failure"
    UNAUTHORIZED_ASSET = "unauthorized_asset"
    OPERATOR_MANUAL_KILL = "operator_manual_kill"


@dataclass
class RiskLimits:
    """Конфігурація лімітів ризику для однієї стратегії або загальна."""
    max_position_size_pct: float = 0.05          # 5% від портфеля
    max_portfolio_exposure_pct: float = 0.30      # 30% всього портфеля
    max_daily_loss_pct: float = 0.02              # 2% на день
    max_drawdown_pct: float = 0.08                # 8% max drawdown
    max_order_frequency_per_hour: int = 10
    max_slippage_bps: float = 50.0                # 50 базисних пунктів
    liquidity_minimum_usd: float = 500_000.0
    volatility_block_percentile: float = 0.95     # блокування якщо vol > 95-й перцентиль


@dataclass
class PortfolioState:
    """Поточний стан портфеля для перевірки лімітів."""
    daily_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    open_position_exposure_pct: float = 0.0
    orders_last_hour: int = 0
    current_volatility_percentile: float = 0.50
    market_data_age_seconds: float = 0.0
    model_state_known: bool = True
    is_authorized_asset: bool = True
    consecutive_rejections: int = 0
    estimated_slippage_bps: float = 0.0
    asset_liquidity_usd: float = 1_000_000.0


class RiskCheckResult:
    def __init__(self, passed: bool, violations: list[str] | None = None):
        self.passed = passed
        self.violations: list[str] = violations or []

    def as_dict(self) -> dict:
        return {"passed": self.passed, "violations": self.violations}


class KillSwitchState:
    """Стан kill switch системи."""

    def __init__(self):
        self._active: bool = False
        self._triggers: list[str] = []
        self._incident_log: list[str] = []

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self, trigger: KillSwitchTrigger, detail: str = "") -> None:
        """Активує kill switch з вказаним тригером."""
        self._active = True
        entry = f"{trigger.value}: {detail}" if detail else trigger.value
        if entry not in self._triggers:
            self._triggers.append(entry)
        self._incident_log.append(f"KILL SWITCH ACTIVATED — {entry}")

    def deactivate_manual(self, operator: str) -> None:
        """Лише оператор може вимкнути kill switch."""
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError("A non-empty operator identity is required.")
        self._active = False
        self._incident_log.append(
            f"KILL SWITCH DEACTIVATED by operator: {operator.strip()}"
        )

    def as_dict(self) -> dict:
        return {
            "is_active": self._active,
            "triggers": self._triggers,
            "incident_log": self._incident_log,
        }


class RiskEngine:
    """
    Движок ризику системи DEAN-OS.

    Перевіряє стан портфеля перед кожною операцією.
    Активує kill switch при порушенні лімітів.
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.kill_switch = KillSwitchState()

    def check(self, state: PortfolioState, strategy_id: str = "unknown") -> RiskCheckResult:
        """
        Виконує повну перевірку ризику для поточного стану портфеля.
        Повертає RiskCheckResult з переліком порушень.
        """
        violations: list[str] = []

        # Kill switch — негайний блок
        if self.kill_switch.is_active:
            violations.append("kill_switch_active")
            return RiskCheckResult(passed=False, violations=violations)

        # Ліміти
        if state.daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            violations.append(f"daily_loss_limit_breached: {state.daily_pnl_pct:.2%}")
            self.kill_switch.activate(
                KillSwitchTrigger.DAILY_LOSS_LIMIT_BREACHED,
                f"strategy={strategy_id}"
            )

        if state.drawdown_pct >= self.limits.max_drawdown_pct:
            violations.append(f"drawdown_limit_breached: {state.drawdown_pct:.2%}")
            self.kill_switch.activate(
                KillSwitchTrigger.DRAWDOWN_LIMIT_BREACHED,
                f"strategy={strategy_id}"
            )

        if state.open_position_exposure_pct > self.limits.max_portfolio_exposure_pct:
            violations.append(f"portfolio_exposure_breached: {state.open_position_exposure_pct:.2%}")

        if state.orders_last_hour > self.limits.max_order_frequency_per_hour:
            violations.append(f"order_frequency_breached: {state.orders_last_hour}/hr")

        if state.current_volatility_percentile >= self.limits.volatility_block_percentile:
            violations.append(f"volatility_block: vol_pct={state.current_volatility_percentile:.2f}")

        if state.estimated_slippage_bps > self.limits.max_slippage_bps:
            violations.append(
                f"slippage_limit_breached: {state.estimated_slippage_bps:.1f}bps"
            )

        if state.asset_liquidity_usd < self.limits.liquidity_minimum_usd:
            violations.append(
                f"liquidity_minimum_breached: {state.asset_liquidity_usd:.2f}usd"
            )

        if state.market_data_age_seconds > 60:
            violations.append(f"stale_market_data: age={state.market_data_age_seconds:.0f}s")
            self.kill_switch.activate(
                KillSwitchTrigger.STALE_MARKET_DATA,
                f"age={state.market_data_age_seconds:.0f}s"
            )

        if not state.model_state_known:
            violations.append("model_state_unknown")
            self.kill_switch.activate(KillSwitchTrigger.MODEL_STATE_UNKNOWN)

        if not state.is_authorized_asset:
            violations.append("unauthorized_asset")
            self.kill_switch.activate(KillSwitchTrigger.UNAUTHORIZED_ASSET)

        if state.consecutive_rejections >= 3:
            violations.append(f"repeated_order_rejection: {state.consecutive_rejections}")
            self.kill_switch.activate(
                KillSwitchTrigger.REPEATED_ORDER_REJECTION,
                f"count={state.consecutive_rejections}"
            )

        return RiskCheckResult(passed=len(violations) == 0, violations=violations)
