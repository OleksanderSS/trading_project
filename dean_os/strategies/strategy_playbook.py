"""
dean_os/strategies/strategy_playbook.py

Схема стратегічного плейбуку. Відповідає STRATEGY_PLAYBOOK_SCHEMA з Codex Phase 7.

Кожна стратегія в системі DEAN-OS описується цим плейбуком.
Вихід стратегії — НЕ є авторитетом виконання.
Авторитет виконання належить лише execution gateway після проходження gates.
"""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StrategyStatus(str, Enum):
    DRAFT = "draft"
    RESEARCH = "research"
    REPLAY = "replay"
    PAPER = "paper"
    SHADOW = "shadow"
    SUPERVISED_LIVE = "supervised_live"
    CONSTRAINED_AUTONOMOUS_CANDIDATE = "constrained_autonomous_candidate"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class MaturityLevel(str, Enum):
    RESEARCH = "research"
    REPLAY = "replay"
    PAPER = "paper"
    SHADOW = "shadow"
    SUPERVISED_LIVE = "supervised_live"
    CONSTRAINED_AUTONOMOUS = "constrained_autonomous"


class RegimeShiftBehavior(str, Enum):
    REDUCE = "reduce"
    PAUSE = "pause"
    BLOCK = "block"
    REQUIRE_REVIEW = "require_review"


class StrategyDescription(BaseModel):
    name: str
    thesis: str
    strategy_family: list[str] = Field(default_factory=list)
    time_horizon: str = "medium_term"
    asset_universe: list[str] = Field(default_factory=list)
    prohibited_assets: list[str] = Field(default_factory=list)


class RegimeCompatibility(BaseModel):
    allowed_regimes: list[str] = Field(default_factory=list)
    forbidden_regimes: list[str] = Field(default_factory=list)
    caution_regimes: list[str] = Field(default_factory=list)
    required_macro_context_fields: list[str] = Field(default_factory=list)
    regime_shift_behavior: RegimeShiftBehavior = RegimeShiftBehavior.REQUIRE_REVIEW


class InputRequirements(BaseModel):
    required_sources: list[str] = Field(default_factory=list)
    required_features: list[str] = Field(default_factory=list)
    required_model_states: list[str] = Field(default_factory=list)
    required_hypothesis_inputs: list[str] = Field(default_factory=list)
    minimum_data_freshness: str = "24h"
    # LLM може постачати ЛИШЕ ці типи входів для стратегії
    allowed_llm_inputs: list[str] = Field(default_factory=lambda: [
        "risk_archetype_tag",
        "evidence_gap",
        "hypothesis_context",
        "review_label",
    ])
    # Ці типи входів від LLM — ЗАБОРОНЕНІ
    forbidden_llm_inputs: list[str] = Field(default_factory=lambda: [
        "direct_order",
        "final_probability_without_base_rate",
        "buy_sell_hold",
        "price_target",
    ])


class RiskPolicy(BaseModel):
    capital_bucket_id: str = "undefined"
    max_position_size: str = "configured_externally"
    max_daily_loss: str = "configured_externally"
    max_drawdown: str = "configured_externally"
    liquidity_filter_required: bool = True
    volatility_filter_required: bool = True


class EvaluationRequirements(BaseModel):
    replay_required: bool = True
    paper_required_before_live: bool = True
    shadow_required_before_supervised_live: bool = True
    out_of_sample_required: bool = True
    walk_forward_required: bool = True
    leakage_checks_required: bool = True
    transaction_costs_required: bool = True
    slippage_model_required: bool = True
    decision_lineage_required: bool = True


class PromotionPolicy(BaseModel):
    current_maturity_level: MaturityLevel = MaturityLevel.RESEARCH
    next_allowed_level: MaturityLevel | None = MaturityLevel.REPLAY
    approval_required: bool = True
    rollback_strategy_id: str | None = None


class StrategyPlaybook(BaseModel):
    """
    Повний опис однієї торгової стратегії в системі DEAN-OS.
    """
    strategy_id: str
    version: str = "1.0"
    status: StrategyStatus = StrategyStatus.DRAFT
    owner: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    description: StrategyDescription
    regime_compatibility: RegimeCompatibility = Field(default_factory=RegimeCompatibility)
    input_requirements: InputRequirements = Field(default_factory=InputRequirements)
    risk_policy: RiskPolicy = Field(default_factory=RiskPolicy)
    evaluation_requirements: EvaluationRequirements = Field(default_factory=EvaluationRequirements)
    promotion_policy: PromotionPolicy = Field(default_factory=PromotionPolicy)

    # Заборонені типи виходів — жорстке обмеження Codex
    forbidden_outputs: list[str] = Field(default_factory=lambda: [
        "LLM_direct_order",
        "broker_order_without_gateway",
        "trade_without_risk_gate",
        "buy_sell_hold_from_analyst",
        "price_target",
    ])

    def is_regime_allowed(self, current_regime: str) -> bool:
        if current_regime in self.regime_compatibility.forbidden_regimes:
            return False
        if self.regime_compatibility.allowed_regimes:
            return current_regime in self.regime_compatibility.allowed_regimes
        return True  # якщо allowed_regimes порожній — всі режими дозволені

    def can_promote_to(
        self,
        target_level: MaturityLevel,
        *,
        approval_present: bool = False,
    ) -> tuple[bool, list[str]]:
        """Перевіряє, чи може стратегія бути промоутована до наступного рівня."""
        issues = []
        if self.promotion_policy.next_allowed_level != target_level:
            issues.append(f"next_allowed_level is {self.promotion_policy.next_allowed_level}, not {target_level}")
        if self.promotion_policy.approval_required and not approval_present:
            issues.append("approval_required: human operator must approve")
        if self.promotion_policy.rollback_strategy_id is None and target_level in (
            MaturityLevel.SHADOW, MaturityLevel.SUPERVISED_LIVE, MaturityLevel.CONSTRAINED_AUTONOMOUS
        ):
            issues.append("rollback_strategy_id required for live-level promotion")
        return len(issues) == 0, issues
