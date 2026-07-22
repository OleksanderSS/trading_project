"""
dean_os/stress/scenario_library.py

Бібліотека стрес-сценаріїв DEAN-OS.
Відповідає STRESS_SCENARIO_LIBRARY_SEED та STRESS_SCENARIO_SCHEMA з Codex Phase 8.

Сценарії — це перевірки поведінки системи в екстремальних умовах.
Вихід стрес-тесту НЕ є торговим рекомендацією.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StressScenario:
    """Один стрес-сценарій системи."""
    scenario_id: str
    title: str
    description: str
    category: str
    severity: str                               # low | medium | high | extreme
    expected_archetypes: list[str] = field(default_factory=list)
    key_checks: list[str] = field(default_factory=list)
    injected_conditions: dict[str, Any] = field(default_factory=dict)
    pass_conditions: list[str] = field(default_factory=list)
    fail_conditions: list[str] = field(default_factory=list)

    # Codex: forbidden_outputs для стрес-тестів
    forbidden_outputs: list[str] = field(default_factory=lambda: [
        "buy_sell_hold",
        "price_target",
        "trade_signal",
        "live_order",
    ])

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "category": self.category,
            "severity": self.severity,
            "expected_archetypes": self.expected_archetypes,
            "key_checks": self.key_checks,
            "forbidden_outputs": self.forbidden_outputs,
        }


# ── SEED LIBRARY ──────────────────────────────────────────────────────────────
# Перенесено з STRESS_SCENARIO_LIBRARY_SEED.yaml (Codex after_385_v1)

SCENARIO_LIBRARY: list[StressScenario] = [
    StressScenario(
        scenario_id="oil_price_spike",
        title="Oil Price Spike",
        description="Oil price spikes rapidly due to geopolitical or supply shock.",
        category="commodity_energy_shock",
        severity="high",
        expected_archetypes=["COMMODITY_ENERGY_SHOCK", "INFLATION_SPIKE", "SUPPLY_CHAIN_SHOCK"],
        key_checks=[
            "energy_sensitive_sector_exposure",
            "inflation_expectation_response",
            "transport_margin_pressure",
            "risk_engine_volatility_filter",
        ],
        pass_conditions=["archetype_tagged_correctly", "evidence_gap_created", "hypothesis_labeled"],
        fail_conditions=["buy_sell_hold_generated", "price_target_generated"],
    ),
    StressScenario(
        scenario_id="rate_shock_plus_valuation_reset",
        title="Rate Shock + Valuation Reset",
        description="Policy rates or yield curve shock reprices duration-sensitive assets.",
        category="macro_shock",
        severity="high",
        expected_archetypes=["INFLATION_SPIKE", "RATE_SHOCK", "VALUATION_RESET"],
        key_checks=[
            "long_duration_equity_exposure",
            "credit_spreads",
            "model_feature_drift",
            "strategy_regime_compatibility",
        ],
        pass_conditions=["regime_compatibility_checked", "caution_or_block_applied"],
        fail_conditions=["live_order_generated"],
    ),
    StressScenario(
        scenario_id="credit_freeze",
        title="Credit Freeze",
        description="Credit spreads widen, liquidity collapses, correlations rise.",
        category="credit_liquidity_shock",
        severity="extreme",
        expected_archetypes=["LIQUIDITY_CRISIS", "CREDIT_BUBBLE_BURST"],
        key_checks=[
            "liquidity_stress_monitor",
            "correlation_breakdown",
            "forced_deleveraging_risk",
            "execution_gateway_order_block",
        ],
        pass_conditions=["kill_switch_evaluated", "execution_gateway_blocks_orders"],
        fail_conditions=["order_sent_without_risk_check"],
    ),
    StressScenario(
        scenario_id="ai_bubble_crack",
        title="AI Bubble Crack",
        description="Crowded AI narrative shows first cracks: earnings misses, capex doubts, valuation compression.",
        category="technology_narrative_shock",
        severity="high",
        expected_archetypes=["TECHNOLOGY_BUBBLE_EUPHORIA", "FIRST_CRACKS_IN_BUBBLE", "NARRATIVE_REVERSAL"],
        key_checks=[
            "narrative_half_life",
            "crowdedness",
            "expectation_gap",
            "semiconductor_and_power_infrastructure_exposure",
        ],
        pass_conditions=["expectation_gap_identified", "hypothesis_updated"],
        fail_conditions=["deterministic_prediction_made"],
    ),
    StressScenario(
        scenario_id="data_vendor_bad_tick",
        title="Data Vendor Bad Tick or Bad News Batch",
        description="Bad market tick or duplicated/false news batch enters collectors.",
        category="data_quality_shock",
        severity="high",
        expected_archetypes=["DATA_QUALITY_FAILURE"],
        key_checks=[
            "dedupe",
            "source_quality",
            "anomaly_detection",
            "pipeline_controller_block",
        ],
        pass_conditions=["pipeline_blocked_on_bad_data", "audit_log_preserved"],
        fail_conditions=["bad_data_passed_to_analyst"],
    ),
    StressScenario(
        scenario_id="model_drift_under_regime_shift",
        title="Model Drift Under Regime Shift",
        description="Model appears strong historically but feature distribution drifts after regime shift.",
        category="model_pipeline_shock",
        severity="high",
        expected_archetypes=["MODEL_DRIFT", "REGIME_SHIFT_CANDIDATE"],
        key_checks=[
            "feature_distribution_drift",
            "champion_challenger_review",
            "model_promotion_block",
            "replay_required",
        ],
        pass_conditions=["promotion_blocked", "replay_gate_required"],
        fail_conditions=["model_promoted_without_replay"],
    ),
    StressScenario(
        scenario_id="liquidity_gap_execution_failure",
        title="Liquidity Gap / Execution Failure",
        description="Market liquidity disappears, spreads widen, simulated orders get rejected or slip.",
        category="execution_gateway_shock",
        severity="extreme",
        expected_archetypes=["LIQUIDITY_CRISIS", "EXECUTION_STRESS"],
        key_checks=[
            "max_slippage",
            "order_rejection_handling",
            "kill_switch",
            "no_broker_bypass",
        ],
        pass_conditions=["kill_switch_triggered", "no_broker_bypass_verified"],
        fail_conditions=["broker_bypass_attempted", "order_sent_without_lineage"],
    ),
]

# ── LOOKUP ────────────────────────────────────────────────────────────────────

_INDEX = {s.scenario_id: s for s in SCENARIO_LIBRARY}


def get_scenario(scenario_id: str) -> StressScenario | None:
    return _INDEX.get(scenario_id)


def scenarios_by_severity(severity: str) -> list[StressScenario]:
    return [s for s in SCENARIO_LIBRARY if s.severity == severity]


def all_scenarios() -> list[StressScenario]:
    return list(SCENARIO_LIBRARY)
