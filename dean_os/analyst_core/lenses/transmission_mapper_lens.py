"""TransmissionMapperLens — maps events to causal transmission channels.

This lens traces how an event transmits through the economy via
first-order, second-order, and third-order effects. It produces
``transmission_channels`` entries on the packet.

Example: oil shock → gasoline cost → headline CPI → Fed repricing → growth multiples
Example: AI capex → HBM demand → foundry capacity → power demand → utility capex

Deterministic, keyword-based mapping. No LLM, no network.
"""
from __future__ import annotations

from typing import Any

from dean_os.analyst_core.lens_contract import AnalysisPacket, AnalystLens, ModuleDelta

# ──────────────────────────────────────────────────────────────────────────────
# Transmission channel definitions (from design notes §6.7)
# Each channel maps an event_class to a chain of economic effects.
# ──────────────────────────────────────────────────────────────────────────────

TRANSMISSION_CHANNELS: dict[str, dict[str, Any]] = {
    "oil_shock": {
        "channel_name": "energy_cost_transmission",
        "chain": [
            "crude_price_increase",
            "gasoline_diesel_cost_rise",
            "transportation_cost_increase",
            "headline_cpi_pressure",
            "fed_rate_expectation_reprice",
            "growth_multiple_pressure",
        ],
        "affected_sectors": ["energy", "transportation", "consumer", "industrial"],
        "intermediate_variables": [
            "crude_spot_price", "gasoline_retail_price", "cpi_energy_component",
            "fed_funds_futures", "treasury_yield",
        ],
        "time_horizon": "1_3_months",
        "counterforces": [
            "strategic_petroleum_release",
            "opec_production_increase",
            "demand_destruction_from_recession",
            "renewable_substitution",
        ],
    },
    "commodity_supply_shock": {
        "channel_name": "commodity_cost_transmission",
        "chain": [
            "commodity_supply_reduction",
            "input_cost_increase",
            "producer_margin_pressure",
            "consumer_price_pass_through",
            "demand_destruction_or_substitution",
        ],
        "affected_sectors": ["industrial", "consumer", "materials"],
        "intermediate_variables": [
            "commodity_spot_price", "futures_curve", "inventory_levels",
            "producer_price_index",
        ],
        "time_horizon": "1_6_months",
        "counterforces": [
            "inventory_release", "substitute_materials",
            "demand_elasticity_response",
        ],
    },
    "ai_capex_announcement": {
        "channel_name": "ai_investment_transmission",
        "chain": [
            "hyperscaler_capex_commitment",
            "gpu_accelerator_order_pipeline",
            "foundry_capacity_utilization",
            "advanced_packaging_demand",
            "memory_hbm_demand",
            "power_infrastructure_requirement",
        ],
        "affected_sectors": ["semiconductor_ai_infrastructure", "energy", "industrial"],
        "intermediate_variables": [
            "gpu_lead_time", "hbm_contract_price", "foundry_utilization_rate",
            "data_center_power_demand",
        ],
        "time_horizon": "6_18_months",
        "counterforces": [
            "capex_guidance_cut", "customer_concentration_risk",
            "ai_roi_disappointment", "regulatory_constraint",
        ],
    },
    "tariff": {
        "channel_name": "trade_policy_transmission",
        "chain": [
            "tariff_implementation",
            "import_cost_increase",
            "supply_chain_reconfiguration",
            "margin_pressure_on_importers",
            "potential_retail_price_increase",
            "demand_shift_to_domestic_alternatives",
        ],
        "affected_sectors": ["semiconductor_ai_infrastructure", "consumer", "industrial"],
        "intermediate_variables": [
            "tariff_rate", "import_volume", "supply_chain_cost",
            "domestic_production_capacity",
        ],
        "time_horizon": "3_12_months",
        "counterforces": [
            "exemption_negotiation", "supply_chain_diversification",
            "currency_offset", "demand_absorption",
        ],
    },
    "sanctions_change": {
        "channel_name": "geopolitical_restriction_transmission",
        "chain": [
            "sanction_implementation",
            "market_access_restriction",
            "supply_chain_disruption",
            "revenue_impact_on_targeted_entities",
            "counter_sanction_risk",
            "allied_supply_chain_reconfiguration",
        ],
        "affected_sectors": ["semiconductor_ai_infrastructure", "energy", "industrial"],
        "intermediate_variables": [
            "sanction_scope", "enforcement_capability",
            "alternative_supply_availability",
        ],
        "time_horizon": "3_24_months",
        "counterforces": [
            "sanction_evasion", "allied_exemptions",
            "domestic_substitution_acceleration",
        ],
    },
    "central_bank_decision": {
        "channel_name": "monetary_policy_transmission",
        "chain": [
            "rate_decision",
            "bond_yield_reprice",
            "credit_spread_adjustment",
            "equity_valuation_multiples",
            "investment_and_consumption_timing",
        ],
        "affected_sectors": ["financials", "consumer", "industrial", "technology"],
        "intermediate_variables": [
            "policy_rate", "yield_curve_slope", "credit_spreads",
            "dollar_index",
        ],
        "time_horizon": "1_6_months",
        "counterforces": [
            "forward_guidance_offset", "market_priced_expectation",
            "fiscal_policy_counter",
        ],
    },
    "inflation_release": {
        "channel_name": "inflation_surprise_transmission",
        "chain": [
            "inflation_data_release",
            "rate_expectation_reprice",
            "real_yield_adjustment",
            "equity_risk_premium_reprice",
            "sector_rotation_pressure",
        ],
        "affected_sectors": ["financials", "consumer", "industrial"],
        "intermediate_variables": [
            "core_cpi", "core_pce", "breakeven_inflation",
            "real_yield",
        ],
        "time_horizon": "1_3_months",
        "counterforces": [
            "transitory_interpretation", "base_effect_normalization",
            "fed_dovish_interpretation",
        ],
    },
    "war_escalation": {
        "channel_name": "geopolitical_shock_transmission",
        "chain": [
            "conflict_escalation",
            "safe_haven_demand_spike",
            "energy_supply_disruption_risk",
            "trade_route_disruption",
            "defense_spending_revision",
            "risk_premium_reprice",
        ],
        "affected_sectors": ["energy", "defense", "industrial", "consumer"],
        "intermediate_variables": [
            "oil_futures", "gold_price", "treasury_bid",
            "shipping_rates", "insurance_costs",
        ],
        "time_horizon": "immediate_6_months",
        "counterforces": [
            "diplomatic_de_escalation", "ceasefire_negotiation",
            "allied_intervention", "market_resilience",
        ],
    },
    "earnings_surprise": {
        "channel_name": "earnings_transmission",
        "chain": [
            "earnings_surprise",
            "analyst_revision_cycle",
            "peer_valuation_reprice",
            "sector_momentum_shift",
            "index_weight_adjustment",
        ],
        "affected_sectors": [],  # sector-specific
        "intermediate_variables": [
            "eps_surprise", "revenue_surprise", "guidance_revision",
            "analyst_consensus_revision",
        ],
        "time_horizon": "1_20_days",
        "counterforces": [
            "mean_reversion", "one_time_item_normalization",
            "guidance_cut_offset",
        ],
    },
    "demand_driver": {
        "channel_name": "semiconductor_demand_transmission",
        "chain": [
            "end_demand_signal",
            "accelerator_and_compute_orders",
            "foundry_and_packaging_utilization",
            "equipment_and_memory_demand",
        ],
        "affected_sectors": ["semiconductor_ai_infrastructure"],
        "intermediate_variables": [
            "order_backlog",
            "lead_times",
            "utilization",
            "inventory",
        ],
        "time_horizon": "30_180_days",
        "counterforces": [
            "customer_inventory_correction",
            "capex_budget_reduction",
            "demand_pull_forward",
        ],
    },
    "supply_disruption": {
        "channel_name": "semiconductor_supply_transmission",
        "chain": [
            "supply_constraint",
            "lead_time_and_input_cost_change",
            "production_mix_adjustment",
            "revenue_and_margin_dispersion",
        ],
        "affected_sectors": ["semiconductor_ai_infrastructure"],
        "intermediate_variables": [
            "lead_times",
            "wafer_capacity",
            "advanced_packaging_capacity",
            "inventory",
        ],
        "time_horizon": "30_180_days",
        "counterforces": [
            "inventory_buffers",
            "second_source_qualification",
            "demand_destruction",
        ],
    },
    "capex_signal": {
        "channel_name": "semiconductor_capex_transmission",
        "chain": [
            "capex_commitment",
            "equipment_orders",
            "capacity_addition",
            "future_supply_and_utilization_change",
        ],
        "affected_sectors": ["semiconductor_ai_infrastructure"],
        "intermediate_variables": [
            "equipment_backlog",
            "construction_timeline",
            "capacity_ramp",
            "utilization",
        ],
        "time_horizon": "90_365_days",
        "counterforces": [
            "project_delay",
            "capex_cancellation",
            "overcapacity",
        ],
    },
    # ── Full-Economy Channels ──────────────────────────────────────────────────
    "climate_disaster": {
        "channel_name": "climate_physical_risk_transmission",
        "chain": [
            "physical_asset_damage_or_disruption",
            "regional_energy_supply_shock",
            "input_cost_spike",
            "production_halt_or_reduction",
            "supply_chain_rerouting",
            "insurance_cost_spike",
            "regional_demand_destruction",
        ],
        "affected_sectors": ["energy", "industrial", "logistics", "consumer",
                             "semiconductor_ai_infrastructure"],
        "intermediate_variables": [
            "regional_power_price", "natural_gas_spot", "insurance_premiums",
            "freight_rates", "crop_prices",
        ],
        "time_horizon": "immediate_3_months",
        "counterforces": [
            "federal_emergency_response", "insurance_payout",
            "production_rerouting", "strategic_reserves_release",
        ],
    },
    "trade_route_disruption": {
        "channel_name": "logistics_chokepoint_transmission",
        "chain": [
            "shipping_lane_disruption",
            "freight_rate_spike",
            "transit_time_increase",
            "inventory_drawdown",
            "input_cost_and_margin_pressure",
            "production_schedule_disruption",
        ],
        "affected_sectors": ["logistics", "industrial", "semiconductor_ai_infrastructure",
                             "consumer", "energy"],
        "intermediate_variables": [
            "baltic_dry_index", "container_rates", "port_congestion",
            "oil_tanker_rates",
        ],
        "time_horizon": "immediate_6_months",
        "counterforces": [
            "alternative_routes", "inventory_buffers",
            "diplomatic_resolution", "naval_escort",
        ],
    },
    "political_transition": {
        "channel_name": "policy_uncertainty_transmission",
        "chain": [
            "policy_uncertainty_spike",
            "investment_deferral",
            "currency_volatility",
            "risk_premium_reprice",
            "regulatory_change_anticipation",
            "capital_flow_reallocation",
        ],
        "affected_sectors": ["financial", "industrial", "energy", "consumer"],
        "intermediate_variables": [
            "policy_uncertainty_index", "fx_volatility",
            "credit_spreads", "equity_risk_premium",
        ],
        "time_horizon": "3_12_months",
        "counterforces": [
            "policy_continuity_signal", "institutional_stability",
            "market_normalization",
        ],
    },
    "debt_crisis": {
        "channel_name": "sovereign_credit_transmission",
        "chain": [
            "credit_downgrade_or_default_event",
            "sovereign_yield_spike",
            "bank_balance_sheet_impairment",
            "credit_tightening",
            "investment_and_consumption_slowdown",
            "global_risk_off_rotation",
        ],
        "affected_sectors": ["financial", "consumer", "industrial", "real_estate"],
        "intermediate_variables": [
            "sovereign_spread", "cds_price", "banking_sector_exposure",
            "interbank_rate",
        ],
        "time_horizon": "1_24_months",
        "counterforces": [
            "imf_bailout", "ecb_backstop",
            "debt_restructuring", "fiscal_consolidation",
        ],
    },
    "pandemic_health_shock": {
        "channel_name": "pandemic_demand_supply_transmission",
        "chain": [
            "mobility_restriction",
            "labour_supply_shock",
            "consumer_demand_collapse_or_shift",
            "supply_chain_disruption",
            "fiscal_stimulus_response",
            "monetary_easing",
        ],
        "affected_sectors": ["consumer", "logistics", "industrial",
                             "healthcare", "financial"],
        "intermediate_variables": [
            "mobility_index", "retail_sales", "unemployment_rate",
            "government_debt_to_gdp",
        ],
        "time_horizon": "immediate_24_months",
        "counterforces": [
            "vaccine_deployment", "stimulus_package",
            "supply_chain_adaptation", "digital_substitution",
        ],
    },
}



class TransmissionMapperLens(AnalystLens):
    """Maps classified events to candidate economic transmission channels.

    Reads ``packet.entity_links`` (which should contain classified events
    from the EventClassifierLens) and produces ``transmission_channels``
    entries describing how each event transmits through the economy.
    """

    lens_name = "transmission_mapper"
    lens_version = "0.1.0"
    event_classes_supported = ("*",)
    can_modify_existing = False

    def analyze(
        self, packet: AnalysisPacket, config: dict[str, Any] | None = None
    ) -> ModuleDelta:
        channels: list[dict[str, Any]] = []

        events = packet.classified_events
        if not events:
            events = [
                item
                for item in [*packet.entity_links, *packet.event_records]
                if isinstance(item, dict)
            ]

        seen_event_ids: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            event_class = str(event.get("event_class", "")).strip()
            if not event_class or event_class == "other":
                continue
            event_id = str(event.get("event_id", event.get("id", "")))
            if event_id and event_id in seen_event_ids:
                continue
            channel = self._build_channel(event, event_class)
            if channel:
                channels.append(channel)
                if event_id:
                    seen_event_ids.add(event_id)

        review_notes: list[str] = []
        if not channels:
            review_notes.append(
                "transmission_mapper: no transmission channels mapped "
                "(no classified events with known event_class)"
            )

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
            as_of=packet.as_of_date,
            transmission_channels_added=channels,
            fields_added=["transmission_channels"],
            confidence=self._overall_confidence(channels, len(events)),
            reason_for_change=(
                f"Mapped {len(channels)} transmission channels from "
                f"{len(events)} classified events."
            ),
            review_notes_added=review_notes,
        )

    def _build_channel(
        self, event: dict[str, Any], event_class: str
    ) -> dict[str, Any] | None:
        template = TRANSMISSION_CHANNELS.get(event_class)
        if template is None:
            return None

        event_id = event.get("event_id", event.get("id", ""))
        affected_sectors = list(template.get("affected_sectors", []))

        # Add sectors from event if available
        for sector in event.get("affected_sectors", []):
            if sector not in affected_sectors:
                affected_sectors.append(sector)

        confidence = self._channel_confidence(event)
        return {
            "channel_id": f"channel_{event_id}",
            "source_event_id": event_id,
            "event_class": event_class,
            "channel_name": template["channel_name"],
            "chain": list(template["chain"]),
            "affected_sectors": affected_sectors,
            "intermediate_variables": list(template.get("intermediate_variables", [])),
            "counterforces": list(template.get("counterforces", [])),
            "causal_metadata": {
                "relation_type": "economic_transmission",
                "identification_method": "assumed_mechanism",
                "causal_claim_allowed": False,
                "confounders": list(template.get("counterforces", [])),
                "mediators": list(template.get("intermediate_variables", [])),
                "colliders": [],
                "intervention": None,
                "counterfactual": None,
                "limitations": [
                    "Template-mapped mechanism, not an identified causal effect",
                    "Event-before-outcome does not establish event-caused-outcome",
                ],
            },
            "dynamics": {
                "strength": None,
                "lag_value": None,
                "lag_unit": "unknown",
                "lag_label": template.get("time_horizon", "unknown"),
                "persistence": None,
                "estimate_confidence": confidence,
                "edge_reliability": confidence,
                "regime_dependencies": list(event.get("regime_dependencies", [])),
                "evidence_count": len(event.get("evidence_ids", [])),
                "last_validated_at": None,
                "decay_function": "unknown",
                "activation_state": "candidate",
            },
            "time_horizon": template.get("time_horizon", "unknown"),
            "confidence": confidence,
            "mapped_by": "transmission_mapper_v0.1",
        }

    def _channel_confidence(self, event: dict[str, Any]) -> float:
        confidence = 0.4
        if event.get("event_class") and event["event_class"] != "other":
            confidence += 0.2
        if event.get("directness") == "direct":
            confidence += 0.15
        elif event.get("directness") == "indirect":
            confidence += 0.08
        if event.get("materiality_score", 0) > 0.5:
            confidence += 0.1
        return min(1.0, confidence)

    def _overall_confidence(
        self, channels: list[dict[str, Any]], total_events: int
    ) -> float:
        if not channels:
            return 0.2
        if total_events == 0:
            return 0.3
        coverage = len(channels) / max(total_events, 1)
        return 0.3 + min(coverage, 1.0) * 0.4


__all__ = ["TransmissionMapperLens", "TRANSMISSION_CHANNELS"]
