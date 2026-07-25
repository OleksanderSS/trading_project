"""Cross-Domain Signal Bus — routes high-materiality events between domains.

Allows a shock detected in one domain (e.g. 'oil_shock' in energy) to be
automatically propagated as contextual evidence (e.g. 'policy_or_geopolitical')
to other domains (e.g. semiconductor_ai_infrastructure).
"""
from __future__ import annotations

from typing import Any

# Mapping: event_class -> how it propagates to other domains
# target_domains must match real registered domain_ids (see
# dean_os.domain_profiles.list_domain_ids()) -- there is no "industrial",
# "consumer", or "financials" domain; the real ids are "industrials",
# "consumer_discretionary"/"consumer_staples", and "liquidity_credit"
# respectively. from_signal_bus() (artifact_evidence_loader.py) silently
# drops any signal whose domain doesn't match a real one, so a typo here
# means the propagation rule just never fires for that domain.
CROSS_DOMAIN_PROPAGATION: dict[str, dict[str, Any]] = {
    "oil_shock": {
        "target_domains": ["semiconductor_ai_infrastructure", "industrials", "consumer_discretionary", "consumer_staples", "logistics"],
        "evidence_type": "policy_or_geopolitical",
        "stance_hint": "negative",
        "strength_multiplier": 0.7,  # attenuate indirect effects
        "required_materiality": 0.5,
    },
    "trade_route_disruption": {
        "target_domains": ["semiconductor_ai_infrastructure", "industrials", "consumer_discretionary", "consumer_staples", "energy"],
        "evidence_type": "supply_chain",
        "stance_hint": "negative",
        "strength_multiplier": 0.8,
        "required_materiality": 0.4,
    },
    "climate_disaster": {
        "target_domains": ["energy", "industrials", "semiconductor_ai_infrastructure", "logistics"],
        "evidence_type": "supply_chain",
        "stance_hint": "negative",
        "strength_multiplier": 0.6,
        "required_materiality": 0.5,
    },
    "war_escalation": {
        "target_domains": ["energy", "semiconductor_ai_infrastructure", "liquidity_credit", "industrials"],
        "evidence_type": "policy_or_geopolitical",
        "stance_hint": "negative",
        "strength_multiplier": 0.9,
        "required_materiality": 0.3,
    },
    "central_bank_decision": {
        "target_domains": ["liquidity_credit", "semiconductor_ai_infrastructure", "consumer_discretionary", "consumer_staples", "real_estate"],
        "evidence_type": "macro_context",
        "stance_hint": "mixed",  # depends on direction
        "strength_multiplier": 0.8,
        "required_materiality": 0.5,
    },
    "pandemic_health_shock": {
        "target_domains": ["consumer_discretionary", "consumer_staples", "logistics", "industrials", "healthcare", "liquidity_credit"],
        "evidence_type": "macro_context",
        "stance_hint": "negative",
        "strength_multiplier": 0.9,
        "required_materiality": 0.6,
    },
    "debt_crisis": {
        "target_domains": ["liquidity_credit", "consumer_discretionary", "consumer_staples", "real_estate", "industrials"],
        "evidence_type": "macro_context",
        "stance_hint": "negative",
        "strength_multiplier": 0.8,
        "required_materiality": 0.5,
    },
    "political_transition": {
        "target_domains": ["liquidity_credit", "industrials", "energy", "consumer_discretionary", "consumer_staples"],
        "evidence_type": "policy_or_geopolitical",
        "stance_hint": "mixed",
        "strength_multiplier": 0.6,
        "required_materiality": 0.6,
    },
}
