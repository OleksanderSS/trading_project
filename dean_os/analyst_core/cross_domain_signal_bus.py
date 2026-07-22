"""Cross-Domain Signal Bus — routes high-materiality events between domains.

Allows a shock detected in one domain (e.g. 'oil_shock' in energy) to be
automatically propagated as contextual evidence (e.g. 'policy_or_geopolitical')
to other domains (e.g. semiconductor_ai_infrastructure).
"""
from __future__ import annotations

from typing import Any

# Mapping: event_class -> how it propagates to other domains
CROSS_DOMAIN_PROPAGATION: dict[str, dict[str, Any]] = {
    "oil_shock": {
        "target_domains": ["semiconductor_ai_infrastructure", "industrial", "consumer", "logistics"],
        "evidence_type": "policy_or_geopolitical",
        "stance_hint": "negative",
        "strength_multiplier": 0.7,  # attenuate indirect effects
        "required_materiality": 0.5,
    },
    "trade_route_disruption": {
        "target_domains": ["semiconductor_ai_infrastructure", "industrial", "consumer", "energy"],
        "evidence_type": "supply_chain",
        "stance_hint": "negative",
        "strength_multiplier": 0.8,
        "required_materiality": 0.4,
    },
    "climate_disaster": {
        "target_domains": ["energy", "industrial", "semiconductor_ai_infrastructure", "logistics"],
        "evidence_type": "supply_chain",
        "stance_hint": "negative",
        "strength_multiplier": 0.6,
        "required_materiality": 0.5,
    },
    "war_escalation": {
        "target_domains": ["energy", "semiconductor_ai_infrastructure", "financials", "industrial"],
        "evidence_type": "policy_or_geopolitical",
        "stance_hint": "negative",
        "strength_multiplier": 0.9,
        "required_materiality": 0.3,
    },
    "central_bank_decision": {
        "target_domains": ["financials", "semiconductor_ai_infrastructure", "consumer", "real_estate"],
        "evidence_type": "macro_context",
        "stance_hint": "mixed",  # depends on direction
        "strength_multiplier": 0.8,
        "required_materiality": 0.5,
    },
    "pandemic_health_shock": {
        "target_domains": ["consumer", "logistics", "industrial", "healthcare", "financials"],
        "evidence_type": "macro_context",
        "stance_hint": "negative",
        "strength_multiplier": 0.9,
        "required_materiality": 0.6,
    },
    "debt_crisis": {
        "target_domains": ["financials", "consumer", "real_estate", "industrial"],
        "evidence_type": "macro_context",
        "stance_hint": "negative",
        "strength_multiplier": 0.8,
        "required_materiality": 0.5,
    },
    "political_transition": {
        "target_domains": ["financials", "industrial", "energy", "consumer"],
        "evidence_type": "policy_or_geopolitical",
        "stance_hint": "mixed",
        "strength_multiplier": 0.6,
        "required_materiality": 0.6,
    },
}
