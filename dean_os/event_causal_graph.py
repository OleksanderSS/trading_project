"""
Event Causal Graph — Probabilistic Propagation from News Events
===============================================================

Implements a candidate transmission graph:
    Trigger event -> intermediate effects -> market outcomes (with probabilities)

Example:
    Headline: "Earthquake hits Taiwan near TSMC fabs"
    Graph:
        earthquake (100%)
          └-> TSMC production halt (80%)
               └-> chip shortage global (65%)
                    └-> NVDA price spike (55%)
                    └-> AAPL supply disruption (45%)
               └-> CoWoS capacity crunch (70%)
                    └-> AI GPU lead times +6w (60%)

This module is standalone (no LLM required). Probabilities are uncalibrated
rule-based review priors from a configurable domain knowledge map. They are not
causal-effect estimates. The legacy output is a ``CausalGraph`` object that:
  - Embeds directly into ``MarketContext.metadata``
  - Converts to ``EvidenceItem`` for DEAN-OS agent reports
  - Produces a human-readable summary for review artifacts

Author: DEAN-OS pipeline — review-only, no trade authority.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from dean_os.causal_contracts import CausalClaimMetadata, GraphEdgeDynamics

# ── Objective event types: shock sentiment is irrelevant to causal chain ─
# For these types the news TONE (positive/negative) doesn't change the
# causal mechanics — an earthquake halts production regardless of headline tone.
_OBJECTIVE_EVENT_TYPES: frozenset[str] = frozenset({
    "natural_disaster", "geopolitical", "regulatory", "supply_chain",
})

# Fallback map: if classify_event_type returns a type with no rules,
# or returns "uncategorized", try these alternative rule sets.
_FALLBACK_RULES: dict[str, str] = {
    "technology":       "supply_chain",
    "corporate":        "macro",
    "commodity":        "energy",
    "credit_financial": "monetary_policy",
    "uncategorized":    "macro",
}

# ── Causal Knowledge Base ─────────────────────────────────────────────────
# Format:
#   trigger_event_type -> [
#       (effect_label, base_probability, affected_sectors, affected_tickers_hint, lag),
#   ]
#
# Probabilities are conservative rule-based estimates, NOT predictions.
# They represent "how often does this trigger historically lead to this effect".

CAUSAL_RULES: dict[str, list[dict[str, Any]]] = {
    "natural_disaster": [
        {
            "effect": "regional_production_halt",
            "probability": 0.75,
            "sectors": ["semiconductor", "logistics", "energy"],
            "tickers_hint": [],
            "lag": "immediate",
            "notes": "Depends on affected region and severity",
        },
        {
            "effect": "supply_chain_disruption",
            "probability": 0.65,
            "sectors": ["logistics", "semiconductor", "consumer"],
            "tickers_hint": [],
            "lag": "days",
        },
        {
            "effect": "insurance_sector_claims_spike",
            "probability": 0.80,
            "sectors": ["finance"],
            "tickers_hint": [],
            "lag": "weeks",
        },
    ],
    "geopolitical": [
        {
            "effect": "export_control_tightening",
            "probability": 0.55,
            "sectors": ["semiconductor", "defense", "technology"],
            "tickers_hint": [],
            "lag": "weeks",
        },
        {
            "effect": "commodity_price_spike",
            "probability": 0.60,
            "sectors": ["energy", "commodity"],
            "tickers_hint": [],
            "lag": "immediate",
        },
        {
            "effect": "risk_off_sentiment",
            "probability": 0.70,
            "sectors": ["finance"],
            "tickers_hint": [],
            "lag": "immediate",
        },
        {
            "effect": "defense_spending_increase",
            "probability": 0.65,
            "sectors": ["defense"],
            "tickers_hint": [],
            "lag": "quarters",
        },
    ],
    "monetary_policy": [
        {
            "effect": "credit_cost_change",
            "probability": 0.90,
            "sectors": ["finance", "real_estate"],
            "tickers_hint": [],
            "lag": "immediate",
        },
        {
            "effect": "growth_sector_rotation",
            "probability": 0.65,
            "sectors": ["technology", "consumer"],
            "tickers_hint": [],
            "lag": "days",
        },
        {
            "effect": "currency_fx_shift",
            "probability": 0.75,
            "sectors": ["finance"],
            "tickers_hint": [],
            "lag": "immediate",
        },
        {
            "effect": "housing_demand_change",
            "probability": 0.60,
            "sectors": ["real_estate"],
            "tickers_hint": [],
            "lag": "months",
        },
    ],
    "supply_chain": [
        {
            "effect": "inventory_drawdown",
            "probability": 0.70,
            "sectors": ["semiconductor", "consumer", "logistics"],
            "tickers_hint": [],
            "lag": "weeks",
        },
        {
            "effect": "lead_time_expansion",
            "probability": 0.75,
            "sectors": ["semiconductor"],
            "tickers_hint": [],
            "lag": "days",
        },
        {
            "effect": "price_premium_spot_market",
            "probability": 0.60,
            "sectors": ["semiconductor", "energy", "commodity"],
            "tickers_hint": [],
            "lag": "weeks",
        },
    ],
    "fiscal_policy": [
        {
            "effect": "sector_capex_incentive",
            "probability": 0.65,
            "sectors": ["semiconductor", "energy", "defense"],
            "tickers_hint": [],
            "lag": "quarters",
        },
        {
            "effect": "consumer_demand_boost",
            "probability": 0.55,
            "sectors": ["consumer"],
            "tickers_hint": [],
            "lag": "months",
        },
    ],
    "corporate": [
        {
            "effect": "earnings_revision_cascade",
            "probability": 0.60,
            "sectors": [],  # depends on company sector
            "tickers_hint": [],
            "lag": "immediate",
        },
        {
            "effect": "sector_sentiment_contagion",
            "probability": 0.50,
            "sectors": [],
            "tickers_hint": [],
            "lag": "days",
        },
    ],
    "macro": [
        {
            "effect": "consumer_confidence_shift",
            "probability": 0.65,
            "sectors": ["consumer", "real_estate"],
            "tickers_hint": [],
            "lag": "weeks",
        },
        {
            "effect": "central_bank_policy_signal",
            "probability": 0.55,
            "sectors": ["finance"],
            "tickers_hint": [],
            "lag": "immediate",
        },
    ],
    "technology": [
        {
            "effect": "semiconductor_demand_acceleration",
            "probability": 0.70,
            "sectors": ["semiconductor", "technology"],
            "tickers_hint": [],
            "lag": "quarters",
        },
        {
            "effect": "cloud_infrastructure_spending",
            "probability": 0.65,
            "sectors": ["technology"],
            "tickers_hint": [],
            "lag": "months",
        },
    ],
    "energy": [
        {
            "effect": "production_cost_shift",
            "probability": 0.70,
            "sectors": ["consumer", "logistics", "semiconductor"],
            "tickers_hint": [],
            "lag": "weeks",
        },
        {
            "effect": "inflation_pressure",
            "probability": 0.60,
            "sectors": ["finance", "consumer"],
            "tickers_hint": [],
            "lag": "months",
        },
    ],
    "regulatory": [
        {
            "effect": "compliance_cost_increase",
            "probability": 0.75,
            "sectors": [],
            "tickers_hint": [],
            "lag": "quarters",
        },
        {
            "effect": "market_access_restriction",
            "probability": 0.55,
            "sectors": [],
            "tickers_hint": [],
            "lag": "months",
        },
    ],
}

# Sector -> ticker candidates (hints for analysts, not signals)
SECTOR_TICKER_HINTS: dict[str, list[str]] = {
    "semiconductor": ["NVDA", "AMD", "INTC", "TSM", "AMAT", "LRCX", "MU"],
    "technology":    ["MSFT", "AAPL", "GOOGL", "META", "AMZN"],
    "energy":        ["XOM", "CVX", "COP", "SLB", "VLO"],
    "finance":       ["JPM", "BAC", "GS", "MS", "BRK.B"],
    "defense":       ["LMT", "RTX", "NOC", "GD", "BA"],
    "consumer":      ["AMZN", "WMT", "TGT", "COST", "HD"],
    "healthcare":    ["JNJ", "PFE", "ABBV", "MRK", "UNH"],
    "logistics":     ["UPS", "FDX", "CHRW", "EXPD"],
    "real_estate":   ["AMT", "PLD", "EQIX", "SPG"],
}


# ── Pydantic Schema ───────────────────────────────────────────────────────


class CausalNode(BaseModel):
    """A single node in the causal propagation graph."""

    node_id: str
    label: str                          # human-readable effect label
    probability: float = Field(ge=0.0, le=1.0)
    probability_kind: str = "heuristic_review_prior"
    estimate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    impact_magnitude: float | None = Field(default=None, ge=0.0, le=1.0)
    market_reaction: float | None = Field(default=None, ge=-1.0, le=1.0)
    fundamental_change: float | None = Field(default=None, ge=-1.0, le=1.0)
    affected_sectors: list[str] = Field(default_factory=list)
    ticker_hints: list[str] = Field(default_factory=list)
    lag: str = "unknown"                # immediate / days / weeks / months / quarters
    direction: str = "unknown"          # bullish / bearish / uncertain
    notes: str = ""
    depth: int = 0                      # 0 = trigger, 1 = first-order, 2 = second-order


class CausalEdge(BaseModel):
    """Directed candidate edge with an epistemic/identification label."""

    source_id: str
    target_id: str
    conditional_probability: float = Field(ge=0.0, le=1.0)
    relationship: str = ""
    causal_metadata: CausalClaimMetadata = Field(
        default_factory=CausalClaimMetadata
    )
    dynamics: GraphEdgeDynamics = Field(default_factory=GraphEdgeDynamics)


class CausalGraph(BaseModel):
    """Legacy name for a review-only candidate transmission graph."""

    graph_id: str
    trigger_headline: str
    trigger_event_type: str
    trigger_shock: str                  # positive / negative / neutral
    trigger_confidence: float = Field(ge=0.0, le=1.0)
    trigger_impact: float = Field(ge=-1.0, le=1.0)
    nodes: list[CausalNode] = Field(default_factory=list)
    edges: list[CausalEdge] = Field(default_factory=list)
    affected_sectors: list[str] = Field(default_factory=list)
    ticker_watch_list: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    review_only: bool = True            # always True — no trade authority

    def to_evidence_value(self) -> dict[str, Any]:
        """Serialize for embedding in an EvidenceItem."""
        return {
            "graph_id": self.graph_id,
            "trigger": self.trigger_headline[:120],
            "event_type": self.trigger_event_type,
            "shock": self.trigger_shock,
            "nodes": len(self.nodes),
            "overall_confidence": round(self.overall_confidence, 3),
            "affected_sectors": self.affected_sectors,
            "ticker_watch_list": self.ticker_watch_list,
            "summary": self.summary,
        }

    def human_readable(self) -> str:
        """Tree-like text representation for review artifacts."""
        lines = [
            f"📰 Trigger: {self.trigger_headline[:100]}",
            f"   Type: {self.trigger_event_type}  Shock: {self.trigger_shock} "
            f"({self.trigger_confidence:.0%})  Impact: {self.trigger_impact:+.2f}",
            "",
            "🔗 Causal Chain:",
        ]
        trigger_node = next((n for n in self.nodes if n.depth == 0), None)
        if trigger_node:
            lines.append(f"  [{trigger_node.probability:.0%}] {trigger_node.label}")

        depth1 = [n for n in self.nodes if n.depth == 1]
        for node in depth1:
            direction_icon = "📈" if node.direction == "bullish" else "📉" if node.direction == "bearish" else "➡️"
            lines.append(f"    └-> [{node.probability:.0%}] {node.label}  {direction_icon}  lag={node.lag}")
            if node.ticker_hints:
                lines.append(f"         🎯 Tickers: {', '.join(node.ticker_hints[:5])}")

        depth2 = [n for n in self.nodes if n.depth == 2]
        for node in depth2:
            direction_icon = "📈" if node.direction == "bullish" else "📉" if node.direction == "bearish" else "➡️"
            lines.append(f"         └-> [{node.probability:.0%}] {node.label}  {direction_icon}  lag={node.lag}")

        if self.ticker_watch_list:
            lines += ["", f"👁  Watch list: {', '.join(self.ticker_watch_list)}"]
        lines += ["", f"⚠️  Review-only — no trade authority.  Confidence: {self.overall_confidence:.0%}"]
        return "\n".join(lines)


# ── Graph Builder ─────────────────────────────────────────────────────────


class EventCausalGraphBuilder:
    """
    Builds a CausalGraph from a classified NewsEvent (from news_event_analyzer).

    Rule-based only — no LLM required.
    Works entirely offline from the CAUSAL_RULES knowledge base.

    Usage:
        from dean_os.agents.news_event_analyzer import NewsEvent
        from dean_os.event_causal_graph import EventCausalGraphBuilder

        event = NewsEvent("Earthquake hits Taiwan near TSMC fabs")
        graph = EventCausalGraphBuilder().build(event)
        print(graph.human_readable())
    """

    def __init__(
        self,
        context_tickers: list[str] | None = None,
        min_probability: float = 0.25,
    ) -> None:
        """
        Parameters
        ----------
        context_tickers:
            Tickers from MarketContext — used to prioritize relevant hints.
        min_probability:
            First-order effects below this threshold are pruned.
        """
        self.context_tickers = [t.upper() for t in (context_tickers or [])]
        self.min_probability = min_probability

    def build(self, event: Any) -> CausalGraph:
        """
        Build a CausalGraph from a NewsEvent (duck-typed to avoid circular imports).

        ``event`` must have:
            .headline, .event_type, .shock, .shock_confidence,
            .impact, .predictability, .affected_sectors
        """
        graph_id = f"cg_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{event.event_type}"
        trigger_node = CausalNode(
            node_id="n0_trigger",
            label=event.headline[:80],
            probability=1.0,
            probability_kind="observed_event",
            estimate_confidence=event.shock_confidence,
            impact_magnitude=abs(event.impact),
            affected_sectors=event.affected_sectors,
            lag="immediate",
            direction=_shock_to_direction(event.shock),
            depth=0,
        )

        # Resolve rules: try direct type, then fallback
        rules = CAUSAL_RULES.get(event.event_type, [])
        if not rules:
            fallback_type = _FALLBACK_RULES.get(event.event_type, "macro")
            rules = CAUSAL_RULES.get(fallback_type, [])

        nodes: list[CausalNode] = [trigger_node]
        edges: list[CausalEdge] = []
        all_sectors: set[str] = set(event.affected_sectors)
        all_tickers: set[str] = set()

        # For objective events (disasters, geopolitical, regulatory) the causal
        # mechanics are real regardless of news tone — use impact magnitude instead
        # of shock_confidence as the confidence multiplier.
        is_objective = event.event_type in _OBJECTIVE_EVENT_TYPES
        confidence_signal = (
            max(abs(event.impact), 0.40)          # at least 0.40 for objective events
            if is_objective
            else event.shock_confidence
        )

        for idx, rule in enumerate(rules):
            base_prob = float(rule["probability"])
            # Adjust by confidence signal and predictability
            adj_prob = base_prob * confidence_signal * (0.7 + 0.3 * event.predictability)
            adj_prob = round(min(adj_prob, 0.95), 3)

            if adj_prob < self.min_probability:
                continue

            sectors: list[str] = list(rule.get("sectors") or event.affected_sectors)
            ticker_hints = _resolve_ticker_hints(sectors, self.context_tickers)
            direction = _effect_direction(rule["effect"], event.shock)

            node = CausalNode(
                node_id=f"n1_{idx}_{rule['effect'][:20]}",
                label=rule["effect"].replace("_", " ").title(),
                probability=adj_prob,
                probability_kind="heuristic_review_prior",
                estimate_confidence=confidence_signal,
                affected_sectors=sectors,
                ticker_hints=ticker_hints[:6],
                lag=str(rule.get("lag", "unknown")),
                direction=direction,
                notes=str(rule.get("notes", "")),
                depth=1,
            )
            nodes.append(node)
            edges.append(CausalEdge(
                source_id=trigger_node.node_id,
                target_id=node.node_id,
                conditional_probability=adj_prob,
                relationship=f"{event.event_type}->{rule['effect']}",
                causal_metadata=CausalClaimMetadata(
                    relation_type="economic_transmission",
                    identification_method="assumed_mechanism",
                    causal_claim_allowed=False,
                    limitations=[
                        "Rule-based transmission prior; not causally identified",
                        "Temporal ordering alone is insufficient for causality",
                    ],
                ),
                dynamics=GraphEdgeDynamics(
                    strength=base_prob,
                    lag_label=str(rule.get("lag", "unknown")),
                    estimate_confidence=confidence_signal,
                    edge_reliability=confidence_signal,
                    regime_dependencies=[event.event_type],
                    evidence_count=0,
                    decay_function="unknown",
                    activation_state="candidate",
                ),
            ))
            all_sectors.update(sectors)
            all_tickers.update(ticker_hints)

        # Context Mesh Recursive Traversal (Depth 2 to 3)
        queue = [n for n in nodes if n.depth == 1]
        visited = set()
        
        while queue:
            parent = queue.pop(0)
            if parent.depth >= 3:
                continue
                
            effect_key = parent.label.lower().replace(" ", "_")
            child_rules = CAUSAL_RULES.get(effect_key, [])
            
            if not child_rules:
                continue
                
            child_rules = sorted(child_rules, key=lambda r: r['probability'], reverse=True)[:3]
            
            for c_idx, rule in enumerate(child_rules):
                base_prob = float(rule['probability'])
                adj_prob = round(parent.probability * base_prob, 3)
                
                if adj_prob < self.min_probability:
                    continue
                    
                effect_name = rule['effect']
                edge_sig = f"{parent.node_id}->{effect_name}"
                if edge_sig in visited:
                    continue
                visited.add(edge_sig)
                
                c_sectors = list(rule.get('sectors') or parent.affected_sectors)
                c_tickers = _resolve_ticker_hints(c_sectors + rule.get('tickers_hint', []), self.context_tickers)
                
                d_node = CausalNode(
                    node_id=f"n{parent.depth+1}_{parent.node_id.split('_')[0]}_{effect_name[:20]}",
                    label=effect_name.replace("_", " ").title(),
                    probability=adj_prob,
                    probability_kind="heuristic_review_prior",
                    estimate_confidence=parent.estimate_confidence,
                    affected_sectors=c_sectors,
                    ticker_hints=c_tickers[:6],
                    lag=_next_lag(parent.lag),
                    direction=parent.direction,
                    depth=parent.depth + 1,
                )
                nodes.append(d_node)
                queue.append(d_node)
                
                edges.append(CausalEdge(
                    source_id=parent.node_id,
                    target_id=d_node.node_id,
                    conditional_probability=adj_prob,
                    relationship=f"{effect_key}->{effect_name}",
                    causal_metadata=CausalClaimMetadata(
                        relation_type="economic_transmission",
                        identification_method="assumed_mechanism",
                        causal_claim_allowed=False,
                        limitations=["Mesh secondary propagation"]
                    ),
                    dynamics=GraphEdgeDynamics(
                        strength=adj_prob,
                        lag_label=_next_lag(parent.lag),
                        estimate_confidence=parent.estimate_confidence,
                        edge_reliability=parent.estimate_confidence,
                        regime_dependencies=[],
                        evidence_count=0,
                        decay_function="unknown",
                        activation_state="candidate",
                    ),
                ))
                all_sectors.update(c_sectors)
                all_tickers.update(c_tickers)

        overall_confidence = (
            sum(n.estimate_confidence for n in nodes) / len(nodes)
        ) if len(nodes) > 1 else 0.0

        watch_list = sorted(all_tickers)

        graph = CausalGraph(
            graph_id=graph_id,
            trigger_headline=event.headline,
            trigger_event_type=event.event_type,
            trigger_shock=event.shock,
            trigger_confidence=round(event.shock_confidence, 3),
            trigger_impact=round(event.impact, 3),
            nodes=nodes,
            edges=edges,
            affected_sectors=sorted(all_sectors),
            ticker_watch_list=watch_list[:12],
            overall_confidence=overall_confidence,
            summary=_build_summary(event, nodes, watch_list),
        )
        return graph

    def build_multi(self, events: list[Any]) -> list[CausalGraph]:
        """Build graphs for a list of NewsEvents. Skips 'neutral' low-impact events."""
        graphs = []
        for event in events:
            if event.shock == "neutral" and abs(event.impact) < 0.2:
                continue
            graphs.append(self.build(event))
        return graphs


# ── Module-level helpers ──────────────────────────────────────────────────


def _shock_to_direction(shock: str) -> str:
    return {"positive": "bullish", "negative": "bearish"}.get(shock, "uncertain")


def _effect_direction(effect: str, shock: str) -> str:
    """Infer bullish/bearish from effect label + trigger shock."""
    bearish_keywords = {"halt", "disruption", "crisis", "shortage", "restriction", "cost", "claims"}
    bullish_keywords = {"incentive", "boost", "acceleration", "spending", "demand_boost"}
    eff_lower = effect.lower()
    if any(kw in eff_lower for kw in bearish_keywords):
        return "bearish"
    if any(kw in eff_lower for kw in bullish_keywords):
        return "bullish"
    return _shock_to_direction(shock)


def _resolve_ticker_hints(sectors: list[str], context_tickers: list[str]) -> list[str]:
    hints: list[str] = []
    for sector in sectors:
        hints.extend(SECTOR_TICKER_HINTS.get(sector, []))
    # Deduplicate, context_tickers first
    seen: set[str] = set()
    ordered: list[str] = []
    for t in context_tickers + hints:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _next_lag(lag: str) -> str:
    order = ["immediate", "days", "weeks", "months", "quarters"]
    try:
        idx = order.index(lag)
        return order[min(idx + 1, len(order) - 1)]
    except ValueError:
        return "months"


def _build_summary(event: Any, nodes: list[CausalNode], watch_list: list[str]) -> str:
    depth1 = [n for n in nodes if n.depth == 1]
    top = sorted(depth1, key=lambda n: n.probability, reverse=True)[:2]
    top_str = " -> ".join(n.label for n in top) if top else "no downstream effects"
    tickers_str = ", ".join(watch_list[:5]) if watch_list else "none identified"
    return (
        f"{event.event_type.upper()} event ({event.shock}): "
        f"{top_str}. Watch: {tickers_str}. "
        f"Review-only — {len(depth1)} first-order effects mapped."
    )


def graphs_to_context_metadata(graphs: list[CausalGraph]) -> dict[str, Any]:
    """
    Serialize a list of CausalGraphs into a dict suitable for
    ``MarketContext.metadata["causal_graphs"]``.
    """
    return {
        "causal_graphs": [g.model_dump(mode="json") for g in graphs],
        "graph_count": len(graphs),
        "all_watch_tickers": sorted({t for g in graphs for t in g.ticker_watch_list}),
        "all_affected_sectors": sorted({s for g in graphs for s in g.affected_sectors}),
        "built_at": datetime.now(UTC).isoformat(),
        "review_only": True,
    }


__all__ = [
    "CausalNode",
    "CausalEdge",
    "CausalGraph",
    "EventCausalGraphBuilder",
    "CAUSAL_RULES",
    "SECTOR_TICKER_HINTS",
    "graphs_to_context_metadata",
]
