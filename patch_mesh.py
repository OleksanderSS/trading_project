import os

file_path = 'd:/trading_project/dean_os/event_causal_graph.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

new_rules = """
    "maritime_chokepoint_blockade": [
        {
            "effect": "energy_cost_spike",
            "probability": 0.85,
            "sectors": ["energy", "logistics"],
            "tickers_hint": [],
            "lag": "days",
            "notes": "Hormuz/Red Sea/Suez blockade"
        },
        {
            "effect": "global_freight_delay",
            "probability": 0.90,
            "sectors": ["logistics", "consumer", "semiconductor"],
            "tickers_hint": [],
            "lag": "weeks",
        }
    ],
    "taiwan_strait_tension": [
        {
            "effect": "advanced_node_supply_shock",
            "probability": 0.70,
            "sectors": ["semiconductor", "technology"],
            "tickers_hint": ["TSM", "NVDA", "AMD", "ASML"],
            "lag": "days",
        }
    ],
    "energy_cost_spike": [
        {
            "effect": "manufacturing_margins_down",
            "probability": 0.75,
            "sectors": ["semiconductor", "industrial", "consumer"],
            "tickers_hint": [],
            "lag": "quarters",
        },
        {
            "effect": "inflation_pressure",
            "probability": 0.80,
            "sectors": ["finance", "macro"],
            "tickers_hint": [],
            "lag": "months",
        }
    ],
    "advanced_node_supply_shock": [
        {
            "effect": "ai_infrastructure_buildout_delay",
            "probability": 0.85,
            "sectors": ["technology", "software"],
            "tickers_hint": ["MSFT", "GOOGL", "META", "AMZN"],
            "lag": "quarters",
        }
    ],
    "global_freight_delay": [
        {
            "effect": "inventory_drawdown",
            "probability": 0.80,
            "sectors": ["retail", "semiconductor"],
            "tickers_hint": [],
            "lag": "months",
        }
    ],
"""

content = content.replace('CAUSAL_RULES: dict[str, list[dict[str, Any]]] = {', 'CAUSAL_RULES: dict[str, list[dict[str, Any]]] = {' + new_rules)

old_build = '''        # Second-order: high-impact first-order nodes get downstream effects
        depth1_high = [n for n in nodes if n.depth == 1 and n.probability >= 0.60]
        for parent in depth1_high[:3]:  # limit explosion
            for sector in parent.affected_sectors[:2]:
                d2_tickers = _resolve_ticker_hints([sector], self.context_tickers)
                if not d2_tickers:
                    continue
                d2_node = CausalNode(
                    node_id=f"n2_{parent.node_id}_{sector}",
                    label=f"{sector.title()} price pressure",
                    probability=round(parent.probability * 0.70, 3),
                    probability_kind="heuristic_review_prior",
                    estimate_confidence=parent.estimate_confidence,
                    affected_sectors=[sector],
                    ticker_hints=d2_tickers[:4],
                    lag=_next_lag(parent.lag),
                    direction=parent.direction,
                    depth=2,
                )
                nodes.append(d2_node)
                edges.append(CausalEdge(
                    source_id=parent.node_id,
                    target_id=d2_node.node_id,
                    conditional_probability=round(parent.probability * 0.70, 3),
                    causal_metadata=CausalClaimMetadata(
                        relation_type="hypothesis_only",
                        identification_method="assumed_mechanism",
                        causal_claim_allowed=False,
                        limitations=["Generic secondary propagation"]
                    ),
                    dynamics=GraphEdgeDynamics(
                        strength=round(parent.probability * 0.70, 3),
                        lag_label=_next_lag(parent.lag),
                        estimate_confidence=parent.estimate_confidence,
                        edge_reliability=parent.estimate_confidence,
                        regime_dependencies=[],
                        evidence_count=0,
                        decay_function="unknown",
                        activation_state="candidate",
                    ),
                ))
                all_sectors.add(sector)
                all_tickers.update(d2_tickers)

        return CausalGraph(
'''

new_build = '''        # Context Mesh Recursive Traversal (Depth 2 to 3)
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

        return CausalGraph(
'''

if old_build in content:
    content = content.replace(old_build, new_build)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully rewrote event_causal_graph.py to support Context Mesh and new chokepoints!")
else:
    print("Could not find the exact old_build block. Exiting.")
