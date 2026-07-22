import os

file_path = 'd:/trading_project/dean_os/event_causal_graph.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "# Second-order: high-impact first-order nodes get downstream effects" in line:
        start_idx = i
    if "graph = CausalGraph(" in line:
        end_idx = i - 2 # go back to avoid removing graph = CausalGraph and watch_list stuff
        break

new_build_lines = '''        # Context Mesh Recursive Traversal (Depth 2 to 3)
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

'''

if start_idx != -1 and end_idx != -1:
    new_content = "".join(lines[:start_idx]) + new_build_lines + "".join(lines[end_idx:])
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully patched line by line!")
else:
    print(f"Failed to find bounds. Start: {start_idx}, End: {end_idx}")
