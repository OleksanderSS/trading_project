"""Economy Regime Brief Formatter.

Formats the deeply structured AnalysisPacket JSON into a readable 
Economy Regime Brief for the user, matching the ChatGPT template style.
"""
from __future__ import annotations

from typing import Any
from dean_os.analyst_core.lens_contract import AnalysisPacket
from dean_os.schemas import utc_now_iso


def format_economy_regime_brief(packet: AnalysisPacket) -> str:
    """Renders an AnalysisPacket into an Economy Regime Brief markdown string."""
    date_str = packet.as_of_date[:10]
    
    lines = [
        "Economy Regime Brief",
        "====================",
        f"Date: {date_str}\n",
    ]

    # 0) Regime Snapshot
    lines.append("## 0) Regime Snapshot")
    if packet.regime_context and packet.regime_context.dimensions:
        dominant_dim = max(packet.regime_context.dimensions.items(), key=lambda x: x[1].intensity)
        dominant_label = dominant_dim[1].state if dominant_dim[1].intensity > 0 else "Mixed / Uncertain"
        lines.append(f"**Overall Regime:** {dominant_label}\n")
        
        lines.append("| Dimension | Current State | Trend |")
        lines.append("|-----------|---------------|-------|")
        for dim, state in packet.regime_context.dimensions.items():
            lines.append(f"| {dim.replace('_', ' ').title()} | {state.state} | {state.trend} |")
    else:
        lines.append("*No regime context generated.*")
    lines.append("")

    # 1) Mandatory Sector Coverage Gate
    lines.append("## 1) Mandatory Sector Coverage Gate")
    lines.append("*(Derived from Evidence Pack and active Context)*\n")
    if hasattr(packet, "evidence_pack") and packet.evidence_pack:
        # Group by category/sector
        lines.append("| Sector / Theme | Signal / Evidence |")
        lines.append("|----------------|-------------------|")
        for ev in packet.evidence_pack[:10]:
            category = ev.get("category") or ev.get("theme") or "General"
            summary = ev.get("summary") or str(ev.get("value", ""))
            lines.append(f"| {category} | {summary[:150]}... |")
    else:
        lines.append("*No specific sector coverage signals found in packet.*")
    lines.append("")

    # 2) Top Situation Developments
    lines.append("## 2) Top Situation Developments")
    if packet.scenario_graph and packet.scenario_graph.nodes:
        # Use nodes from the scenario graph
        for i, node in enumerate(packet.scenario_graph.nodes[:5], 1):
            name = node.name or node.node_id
            lines.append(f"**{i}. {name.replace('_', ' ').title()}**")
            
            # Find evidence supporting this node
            ev_list = []
            if hasattr(packet, "evidence_pack") and packet.evidence_pack:
                for ev in packet.evidence_pack:
                    if node.node_id in str(ev):
                        ev_list.append(ev.get("summary", ""))
            
            if ev_list:
                lines.append(f"> Fact: {ev_list[0]}")
            lines.append("")
    else:
        lines.append("*No structural situation developments found.*")
    lines.append("")

    # 3) News vs Regime Graph
    lines.append("## 3) News vs Regime / Scenario Outcome Graph")
    if packet.transmission_channels:
        lines.append("**Transmission Channels:**")
        for ch in packet.transmission_channels:
            trigger = ch.get("trigger", "Unknown")
            effects = " -> ".join(ch.get("downstream_effects", []))
            lines.append(f"- {trigger} -> {effects}")
        lines.append("")
        
    if packet.expectation_gap:
        lines.append("**Expectation Gaps:**")
        lines.append(f"> {packet.expectation_gap.get('description', 'Unknown gap')}")
        lines.append("")

    if packet.scenario_graph and packet.scenario_graph.edges:
        lines.append("**Scenarios (1-3 months):**")
        for i, edge in enumerate(packet.scenario_graph.edges[:4]):
            desc = edge.rationale or f"{edge.source_node_id} -> {edge.target_node_id}"
            lines.append(f"- Scenario {chr(65+i)}: {desc} ({int(edge.probability_delta * 100)}%)")
    lines.append("")

    # 4) Practical Implications
    lines.append("## 4) Practical Implications")
    if packet.watch_signals:
        lines.append("For the DEAN-OS market layer:")
        for sig in packet.watch_signals:
            lines.append(f"- {sig.get('description', str(sig))}")
    else:
        lines.append("*Implications mapped to review-only evidence blocks.*")
    lines.append("")

    # 5) Risks & Gaps
    lines.append("## 5) Risks and Uncertainty")
    if packet.evidence_gaps:
        for gap in packet.evidence_gaps:
            lines.append(f"- **{gap.expected_source_type.replace('_', ' ').title()}**: {gap.description}")
    else:
        lines.append("*No major evidence gaps identified.*")
    lines.append("")

    # 7) Analyst Journal / Replay
    lines.append("## 7) DEAN-OS Analyst Journal / Learning Notes")
    if packet.hypotheses:
        lines.append("**Observation Horizons:**")
        for h in packet.hypotheses:
            horizons = ", ".join(f"{hrz}d" for hrz in h.horizons_to_check)
            invalidation = "; ".join(h.invalidation_signals) if h.invalidation_signals else "None"
            lines.append(f"- **Horizons ({horizons})**: {invalidation}")
    else:
        lines.append("*No formal hypotheses logged for future review.*")

    return "\n".join(lines)
