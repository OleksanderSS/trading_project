from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.causal_contracts import GraphEdgeDynamics, metadata_for_edge_type
from dean_os.schemas import utc_now_iso
from dean_os.utils import clamp, json_ready

DEFAULT_EVENT_INTERPRETATION_JSON = "reports/dean_os/domain_analyst_event_interpretation_packet_current/latest.json"
DEFAULT_DOMAIN_ID = "semiconductor_ai_infrastructure"
DEFAULT_HORIZONS = ["1d", "5d", "20d", "60d", "120d"]

REGIME_FIELDS = [
    "geopolitical_state",
    "economic_phase",
    "inflation_rates_context",
    "liquidity_credit_context",
    "market_state",
    "commodity_real_economy_stress",
    "ai_tech_cycle",
    "safe_haven_behavior",
]

NEWS_AGAINST_REGIME_QUESTIONS = [
    "Which regime indicators does this news affect?",
    "Does it confirm, weaken, or contradict the current regime?",
    "What is the first-order transmission channel?",
    "What are second-order and third-order effects?",
    "What was likely already priced?",
    "Which scenario probabilities change?",
    "Which horizons should be tracked?",
    "What evidence gaps remain?",
    "What historical analog graphs are relevant?",
    "What would falsify this interpretation?",
]


class DomainAnalystRegimeScenarioPacket:
    """Convert event interpretations into review-only regime and scenario structure.

    This is the deterministic host for the `draft/thinking` ideas: context is a
    vector, news is judged against that context, and scenarios are made
    falsifiable by horizons and evidence gaps. It does not call GPT, FinBERT, or
    a live feed; those can later supply reviewed input artifacts.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_regime_scenario_packet_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        event_interpretation_json: str | Path = DEFAULT_EVENT_INTERPRETATION_JSON,
        domain_id: str = DEFAULT_DOMAIN_ID,
        max_events: int = 20,
        horizons: list[str] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        source = _load_json(event_interpretation_json)
        packets = [
            item
            for item in source.get("event_interpretation_packets", [])
            if isinstance(item, dict)
        ][: max(1, int(max_events))]
        context_snapshot = source.get("context_regime_snapshot", {})
        resolved_horizons = horizons or DEFAULT_HORIZONS
        regime_vector = _build_regime_context_vector(context_snapshot, packets, domain_id=domain_id)
        assessments = _news_vs_regime_assessments(packets, regime_vector, resolved_horizons)
        evidence_gaps = _prioritized_evidence_gaps(packets, assessments)
        scenario_graph = _build_scenario_graph(
            packets=packets,
            regime_vector=regime_vector,
            evidence_gaps=evidence_gaps,
            horizons=resolved_horizons,
            domain_id=domain_id,
        )
        checks = _review_checks(source, packets, regime_vector, scenario_graph, evidence_gaps)
        status = _packet_status(checks, packets)
        payload = {
            "run_id": _run_id("domain_analyst_regime_scenario_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_regime_scenario_packet",
            "inputs": {
                "event_interpretation_json": str(event_interpretation_json),
                "domain_id": domain_id,
                "max_events": max_events,
                "horizons": resolved_horizons,
            },
            "summary": _summary(status, source, packets, regime_vector, scenario_graph, evidence_gaps),
            "thinking_harvest_decisions": _thinking_harvest_decisions(),
            "regime_context_vector": regime_vector,
            "news_against_regime_questions": NEWS_AGAINST_REGIME_QUESTIONS,
            "news_vs_regime_assessments": assessments,
            "scenario_outcome_graph": scenario_graph,
            "evidence_gap_priorities": evidence_gaps,
            "historical_analog_candidates": _historical_analog_candidates(packets),
            "domain_analyst_report_extension": _domain_analyst_report_extension(packets, regime_vector),
            "review_checks": checks,
            "operator_next_steps": _operator_next_steps(status, packets),
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_regime_scenario_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_regime_scenario_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Domain Analyst Regime Scenario Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Source event packets: {summary.get('source_event_packet_count')}",
        f"- Regime fields: {summary.get('regime_field_count')}",
        f"- Scenario nodes: {summary.get('scenario_node_count')}",
        f"- Scenario edges: {summary.get('scenario_edge_count')}",
        f"- Probability mass valid: {summary.get('probability_mass_valid')}",
        f"- Evidence gaps: {summary.get('evidence_gap_count')}",
        f"- Can create regime-context scenario analysis: {summary.get('can_create_regime_context_scenario_analysis')}",
        f"- Can create execution recommendation: {summary.get('can_create_execution_recommendation')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Regime Context Vector",
        "",
    ]
    vector = payload.get("regime_context_vector", {}).get("fields", {})
    for field in REGIME_FIELDS:
        item = vector.get(field, {})
        lines.append(
            f"- `{field}`: state=`{item.get('state')}` intensity={item.get('intensity')} trend=`{item.get('trend')}` confidence=`{item.get('confidence')}`"
        )
    lines.extend(["", "## Top News vs Regime Assessments", ""])
    for item in payload.get("news_vs_regime_assessments", [])[:10]:
        lines.append(f"- `{item.get('event_id')}`: {item.get('relationship_to_regime')} via {item.get('first_order_channel')}")
    lines.extend(["", "## Evidence Gaps", ""])
    for gap in payload.get("evidence_gap_priorities", [])[:12]:
        lines.append(f"- `{gap.get('gap_id')}` priority=`{gap.get('priority')}` - {gap.get('description')}")
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _build_regime_context_vector(
    context_snapshot: dict[str, Any],
    packets: list[dict[str, Any]],
    *,
    domain_id: str,
) -> dict[str, Any]:
    corpus = _context_corpus(context_snapshot, packets)
    fields = {
        "geopolitical_state": _dimension(
            "geopolitical_state",
            corpus,
            packets,
            [
                ("sanctions_chokepoint_risk", ("sanction", "export control", "tariff", "china", "taiwan", "geopolitical")),
                ("localized_war", ("war", "invasion", "conflict")),
                ("escalation", ("escalation", "restriction", "blacklist")),
            ],
            default_state="peace",
        ),
        "economic_phase": _dimension(
            "economic_phase",
            corpus,
            packets,
            [
                ("expansion", ("demand", "growth", "increase", "capex", "orders")),
                ("recession_risk", ("recession", "risk_off", "credit_tightening", "tighten")),
                ("bubble_risk", ("bubble", "crowded", "valuation")),
            ],
            default_state="stagnation",
        ),
        "inflation_rates_context": _dimension(
            "inflation_rates_context",
            corpus,
            packets,
            [
                ("higher_for_longer", ("high rates", "treasury", "yield", "higher_for_longer")),
                ("sticky_inflation", ("inflation", "cpi", "wage", "prices")),
                ("policy_uncertainty", ("fed", "policy", "uncertainty")),
            ],
            default_state="policy_uncertainty",
        ),
        "liquidity_credit_context": _dimension(
            "liquidity_credit_context",
            corpus,
            packets,
            [
                ("tight", ("credit_tightening", "tight", "higher rates")),
                ("stressed", ("stress", "liquidity", "risk_off")),
                ("loose", ("loose", "easing")),
            ],
            default_state="neutral",
        ),
        "market_state": _dimension(
            "market_state",
            corpus,
            packets,
            [
                ("risk_off", ("risk_off", "bear", "volatility", "negative")),
                ("risk_on", ("risk_on", "relative strength", "outperform", "positive")),
                ("crowded_theme", ("crowded", "ai_cycle", "capex wave")),
                ("valuation_reset", ("valuation", "multiple pressure", "rates")),
            ],
            default_state="volatile_resilient",
        ),
        "commodity_real_economy_stress": _dimension(
            "commodity_real_economy_stress",
            corpus,
            packets,
            [
                ("power_stress", ("power", "electricity", "energy")),
                ("strategic_industrial_capacity_stress", ("capacity", "bottleneck", "hbm", "packaging", "foundry")),
                ("freight_stress", ("shipping", "freight")),
            ],
            default_state="strategic_industrial_capacity_stress",
        ),
        "ai_tech_cycle": _dimension(
            "ai_tech_cycle",
            corpus,
            packets,
            [
                ("capex_boom", ("ai", "capex", "hyperscaler", "data center", "gpu")),
                ("memory_bottleneck", ("hbm", "memory")),
                ("infrastructure_bottleneck", ("advanced packaging", "capacity", "bottleneck")),
                ("valuation_bubble_risk", ("valuation", "bubble", "crowded")),
            ],
            default_state="enterprise_adoption",
        ),
        "safe_haven_behavior": _dimension(
            "safe_haven_behavior",
            corpus,
            packets,
            [
                ("cash_preference", ("risk_off", "volatility", "credit_tightening")),
                ("treasury_bid", ("treasury", "yield", "rates")),
                ("dollar_bid", ("dollar", "safe haven")),
            ],
            default_state="cash_preference",
        ),
    }
    return {
        "regime_context_vector_id": f"regime_vector:{domain_id}:{_stable_suffix(fields)}",
        "domain_id": domain_id,
        "as_of_date": _as_of_date(packets),
        "field_shape": {
            "state": "taxonomy_enum",
            "intensity": "float_0_to_1",
            "trend": ["rising", "falling", "stable", "unknown"],
            "confidence": ["low", "medium", "high"],
            "evidence_ids": "list",
            "notes": "string",
        },
        "fields": fields,
        "review_status": "regime_context_vector_ready_for_review",
        "allowed_output": "regime_context_for_review",
        "forbidden_outputs": _forbidden_outputs(),
    }


def _dimension(
    field: str,
    corpus: str,
    packets: list[dict[str, Any]],
    rules: list[tuple[str, tuple[str, ...]]],
    *,
    default_state: str,
) -> dict[str, Any]:
    best_state = default_state
    best_hits: list[str] = []
    for state, keywords in rules:
        hits = [keyword for keyword in keywords if keyword in corpus]
        if len(hits) > len(best_hits):
            best_state = state
            best_hits = hits
    evidence_ids = _matching_event_ids(packets, best_hits)
    intensity = round(clamp(len(best_hits) / 4.0, 0.05 if best_hits else 0.0, 1.0), 3)
    confidence = "high" if len(evidence_ids) >= 3 else "medium" if evidence_ids else "low"
    return {
        "state": best_state,
        "intensity": intensity,
        "trend": _trend(corpus, best_hits),
        "confidence": confidence,
        "evidence_ids": evidence_ids,
        "notes": _dimension_note(field, best_state, best_hits),
    }


def _news_vs_regime_assessments(
    packets: list[dict[str, Any]],
    regime_vector: dict[str, Any],
    horizons: list[str],
) -> list[dict[str, Any]]:
    assessments = []
    active_fields = _active_regime_fields(regime_vector)
    for packet in packets:
        event_id = packet.get("event_id")
        event_type = str(packet.get("event_type") or "other")
        affected = _affected_regime_indicators(packet, active_fields)
        relationship = _relationship_to_regime(packet, affected)
        mechanism = packet.get("mechanism_chain") or []
        first_order = mechanism[0] if mechanism else packet.get("mechanism_summary") or "requires_human_review"
        assessments.append(
            {
                "event_id": event_id,
                "title": packet.get("title"),
                "event_type": event_type,
                "affected_regime_indicators": affected,
                "relationship_to_regime": relationship,
                "first_order_channel": first_order,
                "second_order_channels": mechanism[1:3],
                "third_order_channels": mechanism[3:5],
                "likely_already_priced": _already_priced_view(packet, active_fields),
                "scenario_probability_update": _scenario_update_hint(packet, relationship),
                "horizons_to_track": _horizons_for_event(packet, horizons),
                "evidence_gaps": packet.get("evidence_gaps") or [],
                "historical_analog_candidates": _packet_analogs(packet),
                "falsification_signals": _falsification_signals(packet),
                "allowed_output": "news_vs_regime_assessment_for_review",
                "forbidden_outputs": _forbidden_outputs(),
            }
        )
    return assessments


def _build_scenario_graph(
    *,
    packets: list[dict[str, Any]],
    regime_vector: dict[str, Any],
    evidence_gaps: list[dict[str, Any]],
    horizons: list[str],
    domain_id: str,
) -> dict[str, Any]:
    as_of = regime_vector.get("as_of_date")
    primary_event = packets[0].get("event_id") if packets else "no_event_packet"
    nodes = [
        _node(
            "regime:root",
            "regime_state",
            "Current regime context vector",
            "Multi-dimensional regime state used to interpret news.",
            as_of,
            "medium",
            _all_event_ids(packets),
            "Regime vector is deterministic review structure, not a market truth claim.",
        )
    ]
    edges = []
    scenario_probs = _scenario_probabilities(packets, regime_vector)
    for packet in packets[:10]:
        event_node = f"event:{_safe_id(packet.get('event_id'))}"
        channel_node = f"channel:{_safe_id(packet.get('event_type'))}:{_safe_id(packet.get('event_id'))}"
        nodes.append(
            _node(
                event_node,
                "event",
                str(packet.get("title") or packet.get("event_id") or "event")[:120],
                str(packet.get("summary") or packet.get("mechanism_summary") or "Event requires review."),
                as_of,
                _confidence_label(packet.get("confidence_score")),
                [str(packet.get("event_id"))],
                "Event extracted from an offline event interpretation packet.",
            )
        )
        nodes.append(
            _node(
                channel_node,
                "transmission_channel",
                str(packet.get("event_type") or "transmission channel"),
                str(packet.get("mechanism_summary") or "Transmission channel requires review."),
                as_of,
                _confidence_label(packet.get("confidence_score")),
                [str(packet.get("event_id"))],
                "Mechanism is a candidate channel, not proof.",
            )
        )
        edges.append(_edge("regime:root", event_node, "conditional_update", 0.4, 0.0, "contextual", "News is interpreted against the regime vector.", [str(packet.get("event_id"))]))
        edges.append(_edge(event_node, channel_node, "causal_channel", 0.5, 0.0, "forward", "Event may transmit through this mechanism.", [str(packet.get("event_id"))]))

    for scenario_id, probability in scenario_probs.items():
        node_id = f"scenario:{scenario_id}"
        nodes.append(
            _node(
                node_id,
                "scenario",
                scenario_id.replace("_", " ").title(),
                _scenario_description(scenario_id),
                as_of,
                "medium",
                _all_event_ids(packets),
                "Probability is a review prior from packet structure, not a trading signal.",
                extra={"review_probability": probability},
            )
        )
        edges.append(_edge("regime:root", node_id, "leads_to", 0.35, probability - 0.3333, "conditional", "Scenario branch from current regime vector.", _all_event_ids(packets)))

    for horizon in horizons:
        node_id = f"self_check:{horizon}"
        nodes.append(
            _node(
                node_id,
                "self_check",
                f"Self-check horizon {horizon}",
                "Review whether observed path matched expected mechanisms, not only direction.",
                as_of,
                "medium",
                _all_event_ids(packets),
                "Outcome check is pending until the horizon matures.",
                extra={"horizon": horizon, "review_status": "pending_future_outcome"},
            )
        )
        edges.append(_edge("scenario:base_case_continuation", node_id, "calibrates", 0.2, 0.0, "future_review", "Scenario will be checked at this horizon.", _all_event_ids(packets)))

    graph = {
        "scenario_graph_id": f"scenario_graph:{domain_id}:{_stable_suffix(nodes)}",
        "as_of_date": as_of,
        "root_regime_snapshot_id": regime_vector.get("regime_context_vector_id"),
        "event_id": primary_event,
        "nodes": nodes,
        "edges": edges,
        "horizons": horizons,
        "scenario_probabilities": scenario_probs,
        "probability_mass_check": {
            "sum": round(sum(scenario_probs.values()), 6),
            "valid": abs(sum(scenario_probs.values()) - 1.0) < 0.000001,
            "rule": "scenario sibling probabilities must sum to one",
        },
        "evidence_gaps": evidence_gaps,
        "review_status": "scenario_outcome_graph_ready_for_review",
        "constraints": [
            "acyclic_per_as_of_packet",
            "probability_mass_sums_to_one_for_scenario_siblings",
            "missing_evidence_is_explicit",
            "no_future_evidence_allowed",
            "review_only_no_execution",
        ],
        "allowed_output": "scenario_outcome_graph_for_review",
        "forbidden_outputs": _forbidden_outputs(),
    }
    return graph


def _review_checks(
    source: dict[str, Any],
    packets: list[dict[str, Any]],
    regime_vector: dict[str, Any],
    scenario_graph: dict[str, Any],
    evidence_gaps: list[dict[str, Any]],
) -> list[dict[str, str]]:
    checks = [
        _check("pass" if source.get("mode") == "domain_analyst_event_interpretation_packet" else "warn", "source_event_interpretation_packet", str(source.get("mode"))),
        _check("pass" if packets else "warn", "event_packets_present", f"{len(packets)} event packets available."),
        _check("pass" if set(REGIME_FIELDS).issubset(set(regime_vector.get("fields", {}))) else "fail", "regime_context_vector_required_fields", "Regime vector fields are present."),
        _check("pass" if scenario_graph.get("probability_mass_check", {}).get("valid") else "fail", "scenario_probability_mass_valid", str(scenario_graph.get("probability_mass_check", {}).get("sum"))),
        _check("pass" if _graph_is_acyclic(scenario_graph) else "fail", "scenario_graph_acyclic", "Graph edge order is acyclic."),
        _check("pass" if evidence_gaps else "warn", "evidence_gaps_explicit", f"{len(evidence_gaps)} evidence gaps listed."),
        _check("pass", "review_only_boundary", "Packet forbids execution, buy/sell/hold, sizing, allocation, broker routing, and live trading."),
    ]
    return checks


def _summary(
    status: str,
    source: dict[str, Any],
    packets: list[dict[str, Any]],
    regime_vector: dict[str, Any],
    scenario_graph: dict[str, Any],
    evidence_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    probabilities = scenario_graph.get("scenario_probabilities", {})
    return {
        "packet_status": status,
        "domain_id": regime_vector.get("domain_id"),
        "source_packet_status": source.get("summary", {}).get("packet_status"),
        "source_event_packet_count": len(packets),
        "regime_field_count": len(regime_vector.get("fields", {})),
        "scenario_node_count": len(scenario_graph.get("nodes", [])),
        "scenario_edge_count": len(scenario_graph.get("edges", [])),
        "scenario_probability_count": len(probabilities),
        "probability_mass_sum": scenario_graph.get("probability_mass_check", {}).get("sum"),
        "probability_mass_valid": scenario_graph.get("probability_mass_check", {}).get("valid"),
        "evidence_gap_count": len(evidence_gaps),
        "review_required_event_count": sum(1 for item in packets if item.get("review_required")),
        "can_create_regime_context_vector": True,
        "can_create_news_vs_regime_analysis": True,
        "can_create_scenario_outcome_graph": True,
        "can_create_regime_context_scenario_analysis": True,
        "can_create_self_check_horizons": True,
        "can_use_gpt_or_finbert_inputs_later_if_saved_as_review_evidence": True,
        "can_call_gpt_or_finbert_now": False,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _packet_status(checks: list[dict[str, str]], packets: list[dict[str, Any]]) -> str:
    if any(check.get("status") == "fail" for check in checks):
        return "blocked_domain_analyst_regime_scenario_packet"
    if any(check.get("status") == "warn" for check in checks) or any(packet.get("review_required") for packet in packets):
        return "domain_analyst_regime_scenario_ready_with_review_items"
    return "domain_analyst_regime_scenario_ready"


def _thinking_harvest_decisions() -> list[dict[str, str]]:
    return [
        {
            "source_file": "draft/thinking/.../regime_context_scenario_graph_spec_2026-06-24.json",
            "decision": "integrated_as_review_schema",
            "note": "RegimeContextVector, ScenarioOutcomeGraph, probability mass check, evidence gaps, horizons, and no-execution boundary were integrated.",
        },
        {
            "source_file": "draft/thinking/.../daily_briefing_analyst_notes_spec_2026-06-26.json",
            "decision": "integrated_as_report_extension",
            "note": "Demand/cost/margin/inflation/rates/power/supply-chain/valuation channels and self-check horizons were added as review fields.",
        },
        {
            "source_file": "draft/thinking/.../additional_analyst_observations_spec_2026-06-24.json",
            "decision": "adapted_partially",
            "note": "Already-priced, conflicting news, crowding, and prioritized evidence gaps were kept as review diagnostics.",
        },
        {
            "source_file": "draft/thinking codex module tree",
            "decision": "not_integrated_yet",
            "note": "No graph store, LLM caller, FinBERT caller, autonomous memory writer, or production config hook was added.",
        },
    ]


def _domain_analyst_report_extension(packets: list[dict[str, Any]], regime_vector: dict[str, Any]) -> dict[str, Any]:
    text = _packet_corpus(packets)
    return {
        "demand_channel": _channel_note(text, ("demand", "orders", "gpu", "data center", "hyperscaler")),
        "cost_channel": _channel_note(text, ("cost", "price", "power", "energy", "capacity")),
        "margin_channel": _channel_note(text, ("margin", "pricing", "shortage", "capacity")),
        "inflation_channel": regime_vector.get("fields", {}).get("inflation_rates_context", {}),
        "rates_channel": regime_vector.get("fields", {}).get("liquidity_credit_context", {}),
        "power_channel": regime_vector.get("fields", {}).get("commodity_real_economy_stress", {}),
        "supply_chain_security_channel": regime_vector.get("fields", {}).get("geopolitical_state", {}),
        "valuation_expectation_gap": {
            "status": "requires_market_confirmation",
            "notes": "Check whether AI/capex optimism or policy risk is already reflected in relative performance and earnings revisions.",
        },
        "self_check_horizons": DEFAULT_HORIZONS,
        "allowed_output": "report_extension_for_review",
        "forbidden_outputs": _forbidden_outputs(),
    }


def _prioritized_evidence_gaps(
    packets: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    sources: dict[str, set[str]] = {}
    for packet in packets:
        for gap in packet.get("evidence_gaps") or []:
            text = str(gap)
            counter[text] += 1
            sources.setdefault(text, set()).add(str(packet.get("event_id")))
    for assessment in assessments:
        for signal in assessment.get("falsification_signals") or []:
            text = f"Falsification signal to track: {signal}"
            counter[text] += 1
            sources.setdefault(text, set()).add(str(assessment.get("event_id")))
    if not counter:
        counter["Add corroborating source evidence before changing thesis confidence."] = 1
        sources["Add corroborating source evidence before changing thesis confidence."] = {"packet_level"}
    gaps = []
    for index, (description, count) in enumerate(counter.most_common(), start=1):
        gaps.append(
            {
                "gap_id": f"gap:{index}",
                "description": description,
                "priority": "high" if count >= 3 else "medium" if count == 2 else "low",
                "affected_event_ids": sorted(sources.get(description, set())),
                "importance_to_scenario_probability": "high" if count >= 3 else "medium",
                "allowed_output": "evidence_request_for_review",
            }
        )
    return gaps


def _historical_analog_candidates(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analogs = []
    seen: set[str] = set()
    for packet in packets:
        for analog in _packet_analogs(packet):
            key = str(analog)
            if key in seen:
                continue
            seen.add(key)
            analogs.append(
                {
                    "analog_id": key,
                    "source_event_id": packet.get("event_id"),
                    "false_analogy_risk": "requires_human_review",
                    "allowed_output": "historical_analog_candidate_for_review",
                }
            )
    return analogs


def _packet_analogs(packet: dict[str, Any]) -> list[str]:
    context = packet.get("pipeline_news_context", {})
    analogs = []
    for item in context.get("crisis_pattern_matches") or []:
        if isinstance(item, dict):
            analogs.append(str(item.get("pattern_id") or item.get("display_name") or item))
        else:
            analogs.append(str(item))
    for item in context.get("learned_pattern_matches") or []:
        if isinstance(item, dict):
            analogs.append(str(item.get("pattern_id") or item.get("sample_events") or item))
        else:
            analogs.append(str(item))
    return analogs


def _context_corpus(context_snapshot: dict[str, Any], packets: list[dict[str, Any]]) -> str:
    values = [
        context_snapshot,
        [packet.get("title") for packet in packets],
        [packet.get("summary") for packet in packets],
        [packet.get("context_conditioned_interpretation") for packet in packets],
        [packet.get("pipeline_news_context") for packet in packets],
    ]
    return " ".join(_flatten_strings(values)).lower()


def _packet_corpus(packets: list[dict[str, Any]]) -> str:
    return " ".join(_flatten_strings(packets)).lower()


def _matching_event_ids(packets: list[dict[str, Any]], keywords: list[str]) -> list[str]:
    if not keywords:
        return []
    ids = []
    for packet in packets:
        text = " ".join(_flatten_strings(packet)).lower()
        if any(keyword in text for keyword in keywords):
            ids.append(str(packet.get("event_id")))
    return ids


def _active_regime_fields(regime_vector: dict[str, Any]) -> dict[str, Any]:
    fields = regime_vector.get("fields", {})
    return {
        key: value
        for key, value in fields.items()
        if isinstance(value, dict) and float(value.get("intensity") or 0.0) > 0
    }


def _affected_regime_indicators(packet: dict[str, Any], active_fields: dict[str, Any]) -> list[str]:
    event_type = str(packet.get("event_type") or "")
    mapping = {
        "sanctions": ["geopolitical_state", "liquidity_credit_context", "market_state"],
        "tariff": ["geopolitical_state", "inflation_rates_context", "market_state"],
        "rate_policy": ["inflation_rates_context", "liquidity_credit_context", "market_state"],
        "demand_driver": ["economic_phase", "ai_tech_cycle", "market_state"],
        "capex_signal": ["economic_phase", "ai_tech_cycle", "commodity_real_economy_stress"],
        "capacity_change": ["commodity_real_economy_stress", "ai_tech_cycle", "margin_channel"],
        "supply_disruption": ["commodity_real_economy_stress", "geopolitical_state", "inflation_rates_context"],
    }
    fields = mapping.get(event_type, ["market_state"])
    return [field for field in fields if field in active_fields or field == "margin_channel"]


def _relationship_to_regime(packet: dict[str, Any], affected: list[str]) -> str:
    sentiment = str(packet.get("sentiment", {}).get("label") or "")
    tags = set(packet.get("context_conditioned_interpretation", {}).get("context_tags") or [])
    if "pipeline_macro_overlay_requires_scenario_split" in packet.get("context_conditioned_interpretation", {}).get("review_flags", []):
        return "mixed_requires_scenario_split"
    if sentiment == "mixed":
        return "conflicting_news_requires_review"
    if tags.intersection({"pipeline_risk_off", "inflation_or_rate_pressure", "war_sanctions_tension"}):
        return "confirms_or_amplifies_existing_risk_regime"
    if sentiment == "positive" and affected:
        return "confirms_growth_or_ai_cycle_regime"
    return "ambiguous_requires_more_evidence"


def _already_priced_view(packet: dict[str, Any], active_fields: dict[str, Any]) -> dict[str, Any]:
    tags = set(packet.get("context_conditioned_interpretation", {}).get("context_tags") or [])
    crowded = "market_state" in active_fields and active_fields["market_state"].get("state") in {"crowded_theme", "risk_on"}
    return {
        "status": "requires_market_confirmation" if crowded or "ai_cycle" in tags else "unknown",
        "watch_metrics": sorted(set(packet.get("watch_metrics") or []) | {"sector_relative_performance", "earnings_revisions", "valuation_multiples"}),
        "notes": "Already-priced detector is review-only and needs market/earnings confirmation.",
    }


def _scenario_update_hint(packet: dict[str, Any], relationship: str) -> dict[str, Any]:
    if "risk" in relationship or "conflicting" in relationship:
        return {"base_case_continuation": -0.05, "downside_constraint": 0.05}
    if "growth" in relationship:
        return {"base_case_continuation": -0.05, "upside_acceleration": 0.05}
    return {"base_case_continuation": 0.0}


def _horizons_for_event(packet: dict[str, Any], horizons: list[str]) -> list[str]:
    horizon = str(packet.get("time_horizon") or "")
    if "immediate" in horizon or "short" in horizon:
        return [item for item in horizons if item in {"1d", "5d", "20d"}]
    return horizons


def _falsification_signals(packet: dict[str, Any]) -> list[str]:
    event_type = str(packet.get("event_type") or "")
    base = {
        "demand_driver": ["orders fail to appear in backlog", "earnings revisions do not improve"],
        "capex_signal": ["capex plans are delayed or revised down", "supplier lead times do not tighten"],
        "capacity_change": ["capacity bottleneck resolves faster than expected", "pricing power fades"],
        "rate_policy": ["rates fall without growth confirmation", "credit spreads tighten against the risk thesis"],
        "sanctions": ["licensed sales continue unaffected", "supply chain reroutes without margin impact"],
        "tariff": ["tariff exemptions offset cost impact", "demand absorbs pricing without volume damage"],
    }
    return base.get(event_type, ["source evidence contradicts the mechanism", "observable watch metrics do not move by the review horizon"])


def _scenario_probabilities(packets: list[dict[str, Any]], regime_vector: dict[str, Any]) -> dict[str, float]:
    text = _packet_corpus(packets)
    upside = 0.25
    downside = 0.25
    if any(word in text for word in ("demand", "orders", "capex", "growth", "outperform")):
        upside += 0.05
        downside -= 0.05
    if any(word in text for word in ("risk", "sanction", "tariff", "inflation", "credit", "restriction")):
        downside += 0.05
        upside -= 0.05
    market_state = regime_vector.get("fields", {}).get("market_state", {}).get("state")
    if market_state == "risk_off":
        downside += 0.05
        upside -= 0.05
    upside = round(clamp(upside, 0.15, 0.35), 2)
    downside = round(clamp(downside, 0.15, 0.35), 2)
    base = round(1.0 - upside - downside, 2)
    return {
        "base_case_continuation": base,
        "upside_acceleration": upside,
        "downside_constraint": downside,
    }


def _node(
    node_id: str,
    node_type: str,
    label: str,
    description: str,
    as_of_date: str | None,
    confidence: str,
    evidence_ids: list[str],
    uncertainty_notes: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node = {
        "node_id": node_id,
        "node_type": node_type,
        "label": label,
        "description": description,
        "as_of_date": as_of_date,
        "confidence": confidence,
        "evidence_ids": evidence_ids,
        "uncertainty_notes": uncertainty_notes,
    }
    if extra:
        node.update(extra)
    return node


def _edge(
    source: str,
    target: str,
    edge_type: str,
    weight: float,
    probability_delta: float,
    direction: str,
    rationale: str,
    evidence_ids: list[str],
) -> dict[str, Any]:
    causal_metadata = metadata_for_edge_type(edge_type)
    dynamics = GraphEdgeDynamics(
        strength=weight,
        lag_label="review_horizon_dependent",
        estimate_confidence=0.6,
        edge_reliability=0.6,
        evidence_count=len(evidence_ids),
        activation_state="candidate",
    )
    return {
        "edge_id": f"edge:{_safe_id(source)}:{_safe_id(target)}:{edge_type}",
        "source_node_id": source,
        "target_node_id": target,
        "edge_type": edge_type,
        "weight": round(weight, 3),
        "probability_delta": round(probability_delta, 4),
        "direction": direction,
        "rationale": rationale,
        "evidence_ids": evidence_ids,
        "lag_assumption": "review_horizon_dependent",
        "confidence": "medium",
        "causal_metadata": causal_metadata.model_dump(mode="json"),
        "dynamics": dynamics.model_dump(mode="json"),
    }


def _graph_is_acyclic(graph: dict[str, Any]) -> bool:
    order = {node.get("node_id"): index for index, node in enumerate(graph.get("nodes", []))}
    for edge in graph.get("edges", []):
        source = edge.get("source_node_id")
        target = edge.get("target_node_id")
        if source in order and target in order and order[source] >= order[target]:
            return False
    return True


def _scenario_description(scenario_id: str) -> str:
    descriptions = {
        "base_case_continuation": "Current mixed regime remains dominant; watch whether evidence confirms the thesis mechanism.",
        "upside_acceleration": "AI/capex/demand channels strengthen faster than constraints.",
        "downside_constraint": "Policy, rates, capacity, or crowding constraints dominate the positive thesis.",
    }
    return descriptions.get(scenario_id, "Scenario requires review.")


def _channel_note(text: str, keywords: tuple[str, ...]) -> dict[str, Any]:
    hits = [keyword for keyword in keywords if keyword in text]
    return {
        "status": "present" if hits else "not_observed_in_packet",
        "evidence_keywords": hits,
        "confidence": "medium" if len(hits) >= 2 else "low",
    }


def _dimension_note(field: str, state: str, hits: list[str]) -> str:
    if hits:
        return f"{field} mapped to {state} from packet keywords: {', '.join(hits[:6])}."
    return f"{field} uses low-confidence default state {state}; add direct evidence before increasing confidence."


def _trend(corpus: str, hits: list[str]) -> str:
    if not hits:
        return "unknown"
    if any(word in corpus for word in ("increase", "rising", "expands", "tighten", "escalation")):
        return "rising"
    if any(word in corpus for word in ("falling", "decline", "easing", "de-escalation")):
        return "falling"
    return "stable"


def _confidence_label(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "low"
    if number >= 0.7:
        return "high"
    if number >= 0.45:
        return "medium"
    return "low"


def _all_event_ids(packets: list[dict[str, Any]]) -> list[str]:
    return [str(packet.get("event_id")) for packet in packets if packet.get("event_id")]


def _as_of_date(packets: list[dict[str, Any]]) -> str | None:
    dates = sorted(str(packet.get("event_date") or packet.get("published_at")) for packet in packets if packet.get("event_date") or packet.get("published_at"))
    return dates[-1] if dates else None


def _operator_next_steps(status: str, packets: list[dict[str, Any]]) -> list[str]:
    if not packets:
        return ["Supply or build a DomainAnalystEventInterpretationPacket with local news/report documents before scenario review."]
    return [
        "Review the regime vector and scenario graph before using event mechanisms in the thesis packet.",
        "Use the self-check horizons as future outcome review anchors; do not write learning memory until outcomes and human labels exist.",
        "If GPT or FinBERT is added later, save their outputs as source evidence and rerun this packet rather than letting them mutate memory directly.",
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No GPT, FinBERT, web, or external API call is made.",
        "No source, claim, or entity extraction is executed.",
        "No learning memory, analyst weights, model training, tuning, or production config write is performed.",
        "No execution, buy/sell/hold, allocation, position sizing, price target, paper order, broker route, or live trade recommendation is generated.",
    ]


def _forbidden_outputs() -> list[str]:
    return [
        "buy_sell_hold",
        "position_sizing",
        "allocation",
        "price_target",
        "broker_route",
        "paper_order",
        "live_order",
        "autonomous_learning_write",
        "production_config_write",
    ]


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"mode": "missing_event_interpretation_packet", "summary": {"packet_status": "missing"}, "event_interpretation_packets": []}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _flatten_strings(values: Any) -> list[str]:
    if isinstance(values, dict):
        return [item for value in values.values() for item in _flatten_strings(value)]
    if isinstance(values, (list, tuple, set)):
        return [item for value in values for item in _flatten_strings(value)]
    if isinstance(values, str):
        return [values]
    return []


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _safe_id(value: Any) -> str:
    text = str(value or "unknown").lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:80] or "unknown"


def _stable_suffix(value: Any) -> str:
    text = json.dumps(json_ready(value), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
