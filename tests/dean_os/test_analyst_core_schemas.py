"""Validation tests for the analyst-core schemas nucleus (Phase 1).

These encode the invariants from the analyst design notes
(note 03 §15, note 04 §10, note 07). They are the contract that every
future lens and Phase 2 module must respect.

All tests run with NO LLM and NO network — the analyst core is deterministic.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from dean_os.analyst_core import (
    OUTCOME_HORIZONS,
    REGIME_DIMENSIONS,
    AnalysisPacket,
    AnalystLens,
    Confidence,
    EvidenceGap,
    HistoricalOutcomeCheck,
    HorizonOutcome,
    HypothesisLedgerEntry,
    HypothesisStatus,
    LensRegistry,
    ModuleDelta,
    Priority,
    RegimeContextVector,
    RegimeDimensionState,
    ScenarioEdge,
    ScenarioNode,
    ScenarioOutcomeGraph,
    Trend,
)
from dean_os.analyst_core.lenses.regime_context_lens import (
    DEFAULT_DIMENSION_STATE,
    EVENT_CLASS_TO_DIMENSION,
    RegimeContextLens,
)


# ──────────────────────────────────────────────────────────────────────────────
# RegimeContextVector
# ──────────────────────────────────────────────────────────────────────────────


def test_regime_context_vector_has_required_fields():
    """Vector must carry all 8 dimensions, even when no evidence is supplied."""
    vector = RegimeContextVector()
    assert set(vector.dimensions.keys()) == set(REGIME_DIMENSIONS)
    for dim in vector.dimensions.values():
        assert isinstance(dim, RegimeDimensionState)
        assert 0.0 <= dim.intensity <= 1.0
    # A blank vector is low-confidence by construction.
    assert vector.confidence == Confidence.LOW
    assert vector.safety.get("can_trade") is False


def test_regime_context_vector_rejects_unknown_dimension():
    """Callers cannot silently extend the vector with ad-hoc dimensions."""
    with pytest.raises(ValidationError):
        RegimeContextVector(
            dimensions={
                "made_up_dimension": RegimeDimensionState(state="x"),
            }
        )


# ──────────────────────────────────────────────────────────────────────────────
# ScenarioOutcomeGraph
# ──────────────────────────────────────────────────────────────────────────────


def _scenario_nodes(probabilities: list[float]) -> list[ScenarioNode]:
    return [
        ScenarioNode(node_type="scenario", label=f"scenario_{i}", probability=p)
        for i, p in enumerate(probabilities)
    ]


def test_scenario_graph_probability_mass_sums_to_one():
    """note 03 §15: scenario node probabilities must sum to 1.0."""
    graph = ScenarioOutcomeGraph(nodes=_scenario_nodes([0.45, 0.30, 0.25]))
    assert graph.probability_mass_check is True


def test_scenario_graph_rejects_non_unit_probability_mass():
    """Probabilities that drift from 1.0 must fail loudly."""
    with pytest.raises(ValidationError):
        ScenarioOutcomeGraph(nodes=_scenario_nodes([0.5, 0.5, 0.5]))


def test_scenario_graph_probability_only_on_scenario_nodes():
    """Non-scenario nodes cannot carry a probability (note 03 §4)."""
    with pytest.raises(ValidationError):
        ScenarioNode(node_type="event", label="e", probability=0.5)


def test_scenario_graph_is_acyclic():
    """note 03 §15: the graph must be acyclic."""
    nodes = [
        ScenarioNode(node_type="regime_state", label="root"),
        ScenarioNode(node_type="event", label="event"),
        ScenarioNode(node_type="scenario", label="s1", probability=1.0),
    ]
    edges = [
        ScenarioEdge(source_node_id=nodes[0].node_id, target_node_id=nodes[1].node_id, edge_type="conditional_update"),
        ScenarioEdge(source_node_id=nodes[1].node_id, target_node_id=nodes[2].node_id, edge_type="leads_to"),
    ]
    graph = ScenarioOutcomeGraph(nodes=nodes, edges=edges)
    assert graph.probability_mass_check is True  # valid graph constructs


def test_scenario_graph_rejects_cycle():
    """A cycle must be rejected by Kahn's algorithm."""
    a = ScenarioNode(node_type="regime_state", label="a")
    b = ScenarioNode(node_type="event", label="b")
    edges = [
        ScenarioEdge(source_node_id=a.node_id, target_node_id=b.node_id, edge_type="leads_to"),
        ScenarioEdge(source_node_id=b.node_id, target_node_id=a.node_id, edge_type="leads_to"),
    ]
    with pytest.raises(ValidationError):
        ScenarioOutcomeGraph(nodes=[a, b], edges=edges)


def test_scenario_graph_rejects_self_loop():
    """Edges cannot reference the same node on both ends."""
    a = ScenarioNode(node_type="regime_state", label="a")
    with pytest.raises(ValidationError):
        ScenarioEdge(source_node_id=a.node_id, target_node_id=a.node_id, edge_type="leads_to")


def test_scenario_graph_rejects_dangling_edge():
    """Edges must reference existing node ids (referential integrity)."""
    a = ScenarioNode(node_type="regime_state", label="a")
    with pytest.raises(ValidationError):
        ScenarioOutcomeGraph(
            nodes=[a],
            edges=[ScenarioEdge(source_node_id=a.node_id, target_node_id="ghost", edge_type="leads_to")],
        )


# ──────────────────────────────────────────────────────────────────────────────
# HypothesisLedgerEntry
# ──────────────────────────────────────────────────────────────────────────────


def test_hypothesis_requires_invalidation_signals():
    """note 07 §12: a hypothesis without a falsification path is not testable."""
    with pytest.raises(ValidationError):
        HypothesisLedgerEntry(hypothesis="oil will fall", invalidation_signals=[])


def test_hypothesis_has_invalidation_signals():
    entry = HypothesisLedgerEntry(
        hypothesis="oil risk premium compresses",
        invalidation_signals=["tanker flow resumes disruption", "insurance rates spike"],
        horizons_to_check=(1, 5, 20),
    )
    assert entry.status == HypothesisStatus.OPEN
    assert entry.horizons_to_check == (1, 5, 20)
    assert entry.safety.get("can_trade") is False


# ──────────────────────────────────────────────────────────────────────────────
# EvidenceGap
# ──────────────────────────────────────────────────────────────────────────────


def test_evidence_gap_has_priority():
    """note 07 §10: gaps are ranked by importance to scenario probability."""
    gap = EvidenceGap(
        description="real tanker flow data missing",
        importance_to_scenario_probability=Priority.HIGH,
        priority=Priority.HIGH,
    )
    assert gap.priority == Priority.HIGH
    assert gap.safety.get("can_trade") is False


def test_evidence_gap_rejects_empty_description():
    with pytest.raises(ValidationError):
        EvidenceGap(description="   ")


# ──────────────────────────────────────────────────────────────────────────────
# HistoricalOutcomeCheck
# ──────────────────────────────────────────────────────────────────────────────


def test_historical_outcome_has_fixed_horizons():
    """note 03 §6.10: outcomes are recorded at fixed horizons {1,5,20,60,120}."""
    check = HistoricalOutcomeCheck(
        winning_scenario="oil_falls",
        outcomes_by_horizon={
            1: HorizonOutcome(horizon_days=1, observed={"oil": "down"}),
            120: HorizonOutcome(horizon_days=120, observed={"oil": "down", "inflation": "fade"}),
        },
    )
    assert set(check.outcomes_by_horizon.keys()).issubset(set(OUTCOME_HORIZONS))
    assert check.false_analogy_risk is not None


def test_historical_outcome_rejects_non_fixed_horizon():
    """Arbitrary horizons (e.g. 7d) break the evaluation contract."""
    with pytest.raises(ValidationError):
        HorizonOutcome(horizon_days=7, observed={})


# ──────────────────────────────────────────────────────────────────────────────
# AnalysisPacket + ModuleDelta + review-only boundary
# ──────────────────────────────────────────────────────────────────────────────


def test_analysis_packet_is_always_review_only():
    """note 04 §10: the packet cannot carry execution authority."""
    with pytest.raises(ValidationError):
        AnalysisPacket(packet_id="p1", review_only=False)


def test_module_delta_is_always_review_only():
    with pytest.raises(ValidationError):
        ModuleDelta(module_name="x", review_only=False)


def test_review_only_boundary_blocks_trading_outputs():
    """forbidden_outputs must include all execution-related outputs."""
    packet = AnalysisPacket(packet_id="p1")
    forbidden = set(packet.forbidden_outputs)
    assert {"buy", "sell", "hold", "position_sizing", "live_order"}.issubset(forbidden)


# ──────────────────────────────────────────────────────────────────────────────
# Lens contract: AnalystLens + LensRegistry
# ──────────────────────────────────────────────────────────────────────────────


class _NoopLens(AnalystLens):
    """Minimal lens for testing the plugin contract."""

    lens_name = "noop"
    event_classes_supported = ("oil_shock", "inflation_release")

    def analyze(self, packet, config=None):
        return ModuleDelta(module_name=self.lens_name, fields_added=[])


def test_lens_returns_delta_not_full_state():
    """A lens MUST return a ModuleDelta, never mutate the packet."""
    packet = AnalysisPacket(packet_id="p1")
    delta = _NoopLens().analyze(packet)
    assert isinstance(delta, ModuleDelta)
    # Packet was not mutated by the lens.
    assert packet.regime_context is None


def test_lens_registry_discovers_by_event_class():
    """note 04 §8: registry returns lenses relevant to an event class."""
    registry = LensRegistry()
    registry.register(_NoopLens())
    # oil_shock is explicitly supported.
    assert len(registry.lenses_for_event_class("oil_shock")) == 1
    # An unrelated event class returns no lenses (this lens is not wildcard).
    assert registry.lenses_for_event_class("war_escalation") == []


def test_lens_registry_rejects_non_lens():
    registry = LensRegistry()
    with pytest.raises(TypeError):
        registry.register("not-a-lens")  # type: ignore[arg-type]


def test_lens_registry_rejects_duplicate():
    registry = LensRegistry()
    registry.register(_NoopLens())
    with pytest.raises(ValueError):
        registry.register(_NoopLens())


# ──────────────────────────────────────────────────────────────────────────────
# RegimeContextLens — first concrete lens (proves the pattern end to end)
# ──────────────────────────────────────────────────────────────────────────────


def test_regime_context_lens_produces_full_vector():
    """The lens must fill all 8 dimensions even from sparse input."""
    packet = AnalysisPacket(
        packet_id="p1",
        event_records=[{"event_class": "oil_shock", "intensity": 0.8, "trend": "rising"}],
    )
    delta = RegimeContextLens().analyze(packet)
    assert delta.regime_context is not None
    assert set(delta.regime_context.dimensions.keys()) == set(REGIME_DIMENSIONS)
    # oil_shock maps to commodity_stress.
    commodity = delta.regime_context.dimensions["commodity_stress"]
    assert commodity.trend == Trend.RISING
    assert commodity.intensity == pytest.approx(0.8)
    # Untouched dimensions fall back to conservative defaults.
    geo = delta.regime_context.dimensions["geopolitical_state"]
    assert geo.state == DEFAULT_DIMENSION_STATE["geopolitical_state"]


def test_regime_context_lens_degrades_gracefully_on_empty():
    """No events → low-confidence all-default vector (no crash)."""
    packet = AnalysisPacket(packet_id="p1", event_records=[])
    delta = RegimeContextLens().analyze(packet)
    assert delta.regime_context is not None
    assert delta.regime_context.confidence == Confidence.LOW
    assert delta.regime_context.safety.get("can_trade") is False


def test_event_class_routing_table_covers_note_examples():
    """Sanity: the routing table covers the event classes from note 04 §7."""
    # Every value in the routing table must be a valid regime dimension.
    for dimension in EVENT_CLASS_TO_DIMENSION.values():
        assert dimension in REGIME_DIMENSIONS
