from __future__ import annotations

import pytest
from pydantic import ValidationError

from dean_os.analyst_core.lens_contract import AnalysisPacket
from dean_os.analyst_core.lenses.transmission_mapper_lens import (
    TransmissionMapperLens,
)
from dean_os.analyst_core.schemas import ScenarioEdge, ScenarioNode
from dean_os.causal_contracts import CausalClaimMetadata
from dean_os.dependency_graph import DependencyEdge
from dean_os.event_causal_graph import EventCausalGraphBuilder


def test_association_and_sequence_cannot_authorize_causal_claims() -> None:
    for relation in ("statistical_association", "temporal_sequence"):
        with pytest.raises(ValidationError):
            CausalClaimMetadata(
                relation_type=relation,
                identification_method="event_study",
                causal_claim_allowed=True,
            )


def test_assumed_mechanism_is_not_causal_identification() -> None:
    with pytest.raises(ValidationError):
        CausalClaimMetadata(
            relation_type="economic_transmission",
            identification_method="assumed_mechanism",
            causal_claim_allowed=True,
        )


def test_scenario_separates_probability_confidence_and_impacts() -> None:
    node = ScenarioNode(
        node_type="scenario",
        label="restricted GPU exports",
        probability=0.35,
        confidence="low",
        impact=-0.6,
        market_reaction=-0.2,
        fundamental_change=-0.5,
    )
    edge = ScenarioEdge(
        source_node_id="event",
        target_node_id="scenario",
        edge_type="observed_after",
        causal_metadata=CausalClaimMetadata(
            relation_type="temporal_sequence",
            causal_claim_allowed=False,
        ),
    )

    assert node.probability == 0.35
    assert node.probability_kind == "review_prior"
    assert node.confidence == "low"
    assert node.impact == -0.6
    assert node.market_reaction == -0.2
    assert node.fundamental_change == -0.5
    assert edge.causal_metadata.causal_claim_allowed is False


def test_transmission_lens_labels_template_as_unidentified_mechanism() -> None:
    packet = AnalysisPacket(
        packet_id="packet-causal-contract",
        domain_id="semiconductor_ai_infrastructure",
        classified_events=[
            {
                "event_id": "e1",
                "event_class": "sanctions_change",
                "directness": "direct",
                "materiality_score": 0.8,
                "affected_sectors": ["semiconductors"],
            }
        ],
    )

    delta = TransmissionMapperLens().analyze(packet)
    metadata = delta.transmission_channels_added[0]["causal_metadata"]

    assert metadata["relation_type"] == "economic_transmission"
    assert metadata["identification_method"] == "assumed_mechanism"
    assert metadata["causal_claim_allowed"] is False


def test_legacy_edge_labels_are_conservatively_classified() -> None:
    temporal = ScenarioEdge(
        source_node_id="a",
        target_node_id="b",
        edge_type="observed_after",
    )
    transmission = ScenarioEdge(
        source_node_id="a",
        target_node_id="b",
        edge_type="causal_channel",
    )
    dependency = DependencyEdge.model_validate(
        {"from": "fab", "to": "packaging", "type": "structural"}
    )

    assert temporal.causal_metadata.relation_type == "temporal_sequence"
    assert transmission.causal_metadata.relation_type == "economic_transmission"
    assert transmission.causal_metadata.causal_claim_allowed is False
    assert dependency.causal_metadata.relation_type == "economic_transmission"
    assert dependency.causal_metadata.identification_method == "assumed_mechanism"
    assert dependency.dynamics.strength == 0.5
    assert dependency.dynamics.lag_label == "months"
    assert dependency.dynamics.estimate_confidence == 0.5


def test_event_graph_does_not_use_event_confidence_as_event_probability() -> None:
    class Event:
        headline = "Earthquake interrupts Taiwan production"
        event_type = "natural_disaster"
        shock = "negative"
        shock_confidence = 0.62
        impact = -0.8
        predictability = 0.2
        affected_sectors = ["semiconductors"]

    graph = EventCausalGraphBuilder(min_probability=0.1).build(Event())
    trigger = next(node for node in graph.nodes if node.depth == 0)

    assert trigger.probability == 1.0
    assert trigger.probability_kind == "observed_event"
    assert trigger.estimate_confidence == 0.62
    assert trigger.impact_magnitude == 0.8
    assert all(
        edge.causal_metadata.causal_claim_allowed is False
        for edge in graph.edges
    )
    assert all(edge.dynamics.activation_state == "candidate" for edge in graph.edges)
    assert all(edge.dynamics.lag_label != "unknown" for edge in graph.edges)
    assert all(
        edge.causal_metadata.relation_type
        in {"economic_transmission", "hypothesis_only"}
        for edge in graph.edges
    )
