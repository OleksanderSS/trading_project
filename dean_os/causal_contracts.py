from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


CausalRelationType = Literal[
    "physical_dependency",
    "economic_transmission",
    "statistical_association",
    "temporal_sequence",
    "historical_analogy",
    "hypothesis_only",
]

CausalIdentificationMethod = Literal[
    "none",
    "assumed_mechanism",
    "structural_domain_constraint",
    "event_study",
    "difference_in_differences",
    "instrumental_variable",
    "regression_discontinuity",
    "randomized_intervention",
]


class GraphEdgeDynamics(BaseModel):
    """Operational state of an edge, separate from its causal meaning."""

    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    lag_value: float | None = Field(default=None, ge=0.0)
    lag_unit: Literal[
        "bars", "minutes", "hours", "days", "weeks", "months",
        "quarters", "years", "unknown",
    ] = "unknown"
    lag_label: str = "unknown"
    persistence: float | None = Field(default=None, ge=0.0, le=1.0)
    estimate_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    regime_dependencies: list[str] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    last_validated_at: str | None = None
    decay_function: Literal[
        "none", "exponential", "linear", "step", "empirical", "unknown"
    ] = "unknown"
    activation_state: Literal[
        "inactive", "candidate", "active", "decayed", "invalidated"
    ] = "candidate"


class CausalClaimMetadata(BaseModel):
    """Epistemic contract for a directed graph edge.

    Direction is not proof of causality. Causal language is disabled by default
    and cannot be enabled for association, sequence, analogy, or hypotheses.
    """

    relation_type: CausalRelationType = "hypothesis_only"
    identification_method: CausalIdentificationMethod = "none"
    causal_claim_allowed: bool = False
    confounders: list[str] = Field(default_factory=list)
    mediators: list[str] = Field(default_factory=list)
    colliders: list[str] = Field(default_factory=list)
    intervention: str | None = None
    counterfactual: str | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_causal_language(self) -> "CausalClaimMetadata":
        noncausal = {
            "statistical_association",
            "temporal_sequence",
            "historical_analogy",
            "hypothesis_only",
        }
        if self.causal_claim_allowed and self.relation_type in noncausal:
            raise ValueError(
                f"{self.relation_type} edges cannot authorize a causal claim"
            )
        if self.causal_claim_allowed and self.identification_method in {
            "none",
            "assumed_mechanism",
        }:
            raise ValueError(
                "causal claims require structural or empirical identification"
            )
        return self


def metadata_for_edge_type(edge_type: str) -> CausalClaimMetadata:
    """Conservative mapping for legacy graph edge labels."""

    if edge_type == "causal_channel":
        return CausalClaimMetadata(
            relation_type="economic_transmission",
            identification_method="assumed_mechanism",
            limitations=["Candidate mechanism; causal effect not identified"],
        )
    if edge_type == "observed_after":
        return CausalClaimMetadata(
            relation_type="temporal_sequence",
            limitations=["A occurred before B does not establish A caused B"],
        )
    if edge_type in {"calibrates"}:
        return CausalClaimMetadata(
            relation_type="statistical_association",
            limitations=["Calibration association is not a causal effect"],
        )
    if edge_type in {"supports", "contradicts", "confirms", "invalidates"}:
        return CausalClaimMetadata(
            relation_type="hypothesis_only",
            limitations=["Evidence relation, not a causal relation"],
        )
    return CausalClaimMetadata(
        relation_type="hypothesis_only",
        limitations=["Directed review edge; causal meaning is not established"],
    )


__all__ = [
    "CausalClaimMetadata",
    "CausalIdentificationMethod",
    "CausalRelationType",
    "GraphEdgeDynamics",
    "metadata_for_edge_type",
]
