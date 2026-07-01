"""Core analyst schemas (Phase 1 — schemas nucleus).

These schemas implement the deterministic, review-only data structures described
in the analyst design notes:

- `dean_os/draft/thinking/.../source_notes/03_regime_context_scenario_graph_note_*.md`
- `dean_os/draft/thinking/.../source_notes/07_additional_analyst_observations_*.md`
- `dean_os/draft/thinking/.../00_INDEX.md` (Minimum viable integration target)

They are intentionally deterministic and carry NO LLM calls, NO live trading
authority, and NO model-promotion power. Every object exposes explicit safety
flags so downstream consumers can never mistake review-only analysis for an
execution decision.

These objects are sector-agnostic. Sector specialization comes from the
existing `DomainProfile` + `KnowledgePack` layer; these schemas capture the
*reasoning state* (regime, scenarios, hypotheses, evidence gaps, historical
checks) that any domain analyst produces.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from dean_os.schemas import utc_now_iso
except Exception:  # pragma: no cover - keep analyst_core importable standalone
    from datetime import datetime, timezone

    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Shared enums / taxonomy (note 03 §6, note 07 §5-7)
# ──────────────────────────────────────────────────────────────────────────────

# Fixed outcome horizons. The analyst NEVER re-derives these per analysis —
# they are the contract the evaluation plane keys on (note 03 §6.10).
OUTCOME_HORIZONS: tuple[int, ...] = (1, 5, 20, 60, 120)


class Trend(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HypothesisStatus(str, Enum):
    OPEN = "open"
    CONFIRMED = "confirmed"
    WEAKENED = "weakened"
    FALSIFIED = "falsified"
    UNRESOLVED = "unresolved"


# The eight regime dimensions from note 03 §6. Kept as a tuple so the vector
# can be iterated and validated exhaustively.
REGIME_DIMENSIONS: tuple[str, ...] = (
    "geopolitical_state",
    "economic_phase",
    "inflation_rates_context",
    "liquidity_credit_context",
    "market_state",
    "commodity_stress",
    "ai_tech_cycle",
    "safe_haven_behavior",
)


# ──────────────────────────────────────────────────────────────────────────────
# Safety mixin — every analyst-core object is review-only
# ──────────────────────────────────────────────────────────────────────────────

REVIEW_ONLY_SAFETY: dict[str, bool] = {
    "review_only": True,
    "no_live_execution": True,
    "no_broker_access": True,
    "no_production_config_write": True,
    "no_learning_memory_write": True,
    "no_model_promotion": True,
    "can_trade": False,
}


def _default_safety() -> dict[str, bool]:
    return dict(REVIEW_ONLY_SAFETY)


# ──────────────────────────────────────────────────────────────────────────────
# 1. RegimeContextVector (note 03 §6-7, note 07 §1)
# ──────────────────────────────────────────────────────────────────────────────


class RegimeDimensionState(BaseModel):
    """One axis of the regime vector: state + intensity + trend + confidence.

    A regime is NOT a single label ("war"/"bubble"); it is a graded state
    vector so the model can update context gradually instead of flipping
    between labels (note 07 §1).
    """

    state: str = Field(..., description="Taxonomy value for this dimension (e.g. 'sanctions_chokepoint_risk').")
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    trend: Trend = Trend.UNKNOWN
    confidence: Confidence = Confidence.LOW
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _normalize(self) -> "RegimeDimensionState":
        self.state = self.state.strip()
        if not self.state:
            raise ValueError("RegimeDimensionState.state cannot be empty")
        self.evidence_ids = sorted({eid.strip() for eid in self.evidence_ids if eid and eid.strip()})
        return self


class RegimeContextVector(BaseModel):
    """Date-specific regime snapshot (note 03 §7, note 07 §1).

    All eight dimensions MUST be present (defaulted to intensity 0 / unknown
    if no evidence). This guarantees a stable, comparable shape across
    as_of dates and prevents silent regime drift.
    """

    regime_context_id: str = Field(default_factory=lambda: f"regime_{uuid4().hex}")
    as_of: str = Field(default_factory=utc_now_iso)
    dimensions: dict[str, RegimeDimensionState] = Field(default_factory=dict)
    confidence: Confidence = Confidence.LOW
    evidence_gaps: list[str] = Field(default_factory=list)
    safety: dict[str, bool] = Field(default_factory=_default_safety)

    @model_validator(mode="after")
    def _ensure_all_dimensions(self) -> "RegimeContextVector":
        for dim in REGIME_DIMENSIONS:
            self.dimensions.setdefault(
                dim,
                RegimeDimensionState(state="unknown", intensity=0.0, trend=Trend.UNKNOWN, confidence=Confidence.LOW),
            )
        # Reject unknown dimensions so callers cannot quietly extend the vector.
        unknown = set(self.dimensions) - set(REGIME_DIMENSIONS)
        if unknown:
            raise ValueError(f"Unknown regime dimensions: {sorted(unknown)}")
        return self


# ──────────────────────────────────────────────────────────────────────────────
# 2. ScenarioOutcomeGraph (note 03 §3-5)
# ──────────────────────────────────────────────────────────────────────────────

# Node types from note 03 §4.
SCENARIO_NODE_TYPES: tuple[str, ...] = (
    "regime_state",
    "event",
    "transmission_channel",
    "expectation_gap",
    "scenario",
    "observable_signal",
    "outcome_horizon",
    "invalidation_signal",
    "self_check",
)

# Edge types from note 03 §5.
SCENARIO_EDGE_TYPES: tuple[str, ...] = (
    "causal_channel",
    "conditional_update",
    "supports",
    "contradicts",
    "confirms",
    "invalidates",
    "leads_to",
    "observed_after",
    "calibrates",
)


class ScenarioNode(BaseModel):
    node_id: str = Field(default_factory=lambda: f"node_{uuid4().hex}")
    node_type: str
    label: str
    description: str = ""
    as_of: str = Field(default_factory=utc_now_iso)
    # Probabilities are only meaningful on `scenario` nodes; 0 elsewhere.
    probability: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: Confidence = Confidence.LOW
    evidence_ids: list[str] = Field(default_factory=list)
    uncertainty_notes: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "ScenarioNode":
        if self.node_type not in SCENARIO_NODE_TYPES:
            raise ValueError(f"Unknown scenario node type: {self.node_type!r}")
        if self.node_type != "scenario" and self.probability > 0:
            raise ValueError("probability is only allowed on 'scenario' nodes")
        self.label = self.label.strip()
        if not self.label:
            raise ValueError("ScenarioNode.label cannot be empty")
        return self


class ScenarioEdge(BaseModel):
    edge_id: str = Field(default_factory=lambda: f"edge_{uuid4().hex}")
    source_node_id: str
    target_node_id: str
    edge_type: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    probability_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW

    @model_validator(mode="after")
    def _validate(self) -> "ScenarioEdge":
        if self.edge_type not in SCENARIO_EDGE_TYPES:
            raise ValueError(f"Unknown scenario edge type: {self.edge_type!r}")
        if self.source_node_id == self.target_node_id:
            raise ValueError("self-loops are not allowed (graph must be acyclic)")
        self.source_node_id = self.source_node_id.strip()
        self.target_node_id = self.target_node_id.strip()
        return self


class ScenarioOutcomeGraph(BaseModel):
    """Acyclic potential-outcome graph (note 03 §3).

    Probabilistic map of plausible futures, NOT a forecast. Future updates
    create new graph versions rather than mutating history.
    """

    scenario_graph_id: str = Field(default_factory=lambda: f"scenario_{uuid4().hex}")
    as_of: str = Field(default_factory=utc_now_iso)
    root_regime_snapshot_id: str | None = None
    event_id: str | None = None
    nodes: list[ScenarioNode] = Field(default_factory=list)
    edges: list[ScenarioEdge] = Field(default_factory=list)
    horizons: tuple[int, ...] = OUTCOME_HORIZONS
    probability_mass_check: bool = False
    evidence_gaps: list[str] = Field(default_factory=list)
    review_status: str = "review_only"
    safety: dict[str, bool] = Field(default_factory=_default_safety)

    @model_validator(mode="after")
    def _validate_graph(self) -> "ScenarioOutcomeGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate node_id values in scenario graph")

        for edge in self.edges:
            if edge.source_node_id not in node_ids:
                raise ValueError(f"edge {edge.edge_id} references unknown source node {edge.source_node_id!r}")
            if edge.target_node_id not in node_ids:
                raise ValueError(f"edge {edge.edge_id} references unknown target node {edge.target_node_id!r}")

        # Acyclicity check (note 03 §15: test_scenario_graph_is_acyclic).
        if self.nodes and self.edges and not self._is_acyclic():
            raise ValueError("scenario graph contains a cycle — graphs must be acyclic")

        # Probability mass check (note 03 §15: probability_mass_sums_to_one).
        scenario_probs = [node.probability for node in self.nodes if node.node_type == "scenario"]
        if scenario_probs:
            total = sum(scenario_probs)
            # Tolerate float drift; fail loudly if probabilities do not sum to ~1.
            self.probability_mass_check = abs(total - 1.0) <= 1e-6
        else:
            # No scenario nodes yet: vacuously valid (mass is undefined, not invalid).
            self.probability_mass_check = True

        if not self.probability_mass_check:
            raise ValueError("scenario node probabilities must sum to 1.0")
        return self

    def _is_acyclic(self) -> bool:
        """Kahn's algorithm over node_id → target adjacency."""
        adjacency: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        indegree: dict[str, int] = {node.node_id: 0 for node in self.nodes}
        for edge in self.edges:
            adjacency[edge.source_node_id].append(edge.target_node_id)
            indegree[edge.target_node_id] += 1

        queue = [nid for nid, deg in indegree.items() if deg == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for nxt in adjacency[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return visited == len(self.nodes)


# ──────────────────────────────────────────────────────────────────────────────
# 3. EvidenceGap (note 07 §10)
# ──────────────────────────────────────────────────────────────────────────────


class EvidenceGap(BaseModel):
    """A missing piece of evidence, ranked by how much it would move scenarios."""

    gap_id: str = Field(default_factory=lambda: f"gap_{uuid4().hex}")
    description: str
    importance_to_scenario_probability: Priority = Priority.MEDIUM
    expected_source_type: str = "unknown"
    current_status: str = "missing"
    priority: Priority = Priority.MEDIUM
    safety: dict[str, bool] = Field(default_factory=_default_safety)

    @model_validator(mode="after")
    def _normalize(self) -> "EvidenceGap":
        self.description = self.description.strip()
        if not self.description:
            raise ValueError("EvidenceGap.description cannot be empty")
        return self


# ──────────────────────────────────────────────────────────────────────────────
# 4. HypothesisLedgerEntry (note 07 §12)
# ──────────────────────────────────────────────────────────────────────────────


class HypothesisLedgerEntry(BaseModel):
    """Explicit, falsifiable analyst hypothesis (note 07 §12).

    Prevents analyst reasoning from becoming vague narrative: every
    hypothesis MUST carry invalidation signals and horizons to check, so the
    evaluation plane can later mark it confirmed/weakened/falsified.
    """

    hypothesis_id: str = Field(default_factory=lambda: f"hypothesis_{uuid4().hex}")
    as_of: str = Field(default_factory=utc_now_iso)
    hypothesis: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    expected_observations: list[str] = Field(default_factory=list)
    invalidation_signals: list[str] = Field(default_factory=list)
    horizons_to_check: tuple[int, ...] = OUTCOME_HORIZONS
    status: HypothesisStatus = HypothesisStatus.OPEN
    calibration_note: str = ""
    safety: dict[str, bool] = Field(default_factory=_default_safety)

    @model_validator(mode="after")
    def _validate(self) -> "HypothesisLedgerEntry":
        self.hypothesis = self.hypothesis.strip()
        if not self.hypothesis:
            raise ValueError("HypothesisLedgerEntry.hypothesis cannot be empty")
        # A hypothesis without invalidation signals is not falsifiable and
        # therefore not disciplined analysis (note 07 §12).
        if not self.invalidation_signals:
            raise ValueError(
                "HypothesisLedgerEntry.invalidation_signals cannot be empty — "
                "a hypothesis without a falsification path is not testable"
            )
        self.supporting_evidence_ids = sorted({e.strip() for e in self.supporting_evidence_ids if e and e.strip()})
        self.contradicting_evidence_ids = sorted(
            {e.strip() for e in self.contradicting_evidence_ids if e and e.strip()}
        )
        return self


# ──────────────────────────────────────────────────────────────────────────────
# 5. HistoricalOutcomeCheck (note 03 §11)
# ──────────────────────────────────────────────────────────────────────────────


class HorizonOutcome(BaseModel):
    """Realized outcome at ONE fixed horizon (note 03 §6.10)."""

    horizon_days: int
    observed: dict[str, Any] = Field(default_factory=dict)
    winner_scenario_id: str | None = None

    @field_validator("horizon_days")
    @classmethod
    def _horizon_fixed(cls, value: int) -> int:
        if value not in OUTCOME_HORIZONS:
            raise ValueError(f"horizon_days must be one of the fixed horizons {OUTCOME_HORIZONS}")
        return value


class HistoricalOutcomeCheck(BaseModel):
    """What actually happened after an analogous event, by fixed horizon.

    Used for base rates and self-check (note 03 §11). Always carries a
    `false_analogy_risk` because no two event-regime-outcome paths are
    identical (note 03 §11 `false_analogy_risk`).
    """

    analog_id: str = Field(default_factory=lambda: f"analog_{uuid4().hex}")
    event_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    regime_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome_path_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    key_differences: list[str] = Field(default_factory=list)
    outcomes_by_horizon: dict[int, HorizonOutcome] = Field(default_factory=dict)
    winning_scenario: str | None = None
    false_analogy_risk: Confidence = Confidence.MEDIUM
    cases_where_signal_failed: list[str] = Field(default_factory=list)
    safety: dict[str, bool] = Field(default_factory=_default_safety)

    @model_validator(mode="after")
    def _horizons_are_fixed(self) -> "HistoricalOutcomeCheck":
        for horizon in self.outcomes_by_horizon:
            if horizon not in OUTCOME_HORIZONS:
                raise ValueError(f"outcome horizon {horizon} is not one of the fixed horizons {OUTCOME_HORIZONS}")
        return self


__all__ = [
    "OUTCOME_HORIZONS",
    "REGIME_DIMENSIONS",
    "SCENARIO_NODE_TYPES",
    "SCENARIO_EDGE_TYPES",
    "REVIEW_ONLY_SAFETY",
    "Trend",
    "Confidence",
    "Priority",
    "HypothesisStatus",
    "RegimeDimensionState",
    "RegimeContextVector",
    "ScenarioNode",
    "ScenarioEdge",
    "ScenarioOutcomeGraph",
    "EvidenceGap",
    "HypothesisLedgerEntry",
    "HorizonOutcome",
    "HistoricalOutcomeCheck",
]
