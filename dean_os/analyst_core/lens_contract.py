"""Lens contract — the modular analyst plugin pattern (note 04 §4, §8, §9).

The analyst system must NOT be one monolithic "news -> answer" agent. Instead,
each analytical capability is a *lens*: a small module that reads the shared
analysis state and returns a *delta* describing what it added, revised, or
challenged (note 04 §4):

    analyze(input_packet, analysis_state, config) -> analysis_state_delta

Lenses never overwrite the whole state — they return deltas, which keeps the
system auditable and debuggable (note 04 §9). This lets new analysis
capabilities be added as plugins without rewriting the whole pipeline.

All lenses are review-only: they produce structured reasoning, never trade
instructions (note 04 §10).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, model_validator

from dean_os.analyst_core.schemas import (
    EvidenceGap,
    HypothesisLedgerEntry,
    RegimeContextVector,
    ScenarioOutcomeGraph,
    utc_now_iso,
)


# ──────────────────────────────────────────────────────────────────────────────
# AnalysisPacket — the shared state passed between lenses (note 04 §3.2)
# ──────────────────────────────────────────────────────────────────────────────


class AnalysisPacket(BaseModel):
    """Shared, mutable analysis state that lenses read from and update.

    This is the main object passed through the modular pipeline (note 04 §3.2).
    Lenses NEVER mutate it directly; they read it and return a ``ModuleDelta``
    which the orchestrator applies. This separation is what makes the pipeline
    auditable: every state transition is recorded as a delta.
    """

    packet_id: str
    as_of_date: str = Field(default_factory=utc_now_iso)
    source_packet_ids: list[str] = Field(default_factory=list)
    event_records: list[dict[str, Any]] = Field(default_factory=list)
    entity_links: list[dict[str, Any]] = Field(default_factory=list)

    # Core reasoning objects. ``None`` means "no lens has populated this yet";
    # once populated they are replaced, not overwritten blindly (see ModuleDelta).
    regime_context: RegimeContextVector | None = None
    scenario_graph: ScenarioOutcomeGraph | None = None
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    hypotheses: list[HypothesisLedgerEntry] = Field(default_factory=list)

    transmission_channels: list[dict[str, Any]] = Field(default_factory=list)
    expectation_gap: dict[str, Any] | None = None
    watch_signals: list[dict[str, Any]] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)

    # Review-only invariant is carried on the packet so downstream consumers
    # can never mistake analysis for execution authority.
    review_only: bool = True
    forbidden_outputs: list[str] = Field(
        default_factory=lambda: [
            "live_order",
            "buy",
            "sell",
            "hold",
            "position_sizing",
            "broker_routing",
            "autonomous_execution",
            "production_price_target",
        ]
    )

    @model_validator(mode="after")
    def _enforce_review_only(self) -> "AnalysisPacket":
        # The packet is structurally incapable of carrying execution authority.
        # This is a defense-in-depth invariant, not just documentation.
        if not self.review_only:
            raise ValueError("AnalysisPacket is always review_only — cannot disable")
        return self


# ──────────────────────────────────────────────────────────────────────────────
# ModuleDelta — what one lens changed (note 04 §9)
# ──────────────────────────────────────────────────────────────────────────────


class ModuleDelta(BaseModel):
    """Structured record of what a single lens added/modified.

    Every state transition in the analyst pipeline is recorded as a delta so the
    system is auditable and debuggable (note 04 §9). The orchestrator applies
    deltas to the packet; lenses never overwrite state directly.
    """

    module_name: str
    module_version: str = "0.1.0"
    as_of: str = Field(default_factory=utc_now_iso)

    # Fields the lens set (only present if the lens produced them).
    regime_context: RegimeContextVector | None = None
    scenario_graph: ScenarioOutcomeGraph | None = None
    evidence_gaps_added: list[EvidenceGap] = Field(default_factory=list)
    hypotheses_added: list[HypothesisLedgerEntry] = Field(default_factory=list)
    transmission_channels_added: list[dict[str, Any]] = Field(default_factory=list)
    expectation_gap: dict[str, Any] | None = None
    watch_signals_added: list[dict[str, Any]] = Field(default_factory=list)
    review_notes_added: list[str] = Field(default_factory=list)

    fields_added: list[str] = Field(default_factory=list)
    fields_modified: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason_for_change: str = ""
    review_only: bool = True

    @model_validator(mode="after")
    def _validate_delta(self) -> "ModuleDelta":
        self.module_name = self.module_name.strip()
        if not self.module_name:
            raise ValueError("ModuleDelta.module_name cannot be empty")
        if not self.review_only:
            raise ValueError("ModuleDelta is always review_only")
        self.evidence_ids = sorted({eid.strip() for eid in self.evidence_ids if eid and eid.strip()})
        return self


# ──────────────────────────────────────────────────────────────────────────────
# AnalystLens — the plugin interface (note 04 §4)
# ──────────────────────────────────────────────────────────────────────────────


class AnalystLens(ABC):
    """Base class for a modular analyst lens.

    A lens implements one analytical capability (regime grading, transmission
    mapping, expectation gap, scenario graphing, etc.). It reads the current
    packet and returns a delta describing what it contributed. Lenses are
    sector-agnostic; sector specialization arrives through the DomainProfile
    + KnowledgePack layer feeding the packet's inputs.
    """

    #: Human-readable capability name (used in audit trail).
    lens_name: str = "analyst_lens"
    lens_version: str = "0.1.0"

    #: Event classes this lens should run for (note 04 §8). "*" = always run.
    event_classes_supported: tuple[str, ...] = ("*",)

    #: Whether this lens can mutate existing packet state or only extend it.
    #: Conservative lenses (the default) only extend.
    can_modify_existing: bool = False

    @abstractmethod
    def analyze(self, packet: AnalysisPacket, config: dict[str, Any] | None = None) -> ModuleDelta:
        """Read ``packet`` and return a delta. Must NOT mutate ``packet``.

        Args:
            packet: Current shared analysis state (read-only to the lens).
            config: Optional lens configuration.

        Returns:
            A ModuleDelta describing what this lens added/revised. Returning
            an empty delta is valid (lens had nothing to contribute for this
            input); raising is reserved for genuine lens failures.
        """
        raise NotImplementedError

    def supports_event_class(self, event_class: str) -> bool:
        """True if this lens should run for ``event_class``."""
        if "*" in self.event_classes_supported:
            return True
        return event_class in self.event_classes_supported


# ──────────────────────────────────────────────────────────────────────────────
# LensRegistry — discover & route lenses by event class (note 04 §7, §8)
# ──────────────────────────────────────────────────────────────────────────────


class LensRegistry:
    """Registry of available analyst lenses (note 04 §8).

    Lenses register their name + supported event classes. The orchestrator
    asks the registry which lenses are relevant for a given event class and
    runs them in order, collecting deltas. This lets new analysis lenses be
    added without touching the orchestrator.
    """

    def __init__(self) -> None:
        self._lenses: dict[str, AnalystLens] = {}

    def register(self, lens: AnalystLens) -> None:
        if not isinstance(lens, AnalystLens):
            raise TypeError(f"LensRegistry only accepts AnalystLens instances, got {type(lens).__name__}")
        if not lens.lens_name.strip():
            raise ValueError("Lens.lens_name cannot be empty")
        if lens.lens_name in self._lenses:
            raise ValueError(f"lens {lens.lens_name!r} is already registered")
        self._lenses[lens.lens_name] = lens

    def unregister(self, lens_name: str) -> None:
        self._lenses.pop(lens_name, None)

    def get(self, lens_name: str) -> AnalystLens | None:
        return self._lenses.get(lens_name)

    def all_lenses(self) -> list[AnalystLens]:
        return list(self._lenses.values())

    def lenses_for_event_class(self, event_class: str) -> list[AnalystLens]:
        """Return lenses that should run for ``event_class`` (note 04 §7).

        A lens runs if it explicitly supports the class or supports "*" (all).
        """
        return [lens for lens in self._lenses.values() if lens.supports_event_class(event_class)]

    def __len__(self) -> int:
        return len(self._lenses)

    def __contains__(self, lens_name: str) -> bool:
        return lens_name in self._lenses


__all__ = [
    "AnalysisPacket",
    "ModuleDelta",
    "AnalystLens",
    "LensRegistry",
]
