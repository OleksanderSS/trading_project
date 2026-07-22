"""LensOrchestrator — runs registered lenses sequentially and applies deltas.

The orchestrator is the pipeline engine for the modular analyst. It:
1. Selects lenses relevant to the packet's event classes
2. Runs each lens in registration order
3. Applies each ModuleDelta to the AnalysisPacket
4. Records the full delta trail for auditability

Every lens runs deterministically — no LLM, no network, no side effects.
The orchestrator itself never mutates packet state; it only applies deltas
returned by lenses.
"""
from __future__ import annotations

from typing import Any

from dean_os.analyst_core.lens_contract import (
    AnalysisPacket,
    AnalystLens,
    LensRegistry,
    ModuleDelta,
)


class LensOrchestrator:
    """Sequential lens pipeline that enriches an AnalysisPacket.

    Usage::

        orchestrator = LensOrchestrator(registry)
        enriched_packet, delta_trail = orchestrator.run(packet)
    """

    def __init__(
        self,
        registry: LensRegistry,
        *,
        max_rounds: int = 1,
        config: dict[str, Any] | None = None,
    ):
        """Args:
            registry: LensRegistry with registered lenses.
            max_rounds: How many times to run the full lens set (1 = single pass).
            config: Optional global config passed to every lens.
        """
        self.registry = registry
        self.max_rounds = max(1, max_rounds)
        self.config = config or {}

    def run(
        self,
        packet: AnalysisPacket,
        *,
        event_class: str = "*",
    ) -> tuple[AnalysisPacket, list[ModuleDelta]]:
        """Run all matching lenses on ``packet`` and apply their deltas.

        Args:
            packet: The analysis packet to enrich (read-only to lenses).
            event_class: Event class for lens selection. "*" runs all lenses.

        Returns:
            Tuple of (enriched packet, ordered list of applied deltas).
        """
        delta_trail: list[ModuleDelta] = []

        for _round in range(self.max_rounds):
            lenses = self._select_lenses(event_class, delta_trail)
            if not lenses:
                break

            for lens in lenses:
                delta = lens.analyze(packet, self.config)
                if delta is not None:
                    self._apply_delta(packet, delta)
                    delta_trail.append(delta)

        return packet, delta_trail

    def _select_lenses(
        self,
        event_class: str,
        prior_deltas: list[ModuleDelta],
    ) -> list[AnalystLens]:
        """Select lenses that should run, skipping those already run in this round.

        A lens with ``can_modify_existing=False`` is skipped if a prior delta
        from the same lens name already exists (it cannot modify what it
        already set).
        """
        already_run = {d.module_name for d in prior_deltas}
        selected: list[AnalystLens] = []

        for lens in self.registry.lenses_for_event_class(event_class):
            if not lens.can_modify_existing and lens.lens_name in already_run:
                continue
            selected.append(lens)

        return selected

    def _apply_delta(self, packet: AnalysisPacket, delta: ModuleDelta) -> None:
        """Apply a ModuleDelta to the packet.

        This is the ONLY place where packet state is mutated.
        Each field is applied only if the delta populated it.

        Single-owner invariant: a non-modifying lens MUST NOT overwrite a field
        another lens already set. This fails loudly rather than silently
        dropping the prior lens's output — silently overwriting would make the
        delta trail lie (it would claim both lenses contributed, but only the
        later value survives).
        """
        if delta.regime_context is not None:
            self._assert_can_set("regime_context", packet.regime_context, delta)
            packet.regime_context = delta.regime_context

        if delta.scenario_graph is not None:
            self._assert_can_set("scenario_graph", packet.scenario_graph, delta)
            packet.scenario_graph = delta.scenario_graph

        if delta.classified_events_added:
            packet.classified_events.extend(delta.classified_events_added)

        if getattr(delta, "entity_links_added", None):
            packet.entity_links.extend(delta.entity_links_added)

        if delta.evidence_gaps_added:
            packet.evidence_gaps.extend(delta.evidence_gaps_added)

        if delta.hypotheses_added:
            packet.hypotheses.extend(delta.hypotheses_added)

        if delta.hypothesis_review_proposals_added:
            packet.hypothesis_review_proposals.extend(
                delta.hypothesis_review_proposals_added
            )

        if delta.transmission_channels_added:
            packet.transmission_channels.extend(delta.transmission_channels_added)

        if delta.expectation_gap is not None:
            packet.expectation_gap = delta.expectation_gap

        if delta.watch_signals_added:
            packet.watch_signals.extend(delta.watch_signals_added)

        if delta.review_notes_added:
            packet.review_notes.extend(delta.review_notes_added)

    @staticmethod
    def _assert_can_set(
        field_name: str,
        current_value: Any,
        delta: ModuleDelta,
    ) -> None:
        """Guard the single-owner invariant for overwrite fields.

        Overwrite fields (regime_context, scenario_graph) replace the prior
        value rather than extend it. A non-modifying lens that tries to set
        one already set is a contract violation: refuse loudly so the delta
        trail never lies about who owns the field.
        """
        if current_value is None:
            return
        if delta.can_modify_existing:
            return
        raise ValueError(
            f"Lens {delta.module_name!r} (can_modify_existing=False) tried to "
            f"overwrite packet.{field_name} already set by a prior lens. Either "
            f"mark the lens as can_modify_existing=True or have it return a "
            f"delta on a different field."
        )


__all__ = ["LensOrchestrator"]
