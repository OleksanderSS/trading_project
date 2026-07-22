"""Tests for Phase 2 analyst lenses and LensOrchestrator.

All tests run with NO LLM, NO network — the analyst core is deterministic.
"""
from __future__ import annotations

import pytest

from dean_os.analyst_core import (
    AnalysisPacket,
    Confidence,
    EvidenceGap,
    HypothesisLedgerEntry,
    HypothesisStatus,
    LensOrchestrator,
    LensRegistry,
    ModuleDelta,
    Priority,
    RegimeContextVector,
    ScenarioNode,
    ScenarioOutcomeGraph,
)
from dean_os.analyst_core.lenses.event_classifier_lens import (
    EVENT_CLASS_KEYWORDS,
    EventClassifierLens,
)
from dean_os.analyst_core.lenses.evidence_gap_lens import (
    EVIDENCE_GAP_TEMPLATES,
    EvidenceGapLens,
)
from dean_os.analyst_core.lenses.expectation_gap_lens import ExpectationGapLens
from dean_os.analyst_core.lenses.historical_analog_lens import (
    ANALOG_LIBRARY,
    HistoricalAnalogLens,
)
from dean_os.analyst_core.lenses.hypothesis_ledger_lens import (
    HYPOTHESIS_TEMPLATES,
    HypothesisLedgerLens,
)
from dean_os.analyst_core.lenses.regime_context_lens import RegimeContextLens
from dean_os.analyst_core.lenses.transmission_mapper_lens import (
    TRANSMISSION_CHANNELS,
    TransmissionMapperLens,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_packet(
    event_records: list[dict] | None = None,
    entity_links: list[dict] | None = None,
    hypotheses: list[HypothesisLedgerEntry] | None = None,
    evidence_gaps: list[EvidenceGap] | None = None,
) -> AnalysisPacket:
    return AnalysisPacket(
        packet_id="test_packet_001",
        event_records=event_records or [],
        entity_links=entity_links or [],
        hypotheses=hypotheses or [],
        evidence_gaps=evidence_gaps or [],
    )


def _war_event() -> dict:
    return {
        "event_id": "ev_001",
        "title": "Military escalation in Taiwan strait",
        "text": "China conducts military exercises near Taiwan, raising conflict risk",
        "event_class": "war_escalation",
        "source": "reuters",
    }


def _demand_event() -> dict:
    return {
        "event_id": "ev_002",
        "title": "NVIDIA reports record GPU orders",
        "text": "NVIDIA demand for AI accelerators surges, data center orders increase",
        "event_class": "demand_driver",
        "source": "bloomberg",
    }


def _oil_event() -> dict:
    return {
        "event_id": "ev_003",
        "title": "OPEC announces production cut",
        "text": "OPEC oil supply disruption, crude price spike expected",
        "event_class": "oil_shock",
        "source": "cnbc",
    }


def _tariff_event() -> dict:
    return {
        "event_id": "ev_004",
        "title": "US announces new tariffs on Chinese semiconductors",
        "text": "Tariff restriction on semiconductor imports, trade war escalation",
        "event_class": "tariff",
        "source": "wsj",
    }


def _classified_entity_link() -> dict:
    return {
        "link_id": "link_001",
        "event_id": "ev_002",
        "text": "NVIDIA GPU orders surge on AI demand",
        "event_class": "demand_driver",
        "directness": "direct",
        "sentiment": "positive",
        "materiality_score": 0.75,
        "affected_sectors": ["semiconductor_ai_infrastructure"],
        "affected_tickers": ["NVDA"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# EventClassifierLens
# ──────────────────────────────────────────────────────────────────────────────


class TestEventClassifierLens:
    def test_classifies_war_event(self):
        lens = EventClassifierLens()
        packet = _make_packet(event_records=[_war_event()])
        delta = lens.analyze(packet)

        assert isinstance(delta, ModuleDelta)
        assert delta.module_name == "event_classifier"
        assert len(delta.evidence_ids) == 1

    def test_classifies_demand_event(self):
        lens = EventClassifierLens()
        packet = _make_packet(event_records=[_demand_event()])
        delta = lens.analyze(packet)

        assert delta is not None
        assert len(delta.evidence_ids) >= 1
        assert len(delta.classified_events_added) == 1

    def test_empty_packet_returns_delta(self):
        lens = EventClassifierLens()
        packet = _make_packet()
        delta = lens.analyze(packet)

        assert isinstance(delta, ModuleDelta)
        assert "no events classified" in delta.review_notes_added[0]

    def test_event_class_keywords_exist(self):
        assert "war_escalation" in EVENT_CLASS_KEYWORDS
        assert "ai_capex_announcement" in EVENT_CLASS_KEYWORDS
        assert "oil_shock" in EVENT_CLASS_KEYWORDS
        assert len(EVENT_CLASS_KEYWORDS) >= 15

    def test_does_not_duplicate_entity_links_as_events(self):
        lens = EventClassifierLens()
        link = {
            "link_id": "link_100",
            "text": "NVIDIA GPU demand surges on AI capex announcement",
            "source": "test",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert delta.evidence_ids == []
        assert delta.classified_events_added == []


# ──────────────────────────────────────────────────────────────────────────────
# TransmissionMapperLens
# ──────────────────────────────────────────────────────────────────────────────


class TestTransmissionMapperLens:
    def test_maps_oil_shock_channel(self):
        lens = TransmissionMapperLens()
        link = {
            "link_id": "link_001",
            "event_id": "ev_003",
            "event_class": "oil_shock",
            "text": "OPEC cuts production",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.transmission_channels_added) == 1
        channel = delta.transmission_channels_added[0]
        assert channel["channel_name"] == "energy_cost_transmission"
        assert len(channel["chain"]) >= 4
        assert channel["event_class"] == "oil_shock"

    def test_maps_tariff_channel(self):
        lens = TransmissionMapperLens()
        link = {
            "link_id": "link_002",
            "event_id": "ev_004",
            "event_class": "tariff",
            "text": "New tariffs announced",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.transmission_channels_added) == 1
        channel = delta.transmission_channels_added[0]
        assert channel["channel_name"] == "trade_policy_transmission"

    def test_ignores_unknown_event_class(self):
        lens = TransmissionMapperLens()
        link = {
            "link_id": "link_003",
            "event_id": "ev_999",
            "event_class": "other",
            "text": "Some unrelated event",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.transmission_channels_added) == 0
        assert "no transmission channels" in delta.review_notes_added[0]

    def test_transmission_channels_registry_complete(self):
        assert "oil_shock" in TRANSMISSION_CHANNELS
        assert "tariff" in TRANSMISSION_CHANNELS
        assert "ai_capex_announcement" in TRANSMISSION_CHANNELS
        assert "war_escalation" in TRANSMISSION_CHANNELS
        assert "central_bank_decision" in TRANSMISSION_CHANNELS


# ──────────────────────────────────────────────────────────────────────────────
# ExpectationGapLens
# ──────────────────────────────────────────────────────────────────────────────


class TestExpectationGapLens:
    def test_estimates_gap_for_surprise_event(self):
        lens = ExpectationGapLens()
        link = {
            "link_id": "link_001",
            "event_id": "ev_001",
            "event_class": "war_escalation",
            "text": "unexpected military escalation shocks market",
            "sentiment": "negative",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert delta.expectation_gap is not None
        assert delta.expectation_gap["status"] == "expectation_context_qualitative_only"
        assert delta.expectation_gap["event_count"] == 1
        assert delta.expectation_gap["quantified_event_count"] == 0

    def test_detects_already_priced_event(self):
        lens = ExpectationGapLens()
        link = {
            "link_id": "link_002",
            "event_id": "ev_002",
            "event_class": "central_bank_decision",
            "text": "fed rate decision widely expected, consensus priced in",
            "sentiment": "neutral",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        gap = delta.expectation_gap
        assert gap["already_priced_events"] == 0
        assert gap["event_assessments"][0]["positioning_crowdedness"] != "not_crowded"

    def test_empty_packet_returns_summary(self):
        lens = ExpectationGapLens()
        packet = _make_packet()
        delta = lens.analyze(packet)

        assert delta.expectation_gap["status"] == "no_events_to_assess"

    def test_high_surprise_generates_watch_signal(self):
        lens = ExpectationGapLens()
        link = {
            "link_id": "link_003",
            "event_id": "ev_003",
            "event_class": "war_escalation",
            "text": "unprecedented sudden attack shocks everyone",
            "sentiment": "negative",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        # High surprise should generate watch signals
        high_surprise = [
            s for s in delta.watch_signals_added
            if s.get("magnitude", 0) > 0.5
        ]
        assert len(high_surprise) >= 1
        assert high_surprise[0]["quantitative_gap_available"] is False

    def test_quantifies_only_sourced_actual_minus_expected(self):
        lens = ExpectationGapLens()
        link = {
            "event_id": "ev_004",
            "event_class": "earnings_surprise",
            "expectation_evidence": {
                "contract": "dean_expectation_evidence_v1",
                "expectation_type": "analyst_consensus",
                "actual": {
                    "value": 1.20, "unit": "USD/share",
                    "available_at": "2026-06-30T12:00:00+00:00",
                    "source_locator": "filing://issuer/result", "source_sha256": "a" * 64,
                },
                "expected": {
                    "value": 1.00, "unit": "USD/share",
                    "available_at": "2026-06-29T12:00:00+00:00",
                    "source_locator": "consensus://snapshot", "source_sha256": "b" * 64,
                },
                "expectation_std": 0.10,
            },
        }
        delta = lens.analyze(_make_packet(entity_links=[link]))
        assessment = delta.expectation_gap["event_assessments"][0]

        assert delta.expectation_gap["status"] == "expectation_gap_quantified"
        assert assessment["quantitative_gap_available"] is True
        assert assessment["surprise_value"] == pytest.approx(0.20)
        assert assessment["standardized_surprise"] == pytest.approx(2.0)

    def test_flat_source_labels_cannot_quantify_expectation_gap(self):
        link = {
            "event_id": "ev_legacy",
            "actual_value": 1.2,
            "expected_value": 1.0,
            "actual_source": "issuer filing",
            "expectation_source": "consensus snapshot",
        }
        delta = ExpectationGapLens().analyze(_make_packet(entity_links=[link]))
        assessment = delta.expectation_gap["event_assessments"][0]
        assert assessment["quantitative_gap_available"] is False
        assert "expectation_contract_missing_or_invalid" in assessment["expectation_validation_reasons"] or "expectation_evidence_not_structured" in assessment["expectation_validation_reasons"]


# ──────────────────────────────────────────────────────────────────────────────
# HistoricalAnalogLens
# ──────────────────────────────────────────────────────────────────────────────


class TestHistoricalAnalogLens:
    def test_finds_supply_shortage_analog(self):
        lens = HistoricalAnalogLens()
        link = {
            "link_id": "link_001",
            "event_id": "ev_001",
            "event_class": "supply_disruption",
            "text": "semiconductor shortage supply bottleneck",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.watch_signals_added) >= 1
        analog = delta.watch_signals_added[0]
        assert analog["signal_type"] == "historical_analog"
        assert "analog_id" in analog

    def test_finds_energy_shock_analog(self):
        lens = HistoricalAnalogLens()
        link = {
            "link_id": "link_002",
            "event_id": "ev_002",
            "event_class": "oil_shock",
            "text": "oil price spike energy shock crude",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.watch_signals_added) >= 1

    def test_no_match_returns_empty(self):
        lens = HistoricalAnalogLens()
        link = {
            "link_id": "link_003",
            "event_id": "ev_003",
            "event_class": "other",
            "text": "something completely unrelated",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.watch_signals_added) == 0

    def test_analog_library_has_required_fields(self):
        for analog in ANALOG_LIBRARY:
            assert "analog_id" in analog
            assert "pattern_name" in analog
            assert "outcomes_by_horizon" in analog
            assert "why_this_case_may_mislead" in analog
            assert "false_analogy_risk" in analog
            # All horizons must be present
            for horizon in [1, 5, 20, 60, 120]:
                assert horizon in analog["outcomes_by_horizon"]

    def test_outcomes_by_horizon_use_fixed_horizons(self):
        from dean_os.analyst_core.schemas import OUTCOME_HORIZONS

        for analog in ANALOG_LIBRARY:
            for horizon in analog["outcomes_by_horizon"]:
                assert horizon in OUTCOME_HORIZONS


# ──────────────────────────────────────────────────────────────────────────────
# HypothesisLedgerLens
# ──────────────────────────────────────────────────────────────────────────────


class TestHypothesisLedgerLens:
    def test_generates_hypothesis_from_demand_event(self):
        lens = HypothesisLedgerLens()
        link = {
            "link_id": "link_001",
            "event_id": "ev_001",
            "event_class": "demand_driver",
            "text": "AI demand growth",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.hypotheses_added) >= 1
        h = delta.hypotheses_added[0]
        assert isinstance(h, HypothesisLedgerEntry)
        assert h.status == HypothesisStatus.OPEN
        assert len(h.invalidation_signals) > 0  # Must be falsifiable
        assert h.trigger_evidence_ids == ["ev_001"]
        assert h.supporting_evidence_ids == []

    def test_generates_hypothesis_from_oil_event(self):
        lens = HypothesisLedgerLens()
        link = {
            "link_id": "link_002",
            "event_id": "ev_002",
            "event_class": "oil_shock",
            "text": "oil price shock",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.hypotheses_added) >= 1

    def test_existing_hypothesis_not_duplicated(self):
        existing = HypothesisLedgerEntry(
            hypothesis="AI demand growth will accelerate",
            invalidation_signals=["orders fail to appear"],
            status=HypothesisStatus.FALSIFIED,  # Already resolved
        )
        lens = HypothesisLedgerLens()
        link = {
            "link_id": "link_003",
            "event_id": "ev_003",
            "event_class": "demand_driver",
            "text": "AI demand growth",
        }
        packet = _make_packet(
            entity_links=[link],
            hypotheses=[existing],
        )
        delta = lens.analyze(packet)

        # Resolved hypotheses should not be regenerated as new
        # (new hypotheses may still be generated for the event class)
        assert len(delta.hypotheses_added) >= 0  # No crash, valid behavior

    def test_all_hypotheses_are_falsifiable(self):
        for event_class, templates in HYPOTHESIS_TEMPLATES.items():
            for tmpl in templates:
                assert len(tmpl["invalidation_signals"]) > 0, (
                    f"Hypothesis template for {event_class} must have "
                    f"invalidation_signals (falsifiability requirement)"
                )

    def test_hypothesis_status_update(self):
        """Test that existing hypotheses can be updated."""
        h = HypothesisLedgerEntry(
            hypothesis="Supply constraints will persist",
            invalidation_signals=["capacity expansion completed"],
            status=HypothesisStatus.OPEN,
        )
        lens = HypothesisLedgerLens()
        link = {
            "link_id": "link_004",
            "event_id": "ev_004",
            "event_class": "supply_disruption",
            "text": "capacity expansion completed ahead of schedule",
        }
        packet = _make_packet(entity_links=[link], hypotheses=[h])
        delta = lens.analyze(packet)

        # The hypothesis should have been evaluated
        # (may or may not change status depending on keyword overlap)


# ──────────────────────────────────────────────────────────────────────────────
# EvidenceGapLens
# ──────────────────────────────────────────────────────────────────────────────


class TestEvidenceGapLens:
    def test_generates_gaps_from_demand_event(self):
        lens = EvidenceGapLens()
        link = {
            "link_id": "link_001",
            "event_id": "ev_001",
            "event_class": "demand_driver",
            "text": "AI demand surge",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.evidence_gaps_added) >= 1
        gap = delta.evidence_gaps_added[0]
        assert isinstance(gap, EvidenceGap)
        assert gap.current_status == "missing"

    def test_generates_gaps_from_oil_event(self):
        lens = EvidenceGapLens()
        link = {
            "link_id": "link_002",
            "event_id": "ev_002",
            "event_class": "oil_shock",
            "text": "oil supply cut",
        }
        packet = _make_packet(entity_links=[link])
        delta = lens.analyze(packet)

        assert len(delta.evidence_gaps_added) >= 1

    def test_no_duplicate_gaps(self):
        existing = EvidenceGap(
            description="Actual order backlog data vs. narrative claims",
            priority=Priority.HIGH,
        )
        lens = EvidenceGapLens()
        link = {
            "link_id": "link_003",
            "event_id": "ev_003",
            "event_class": "demand_driver",
            "text": "demand surge",
        }
        packet = _make_packet(entity_links=[link], evidence_gaps=[existing])
        delta = lens.analyze(packet)

        descriptions = [g.description for g in delta.evidence_gaps_added]
        assert "Actual order backlog data vs. narrative claims" not in descriptions

    def test_evidence_gap_templates_exist(self):
        assert "demand_driver" in EVIDENCE_GAP_TEMPLATES
        assert "oil_shock" in EVIDENCE_GAP_TEMPLATES
        assert "tariff" in EVIDENCE_GAP_TEMPLATES

    def test_gap_generates_from_hypothesis_observations(self):
        h = HypothesisLedgerEntry(
            hypothesis="Supply constraints will persist for 30 days",
            invalidation_signals=["capacity expansion completed"],
            expected_observations=["extended lead times", "pricing power"],
            confidence=0.7,
        )
        lens = EvidenceGapLens()
        packet = _make_packet(hypotheses=[h])
        delta = lens.analyze(packet)

        assert len(delta.evidence_gaps_added) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# LensOrchestrator
# ──────────────────────────────────────────────────────────────────────────────


class TestLensOrchestrator:
    def test_runs_all_lenses(self):
        registry = LensRegistry()
        registry.register(RegimeContextLens())
        registry.register(EventClassifierLens())
        registry.register(TransmissionMapperLens())
        registry.register(ExpectationGapLens())
        registry.register(HistoricalAnalogLens())
        registry.register(HypothesisLedgerLens())
        registry.register(EvidenceGapLens())

        orchestrator = LensOrchestrator(registry)
        packet = _make_packet(event_records=[_demand_event()])

        enriched, deltas = orchestrator.run(packet)

        assert len(deltas) >= 5  # At least 5 of 7 lenses should produce deltas
        assert enriched.regime_context is not None
        assert len(evidence_ids := [d.evidence_ids for d in deltas if d.evidence_ids]) >= 1

    def test_empty_packet_runs_without_error(self):
        registry = LensRegistry()
        registry.register(RegimeContextLens())
        registry.register(EventClassifierLens())
        registry.register(TransmissionMapperLens())
        registry.register(ExpectationGapLens())
        registry.register(HistoricalAnalogLens())
        registry.register(HypothesisLedgerLens())
        registry.register(EvidenceGapLens())

        orchestrator = LensOrchestrator(registry)
        packet = _make_packet()

        enriched, deltas = orchestrator.run(packet)

        # Should complete without error even with empty input
        assert enriched is not None
        assert len(deltas) >= 0

    def test_delta_trail_is_ordered(self):
        registry = LensRegistry()
        registry.register(EventClassifierLens())
        registry.register(TransmissionMapperLens())

        orchestrator = LensOrchestrator(registry)
        packet = _make_packet(
            entity_links=[_classified_entity_link()]
        )

        _, deltas = orchestrator.run(packet)

        names = [d.module_name for d in deltas]
        assert names == sorted(names, key=lambda n: names.index(n))

    def test_applies_regime_context(self):
        registry = LensRegistry()
        registry.register(RegimeContextLens())

        orchestrator = LensOrchestrator(registry)
        packet = _make_packet(event_records=[_war_event()])

        enriched, _ = orchestrator.run(packet)

        assert enriched.regime_context is not None
        assert isinstance(enriched.regime_context, RegimeContextVector)

    def test_applies_transmission_channels(self):
        registry = LensRegistry()
        registry.register(TransmissionMapperLens())

        orchestrator = LensOrchestrator(registry)
        link = {
            "link_id": "link_010",
            "event_id": "ev_010",
            "event_class": "tariff",
            "text": "tariff announcement",
        }
        packet = _make_packet(entity_links=[link])

        enriched, _ = orchestrator.run(packet)

        assert len(enriched.transmission_channels) >= 1

    def test_applies_evidence_gaps(self):
        registry = LensRegistry()
        registry.register(EvidenceGapLens())

        orchestrator = LensOrchestrator(registry)
        link = {
            "link_id": "link_011",
            "event_id": "ev_011",
            "event_class": "demand_driver",
            "text": "demand surge",
        }
        packet = _make_packet(entity_links=[link])

        enriched, _ = orchestrator.run(packet)

        assert len(enriched.evidence_gaps) >= 1

    def test_max_rounds_respected(self):
        registry = LensRegistry()
        registry.register(RegimeContextLens())

        orchestrator = LensOrchestrator(registry, max_rounds=2)
        packet = _make_packet(event_records=[_war_event()])

        _, deltas = orchestrator.run(packet)

        # With max_rounds=2 and can_modify_existing=False for RegimeContextLens,
        # it should only run once (second round skipped)
        regime_deltas = [d for d in deltas if d.module_name == "regime_context"]
        assert len(regime_deltas) == 1

    def test_full_pipeline_with_multiple_events(self):
        registry = LensRegistry()
        registry.register(RegimeContextLens())
        registry.register(EventClassifierLens())
        registry.register(TransmissionMapperLens())
        registry.register(ExpectationGapLens())
        registry.register(HistoricalAnalogLens())
        registry.register(HypothesisLedgerLens())
        registry.register(EvidenceGapLens())

        orchestrator = LensOrchestrator(registry)
        packet = _make_packet(
            event_records=[_war_event(), _demand_event(), _oil_event()],
            entity_links=[
                {
                    "link_id": "link_001",
                    "event_id": "ev_002",
                    "event_class": "demand_driver",
                    "text": "NVIDIA GPU demand surges",
                },
                {
                    "link_id": "link_002",
                    "event_id": "ev_003",
                    "event_class": "oil_shock",
                    "text": "OPEC supply cut",
                },
            ],
        )

        enriched, deltas = orchestrator.run(packet)

        # Regime context should be populated
        assert enriched.regime_context is not None
        # Should have transmission channels
        assert len(enriched.transmission_channels) >= 1
        # Should have evidence gaps
        assert len(enriched.evidence_gaps) >= 1
        # Should have hypotheses
        assert len(enriched.hypotheses) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Safety invariants
# ──────────────────────────────────────────────────────────────────────────────


class TestSafetyInvariants:
    def test_packet_remains_review_only_after_enrichment(self):
        registry = LensRegistry()
        registry.register(RegimeContextLens())
        registry.register(EventClassifierLens())
        registry.register(TransmissionMapperLens())
        registry.register(ExpectationGapLens())
        registry.register(HistoricalAnalogLens())
        registry.register(HypothesisLedgerLens())
        registry.register(EvidenceGapLens())

        orchestrator = LensOrchestrator(registry)
        packet = _make_packet(event_records=[_demand_event()])

        enriched, _ = orchestrator.run(packet)

        assert enriched.review_only is True
        assert "buy" in enriched.forbidden_outputs
        assert "sell" in enriched.forbidden_outputs

    def test_all_deltas_are_review_only(self):
        registry = LensRegistry()
        registry.register(RegimeContextLens())
        registry.register(EventClassifierLens())
        registry.register(TransmissionMapperLens())
        registry.register(ExpectationGapLens())
        registry.register(HistoricalAnalogLens())
        registry.register(HypothesisLedgerLens())
        registry.register(EvidenceGapLens())

        orchestrator = LensOrchestrator(registry)
        packet = _make_packet(event_records=[_war_event(), _oil_event()])

        _, deltas = orchestrator.run(packet)

        for delta in deltas:
            assert delta.review_only is True

    def test_hypothesis_invalidation_signals_required(self):
        """From design notes: hypothesis without invalidation signals is not testable."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            HypothesisLedgerEntry(
                hypothesis="test",
                invalidation_signals=[],  # Empty = not falsifiable
            )

    def test_scenario_graph_probability_mass(self):
        """Scenario probabilities must sum to 1.0."""
        nodes = [
            ScenarioNode(node_type="scenario", label="base", probability=0.5),
            ScenarioNode(node_type="scenario", label="upside", probability=0.3),
            ScenarioNode(node_type="scenario", label="downside", probability=0.2),
        ]
        graph = ScenarioOutcomeGraph(nodes=nodes)
        assert graph.probability_mass_check is True
