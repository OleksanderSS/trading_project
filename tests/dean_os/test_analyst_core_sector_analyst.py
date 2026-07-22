"""Tests for SectorAnalyst — the unified sector analysis orchestrator.

All tests run with NO LLM, NO network — deterministic review-only analysis.
"""
from __future__ import annotations

import pytest

from dean_os.analyst_core.sector_analyst import (
    SectorAnalyst,
    SectorReport,
    _evidence_to_entity_links,
    _evidence_to_event_records,
)
from dean_os.analyst_core.lens_contract import AnalysisPacket, ModuleDelta
from dean_os.analyst_core.schemas import (
    EvidenceGap,
    HypothesisLedgerEntry,
    RegimeContextVector,
    ScenarioOutcomeGraph,
)
from dean_os.analysts.schemas import AnalystEvidenceItem, AnalystReport


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_evidence(
    evidence_id: str = "ev_001",
    evidence_type: str = "sector_demand",
    summary: str = "AI demand accelerates",
    stance_hint: str = "positive",
    directness: str = "sector",
    tickers: list[str] | None = None,
    strength: float = 0.7,
) -> AnalystEvidenceItem:
    return AnalystEvidenceItem(
        evidence_id=evidence_id,
        source_type="news",
        source="reuters",
        as_of="2026-07-01T00:00:00Z",
        domain_id="semiconductor_ai_infrastructure",
        tickers=tickers or [],
        sectors=["semiconductor_ai_infrastructure"],
        evidence_type=evidence_type,
        summary=summary,
        stance_hint=stance_hint,
        strength=strength,
        freshness_score=0.8,
        directness=directness,
        reliability_score=0.7,
    )


def _make_tickerevidence() -> AnalystEvidenceItem:
    return _make_evidence(
        evidence_id="ev_ticker_001",
        evidence_type="sector_demand",
        summary="NVIDIA reports record data center revenue",
        stance_hint="positive",
        directness="ticker",
        tickers=["NVDA"],
        strength=0.85,
    )


def _make_context_with_news():
    """Build a minimal MarketContext-like object for testing."""
    from dean_os.schemas import MarketContext

    return MarketContext(
        phase="pre_pipeline",
        as_of="2026-07-01T00:00:00Z",
        tickers=["NVDA", "AMD", "TSM"],
        news=[
            {
                "title": "AI demand accelerates",
                "text": "NVIDIA reports record GPU orders on AI infrastructure spending",
                "source": "reuters",
            },
            {
                "title": "OPEC announces production cut",
                "text": "OPEC oil supply disruption, crude price spike",
                "source": "cnbc",
            },
        ],
        fundamentals={
            "NVDA": {"revenue": 40000000000, "pe_ratio": 65.0},
            "AMD": {"revenue": 7000000000, "pe_ratio": 45.0},
        },
        macro={
            "cpi_yoy": 3.2,
            "fed_rate": 5.25,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper function tests
# ──────────────────────────────────────────────────────────────────────────────


class TestHelperFunctions:
    def test_evidence_to_event_records(self):
        evidence = [_make_evidence(), _make_tickerevidence()]
        records = _evidence_to_event_records(evidence)

        assert len(records) == 2
        assert records[0]["event_id"] == "ev_001"
        assert records[0]["event_class"] == "sector_demand"
        assert records[1]["event_id"] == "ev_ticker_001"
        assert records[1]["tickers"] == ["NVDA"]

    def test_evidence_to_entity_links(self):
        evidence = [_make_tickerevidence()]
        links = _evidence_to_entity_links(evidence)

        assert len(links) == 2
        assert {link["entity_type"] for link in links} == {"sector", "ticker"}
        ticker_link = next(link for link in links if link["entity_type"] == "ticker")
        assert ticker_link["entity_id"] == "NVDA"
        assert ticker_link["relationship"] == "explicit_runtime_attribution"

    def test_empty_evidence_produces_empty_records(self):
        assert _evidence_to_event_records([]) == []
        assert _evidence_to_entity_links([]) == []


# ──────────────────────────────────────────────────────────────────────────────
# SectorAnalyst construction
# ──────────────────────────────────────────────────────────────────────────────


class TestSectorAnalystConstruction:
    def test_creates_for_semiconductor(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        assert analyst.domain_id == "semiconductor_ai_infrastructure"
        assert analyst.profile.display_name == "Semiconductors & AI Infrastructure"

    def test_creates_for_energy(self):
        analyst = SectorAnalyst(domain_id="energy")
        assert analyst.domain_id == "energy"
        assert analyst.profile.display_name == "Energy"

    def test_creates_for_macro(self):
        analyst = SectorAnalyst(domain_id="macro_policy")
        assert analyst.domain_id == "macro_policy"

    def test_creates_for_geopolitics(self):
        analyst = SectorAnalyst(domain_id="geopolitics")
        assert analyst.domain_id == "geopolitics"

    def test_creates_for_liquidity(self):
        analyst = SectorAnalyst(domain_id="liquidity_credit")
        assert analyst.domain_id == "liquidity_credit"

    def test_rejects_unknown_domain(self):
        with pytest.raises(KeyError):
            SectorAnalyst(domain_id="nonexistent_sector")

    def test_custom_agent_name(self):
        analyst = SectorAnalyst(
            domain_id="energy",
            agent_name="my_energy_analyst",
        )
        assert analyst.agent_name == "my_energy_analyst"

    def test_default_agent_name(self):
        analyst = SectorAnalyst(domain_id="energy")
        assert analyst.agent_name == "energy_sector_analyst"


# ──────────────────────────────────────────────────────────────────────────────
# SectorAnalyst.run with evidence list
# ──────────────────────────────────────────────────────────────────────────────


class TestSectorAnalystRunEvidence:
    def test_run_merges_context_and_pre_adapted_evidence(self, monkeypatch):
        analyst = SectorAnalyst(
            domain_id="semiconductor_ai_infrastructure"
        )
        context_item = _make_evidence(
            evidence_id="context_news",
            evidence_type="sector_demand",
        )
        runtime_item = _make_evidence(
            evidence_id="runtime_pipeline",
            evidence_type="market_confirmation",
            summary="Clean three-lane pipeline context confirmed",
        )
        monkeypatch.setattr(
            analyst.evidence_adapter,
            "adapt",
            lambda context, as_of: {
                "evidence": [context_item],
                "exclusions": [],
            },
        )

        report = analyst.run(
            context=object(),
            as_of="2026-07-01T00:00:00Z",
            pre_adapted_evidence=[runtime_item],
        )

        assert report.evidence_count == 2
        assert {item.evidence_id for item in report.evidence} == {
            "context_news",
            "runtime_pipeline",
        }

    def test_run_deduplicates_same_evidence_across_streams(self, monkeypatch):
        analyst = SectorAnalyst(
            domain_id="semiconductor_ai_infrastructure"
        )
        context_item = _make_evidence(evidence_id="context_copy")
        runtime_item = context_item.model_copy(
            update={"evidence_id": "runtime_copy"}
        )
        monkeypatch.setattr(
            analyst.evidence_adapter,
            "adapt",
            lambda context, as_of: {
                "evidence": [context_item],
                "exclusions": [],
            },
        )

        report = analyst.run(
            context=object(),
            as_of="2026-07-01T00:00:00Z",
            pre_adapted_evidence=[runtime_item],
        )

        assert report.evidence_count == 1
        assert report.evidence_exclusion_count == 1
        assert report.evidence[0].evidence_id == "context_copy"

    def test_run_with_evidence_items(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand", stance_hint="positive"),
            _make_evidence(
                evidence_id="ev_002",
                evidence_type="capex_cycle",
                summary="Hyperscaler capex increases",
                stance_hint="positive",
            ),
            _make_evidence(
                evidence_id="ev_003",
                evidence_type="supply_chain",
                summary="Foundry capacity expansion",
                stance_hint="positive",
            ),
            _make_evidence(
                evidence_id="ev_004",
                evidence_type="policy_or_geopolitical",
                summary="Export controls tighten",
                stance_hint="negative",
            ),
            _make_evidence(
                evidence_id="ev_005",
                evidence_type="market_confirmation",
                summary="Relative strength breakout",
                stance_hint="positive",
            ),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        assert isinstance(report, SectorReport)
        assert report.domain_id == "semiconductor_ai_infrastructure"
        assert report.as_of == "2026-07-01T00:00:00Z"
        assert report.evidence_count == 5

    def test_report_has_thesis(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand"),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        assert report.thesis is not None
        assert report.thesis.stance in (
            "constructive",
            "risk_heavy",
            "neutral",
            "mixed",
            "insufficient_data",
        )
        assert 0.0 <= report.thesis.confidence <= 1.0

    def test_report_has_ticker_basket(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand"),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        basket = report.ticker_basket
        assert basket is not None
        assert len(basket.candidates) > 0
        # All candidates should have valid statuses
        for c in basket.candidates:
            assert c.candidate_status in (
                "direct_ticker_thesis",
                "basket_candidate",
                "blocked_missing_evidence",
            )

    def test_report_has_lens_output(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand"),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        # Lenses should have produced output
        assert report.lens_count > 0
        assert len(report.delta_trail) > 0
        for delta in report.delta_trail:
            assert isinstance(delta, ModuleDelta)
            assert delta.review_only is True

    def test_report_has_regime_context(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand"),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        assert report.regime_context is not None
        assert isinstance(report.regime_context, RegimeContextVector)

    def test_report_has_hypotheses(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            # Use evidence types that match hypothesis/evidence-gap templates
            _make_evidence(evidence_type="demand_driver"),
            _make_evidence(evidence_id="ev_002", evidence_type="supply_disruption"),
            _make_evidence(evidence_id="ev_003", evidence_type="oil_shock"),
            _make_evidence(evidence_id="ev_004", evidence_type="tariff"),
            _make_evidence(evidence_id="ev_005", evidence_type="central_bank_decision"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        assert len(report.hypotheses) > 0
        for h in report.hypotheses:
            assert isinstance(h, HypothesisLedgerEntry)
            assert len(h.invalidation_signals) > 0  # Must be falsifiable

    def test_report_has_evidence_gaps(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="demand_driver"),
            _make_evidence(evidence_id="ev_002", evidence_type="supply_disruption"),
            _make_evidence(evidence_id="ev_003", evidence_type="oil_shock"),
            _make_evidence(evidence_id="ev_004", evidence_type="tariff"),
            _make_evidence(evidence_id="ev_005", evidence_type="central_bank_decision"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        assert len(report.evidence_gaps) > 0
        for gap in report.evidence_gaps:
            assert isinstance(gap, EvidenceGap)

    def test_report_always_review_only(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [_make_evidence(evidence_type="sector_demand")]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        assert report.review_required is True
        assert report.live_execution_allowed is False

    def test_report_summary(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand"),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")
        summary = report.summary()

        assert "semiconductor_ai_infrastructure" in summary
        assert "evidence" in summary
        assert "lens" in summary

    def test_report_to_dict(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand"),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")
        d = report.to_dict()

        assert d["report_type"] == "sector_analysis"
        assert d["domain_id"] == "semiconductor_ai_infrastructure"
        assert d["review_required"] is True
        assert d["live_execution_allowed"] is False
        assert "thesis" in d
        assert "ticker_basket" in d
        assert "regime_context" in d
        assert "hypotheses" in d
        assert "evidence_gaps" in d
        assert "stats" in d


# ──────────────────────────────────────────────────────────────────────────────
# SectorAnalyst with ticker evidence
# ──────────────────────────────────────────────────────────────────────────────


class TestSectorAnalystTickerEvidence:
    def test_direct_ticker_thesis_when_ticker_evidence_present(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_tickerevidence(),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        nvda_candidates = [
            c for c in report.ticker_basket.candidates if c.ticker == "NVDA"
        ]
        assert len(nvda_candidates) == 1
        assert nvda_candidates[0].candidate_status == "direct_ticker_thesis"
        assert len(nvda_candidates[0].ticker_specific_evidence_ids) > 0

    def test_basket_candidate_without_ticker_evidence(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand"),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        # Without ticker-specific evidence, all should be basket_candidates
        for c in report.ticker_basket.candidates:
            assert c.candidate_status in ("basket_candidate", "blocked_missing_evidence")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-sector replication
# ──────────────────────────────────────────────────────────────────────────────


class TestCrossSectorReplication:
    def test_energy_analyst_works(self):
        analyst = SectorAnalyst(domain_id="energy")
        evidence = [
            AnalystEvidenceItem(
                evidence_id="en_001",
                source_type="news",
                source="cnbc",
                as_of="2026-07-01T00:00:00Z",
                domain_id="energy",
                tickers=[],
                sectors=["energy"],
                evidence_type="supply",
                summary="OPEC announces production cut",
                stance_hint="positive",
                strength=0.8,
                freshness_score=0.9,
                directness="sector",
                reliability_score=0.7,
            ),
            AnalystEvidenceItem(
                evidence_id="en_002",
                source_type="news",
                source="reuters",
                as_of="2026-07-01T00:00:00Z",
                domain_id="energy",
                tickers=[],
                sectors=["energy"],
                evidence_type="demand",
                summary="Global oil demand grows",
                stance_hint="positive",
                strength=0.7,
                freshness_score=0.8,
                directness="sector",
                reliability_score=0.6,
            ),
            AnalystEvidenceItem(
                evidence_id="en_003",
                source_type="news",
                source="eia",
                as_of="2026-07-01T00:00:00Z",
                domain_id="energy",
                tickers=[],
                sectors=["energy"],
                evidence_type="inventories",
                summary="US crude inventories draw",
                stance_hint="positive",
                strength=0.6,
                freshness_score=0.7,
                directness="sector",
                reliability_score=0.8,
            ),
            AnalystEvidenceItem(
                evidence_id="en_004",
                source_type="news",
                source="bloomberg",
                as_of="2026-07-01T00:00:00Z",
                domain_id="energy",
                tickers=[],
                sectors=["energy"],
                evidence_type="policy_or_geopolitical",
                summary="Middle East tensions rise",
                stance_hint="negative",
                strength=0.7,
                freshness_score=0.9,
                directness="sector",
                reliability_score=0.6,
            ),
            AnalystEvidenceItem(
                evidence_id="en_005",
                source_type="news",
                source="wsj",
                as_of="2026-07-01T00:00:00Z",
                domain_id="energy",
                tickers=[],
                sectors=["energy"],
                evidence_type="market_confirmation",
                summary="Brent crude rallies",
                stance_hint="positive",
                strength=0.6,
                freshness_score=0.8,
                directness="sector",
                reliability_score=0.5,
            ),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        assert report.domain_id == "energy"
        assert report.thesis is not None
        assert len(report.ticker_basket.candidates) > 0
        assert report.lens_count > 0

    def test_geopolitics_analyst_works(self):
        analyst = SectorAnalyst(domain_id="geopolitics")
        evidence = [
            AnalystEvidenceItem(
                evidence_id="geo_001",
                source_type="news",
                source="reuters",
                as_of="2026-07-01T00:00:00Z",
                domain_id="geopolitics",
                tickers=[],
                sectors=["geopolitics"],
                evidence_type="geopolitical_event",
                summary="Trade tensions between US and China escalate",
                stance_hint="negative",
                strength=0.8,
                freshness_score=0.9,
                directness="sector",
                reliability_score=0.7,
            ),
            AnalystEvidenceItem(
                evidence_id="geo_002",
                source_type="news",
                source="bloomberg",
                as_of="2026-07-01T00:00:00Z",
                domain_id="geopolitics",
                tickers=[],
                sectors=["geopolitics"],
                evidence_type="policy_or_sanctions",
                summary="New semiconductor export controls announced",
                stance_hint="negative",
                strength=0.85,
                freshness_score=0.9,
                directness="sector",
                reliability_score=0.8,
            ),
            AnalystEvidenceItem(
                evidence_id="geo_003",
                source_type="news",
                source="wsj",
                as_of="2026-07-01T00:00:00Z",
                domain_id="geopolitics",
                tickers=[],
                sectors=["geopolitics"],
                evidence_type="exposure_mapping",
                summary="Taiwan supply chain concentration risk",
                stance_hint="negative",
                strength=0.7,
                freshness_score=0.8,
                directness="sector",
                reliability_score=0.6,
            ),
            AnalystEvidenceItem(
                evidence_id="geo_004",
                source_type="news",
                source="cnbc",
                as_of="2026-07-01T00:00:00Z",
                domain_id="geopolitics",
                tickers=[],
                sectors=["geopolitics"],
                evidence_type="market_confirmation",
                summary="Defense stocks rally on geopolitical risk",
                stance_hint="positive",
                strength=0.6,
                freshness_score=0.8,
                directness="sector",
                reliability_score=0.5,
            ),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        assert report.domain_id == "geopolitics"
        assert report.thesis is not None
        assert report.regime_context is not None


# ──────────────────────────────────────────────────────────────────────────────
# Safety invariants
# ──────────────────────────────────────────────────────────────────────────────


class TestSectorAnalystSafety:
    def test_all_deltas_are_review_only(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand"),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        for delta in report.delta_trail:
            assert delta.review_only is True

    def test_report_cannot_enable_live_execution(self):
        """AnalystReport validator blocks live_execution_allowed=True."""
        with pytest.raises(Exception):
            AnalystReport(
                agent_name="test",
                domain_id="test",
                as_of="2026-07-01",
                horizon_days=180,
                domain_profile_version="0.1.0",
                thesis=None,
                ticker_basket=None,
                recommendation="blocked",
                live_execution_allowed=True,  # This should fail
            )

    def test_hypotheses_must_be_falsifiable(self):
        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        evidence = [
            _make_evidence(evidence_type="sector_demand"),
            _make_evidence(evidence_id="ev_002", evidence_type="capex_cycle"),
            _make_evidence(evidence_id="ev_003", evidence_type="supply_chain"),
            _make_evidence(evidence_id="ev_004", evidence_type="policy_or_geopolitical"),
            _make_evidence(evidence_id="ev_005", evidence_type="market_confirmation"),
        ]

        report = analyst.run_from_evidence(evidence, as_of="2026-07-01T00:00:00Z")

        for h in report.hypotheses:
            assert len(h.invalidation_signals) > 0, (
                f"Hypothesis '{h.hypothesis[:50]}' must have invalidation_signals"
            )
