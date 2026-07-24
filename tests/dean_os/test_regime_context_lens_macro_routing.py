"""Tests for RegimeContextLens's macro-evidence routing fix.

Context: SavedMacroEvidenceProducer (dean_os/analysts/_producers/macro.py) and
MarketContextEvidenceAdapter (dean_os/analysts/context_adapter.py) already
turn FRED series (FEDFUNDS, T10Y2Y, VIXCLS, ...) into AnalysisPacket
event_records with event_class values like "rates_policy"/"inflation"/
"market_confirmation" (MarketContextEvidenceAdapter.MACRO_SERIES_EVIDENCE_MAP).
None of those strings appeared in RegimeContextLens.EVENT_CLASS_TO_DIMENSION
(which was built for discrete news events: "central_bank_decision",
"inflation_release", ...), so every macro observation that reached the lens
was silently dropped (`if dimension is None: continue`) -- market_state,
liquidity_credit_context, and safe_haven_behavior were structurally always
"unknown" regardless of how much real macro evidence existed.

This fix routes by the registry's context_key (read from
event["provenance"]["name"], which structured_context_provenance.
audit_structured_context populates from SavedMacroEvidenceProducer's
market_context_fragment.macro dict -- keyed by context_key, NOT the raw FRED
series id) directly to the regime dimension it speaks to, and adds
single-snapshot numeric-threshold grading for the two series where one
observation is enough to say something directional (yield curve sign, VIX
level). This was verified end-to-end against a real SavedMacroEvidenceProducer
artifact built from data/processed/features/macro_data.parquet, which is how
the context_key-vs-series-id shape mismatch was caught in the first place --
an earlier version of this fix used the raw series id and matched nothing.
"""
from __future__ import annotations

from dean_os.analyst_core.lens_contract import AnalysisPacket
from dean_os.analyst_core.lenses.regime_context_lens import (
    MACRO_SERIES_TO_DIMENSION,
    RegimeContextLens,
)
from dean_os.analyst_core.schemas import Trend


def _macro_event(context_key: str, value: float, *, event_id: str = "ev_macro_1") -> dict:
    """Shape-matches what MarketContextEvidenceAdapter._structured_context_evidence
    + sector_analyst._evidence_to_event_records actually produce for a macro
    observation: provenance["name"] is the registry context_key (e.g.
    "fed_funds_rate"), not the raw FRED series id (e.g. "FEDFUNDS")."""
    return {
        "event_id": event_id,
        "evidence_id": event_id,
        "title": f"macro observation {context_key}={value}",
        "text": f"macro observation {context_key}={value}",
        "event_class": "rates_policy",  # the coarse, non-matching evidence_type
        "evidence_type": "rates_policy",
        "source_type": "macro",
        "provenance": {"family": "macro", "scope": "macro", "name": context_key, "value": value},
    }


class TestMacroSeriesRouting:
    def test_fed_funds_rate_routes_to_liquidity_credit_context(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p1", event_records=[_macro_event("fed_funds_rate", 4.5)])
        delta = lens.analyze(packet)
        dim = delta.regime_context.dimensions["liquidity_credit_context"]
        assert dim.state == "fed_funds_rate_signal"
        assert dim.confidence.value in {"low", "medium"}

    def test_cpi_routes_to_inflation_rates_context(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p2", event_records=[_macro_event("cpi", 332.0)])
        delta = lens.analyze(packet)
        assert delta.regime_context.dimensions["inflation_rates_context"].state == "cpi_signal"

    def test_vix_routes_to_safe_haven_behavior(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p3", event_records=[_macro_event("vix", 30.0)])
        delta = lens.analyze(packet)
        assert delta.regime_context.dimensions["safe_haven_behavior"].state == "vix_elevated_fear_signal"

    def test_oil_routes_to_commodity_stress(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p4", event_records=[_macro_event("wti_crude_oil", 85.0)])
        delta = lens.analyze(packet)
        assert delta.regime_context.dimensions["commodity_stress"].state == "wti_crude_oil_signal"

    def test_unmapped_fx_series_is_dropped_not_guessed(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p5", event_records=[_macro_event("usd_cny", 7.2)])
        delta = lens.analyze(packet)
        # No dimension should be touched by an unmapped series.
        for dimension in delta.regime_context.dimensions.values():
            assert dimension.state == "unknown"

    def test_all_registry_series_except_fx_pairs_are_mapped(self):
        assert "usd_cny" not in MACRO_SERIES_TO_DIMENSION
        assert "usd_eur" not in MACRO_SERIES_TO_DIMENSION
        assert len(MACRO_SERIES_TO_DIMENSION) == 25

    def test_non_macro_event_unaffected_by_grading(self):
        """Regression: news events (source_type != 'macro') must route exactly
        as before -- this fix must not touch the existing news path."""
        lens = RegimeContextLens()
        news_event = {
            "event_id": "ev_news_1",
            "event_class": "war_escalation",
            "intensity": 0.8,
        }
        packet = AnalysisPacket(packet_id="p6", event_records=[news_event])
        delta = lens.analyze(packet)
        assert delta.regime_context.dimensions["geopolitical_state"].state == "war_escalation_signal"

    def test_grading_does_not_mutate_original_event_dict(self):
        lens = RegimeContextLens()
        original = _macro_event("fed_funds_rate", 4.5)
        snapshot = dict(original)
        packet = AnalysisPacket(packet_id="p7", event_records=[original])
        lens.analyze(packet)
        assert original == snapshot


class TestYieldCurveThresholdGrading:
    def test_negative_slope_is_inverted(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p8", event_records=[_macro_event("yield_curve_10y_2y", -0.15)])
        delta = lens.analyze(packet)
        dim = delta.regime_context.dimensions["liquidity_credit_context"]
        assert dim.state == "yield_curve_inverted_signal"
        assert dim.trend == Trend.FALLING

    def test_positive_slope_is_normal(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p9", event_records=[_macro_event("yield_curve_10y_2y", 0.4)])
        delta = lens.analyze(packet)
        dim = delta.regime_context.dimensions["liquidity_credit_context"]
        assert dim.state == "yield_curve_normal_signal"
        assert dim.trend == Trend.STABLE

    def test_zero_slope_is_not_inverted(self):
        """0.0 is the textbook inversion threshold: exactly 0 is not < 0."""
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p10", event_records=[_macro_event("yield_curve_10y_2y", 0.0)])
        delta = lens.analyze(packet)
        assert delta.regime_context.dimensions["liquidity_credit_context"].state == "yield_curve_normal_signal"


class TestVixThresholdGrading:
    def test_elevated_fear_band(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p11", event_records=[_macro_event("vix", 28.0)])
        delta = lens.analyze(packet)
        dim = delta.regime_context.dimensions["safe_haven_behavior"]
        assert dim.state == "vix_elevated_fear_signal"
        assert dim.trend == Trend.RISING

    def test_complacency_band(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p12", event_records=[_macro_event("vix", 12.0)])
        delta = lens.analyze(packet)
        dim = delta.regime_context.dimensions["safe_haven_behavior"]
        assert dim.state == "vix_complacent_signal"
        assert dim.trend == Trend.FALLING

    def test_neutral_band(self):
        lens = RegimeContextLens()
        packet = AnalysisPacket(packet_id="p13", event_records=[_macro_event("vix", 18.0)])
        delta = lens.analyze(packet)
        dim = delta.regime_context.dimensions["safe_haven_behavior"]
        assert dim.state == "vix_neutral_signal"
        assert dim.trend == Trend.STABLE


class TestMultipleMacroSeriesAggregation:
    def test_two_series_in_same_dimension_produce_mixed_signals(self):
        """fed_funds_rate and an inverted yield curve both land in
        liquidity_credit_context with different (synthetic) event classes ->
        mixed_signals, matching the existing aggregation rule for any
        multi-class dimension."""
        lens = RegimeContextLens()
        packet = AnalysisPacket(
            packet_id="p14",
            event_records=[
                _macro_event("fed_funds_rate", 4.5, event_id="ev_1"),
                _macro_event("yield_curve_10y_2y", -0.2, event_id="ev_2"),
            ],
        )
        delta = lens.analyze(packet)
        assert delta.regime_context.dimensions["liquidity_credit_context"].state == "mixed_signals"


class TestLiveMacroEvidenceIntegration:
    """End-to-end against the real evidence-adaptation path (not hand-built
    event dicts), to guard against the exact context_key-vs-series-id shape
    mismatch this fix's first version had."""

    def test_saved_producer_fragment_flows_through_adapter_into_graded_dimensions(self):
        from dean_os.analyst_core.sector_analyst import _evidence_to_event_records
        from dean_os.analysts.context_adapter import MarketContextEvidenceAdapter
        from dean_os.schemas import MarketContext

        as_of = "2026-07-23T23:59:59+00:00"
        # Mirrors SavedMacroEvidenceProducer.build()'s market_context_fragment.macro
        # shape: keyed by context_key, each value a dict with value/unit/period/
        # available_at/source_url/metadata.
        macro_fragment = {
            "fed_funds_rate": {
                "value": 4.5,
                "unit": "percent",
                "period": "2026-06-01",
                "available_at": "2026-07-01T23:59:59.999999+00:00",
                "source_url": "https://fred.stlouisfed.org/series/FEDFUNDS",
                "metadata": {"series_id": "FEDFUNDS", "series_name": "Effective Federal Funds Rate"},
            },
            "yield_curve_10y_2y": {
                "value": -0.2,
                "unit": "percentage_points",
                "period": "2026-07-20",
                "available_at": "2026-07-21T23:59:59.999999+00:00",
                "source_url": "https://fred.stlouisfed.org/series/T10Y2Y",
                "metadata": {"series_id": "T10Y2Y", "series_name": "10Y-2Y Treasury Spread"},
            },
        }
        context = MarketContext(macro=macro_fragment, as_of=as_of)
        adapted = MarketContextEvidenceAdapter("macro_policy").adapt(context, as_of=as_of)
        assert len(adapted["evidence"]) == 2

        records = _evidence_to_event_records(list(adapted["evidence"]))
        packet = AnalysisPacket(packet_id="live_integration", event_records=records, as_of_date=as_of)
        delta = RegimeContextLens().analyze(packet)

        dim = delta.regime_context.dimensions["liquidity_credit_context"]
        assert dim.state == "mixed_signals"  # fed_funds_rate_signal + yield_curve_inverted_signal
        assert dim.confidence.value in {"low", "medium"}
