"""Tests for MarketContextEvidenceAdapter's macro evidence_type routing.

Context: MACRO_SERIES_EVIDENCE_MAP was keyed by raw FRED series id (e.g.
"FEDFUNDS", "CPIAUCSL") but _macro_series_evidence_type is always called with
observation["name"], which structured_context_provenance.audit_structured_context
populates from the *registry context_key* (e.g. "fed_funds_rate", "cpi") --
SavedMacroEvidenceProducer's market_context_fragment.macro dict is keyed by
context_key, not series id. The mismatch meant every macro observation, for
every domain, silently fell through to the single generic domain-default
evidence_type -- confirmed live against a real SavedMacroEvidenceProducer
artifact, where all 19 accepted series came back with evidence_type=="inflation"
regardless of which series they actually were.

This is the second instance of the exact same key-shape bug (the first was
RegimeContextLens.MACRO_SERIES_TO_DIMENSION) -- both times a hand-built unit
test with the wrong key assumption would have passed anyway, so these tests
exercise the real context_key strings from dean_os/config/macro_series_registry.yaml
rather than inventing fixture keys.

A third, deeper bug in the same chain: for a macro series NOT in
MACRO_SERIES_EVIDENCE_MAP for the given domain, evidence_type used to fall
back to ``self._first_required_or(self.profile.macro_evidence_type or
"market_confirmation")`` -- and _first_required_or always prefers
profile.required_evidence_types[0] over the fallback it was given whenever
that list is non-empty (true for every real domain profile). Every domain
profile sets macro_evidence_type: "macro_context" as a deliberate sentinel
that EventClassifierLens._classify_macro_observation checks for by exact
string match to keyword-classify the series by name instead of falling
through to generic text detection -- so the override silently (a) defeated
that classifier branch for every unmapped macro series, for every domain,
and (b) let irrelevant macro noise falsely "satisfy" whatever the domain's
first required evidence lane happened to be.
"""
from __future__ import annotations

import yaml

from dean_os.analyst_core.lens_contract import AnalysisPacket
from dean_os.analyst_core.lenses.event_classifier_lens import EventClassifierLens
from dean_os.analysts.context_adapter import (
    MACRO_SERIES_EVIDENCE_MAP,
    MarketContextEvidenceAdapter,
    _macro_series_evidence_type,
)
from dean_os.schemas import MarketContext

REGISTRY_PATH = "dean_os/config/macro_series_registry.yaml"


def _registry_context_keys() -> set[str]:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    return {entry["context_key"] for entry in registry["series"].values()}


class TestMacroSeriesEvidenceMapKeys:
    def test_every_map_key_is_a_real_registry_context_key(self):
        """Guards against reintroducing FRED series ids (e.g. "FEDFUNDS")
        as keys instead of registry context_keys (e.g. "fed_funds_rate")."""
        registry_keys = _registry_context_keys()
        for key in MACRO_SERIES_EVIDENCE_MAP:
            assert key in registry_keys, f"{key!r} is not a registered context_key"

    def test_fed_funds_rate_resolves_for_macro_policy_domain(self):
        assert _macro_series_evidence_type("fed_funds_rate", "macro_policy") == "rates_policy"

    def test_unmapped_context_key_returns_none(self):
        assert _macro_series_evidence_type("not_a_real_key", "macro_policy") is None


class TestAdapterMacroEvidenceTypeVaries:
    def test_evidence_type_varies_by_series_not_constant(self):
        """Regression for the exact live-data symptom: every macro
        observation getting the same evidence_type regardless of series."""
        macro = {
            "fed_funds_rate": {
                "value": 4.5, "unit": "percent", "period": "2026-06-01",
                "available_at": "2026-07-01T00:00:00+00:00",
                "source_url": "https://fred.stlouisfed.org/series/FEDFUNDS",
                "metadata": {"series_id": "FEDFUNDS"},
            },
            "cpi": {
                "value": 332.0, "unit": "index_1982_1984_100", "period": "2026-06-01",
                "available_at": "2026-07-01T00:00:00+00:00",
                "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
                "metadata": {"series_id": "CPIAUCSL"},
            },
            "vix": {
                "value": 18.0, "unit": "index_points", "period": "2026-06-30",
                "available_at": "2026-07-01T00:00:00+00:00",
                "source_url": "https://fred.stlouisfed.org/series/VIXCLS",
                "metadata": {"series_id": "VIXCLS"},
            },
        }
        as_of = "2026-07-02T00:00:00+00:00"
        context = MarketContext(macro=macro, as_of=as_of)
        adapted = MarketContextEvidenceAdapter("macro_policy").adapt(context, as_of=as_of)
        evidence_types = {item.summary.split("=")[0].split()[-1]: item.evidence_type for item in adapted["evidence"]}

        assert evidence_types.get("fed_funds_rate") == "rates_policy"
        assert evidence_types.get("cpi") == "inflation"
        assert evidence_types.get("vix") == "market_confirmation"
        # The whole point: not every series collapsing to one shared value.
        assert len(set(evidence_types.values())) > 1


class TestUnmappedMacroSeriesFallback:
    """A macro series with no MACRO_SERIES_EVIDENCE_MAP entry for the domain
    must fall back to the domain's own macro_evidence_type ("macro_context")
    -- never to an unrelated required_evidence_types entry, which would mean
    unclassified macro noise falsely satisfying an unrelated required lane."""

    def _adapt_single_observation(self, context_key: str, domain_id: str):
        macro = {
            context_key: {
                "value": 1.0, "unit": "index_points", "period": "2026-06-30",
                "available_at": "2026-07-01T00:00:00+00:00",
                "source_url": "https://fred.stlouisfed.org/series/UNMAPPED",
            }
        }
        as_of = "2026-07-02T00:00:00+00:00"
        context = MarketContext(macro=macro, as_of=as_of)
        adapted = MarketContextEvidenceAdapter(domain_id).adapt(context, as_of=as_of)
        return adapted["evidence"][0]

    def test_unmapped_series_falls_back_to_macro_context_sentinel(self):
        item = self._adapt_single_observation("not_a_registered_series", "semiconductor_ai_infrastructure")
        assert item.evidence_type == "macro_context"

    def test_unmapped_series_never_borrows_a_required_lane_name(self):
        item = self._adapt_single_observation("not_a_registered_series", "semiconductor_ai_infrastructure")
        assert item.evidence_type != "sector_demand"


class TestEventClassifierMacroContextBranchIsReachable:
    """EventClassifierLens._classify_macro_observation only fires when
    evidence_type == "macro_context" exactly -- this is a regression test
    that the branch is actually reachable end-to-end (adapter -> event
    record -> classifier), not just correct in isolation."""

    def _classify_via_full_adapter_path(self, context_key: str, domain_id: str) -> str:
        macro = {
            context_key: {
                "value": 1.0, "unit": "percent", "period": "2026-06-30",
                "available_at": "2026-07-01T00:00:00+00:00",
                "source_url": "https://fred.stlouisfed.org/series/UNMAPPED",
            }
        }
        as_of = "2026-07-02T00:00:00+00:00"
        context = MarketContext(macro=macro, as_of=as_of)
        adapted = MarketContextEvidenceAdapter(domain_id).adapt(context, as_of=as_of)
        item = adapted["evidence"][0]
        record = {
            "event_id": item.evidence_id,
            "evidence_id": item.evidence_id,
            "title": item.summary[:120],
            "text": item.summary,
            "event_class": item.evidence_type,
            "evidence_type": item.evidence_type,
            "source_type": item.source_type,
            "provenance": item.provenance,
        }
        delta = EventClassifierLens().analyze(
            AnalysisPacket(packet_id="p1", event_records=[record])
        )
        return delta.classified_events_added[0]["event_class"]

    def test_unmapped_inflation_like_series_reaches_inflation_observation(self):
        event_class = self._classify_via_full_adapter_path("core_cpi_unmapped_variant", "semiconductor_ai_infrastructure")
        assert event_class == "inflation_observation"

    def test_unmapped_central_bank_like_series_reaches_liquidity_observation(self):
        event_class = self._classify_via_full_adapter_path("central_bank_balance_sheet_unmapped", "semiconductor_ai_infrastructure")
        assert event_class == "liquidity_observation"
