"""SectorAnalyst — unified orchestrator for any economic sector.

This is the single entry point for running a full domain analysis. It wires:
    MarketContext → EvidenceAdapter → AnalysisPacket → LensOrchestrator → BaseAnalystAgent → Report

One class, many sectors. Swap the sector profile, get a different analyst.

Design principles (from draft/thinking):
- Review-only: no trades, no live execution, no config writes
- Evidence-gated: missing evidence blocks the pipeline
- Modular: lenses are plugins, not hardcoded steps
- Auditable: every state change is a ModuleDelta in the delta trail
- Falsifiable: hypotheses carry invalidation signals, analogs carry false-analogy risk
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
import json
from pathlib import Path

from dean_os.analyst_core.lens_contract import AnalysisPacket, LensRegistry, ModuleDelta
from dean_os.analyst_core.lens_orchestrator import LensOrchestrator
from dean_os.analyst_core.schemas import EvidenceGap, HypothesisLedgerEntry
from dean_os.analysts.base import BaseAnalystAgent
from dean_os.analysts.context_adapter import MarketContextEvidenceAdapter
from dean_os.analysts.profiles import get_domain_profile
from dean_os.analyst_core.cross_domain_signal_bus import CROSS_DOMAIN_PROPAGATION
from dean_os.analysts.schemas import (
    AnalystEvidenceItem,
    AnalystReport,
    DomainProfile,
)
from dean_os.utils import sha256_json


def _build_default_registry() -> LensRegistry:
    """Build the verified production registry.

    Probability-like expectation-gap heuristics and static historical analog
    templates remain available as experimental lenses, but are intentionally
    excluded from the real artifact path until empirical inputs exist.
    """
    from dean_os.analyst_core.lenses.event_classifier_lens import EventClassifierLens
    from dean_os.analyst_core.lenses.evidence_gap_lens import EvidenceGapLens
    from dean_os.analyst_core.lenses.hypothesis_ledger_lens import HypothesisLedgerLens
    from dean_os.analyst_core.lenses.regime_context_lens import RegimeContextLens
    from dean_os.analyst_core.lenses.transmission_mapper_lens import TransmissionMapperLens

    registry = LensRegistry()
    # Order matters: classifier first, then regime, then the rest
    registry.register(EventClassifierLens())
    registry.register(RegimeContextLens())
    registry.register(TransmissionMapperLens())
    registry.register(HypothesisLedgerLens())
    registry.register(EvidenceGapLens())
    return registry


def _evidence_to_event_records(
    evidence: list[AnalystEvidenceItem],
) -> list[dict[str, Any]]:
    """Convert AnalystEvidenceItem list to event_records for the AnalysisPacket.

    Each evidence item becomes an event record that lenses can classify
    and reason about.
    """
    records: list[dict[str, Any]] = []
    for item in evidence:
        records.append({
            "event_id": item.evidence_id,
            "evidence_id": item.evidence_id,
            "title": item.summary[:120],
            "text": item.summary,
            "event_class": item.evidence_type,
            "evidence_type": item.evidence_type,
            "source": item.source,
            "source_type": item.source_type,
            "tickers": item.tickers,
            "sectors": item.sectors,
            "stance_hint": item.stance_hint,
            "directness": item.directness,
            "strength": item.strength,
            "reliability_score": item.reliability_score,
            "freshness_score": item.freshness_score,
            "required_lane_eligible": bool(
                item.provenance.get("required_lane_eligible", False)
            ),
            "provenance": item.provenance,
            "point_in_time": item.point_in_time,
            "published_at": item.published_at,
            "as_of": item.as_of,
        })
    return records


def _evidence_to_entity_links(
    evidence: list[AnalystEvidenceItem],
) -> list[dict[str, Any]]:
    """Convert AnalystEvidenceItem list to entity_links for the AnalysisPacket.

    Entity links connect evidence to specific entities (companies, sectors,
    commodities) and carry classification metadata for lenses.
    """
    links: list[dict[str, Any]] = []
    for item in evidence:
        for sector in item.sectors:
            links.append({
                "link_id": f"link_{item.evidence_id}_sector_{sector}",
                "event_id": item.evidence_id,
                "evidence_id": item.evidence_id,
                "entity_type": "sector",
                "entity_id": sector,
                "relationship": "evidence_applies_to",
            })
        for ticker in item.tickers:
            links.append({
                "link_id": f"link_{item.evidence_id}_ticker_{ticker}",
                "event_id": item.evidence_id,
                "evidence_id": item.evidence_id,
                "entity_type": "ticker",
                "entity_id": ticker,
                "relationship": "explicit_runtime_attribution",
            })
    return links


class SectorAnalyst:
    """Unified sector analyst — one class, any sector.

    Usage::

        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        report = analyst.run(market_context, as_of="2026-07-01")

    The report contains:
    - domain thesis (from BaseAnalystAgent)
    - ticker basket (undervalued candidates with guardrails)
    - lens analysis (regime context, scenario graph, hypotheses, evidence gaps)
    - quality gates and review packet
    - outcome tracking plan

    To create a new sector analyst, just pass a different domain_id::

        analyst = SectorAnalyst(domain_id="energy")
        report = analyst.run(energy_context, as_of="2026-07-01")
    """

    def __init__(
        self,
        domain_id: str,
        *,
        agent_name: str | None = None,
        registry: LensRegistry | None = None,
        lens_config: dict[str, Any] | None = None,
    ):
        """Initialize the sector analyst.

        Args:
            domain_id: Sector identifier (e.g. "semiconductor_ai_infrastructure").
            agent_name: Optional human-readable name for this analyst instance.
            registry: Optional pre-built LensRegistry. If None, builds default.
            lens_config: Optional config passed to all lenses.
        """
        self.domain_id = domain_id
        self.profile: DomainProfile = get_domain_profile(domain_id)
        self.agent_name = agent_name or f"{domain_id}_sector_analyst"

        # Evidence adapter: MarketContext → AnalystEvidenceItem[]
        self.evidence_adapter = MarketContextEvidenceAdapter(domain_id)

        # Base analyst agent: evidence → thesis → ticker basket → report
        self.base_agent = BaseAnalystAgent(
            domain_id=domain_id,
            agent_name=self.agent_name,
        )

        # Lens orchestrator: AnalysisPacket → enriched AnalysisPacket
        self.registry = registry or _build_default_registry()
        effective_lens_config = dict(lens_config or {})
        effective_lens_config.setdefault("domain_id", domain_id)
        effective_lens_config.setdefault(
            "default_horizon_days",
            self.profile.horizon_days_default,
        )
        effective_lens_config.setdefault(
            "checkpoint_horizons",
            [
                value
                for value in (30, 90, 180)
                if value in self.profile.allowed_horizons
            ],
        )
        effective_lens_config.setdefault("sector_keywords", self.profile.sector_keywords)
        effective_lens_config.setdefault("ticker_universe", self.profile.ticker_universe_hint)
        self.lens_orchestrator = LensOrchestrator(
            self.registry,
            config=effective_lens_config,
        )
        self.enable_cross_domain_signal_bus = bool(
            effective_lens_config.get("enable_cross_domain_signal_bus", False)
        )
        self.signal_bus_dir = Path(
            effective_lens_config.get(
                "signal_bus_dir", "reports/dean_os/signal_bus"
            )
        )

    def clone(
        self,
        domain_id: str,
        *,
        agent_name: str | None = None,
        ticker_universe: list[str] | None = None,
        sector_keywords: list[str] | None = None,
        core_questions: list[str] | None = None,
        required_evidence_types: list[str] | None = None,
        lens_config: dict[str, Any] | None = None,
    ) -> SectorAnalyst:
        """Clone this analyst with different domain parameters.

        Creates a new SectorAnalyst instance, reusing the lens registry
        but replacing the domain profile parameters.

        Usage::

            semiconductor = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
            energy = semiconductor.clone(
                domain_id="energy",
                ticker_universe=["XLE", "USO", "XOM", "CVX"],
                sector_keywords=["oil", "gas", "OPEC", "inventories"],
            )

        Args:
            domain_id: New domain identifier.
            agent_name: Optional name for the new analyst.
            ticker_universe: Override ticker list for the new domain.
            sector_keywords: Override sector keywords for the new domain.
            core_questions: Override core questions for the new domain.
            required_evidence_types: Override required evidence types.
            lens_config: Override lens configuration.

        Returns:
            New SectorAnalyst instance with updated profile.
        """
        # Profiles returned by get_domain_profile are shared registry
        # singletons. Build one private profile first, then bind every
        # profile-consuming component to that same private copy.
        private_profile = get_domain_profile(domain_id).model_copy(deep=True)
        if ticker_universe is not None:
            private_profile.ticker_universe_hint = list(ticker_universe)
        if sector_keywords is not None:
            private_profile.sector_keywords = list(sector_keywords)
        if core_questions is not None:
            private_profile.core_questions = list(core_questions)
        if required_evidence_types is not None:
            private_profile.required_evidence_types = list(
                required_evidence_types
            )

        effective_lens_config = dict(lens_config or {})
        effective_lens_config.setdefault(
            "ticker_universe",
            private_profile.ticker_universe_hint,
        )
        effective_lens_config.setdefault(
            "sector_keywords",
            private_profile.sector_keywords,
        )
        new_analyst = SectorAnalyst(
            domain_id=domain_id,
            agent_name=agent_name,
            registry=self.registry,
            lens_config=effective_lens_config,
        )
        new_analyst.profile = private_profile
        new_analyst.base_agent.profile = private_profile
        new_analyst.evidence_adapter.profile = private_profile

        return new_analyst

    def run(
        self,
        context: Any,
        *,
        as_of: str | None = None,
        tickers: list[str] | None = None,
        horizon_days: int | None = None,
        pre_adapted_evidence: list[AnalystEvidenceItem] | None = None,
        prior_hypotheses: list[HypothesisLedgerEntry] | None = None,
    ) -> SectorReport:
        """Run the full sector analysis pipeline.

        Args:
            context: MarketContext object with news, research, fundamentals, macro.
            as_of: Point-in-time cutoff (ISO format). Defaults to now.
            tickers: Optional ticker override. Uses profile universe if None.
            horizon_days: Optional horizon override. Uses profile default if None.
            pre_adapted_evidence: Optional verified evidence merged additively
                with evidence adapted from MarketContext.

        Returns:
            SectorReport with everything: thesis, ticker basket, lens analysis,
            quality gates, review packet, outcome tracking.
        """
        as_of = as_of or datetime.now(UTC).isoformat()
        horizon_days = horizon_days or self.profile.horizon_days_default

        # ── Step 1: Evidence Adaptation ──────────────────────────────────
        # MarketContext → AnalystEvidenceItem[]
        adapted = self.evidence_adapter.adapt(context, as_of=as_of)
        
        from dean_os.analyst_core.artifact_evidence_loader import ArtifactEvidenceLoader
        bus_evidence = (
            ArtifactEvidenceLoader().from_signal_bus(
                self.domain_id,
                as_of,
                bus_dir=self.signal_bus_dir,
            )
            if self.enable_cross_domain_signal_bus
            else []
        )

        evidence, merge_exclusions = _merge_evidence_streams(
            list(adapted["evidence"]) + bus_evidence,
            list(pre_adapted_evidence or []),
        )
        evidence_exclusions = [
            *adapted.get("exclusions", []),
            *merge_exclusions,
        ]

        # ── Step 2: Build AnalysisPacket for Lenses ──────────────────────
        # Convert evidence items into event_records and entity_links
        # that the lens pipeline can reason about.
        event_records = _evidence_to_event_records(evidence)
        entity_links = _evidence_to_entity_links(evidence)

        analysis_input_sha256 = sha256_json(
            {
                "domain_id": self.domain_id,
                "as_of": as_of,
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "prior_hypotheses": [
                    item.model_dump(mode="json")
                    for item in (prior_hypotheses or [])
                ],
            }
        )
        packet = AnalysisPacket(
            packet_id=f"packet_{analysis_input_sha256[:24]}",
            as_of_date=as_of,
            source_packet_ids=[e.evidence_id for e in evidence],
            event_records=event_records,
            entity_links=entity_links,
            hypotheses=list(prior_hypotheses or []),
        )

        # ── Step 3: Lens Pipeline ────────────────────────────────────────
        if not event_records:
            enriched_packet = packet
            delta_trail = []
        else:
            enriched_packet, delta_trail = self.lens_orchestrator.run(packet)
            if self.enable_cross_domain_signal_bus:
                _publish_signals_to_bus(
                    self.domain_id,
                    enriched_packet.classified_events,
                    as_of,
                    bus_dir=self.signal_bus_dir,
                )

        # ── Step 4: Base Analyst Agent ───────────────────────────────────
        # evidence → domain thesis → ticker basket → quality gates → report
        base_report = self.base_agent.run(
            evidence=evidence,
            tickers=tickers or self.profile.ticker_universe_hint,
            horizon_days=horizon_days,
            as_of=as_of,
        )

        from dean_os.analyst_core.formatters.regime_brief_formatter import format_economy_regime_brief
        economy_regime_brief = format_economy_regime_brief(enriched_packet)

        # ── Step 5: Compose SectorReport ─────────────────────────────────
        # Combine base analyst output with lens pipeline output.
        return SectorReport(
            # From BaseAnalystAgent
            base_report=base_report,
            # From LensOrchestrator
            classified_events=enriched_packet.classified_events,
            regime_context=enriched_packet.regime_context,
            scenario_graph=enriched_packet.scenario_graph,
            transmission_channels=enriched_packet.transmission_channels,
            expectation_gap=enriched_packet.expectation_gap,
            hypotheses=enriched_packet.hypotheses,
            hypothesis_review_proposals=(
                enriched_packet.hypothesis_review_proposals
            ),
            evidence_gaps=enriched_packet.evidence_gaps,
            watch_signals=enriched_packet.watch_signals,
            review_notes=enriched_packet.review_notes,
            delta_trail=delta_trail,
            # Metadata
            evidence_count=len(evidence),
            evidence_exclusion_count=len(evidence_exclusions),
            lens_count=len(delta_trail),
            economy_regime_brief=economy_regime_brief,
            analysis_input_sha256=analysis_input_sha256,
            analysis_output_sha256=sha256_json(
                enriched_packet.model_dump(mode="json")
            ),
        )


    def run_from_evidence(
        self,
        evidence: list[AnalystEvidenceItem | dict],
        *,
        as_of: str | None = None,
        tickers: list[str] | None = None,
        horizon_days: int | None = None,
        prior_hypotheses: list[HypothesisLedgerEntry] | None = None,
    ) -> SectorReport:
        """Run analysis from pre-built evidence items (skip adapter).

        Useful when evidence is already adapted (e.g., from saved producers).
        """
        as_of = as_of or datetime.now(UTC).isoformat()
        horizon_days = horizon_days or self.profile.horizon_days_default

        # Normalize evidence
        normalized = self.base_agent.normalize_evidence(evidence, as_of=as_of)

        # Build packet for lenses
        event_records = _evidence_to_event_records(normalized)
        entity_links = _evidence_to_entity_links(normalized)

        analysis_input_sha256 = sha256_json(
            {
                "domain_id": self.domain_id,
                "as_of": as_of,
                "evidence": [item.model_dump(mode="json") for item in normalized],
                "prior_hypotheses": [
                    item.model_dump(mode="json")
                    for item in (prior_hypotheses or [])
                ],
            }
        )
        packet = AnalysisPacket(
            packet_id=f"packet_{analysis_input_sha256[:24]}",
            as_of_date=as_of,
            source_packet_ids=[e.evidence_id for e in normalized],
            event_records=event_records,
            entity_links=entity_links,
            hypotheses=list(prior_hypotheses or []),
        )

        if not normalized:
            enriched_packet = packet
            delta_trail = []
        else:
            enriched_packet, delta_trail = self.lens_orchestrator.run(packet)
            if self.enable_cross_domain_signal_bus:
                _publish_signals_to_bus(
                    self.domain_id,
                    enriched_packet.classified_events,
                    as_of,
                    bus_dir=self.signal_bus_dir,
                )

        # Run base agent
        base_report = self.base_agent.run(
            evidence=normalized,
            tickers=tickers or self.profile.ticker_universe_hint,
            horizon_days=horizon_days,
            as_of=as_of,
        )

        from dean_os.analyst_core.formatters.regime_brief_formatter import format_economy_regime_brief
        economy_regime_brief = format_economy_regime_brief(enriched_packet)

        return SectorReport(
            base_report=base_report,
            classified_events=enriched_packet.classified_events,
            regime_context=enriched_packet.regime_context,
            scenario_graph=enriched_packet.scenario_graph,
            transmission_channels=enriched_packet.transmission_channels,
            expectation_gap=enriched_packet.expectation_gap,
            hypotheses=enriched_packet.hypotheses,
            hypothesis_review_proposals=(
                enriched_packet.hypothesis_review_proposals
            ),
            evidence_gaps=enriched_packet.evidence_gaps,
            watch_signals=enriched_packet.watch_signals,
            review_notes=enriched_packet.review_notes,
            delta_trail=delta_trail,
            evidence_count=len(normalized),
            evidence_exclusion_count=0,
            lens_count=len(delta_trail),
            economy_regime_brief=economy_regime_brief,
            analysis_input_sha256=analysis_input_sha256,
            analysis_output_sha256=sha256_json(
                enriched_packet.model_dump(mode="json")
            ),
        )


def _merge_evidence_streams(
    context_evidence: list[AnalystEvidenceItem],
    pre_adapted_evidence: list[AnalystEvidenceItem],
) -> tuple[list[AnalystEvidenceItem], list[dict[str, Any]]]:
    """Merge evidence without allowing one source family to replace another."""
    merged: list[AnalystEvidenceItem] = []
    exclusions: list[dict[str, Any]] = []
    fingerprint_to_item: dict[str, AnalystEvidenceItem] = {}
    evidence_id_to_fingerprint: dict[str, str] = {}

    for stream_name, items in (
        ("market_context", context_evidence),
        ("pre_adapted", pre_adapted_evidence),
    ):
        for item in items:
            fingerprint = _evidence_fingerprint(item)
            existing_fingerprint = evidence_id_to_fingerprint.get(
                item.evidence_id
            )
            if (
                existing_fingerprint is not None
                and existing_fingerprint != fingerprint
            ):
                raise ValueError(
                    "Evidence ID collision with different content: "
                    f"{item.evidence_id}"
                )
            evidence_id_to_fingerprint[item.evidence_id] = fingerprint

            existing = fingerprint_to_item.get(fingerprint)
            if existing is not None:
                exclusions.append(
                    {
                        "reason": "duplicate_evidence_across_streams",
                        "stream": stream_name,
                        "evidence_id": item.evidence_id,
                        "kept_evidence_id": existing.evidence_id,
                        "fingerprint": fingerprint,
                    }
                )
                continue
            fingerprint_to_item[fingerprint] = item
            merged.append(item)

    return merged, exclusions


def _evidence_fingerprint(item: AnalystEvidenceItem) -> str:
    lineage_hash = _find_lineage_hash(item.provenance)
    if lineage_hash:
        return sha256_json(
            {
                "lineage_hash": lineage_hash,
                "source_type": item.source_type,
                "evidence_type": item.evidence_type,
                "domain_id": item.domain_id,
            }
        )
    payload = item.model_dump(mode="json")
    for field in (
        "evidence_id",
        "strength",
        "freshness_score",
        "reliability_score",
        "limitations",
        "blocked_windows",
    ):
        payload.pop(field, None)
        payload.pop(field, None)
    return sha256_json(payload)


def _publish_signals_to_bus(
    domain_id: str,
    classified_events: list[dict[str, Any]],
    as_of: str,
    *,
    bus_dir: Path,
) -> None:
    """Publish verified, point-in-time-safe signals to an explicitly enabled bus."""
    bus_dir.mkdir(parents=True, exist_ok=True)
    analysis_cutoff = _parse_aware_timestamp(as_of)
    
    for event in classified_events:
        cls = event.get("event_class")
        prop = CROSS_DOMAIN_PROPAGATION.get(cls)
        if not prop:
            continue
            
        mat = event.get("materiality_score", 0.0)
        if mat < prop["required_materiality"]:
            continue

        event_id = str(event.get("event_id") or event.get("evidence_id") or "").strip()
        source_hash = _find_lineage_hash(event.get("provenance", {}))
        available_at = _event_available_at(event)
        if not event_id or not source_hash or not available_at:
            continue
        if _parse_aware_timestamp(available_at) > analysis_cutoff:
            continue

        signal = {
            "contract": "dean_cross_domain_signal_v1",
            "source_domain": domain_id,
            "event_class": cls,
            "event_id": event_id,
            "source_evidence_id": event.get("evidence_id") or event_id,
            "source_evidence_sha256": source_hash,
            "title": event.get("title", ""),
            "text": event.get("text_preview", ""),
            "materiality": mat,
            "as_of": as_of,
            "available_at": available_at,
            "source_reliability": float(event.get("reliability_score", 0.0) or 0.0),
            "propagation_rules": prop,
        }
        signal_sha256 = sha256_json(signal)
        signal["signal_sha256"] = signal_sha256
        filepath = bus_dir / f"signal_{signal_sha256}.json"
        temporary = filepath.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(signal, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.replace(filepath)


def _event_available_at(event: dict[str, Any]) -> str | None:
    point_in_time = event.get("point_in_time")
    if isinstance(point_in_time, dict):
        for key in ("available_at", "availability_at", "published_at"):
            value = str(point_in_time.get(key) or "").strip()
            if value:
                return value
    value = str(event.get("published_at") or "").strip()
    return value or None


def _parse_aware_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timezone-aware timestamp required: {value!r}")
    return parsed


def _find_lineage_hash(value: Any) -> str | None:
    hash_keys = (
        "canonical_record_sha256",
        "record_sha256",
        "candidate_sha256",
        "observation_sha256",
        "content_sha256",
        "source_sha256",
    )
    if isinstance(value, dict):
        for key in hash_keys:
            candidate = str(value.get(key) or "").strip().lower()
            if candidate:
                return candidate
        for child in value.values():
            candidate = _find_lineage_hash(child)
            if candidate:
                return candidate
    elif isinstance(value, (list, tuple)):
        for child in value:
            candidate = _find_lineage_hash(child)
            if candidate:
                return candidate
    return None


class SectorReport:
    """Unified output from a sector analysis run.

    Combines the BaseAnalystAgent report (thesis, ticker basket, quality gates)
    with the lens pipeline output (regime context, scenario graph, hypotheses,
    evidence gaps, watch signals).
    """

    def __init__(
        self,
        *,
        base_report: AnalystReport,
        classified_events: list[dict[str, Any]] | None = None,
        regime_context: Any | None = None,
        scenario_graph: Any | None = None,
        transmission_channels: list[dict[str, Any]] | None = None,
        expectation_gap: dict[str, Any] | None = None,
        hypotheses: list[HypothesisLedgerEntry] | None = None,
        hypothesis_review_proposals: list[dict[str, Any]] | None = None,
        evidence_gaps: list[EvidenceGap] | None = None,
        watch_signals: list[dict[str, Any]] | None = None,
        review_notes: list[str] | None = None,
        delta_trail: list[ModuleDelta] | None = None,
        evidence_count: int = 0,
        evidence_exclusion_count: int = 0,
        lens_count: int = 0,
        economy_regime_brief: str | None = None,
        analysis_input_sha256: str = "",
        analysis_output_sha256: str = "",
    ):
        self.base_report = base_report
        self.classified_events = classified_events or []
        self.regime_context = regime_context
        self.scenario_graph = scenario_graph
        self.transmission_channels = transmission_channels or []
        self.expectation_gap = expectation_gap
        self.hypotheses = hypotheses or []
        self.hypothesis_review_proposals = hypothesis_review_proposals or []
        self.evidence_gaps = evidence_gaps or []
        self.watch_signals = watch_signals or []
        self.review_notes = review_notes or []
        self.delta_trail = delta_trail or []
        self.evidence_count = evidence_count
        self.evidence_exclusion_count = evidence_exclusion_count
        self.lens_count = lens_count
        self.economy_regime_brief = economy_regime_brief
        self.analysis_input_sha256 = analysis_input_sha256
        self.analysis_output_sha256 = analysis_output_sha256

    # ── Convenience accessors ────────────────────────────────────────────

    @property
    def domain_id(self) -> str:
        return self.base_report.domain_id

    @property
    def as_of(self) -> str:
        return self.base_report.as_of

    @property
    def thesis(self):
        return self.base_report.thesis

    @property
    def ticker_basket(self):
        return self.base_report.ticker_basket

    @property
    def recommendation(self) -> str:
        return self.base_report.recommendation

    @property
    def quality_gates(self) -> dict[str, Any]:
        return self.base_report.quality_gates

    @property
    def review_packet(self) -> dict[str, Any]:
        return self.base_report.review_packet

    @property
    def outcome_tracking_plan(self) -> dict[str, Any]:
        return self.base_report.outcome_tracking_plan

    @property
    def evidence(self) -> list[AnalystEvidenceItem]:
        return self.base_report.evidence

    @property
    def review_required(self) -> bool:
        return True

    @property
    def live_execution_allowed(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full report to a dict (for JSON/artifact writing)."""
        return {
            "report_type": "sector_analysis",
            "domain_id": self.domain_id,
            "as_of": self.as_of,
            "recommendation": self.recommendation,
            "review_required": self.review_required,
            "live_execution_allowed": self.live_execution_allowed,
            # Thesis
            "thesis": {
                "stance": self.thesis.stance,
                "expected_direction": self.thesis.expected_direction,
                "confidence": self.thesis.confidence,
                "thesis": self.thesis.thesis,
                "key_drivers": self.thesis.key_drivers,
                "risks": self.thesis.risks,
                "blind_spots": self.thesis.blind_spots,
                "data_quality": self.thesis.data_quality,
            },
            # Ticker basket
            "ticker_basket": {
                "basket_status": self.ticker_basket.basket_status,
                "candidates": [
                    {
                        "ticker": c.ticker,
                        "status": c.candidate_status,
                        "direction": c.expected_direction,
                        "confidence": c.confidence,
                        "blocked_reasons": c.blocked_reasons,
                    }
                    for c in self.ticker_basket.candidates
                ],
            },
            # Lens analysis
            "regime_context": (
                self.regime_context.model_dump()
                if self.regime_context is not None
                else None
            ),
            "scenario_graph": (
                self.scenario_graph.model_dump()
                if self.scenario_graph is not None
                else None
            ),
            "classified_events": self.classified_events,
            "transmission_channels": self.transmission_channels,
            "expectation_gap": self.expectation_gap,
            "hypotheses": [h.model_dump() for h in self.hypotheses],
            "hypothesis_review_proposals": self.hypothesis_review_proposals,
            "evidence_gaps": [g.model_dump() for g in self.evidence_gaps],
            "watch_signals": self.watch_signals,
            "review_notes": self.review_notes,
            # Stats
            "stats": {
                "evidence_count": self.evidence_count,
                "evidence_exclusion_count": self.evidence_exclusion_count,
                "lens_count": self.lens_count,
            },
            # Quality gates
            "quality_gates": self.quality_gates,
            "economy_regime_brief": self.economy_regime_brief,
            "audit": {
                "analysis_input_sha256": self.analysis_input_sha256,
                "analysis_output_sha256": self.analysis_output_sha256,
                "source_evidence_ids": [
                    item.evidence_id for item in self.evidence
                ],
                "delta_trail": [
                    {
                        **delta.model_dump(mode="json"),
                        "delta_sha256": sha256_json(
                            delta.model_dump(mode="json")
                        ),
                    }
                    for delta in self.delta_trail
                ],
            },
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        t = self.thesis
        candidates = self.ticker_basket.candidates
        direct = [c for c in candidates if c.candidate_status == "direct_ticker_thesis"]
        return (
            f"[{self.domain_id}] {t.stance} (confidence={t.confidence:.2f}) | "
            f"{len(candidates)} tickers ({len(direct)} direct) | "
            f"{self.evidence_count} evidence items | "
            f"{self.lens_count} lens deltas | "
            f"recommendation={self.recommendation}"
        )


__all__ = ["SectorAnalyst", "SectorReport"]
