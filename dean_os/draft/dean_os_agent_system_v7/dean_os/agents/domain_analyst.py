from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.analyst_core.artifact_evidence_loader import (
    ArtifactEvidenceLoader,
)
from dean_os.analyst_core.sector_analyst import SectorAnalyst, SectorReport
from dean_os.analyst_core.pipeline_context_evidence_loader import (
    PipelineContextEvidenceLoader,
)
from dean_os.analyst_core.research_corpus_evidence_loader import (
    DEFAULT_RESEARCH_QUERY,
    ResearchCorpusEvidenceLoader,
)
from dean_os.analysts.schemas import AnalystEvidenceItem
from dean_os.base import BaseAgent
from dean_os.schemas import EvidenceItem, MarketContext, PipelineReport
from dean_os.utils import sha256_json


class DomainAnalystAgent(BaseAgent):
    version = "0.2.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        domain_id: str = self.config.get("domain_id", "semiconductor_ai_infrastructure")
        as_of: str = context.as_of or self.config.get("as_of", "")
        tickers: list[str] | None = context.tickers or self.config.get("tickers")
        horizon_days: int | None = self.config.get("horizon_days")
        if not _timezone_aware_as_of(as_of):
            report = _input_gap_report(
                self.name,
                self.version,
                domain_id,
                "timezone-aware as_of is required",
            )
            report.input_hash = self.context_hash(context)
            report.config_hash = sha256_json(self.config)
            return report

        try:
            pre_adapted = _load_configured_evidence(
                self.config,
                domain_id=domain_id,
                as_of=as_of,
                tickers=tickers,
            )
        except Exception as exc:
            report = _input_gap_report(
                self.name,
                self.version,
                domain_id,
                f"configured runtime artifact was rejected: {exc}",
            )
            report.input_hash = self.context_hash(context)
            report.config_hash = sha256_json(self.config)
            return report
        if pre_adapted is None and not _has_context_evidence(context):
            report = _input_gap_report(
                self.name,
                self.version,
                domain_id,
                (
                    "at least one populated MarketContext evidence source "
                    "or a verified runtime artifact is required"
                ),
            )
            report.input_hash = self.context_hash(context)
            report.config_hash = sha256_json(self.config)
            return report

        analyst = SectorAnalyst(
            domain_id=domain_id,
            agent_name=self.name,
        )
        report: SectorReport = analyst.run(
            context=context,
            as_of=as_of or None,
            tickers=tickers,
            horizon_days=horizon_days,
            pre_adapted_evidence=pre_adapted,
        )

        pipeline_report = _report_to_pipeline(
            self.name,
            self.version,
            report,
            domain_id,
        )
        pipeline_report.input_hash = sha256_json(
            {
                "context_hash": self.context_hash(context),
                "adapted_evidence": [
                    item.model_dump(mode="json")
                    for item in report.evidence
                ],
            }
        )
        pipeline_report.config_hash = sha256_json(self.config)
        return pipeline_report


def _report_to_pipeline(
    agent_name: str,
    agent_version: str,
    report: SectorReport,
    domain_id: str,
) -> PipelineReport:
    t = report.thesis
    basket = report.ticker_basket

    verdict = _verdict_from_recommendation(report.recommendation)
    confidence = t.confidence
    data_quality = _quality_score(t.data_quality)
    signal = _signal_from_verdict(verdict)

    reasons: list[str] = []
    if t.thesis:
        reasons.append(f"Thesis: {t.thesis}")
    if t.key_drivers:
        reasons.append(f"Key drivers: {'; '.join(t.key_drivers[:3])}")
    if t.risks:
        reasons.append(f"Risks: {'; '.join(t.risks[:3])}")
    if t.blind_spots:
        reasons.append(f"Blind spots: {'; '.join(t.blind_spots[:2])}")
    if not reasons:
        reasons.append(f"Domain {domain_id} analysis complete — review required")

    risks = list(t.risks or [])
    if report.review_required:
        risks.append("Review-only artifact — human review required before any action")

    evidence = [
        _ev(agent_name, "metric", "sector_report", "evidence_count", report.evidence_count),
        _ev(agent_name, "metric", "sector_report", "lens_count", report.lens_count),
        _ev(agent_name, "metric", "sector_report", "stance", t.stance),
        _ev(agent_name, "metric", "sector_report", "recommendation", report.recommendation),
    ]

    return PipelineReport(
        agent_name=agent_name,
        agent_version=agent_version,
        verdict=verdict,
        confidence=confidence,
        data_quality_score=data_quality,
        signal_strength=signal,
        reasons=reasons,
        risks=risks,
        blind_spots=[
            "DomainAnalystAgent is review-only — no trade signals, no live execution",
            "Sector/domain evidence alone cannot support a direct ticker thesis",
        ],
        evidence=evidence,
        metrics_snapshot={
            "domain_id": domain_id,
            "agent_role": "standalone_domain_analysis",
            "decision_influence": False,
            "supporting_review_only": True,
            "can_create_ticker_forecast": False,
            "can_trade": False,
            "recommendation": report.recommendation,
            "review_required": report.review_required,
            "live_execution_allowed": report.live_execution_allowed,
            "evidence_count": report.evidence_count,
            "lens_count": report.lens_count,
            "classified_event_count": len(report.classified_events),
            "hypothesis_count": len(report.hypotheses),
            "evidence_gap_count": len(report.evidence_gaps),
            "hypotheses": [
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else item
                for item in report.hypotheses
            ],
            "evidence_gaps": [
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else item
                for item in report.evidence_gaps
            ],
            "watch_signals": report.watch_signals,
            "regime_context": (
                report.regime_context.model_dump(mode="json")
                if hasattr(report.regime_context, "model_dump")
                else report.regime_context
            ),
            "expectation_gap": report.expectation_gap,
            "thesis": t.thesis,
            "stance": t.stance,
            "confidence": t.confidence,
            "ticker_basket_status": basket.basket_status,
            "ticker_candidates": len(basket.candidates),
        },
    )


def _input_gap_report(
    agent_name: str,
    agent_version: str,
    domain_id: str,
    reason: str,
) -> PipelineReport:
    return PipelineReport(
        agent_name=agent_name,
        agent_version=agent_version,
        verdict="needs_more_data",
        confidence=1.0,
        data_quality_score=0.0,
        signal_strength=0.0,
        reasons=[reason],
        risks=["Domain analysis did not run"],
        evidence=[
            _ev(
                agent_name,
                "audit_finding",
                "domain_analyst_input",
                "missing_input",
                reason,
            )
        ],
        metrics_snapshot={
            "domain_id": domain_id,
            "agent_role": "standalone_domain_analysis",
            "decision_influence": False,
            "supporting_review_only": True,
            "analysis_executed": False,
            "can_create_ticker_forecast": False,
            "can_trade": False,
        },
    )


def _timezone_aware_as_of(value: str | None) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() is not None
    )


def _verdict_from_recommendation(rec: str) -> str:
    return {
        "ready_for_review": "clear",
        "partial_ready_for_review": "caution",
        "needs_more_data": "needs_more_data",
        "blocked": "blocked",
    }.get(rec, "caution")


def _quality_score(quality: str | None) -> float:
    return {"strong": 0.9, "medium": 0.65, "weak": 0.35}.get(str(quality).lower(), 0.5)


def _signal_from_verdict(verdict: str) -> float:
    return {
        "clear": 0.3,
        "caution": 0.0,
        "needs_more_data": -0.25,
        "blocked": -0.5,
    }.get(verdict, 0.0)


def _ev(
    agent_name: str,
    source_type: str,
    source: str,
    key: str,
    value: Any,
) -> EvidenceItem:
    return EvidenceItem(source_type=source_type, source=source, key=key, value=value)


def _load_configured_evidence(
    config: dict[str, Any],
    *,
    domain_id: str,
    as_of: str,
    tickers: list[str] | None = None,
) -> list[AnalystEvidenceItem] | None:
    evidence: list[AnalystEvidenceItem] = []
    runtime_path = config.get("runtime_artifact_path")
    producer_paths = config.get("producer_artifact_paths") or {}
    if runtime_path and producer_paths:
        raise ValueError(
            "runtime_artifact_path and producer_artifact_paths cannot be combined; "
            "the runtime already contains adapted producer evidence"
        )
    if runtime_path:
        evidence.extend(
            ArtifactEvidenceLoader().from_runtime_artifact(
                Path(runtime_path),
                domain_id=domain_id,
                as_of=as_of,
            )
        )
    if producer_paths:
        if not isinstance(producer_paths, dict):
            raise ValueError("producer_artifact_paths must be a mapping")
        supported = {
            "news", "macro", "sector_market", "policy", "fundamental"
        }
        unknown = sorted(set(producer_paths) - supported)
        if unknown:
            raise ValueError(
                "unsupported producer artifact types: " + ", ".join(unknown)
            )
        evidence.extend(
            ArtifactEvidenceLoader().from_producer_artifacts(
                news_path=_optional_path(producer_paths, "news"),
                macro_path=_optional_path(producer_paths, "macro"),
                sector_market_path=_optional_path(producer_paths, "sector_market"),
                policy_path=_optional_path(producer_paths, "policy"),
                fundamental_path=_optional_path(producer_paths, "fundamental"),
                domain_id=domain_id,
                as_of=as_of,
            )
        )
    pipeline_context_path = config.get("pipeline_context_artifact_path")
    if pipeline_context_path:
        evidence.extend(
            PipelineContextEvidenceLoader().load(
                pipeline_context_path,
                domain_id=domain_id,
                as_of=as_of,
                tickers=tickers,
            )
        )
    research_corpus_path = config.get("research_corpus_path")
    if research_corpus_path:
        evidence.extend(
            ResearchCorpusEvidenceLoader().load(
                research_corpus_path,
                domain_id=domain_id,
                as_of=as_of,
                tickers=tickers,
                query=str(config.get("research_query") or DEFAULT_RESEARCH_QUERY),
                top_k=int(config.get("research_top_k") or 20),
            )
        )
    return evidence or None


def _optional_path(values: dict[str, Any], key: str) -> Path | None:
    value = str(values.get(key) or "").strip()
    return Path(value) if value else None


def _has_context_evidence(context: MarketContext) -> bool:
    return any(
        bool(value)
        for value in (
            context.news,
            context.fundamentals,
            context.macro,
            context.sector_data,
            context.research_documents,
            context.research_notes,
            context.nlp_results,
        )
    )
