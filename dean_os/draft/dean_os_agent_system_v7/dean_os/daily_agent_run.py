from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from dean_os.draft.dean_os_agent_system_v7.dean_os.briefing_contract import DailyBriefing, DailyBriefingBuilder
from dean_os.draft.dean_os_agent_system_v7.dean_os.briefing_renderer import DailyBriefingRenderer
from dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_dedup import EvidenceDedupResult, SemanticEvidenceDeduplicator
from dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_gap_planner_v2 import EvidenceGapPlan, EvidenceGapPlanner
from dean_os.draft.dean_os_agent_system_v7.dean_os.operator_review_inbox_v2 import (
    OperatorReviewInboxBuilder,
    ReviewInboxItem,
    ReviewInboxProtocol,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_catalog import (
    CatalogEvidenceRecord,
    EvidenceAcquisitionRunManifest,
    EvidenceCatalogBuilder,
    EvidenceCatalogProtocol,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.replay_scheduler import ReplayScheduleItem, ReplayScheduler
from dean_os.draft.dean_os_agent_system_v7.dean_os.daily_run_store import DailyRunRecord, DailyRunRecordBuilder, DailyRunStoreProtocol
from dean_os.schemas import MarketContext, ResearchDocument
from dean_os.analysts.profiles import get_domain_profile


class DailyAgentRunResult(BaseModel):
    daily_run_id: str = Field(default_factory=lambda: f"daily_agent_run_{uuid4().hex}")
    status: str
    domain_id: str
    as_of: str
    knowledge_cutoff: str
    evidence_manifest: EvidenceAcquisitionRunManifest
    evidence_records: list[CatalogEvidenceRecord] = Field(default_factory=list)
    system_result: dict[str, Any]
    briefing: DailyBriefing
    replay_schedule: list[ReplayScheduleItem] = Field(default_factory=list)
    due_replay_tasks: list[ReplayScheduleItem] = Field(default_factory=list)
    persisted_run_record: DailyRunRecord | None = None
    evidence_dedup: EvidenceDedupResult | None = None
    evidence_gap_plan: EvidenceGapPlan | None = None
    review_inbox_items: list[ReviewInboxItem] = Field(default_factory=list)
    rendered_artifacts: dict[str, str] = Field(default_factory=dict)
    safety: dict[str, bool] = Field(default_factory=lambda: {
        "can_trade": False,
        "can_write_production_config": False,
        "can_promote_model": False,
        "can_write_learning_memory": False,
        "human_review_required": True,
    })


class DailyAgentRun:
    """Network-agnostic daily composition layer for the analytical system.

    Collectors may run elsewhere. This class receives bounded source payloads,
    catalogs them, executes the agent system, emits a briefing contract, and
    schedules replay observations.
    """

    def __init__(
        self,
        system: Any,
        *,
        domain_id: str,
        evidence_catalog: EvidenceCatalogProtocol | None = None,
        required_coverage: list[dict[str, str]] | None = None,
        daily_run_store: DailyRunStoreProtocol | None = None,
        review_inbox: ReviewInboxProtocol | None = None,
        briefing_output_dir: str | None = None,
    ):
        self.system = system
        self.domain_id = domain_id
        self.evidence_catalog = evidence_catalog
        self.required_coverage = required_coverage or _domain_coverage(domain_id)
        self.daily_run_store = daily_run_store
        self.run_record_builder = DailyRunRecordBuilder()
        self.catalog_builder = EvidenceCatalogBuilder()
        self.briefing_builder = DailyBriefingBuilder()
        self.replay_scheduler = ReplayScheduler()
        self.deduplicator = SemanticEvidenceDeduplicator()
        self.evidence_gap_planner = EvidenceGapPlanner(domain_id)
        self.review_inbox = review_inbox
        self.review_inbox_builder = OperatorReviewInboxBuilder()
        self.briefing_renderer = DailyBriefingRenderer()
        self.briefing_output_dir = briefing_output_dir

    async def run(
        self,
        context: MarketContext,
        *,
        evidence_payloads: list[dict[str, Any]] | None = None,
        prior_replay_tasks: list[ReplayScheduleItem | dict[str, Any]] | None = None,
    ) -> DailyAgentRunResult:
        as_of = str(context.as_of or datetime.now(UTC).isoformat())
        context.as_of = as_of
        knowledge_cutoff = str(context.metadata.get("knowledge_cutoff") or as_of)
        context.metadata["knowledge_cutoff"] = knowledge_cutoff
        started_at = datetime.now(UTC).isoformat()

        payloads = list(evidence_payloads or [])
        payloads.extend(_document_payload(document) for document in context.research_documents)
        payloads.extend(_news_payload(item) for item in context.news if isinstance(item, dict))

        records: list[CatalogEvidenceRecord] = []
        accepted_payloads: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        warnings: list[str] = []
        dedup = self.deduplicator.analyze(payloads)
        decisions = {item.input_index: item for item in dedup.decisions}
        for index, payload in enumerate(payloads):
            decision = decisions[index]
            if index in dedup.suppressed_indices:
                suppressed.append({
                    "index": index,
                    "reason": decision.status,
                    "canonical_index": decision.canonical_index,
                    "similarity": decision.similarity,
                    "title": payload.get("title"),
                })
                continue
            enriched = dict(payload)
            enriched["duplicate_cluster_id"] = decision.cluster_id
            enriched["duplicate_status"] = decision.status
            if decision.canonical_index != index:
                enriched["duplicate_of"] = str(payloads[decision.canonical_index].get("evidence_id") or payloads[decision.canonical_index].get("document_id") or decision.canonical_index)
            try:
                record = self.catalog_builder.build_record(
                    enriched,
                    domain_id=self.domain_id,
                    as_of=as_of,
                    knowledge_cutoff=knowledge_cutoff,
                )
                if record.point_in_time_status == "invalid":
                    rejected.append({"index": index, "reason": "evidence_available_after_knowledge_cutoff", "title": record.title})
                    continue
                records.append(record)
                accepted_payload = dict(enriched)
                accepted_payload["evidence_id"] = record.evidence_id
                accepted_payload["catalog_record"] = record.model_dump(mode="json")
                accepted_payloads.append(accepted_payload)
                if self.evidence_catalog is not None:
                    self.evidence_catalog.append(record)
            except Exception as exc:
                rejected.append({"index": index, "reason": f"{type(exc).__name__}: {exc}"})

        context.metadata["evidence_catalog_records"] = [item.model_dump(mode="json") for item in records]
        context.metadata["accepted_evidence_payloads"] = accepted_payloads
        # Critical point-in-time boundary: downstream agents must not retain raw
        # items that the catalog rejected or duplicate suppression removed.
        context.news = [
            dict(item)
            for item in accepted_payloads
            if str(item.get("source_type") or "").lower() == "news"
        ]
        accepted_document_ids = {
            str(item.get("document_id") or item.get("evidence_id") or "")
            for item in accepted_payloads
        }
        context.research_documents = [
            document
            for document in context.research_documents
            if document.document_id in accepted_document_ids
        ]
        system_result = await self.system.run(context)
        replay_schedule = self.replay_scheduler.build_from_run_result(system_result)
        due = self.replay_scheduler.due(prior_replay_tasks or [], as_of=as_of)
        completed_at = datetime.now(UTC).isoformat()
        manifest = self.catalog_builder.build_manifest(
            domain_id=self.domain_id,
            as_of=as_of,
            knowledge_cutoff=knowledge_cutoff,
            started_at=started_at,
            completed_at=completed_at,
            records=records,
            rejected_items=rejected,
            suppressed_items=suppressed,
            warnings=warnings,
        )
        briefing = self.briefing_builder.build(
            run_result=system_result,
            required_coverage=self.required_coverage,
            evidence_records=[item.model_dump(mode="json") for item in records],
            replay_due=[item.model_dump(mode="json") for item in due],
        )
        evidence_gap_plan = self.evidence_gap_planner.build(
            briefing=briefing,
            evidence_records=[item.model_dump(mode="json") for item in records],
        )
        status = "partial" if system_result.status != "completed" or manifest.status != "completed" else "completed"
        result = DailyAgentRunResult(
            status=status,
            domain_id=self.domain_id,
            as_of=as_of,
            knowledge_cutoff=knowledge_cutoff,
            evidence_manifest=manifest,
            evidence_records=records,
            system_result=system_result.model_dump(mode="json"),
            briefing=briefing,
            replay_schedule=replay_schedule,
            due_replay_tasks=due,
            evidence_dedup=dedup,
            evidence_gap_plan=evidence_gap_plan,
        )
        review_items = self.review_inbox_builder.build(result, evidence_gap_plan=evidence_gap_plan)
        result.review_inbox_items = review_items
        if self.review_inbox is not None:
            for item in review_items:
                self.review_inbox.append(item)
        if self.briefing_output_dir:
            md_path, html_path = self.briefing_renderer.save(
                briefing,
                self.briefing_output_dir,
                evidence_gap_plan=evidence_gap_plan,
            )
            result.rendered_artifacts = {"markdown": str(md_path), "html": str(html_path)}
        if self.daily_run_store is not None:
            record = self.run_record_builder.build(result)
            self.daily_run_store.append(record)
            result.persisted_run_record = record
        return result


def _document_payload(document: ResearchDocument) -> dict[str, Any]:
    return {
        "evidence_id": document.document_id,
        "source_type": document.source_type,
        "source": document.metadata.get("source", document.source_type),
        "title": document.title,
        "text": document.text,
        "uri": document.uri,
        "published_at": document.published_at,
        "available_at": document.metadata.get("available_at") or document.published_at or document.ingested_at,
        "ingested_at": document.ingested_at,
        "tickers": document.tickers,
        "sectors": document.sectors,
        "tags": document.tags,
        "quality_score": document.metadata.get("quality_score", 0.5),
        "quarantine_flags": document.quarantine_flags,
    }


def _news_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    payload.setdefault("source_type", "news")
    return payload


def _domain_coverage(domain_id: str) -> list[dict[str, Any]]:
    profile = get_domain_profile(domain_id)
    if profile.coverage_gate:
        return [dict(item) for item in profile.coverage_gate]
    return [
        {"coverage_id": evidence_type, "label": evidence_type.replace("_", " ").title(), "aliases": [evidence_type]}
        for evidence_type in profile.required_evidence_types
    ]
