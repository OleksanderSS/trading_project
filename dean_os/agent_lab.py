from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from dean_os.agents.domain_research import (
    MacroPolicyAgent,
    ValueScreeningAgent,
)
from dean_os.agents.financial_nlp import FinancialNLPAgent
from dean_os.agents.operations import OperationsProposalAgent
from dean_os.agents.research_agents import ResearchIngestionAgent, SpecialistResearchAgent
from dean_os.agents.synthesis import EvidenceSynthesisAgent
from dean_os.event_log import EventLog
from dean_os.learning import LearningStore, direction_from_note
from dean_os.material_loaders import MaterialLoadError, load_research_directory
from dean_os.operation_queue import OperationQueue
from dean_os.recommendation_memory import RecommendationMemoryStore
from dean_os.regime_context import normalize_context_tags
from dean_os.research_corpus import ResearchCorpus
from dean_os.schemas import (
    AgentLabRunReport,
    MarketContext,
    MarketRegimeSnapshot,
    ResearchDocument,
    utc_now_iso,
)
from dean_os.structured_context_provenance import (
    apply_market_context_structured_boundary,
)
from dean_os.utils import json_ready


class AgentLabRunner:
    """Isolated research lab runner for specialist agents.

    This runner never starts the trading pipeline. It ingests research materials,
    runs research agents, and writes a reviewable report.
    """

    def __init__(
        self,
        corpus_path: str | Path = "data/dean_os/research_corpus.sqlite",
        learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
        output_dir: str | Path = "reports/dean_os/agent_lab",
        operation_queue_path: str | Path | None = None,
        memory_path: str | Path = "data/dean_os/recommendation_memory.sqlite",
        log_path: str | Path | None = None,
        chunk_size: int = 1200,
    ):
        self.corpus_path = Path(corpus_path)
        self.learning_path = Path(learning_path)
        self.output_dir = Path(output_dir)
        self.operation_queue_path = Path(operation_queue_path) if operation_queue_path else None
        self.memory_path = Path(memory_path)
        self.log_path = Path(log_path) if log_path else None
        self.event_log = EventLog(self.log_path) if self.log_path else None
        self.chunk_size = chunk_size
        self.corpus = ResearchCorpus(self.corpus_path)
        self.learning_store = LearningStore(self.learning_path)
        self.memory_store = RecommendationMemoryStore(self.memory_path)

    async def run(
        self,
        materials_path: str | Path | None = None,
        documents: list[ResearchDocument] | None = None,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        tags: list[str] | None = None,
        regime_tags: list[str] | None = None,
        regime_context: MarketRegimeSnapshot | dict[str, Any] | None = None,
        source_type: str | None = None,
        fundamentals: dict[str, dict[str, Any]] | None = None,
        fundamental_gate: dict[str, Any] | None = None,
        fundamental_provenance: dict[str, Any] | None = None,
        macro: dict[str, Any] | None = None,
        macro_provenance: dict[str, Any] | None = None,
        as_of: str | None = None,
        include_financial_nlp: bool = True,
        include_synthesis: bool = True,
        create_learning_records: bool = True,
        include_operations_proposals: bool = True,
    ) -> AgentLabRunReport:
        run_id = uuid4().hex
        regime_snapshot = self._coerce_regime_context(regime_context)
        normalized_regime_tags = normalize_context_tags(
            [*(regime_tags or []), *((regime_snapshot.context_tags if regime_snapshot else []) or [])]
        )
        context_tags = normalize_context_tags([*(tags or []), *normalized_regime_tags])
        fundamental_gate_summary = _fundamental_gate_summary(
            fundamental_gate,
            has_fundamentals=bool(fundamentals),
        )
        self._log(
            "agent_lab_run_started",
            run_id,
            {
                "materials_path": str(materials_path) if materials_path is not None else None,
                "preloaded_document_count": len(documents or []),
                "tickers": tickers or [],
                "sectors": sectors or [],
                "tags": tags or [],
                "regime_tags": normalized_regime_tags,
                "context_tags": context_tags,
            },
        )
        loaded_documents, load_errors = self._load_documents(
            materials_path=materials_path,
            documents=documents,
            tickers=tickers,
            sectors=sectors,
            tags=tags,
            source_type=source_type,
        )
        self._log(
            "agent_lab_materials_loaded",
            run_id,
            {
                "document_count": len(loaded_documents),
                "load_error_count": len(load_errors),
                "load_errors": load_errors,
            },
        )
        analysis_as_of = as_of or utc_now_iso()
        context = MarketContext(
            as_of=analysis_as_of,
            tickers=tickers or [],
            research_documents=loaded_documents,
            fundamentals=fundamentals or {},
            macro=macro or {},
            metadata={
                "agent_lab": True,
                "analysis_as_of": analysis_as_of,
                "load_errors": load_errors,
                "fundamental_input_readiness_gate": fundamental_gate_summary,
                "fundamental_evidence_provenance": (
                    fundamental_provenance or {}
                ),
                "macro_evidence_provenance": macro_provenance or {},
                "context_tags": context_tags,
                "material_tags": normalize_context_tags(tags or []),
                "regime_tags": normalized_regime_tags,
                "regime_context": json_ready(regime_snapshot) if regime_snapshot else None,
                "recommendation_memory": self.memory_store.context_snapshot(
                    context_tags=context_tags,
                    tickers=tickers or [],
                    sectors=sectors or [],
                ),
            },
        )
        structured_audit = apply_market_context_structured_boundary(
            context
        )
        fundamental_gate_summary[
            "structured_point_in_time_status"
        ] = structured_audit["status"]
        fundamental_gate_summary[
            "structured_accepted_fundamental_count"
        ] = structured_audit.get("family_counts", {}).get(
            "fundamental",
            0,
        )
        fundamental_gate_summary[
            "context_structured_accepted_fingerprint"
        ] = structured_audit["accepted_fingerprint"]
        gate_fingerprint = fundamental_gate_summary.get(
            "gate_structured_accepted_fingerprint"
        )
        fundamental_gate_summary[
            "structured_fingerprint_matches_context"
        ] = bool(
            gate_fingerprint
            and gate_fingerprint
            == structured_audit["accepted_fingerprint"]
        )
        if fundamentals and not context.fundamentals:
            fundamental_gate_summary[
                "can_feed_value_screening_after_manual_review"
            ] = False
            fundamental_gate_summary["readiness_status"] = (
                "blocked_structured_point_in_time_contract"
            )
        elif (
            fundamentals
            and fundamental_gate_summary.get(
                "can_feed_value_screening_after_manual_review"
            )
            is True
            and not fundamental_gate_summary[
                "structured_fingerprint_matches_context"
            ]
        ):
            fundamental_gate_summary[
                "can_feed_value_screening_after_manual_review"
            ] = False
            fundamental_gate_summary["readiness_status"] = (
                "blocked_fundamental_gate_context_fingerprint_mismatch"
            )

        ingestion = ResearchIngestionAgent(
            name="research_ingestion",
            config={"corpus_path": str(self.corpus_path), "chunk_size": self.chunk_size},
        )
        specialist = SpecialistResearchAgent(
            name="specialist_research",
            config={"corpus_path": str(self.corpus_path), "horizon_days": 365},
        )
        reports = [await ingestion.run(context)]

        if include_financial_nlp:
            nlp_agent = FinancialNLPAgent(name="financial_nlp", config={})
            reports.append(await nlp_agent.run(context))

        reports.append(await specialist.run(context))

        if context.macro:
            macro_policy = MacroPolicyAgent(
                name="macro_policy",
                config={},
            )
            reports.append(await macro_policy.run(context))

        if fundamentals:
            value_screen = ValueScreeningAgent(name="value_screening", config={})
            reports.append(await value_screen.run(context))

        if include_synthesis:
            synthesis = EvidenceSynthesisAgent(name="evidence_synthesis", config={})
            reports.append(await synthesis.run(context))

        if include_operations_proposals:
            operations = OperationsProposalAgent(name="operations_proposal", config={"proposal_only": True})
            reports.append(await operations.run(context))
        self._log(
            "agent_lab_agents_finished",
            run_id,
            {
                "report_count": len(reports),
                "note_count": len(context.research_notes),
                "nlp_result_count": len(context.nlp_results),
                "proposal_count": len(context.action_proposals),
                "memory_relevant_count": context.metadata.get("recommendation_memory", {}).get("relevant_count", 0),
            },
        )

        learning_records = []
        if create_learning_records:
            learning_records = [
                self.learning_store.create_record_from_note(
                    note=note,
                    expected_direction=direction_from_note(note),
                    metadata={
                        "agent_lab_run": True,
                        "tickers": tickers or [],
                        "sectors": sectors or [],
                        "context_tags": context_tags,
                        "regime_tags": normalized_regime_tags,
                    },
                )
                for note in context.research_notes
                if note.data_quality != "weak"
            ]
        self._log(
            "agent_lab_learning_records_created",
            run_id,
            {"learning_record_count": len(learning_records)},
        )

        queued_proposal_count = 0
        if self.operation_queue_path and context.action_proposals:
            queued_proposal_count = len(
                OperationQueue(self.operation_queue_path, event_log_path=self.log_path).add_many(context.action_proposals)
            )
        self._log(
            "agent_lab_operation_proposals_queued",
            run_id,
            {
                "proposal_count": len(context.action_proposals),
                "queued_proposal_count": queued_proposal_count,
            },
        )

        report = AgentLabRunReport(
            run_id=run_id,
            corpus_path=str(self.corpus_path),
            document_count=len(loaded_documents),
            chunk_count=self._count_chunks(loaded_documents),
            note_count=len(context.research_notes),
            reports=reports,
            research_notes=context.research_notes,
            learning_records=learning_records,
            action_proposals=context.action_proposals,
            summary=self._summary(context, load_errors, learning_records, queued_proposal_count),
        )
        self.save_report(report)
        self._log("agent_lab_run_completed", run_id, report.summary)
        return report

    def save_report(self, report: AgentLabRunReport) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{report.run_id}.json"
        md_path = self.output_dir / f"{report.run_id}.md"
        json_path.write_text(json.dumps(json_ready(report), indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(render_agent_lab_markdown(report), encoding="utf-8")
        return json_path, md_path

    def _load_documents(
        self,
        materials_path: str | Path | None,
        documents: list[ResearchDocument] | None,
        tickers: list[str] | None,
        sectors: list[str] | None,
        tags: list[str] | None,
        source_type: str | None,
    ) -> tuple[list[ResearchDocument], list[str]]:
        loaded = list(documents or [])
        errors: list[str] = []
        if materials_path is not None:
            try:
                directory_docs, directory_errors = load_research_directory(
                    materials_path,
                    source_type=source_type,
                    tickers=tickers or [],
                    sectors=sectors or [],
                    tags=tags or [],
                    recursive=True,
                    ignore_errors=True,
                )
            except MaterialLoadError as exc:
                directory_docs = []
                directory_errors = [str(exc)]
            loaded.extend(directory_docs)
            errors.extend(directory_errors)
        return loaded, errors

    def _count_chunks(self, documents: list[ResearchDocument]) -> int:
        # Ingestion agent writes the actual chunks; this count is for the report summary.
        return sum(max(1, len(document.text.split()) // max(1, self.chunk_size // 8)) for document in documents)

    def _summary(
        self,
        context: MarketContext,
        load_errors: list[str],
        learning_records: list,
        queued_proposal_count: int = 0,
    ) -> dict[str, Any]:
        patterns: dict[str, int] = {}
        for note in context.research_notes:
            for pattern in note.patterns:
                patterns[pattern] = patterns.get(pattern, 0) + 1
        latest_note = context.research_notes[-1] if context.research_notes else None
        return {
            "top_patterns": sorted(patterns, key=patterns.get, reverse=True),
            "load_error_count": len(load_errors),
            "load_errors": load_errors,
            "latest_thesis": latest_note.thesis if latest_note else "",
            "synthesis_count": len([note for note in context.research_notes if note.agent_name == "evidence_synthesis"]),
            "learning_record_count": len(learning_records),
            "nlp_result_count": len(context.nlp_results),
            "avg_nlp_sentiment": self._avg_nlp_sentiment(context),
            "proposal_count": len(context.action_proposals),
            "queued_proposal_count": queued_proposal_count,
            "memory_relevant_count": context.metadata.get("recommendation_memory", {}).get("relevant_count", 0),
            "memory_miss_count": context.metadata.get("recommendation_memory", {}).get("miss_count", 0),
            "fundamental_input_readiness_gate": context.metadata.get("fundamental_input_readiness_gate", {}),
            "fundamental_evidence_provenance": context.metadata.get(
                "fundamental_evidence_provenance",
                {},
            ),
            "context_tags": context.metadata.get("context_tags", []),
            "regime_tags": context.metadata.get("regime_tags", []),
            "regime_context": context.metadata.get("regime_context"),
            "analysis_as_of": context.as_of,
            "news_point_in_time_audit": context.metadata.get(
                "news_point_in_time_audit",
                {},
            ),
            "structured_context_point_in_time_audit": (
                context.metadata.get(
                    "structured_context_point_in_time_audit",
                    {},
                )
            ),
            "macro_evidence_provenance": context.metadata.get(
                "macro_evidence_provenance",
                {},
            ),
        }

    def _avg_nlp_sentiment(self, context: MarketContext) -> float | None:
        if not context.nlp_results:
            return None
        return sum(result.sentiment_score for result in context.nlp_results) / len(context.nlp_results)

    def _log(self, event_type: str, run_id: str, payload: dict[str, Any]) -> None:
        if self.event_log:
            self.event_log.write(event_type=event_type, source="agent_lab", payload=payload, run_id=run_id)

    def _coerce_regime_context(
        self,
        regime_context: MarketRegimeSnapshot | dict[str, Any] | None,
    ) -> MarketRegimeSnapshot | None:
        if regime_context is None:
            return None
        if isinstance(regime_context, MarketRegimeSnapshot):
            return regime_context
        return MarketRegimeSnapshot(**regime_context)


def _fundamental_gate_summary(fundamental_gate: dict[str, Any] | None, has_fundamentals: bool) -> dict[str, Any]:
    if not has_fundamentals:
        return {
            "gate_attached": bool(fundamental_gate),
            "readiness_status": "not_applicable_no_fundamentals",
            "can_feed_value_screening_after_manual_review": False,
            "warning_count": 0,
            "fail_count": 0,
        }
    if not fundamental_gate:
        return {
            "gate_attached": False,
            "readiness_status": "not_attached",
            "can_feed_value_screening_after_manual_review": None,
            "warning_count": None,
            "fail_count": None,
        }
    summary = fundamental_gate.get("summary", {})
    guidance = fundamental_gate.get("decision_guidance", {})
    return {
        "gate_attached": True,
        "run_id": fundamental_gate.get("run_id"),
        "readiness_status": summary.get("readiness_status"),
        "can_enter_manual_fundamental_review": summary.get("can_enter_manual_fundamental_review"),
        "can_feed_value_screening_after_manual_review": summary.get("can_feed_value_screening_after_manual_review"),
        "metric_count": summary.get("metric_count"),
        "source_citation_missing_count": summary.get("source_citation_missing_count"),
        "period_missing_count": summary.get("period_missing_count"),
        "availability_timestamp_missing_count": summary.get(
            "availability_timestamp_missing_count"
        ),
        "structured_point_in_time_status": summary.get(
            "structured_point_in_time_status"
        ),
        "gate_structured_accepted_fingerprint": summary.get(
            "structured_accepted_fingerprint"
        ),
        "warning_count": guidance.get("warning_count"),
        "fail_count": guidance.get("fail_count"),
    }


def render_agent_lab_markdown(report: AgentLabRunReport) -> str:
    lines = [
        "# DEAN-OS Agent Lab Report",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Corpus: `{report.corpus_path}`",
        f"- Documents: {report.document_count}",
        f"- Estimated chunks: {report.chunk_count}",
        f"- Research notes: {report.note_count}",
        "",
        "## Summary",
        "",
        f"- Latest thesis: {report.summary.get('latest_thesis', '')}",
        f"- Top patterns: {', '.join(report.summary.get('top_patterns', [])) or 'none'}",
        f"- Context tags: {', '.join(report.summary.get('context_tags', [])) or 'none'}",
        f"- Regime tags: {', '.join(report.summary.get('regime_tags', [])) or 'none'}",
        f"- Evidence syntheses: {report.summary.get('synthesis_count', 0)}",
        f"- Learning records: {report.summary.get('learning_record_count', 0)}",
        f"- NLP results: {report.summary.get('nlp_result_count', 0)}",
        f"- Average NLP sentiment: {report.summary.get('avg_nlp_sentiment')}",
        f"- Action proposals: {report.summary.get('proposal_count', 0)}",
        f"- Queued proposals: {report.summary.get('queued_proposal_count', 0)}",
        f"- Fundamental gate: {report.summary.get('fundamental_input_readiness_gate', {}).get('readiness_status')}",
        "",
        "## Research Notes",
        "",
    ]
    for note in report.research_notes:
        lines.extend(
            [
                f"### {note.topic}",
                "",
                f"- Agent: `{note.agent_name}`",
                f"- Confidence: {note.confidence:.2f}",
                f"- Data quality: `{note.data_quality}`",
                f"- Thesis: {note.thesis}",
                f"- Patterns: {', '.join(note.patterns) or 'none'}",
                f"- Risks: {'; '.join(note.risks) or 'none'}",
                "",
            ]
        )
    if report.action_proposals:
        lines.extend(["## Action Proposals", ""])
        for proposal in report.action_proposals:
            lines.extend(
                [
                    f"- `{proposal.action_type}` -> `{proposal.target}`",
                    f"  Reason: {proposal.reason}",
                    f"  Dry run: {proposal.dry_run}",
                ]
            )
    return "\n".join(lines).strip() + "\n"
