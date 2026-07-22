from __future__ import annotations

from collections import Counter
from typing import Any

from dean_os.base import AnalyticalAgent
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import (
    audit_market_context_news,
    audit_research_documents,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.research_corpus import ResearchCorpus, chunk_document
from dean_os.schemas import AnalyticalReport, MarketContext, ResearchDocument, ResearchNote, SourceCitation
from dean_os.draft.dean_os_agent_system_v7.dean_os.structured_context_provenance import (
    apply_market_context_structured_boundary,
)
from dean_os.utils import clamp

PATTERN_TERMS: dict[str, tuple[str, ...]] = {
    "defense_rearmament": ("defense budget", "rearmament", "munitions", "missile", "defense contract"),
    "ai_compute_cycle": ("ai", "accelerator", "gpu", "semiconductor", "data center", "compute"),
    "energy_security": ("energy security", "lng", "grid", "uranium", "pipeline", "power demand"),
    "policy_easing": ("rate cut", "easing", "stimulus", "liquidity", "soft landing"),
    "supply_chain_reshoring": ("reshoring", "friendshoring", "supply chain", "domestic capacity"),
    "value_margin_safety": ("margin of safety", "undervalued", "discount", "free cash flow", "book value"),
    "pricing_power": ("pricing power", "backlog", "long-term contract", "recurring revenue"),
    "regulatory_risk": ("antitrust", "export control", "sanction", "regulatory pressure", "tariff"),
    "balance_sheet_stress": ("debt burden", "refinancing", "liquidity crunch", "covenant", "downgrade"),
    "capacity_pressure": ("shortage", "bottleneck", "capacity constraint", "lead time", "inventory glut"),
}

BULLISH_PATTERNS = {
    "defense_rearmament",
    "ai_compute_cycle",
    "energy_security",
    "policy_easing",
    "supply_chain_reshoring",
    "value_margin_safety",
    "pricing_power",
}

RISK_PATTERNS = {"regulatory_risk", "balance_sheet_stress", "capacity_pressure"}


class ResearchIngestionAgent(AnalyticalAgent):
    version = "0.1.0"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        documents = material_documents(context)
        corpus_path = self.config.get("corpus_path")
        chunk_count = 0
        if corpus_path:
            corpus = ResearchCorpus(corpus_path)
            for document in documents:
                chunk_count += len(corpus.add_document(document, chunk_size=int(self.config.get("chunk_size", 1200))))
        else:
            chunk_count = sum(len(chunk_document(document, chunk_size=int(self.config.get("chunk_size", 1200)))) for document in documents)

        data_quality = "partial" if documents else "weak"
        note = ResearchNote(
            agent_name=self.name,
            topic="research_ingestion",
            thesis=f"Ingested {len(documents)} research documents into the agent context.",
            patterns=["research_corpus_ingestion"] if documents else [],
            tickers=context.tickers,
            confidence=0.75 if documents else 0.2,
            data_quality=data_quality,
            evidence=[
                self.evidence("document", "context.research_documents", "document_count", len(documents)),
                self.evidence("document", "research_corpus", "chunk_count", chunk_count),
            ],
            citations=[citation_for_document(document) for document in documents[:10]],
            risks=["Ingestion stores text and metadata; it does not validate factual correctness"],
            blind_spots=["PDF/OCR extraction is not implemented in this MVP"],
        )
        context.research_notes.append(note)

        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict="neutral" if documents else "needs_more_data",
            confidence=note.confidence,
            data_quality_score=0.6 if documents else 0.2,
            signal_strength=0.0,
            ticker=",".join(context.tickers) if context.tickers else None,
            asset_or_sector="research_corpus",
            horizon_years=0.0,
            thesis=note.thesis,
            data_quality=data_quality,
            position_bias="neutral" if documents else "insufficient_data",
            watchlist_score=0.0,
            reasons=[note.thesis],
            risks=note.risks,
            blind_spots=note.blind_spots,
            evidence=note.evidence,
            input_hash=self.context_hash(context),
        )


class SpecialistResearchAgent(AnalyticalAgent):
    version = "0.1.0"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        apply_market_context_structured_boundary(context)
        documents = material_documents(context)
        corpus_chunks = self._load_corpus_chunks(context)
        texts = [document.text.lower() for document in documents]
        texts.extend(chunk.text.lower() for chunk in corpus_chunks)

        pattern_counts = extract_pattern_counts(texts)
        metric_patterns = metric_patterns_from_context(context)
        pattern_counts.update(metric_patterns)
        top_patterns = [name for name, count in pattern_counts.most_common() if count > 0]
        citations = collect_pattern_citations(documents, top_patterns)
        data_quality = self._data_quality(documents, corpus_chunks, context, top_patterns)
        bullish_score = sum(pattern_counts.get(pattern, 0) for pattern in BULLISH_PATTERNS)
        risk_score = sum(pattern_counts.get(pattern, 0) for pattern in RISK_PATTERNS)
        position_bias = self._position_bias(bullish_score, risk_score, data_quality)
        confidence = self._confidence(top_patterns, documents, corpus_chunks, context)
        thesis = self._thesis(top_patterns, position_bias, data_quality)
        memory_snapshot = context.metadata.get("recommendation_memory", {})
        memory_risks, memory_blind_spots = memory_context_warnings(memory_snapshot)

        note = ResearchNote(
            agent_name=self.name,
            topic=self.config.get("topic", "specialist_research"),
            thesis=thesis,
            patterns=top_patterns,
            catalysts=self._catalysts(top_patterns),
            tailwinds=[pattern for pattern in top_patterns if pattern in BULLISH_PATTERNS],
            headwinds=[pattern for pattern in top_patterns if pattern in RISK_PATTERNS],
            tickers=context.tickers,
            sectors=list({sector for document in documents for sector in document.sectors}),
            horizon_days=int(self.config.get("horizon_days", 365)),
            confidence=confidence,
            data_quality=data_quality,
            evidence=[
                self.evidence("pattern", "research_materials", "pattern_counts", dict(pattern_counts)),
                self.evidence("document", "research_materials", "document_count", len(documents)),
                self.evidence("metric", "context.fundamentals", "ticker_count", len(context.fundamentals)),
                self.evidence("memory", "context.metadata.recommendation_memory", "relevant_count", memory_snapshot.get("relevant_count", 0)),
            ],
            citations=citations,
            risks=[*self._risks(data_quality, risk_score), *memory_risks],
            blind_spots=[
                "This MVP extracts rule-based patterns; no LLM synthesis or FinBERT scoring is active yet",
                "Research material quality depends on caller-supplied documents and corpus state",
                *memory_blind_spots,
            ],
        )
        context.research_notes.append(note)
        if self.config.get("corpus_path"):
            ResearchCorpus(self.config["corpus_path"]).add_note(note)

        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=self._verdict(position_bias, data_quality),
            confidence=confidence,
            data_quality_score={"strong": 0.9, "partial": 0.6, "weak": 0.2}[data_quality],
            signal_strength=self._signal_strength(position_bias, confidence),
            ticker=",".join(context.tickers) if context.tickers else None,
            asset_or_sector=note.topic,
            horizon_years=(note.horizon_days or 365) / 365,
            thesis=thesis,
            data_quality=data_quality,
            position_bias=position_bias,
            catalysts=note.catalysts,
            tailwinds=note.tailwinds,
            headwinds=note.headwinds,
            watchlist_score=self._watchlist_score(top_patterns, confidence, data_quality),
            reasons=[f"Extracted research patterns: {', '.join(top_patterns[:6])}" if top_patterns else "No durable patterns found"],
            risks=note.risks,
            blind_spots=note.blind_spots,
            evidence=note.evidence,
            input_hash=self.context_hash(context),
        )

    def _load_corpus_chunks(self, context: MarketContext):
        corpus_path = self.config.get("corpus_path")
        if not corpus_path:
            return []
        query_terms = " ".join([*context.tickers, *context.sector_data.keys(), *context.macro.keys()])
        if not query_terms.strip():
            query_terms = " ".join(PATTERN_TERMS)
        return ResearchCorpus(corpus_path).search_chunks(query_terms, limit=int(self.config.get("corpus_search_limit", 20)))

    def _data_quality(self, documents, corpus_chunks, context: MarketContext, patterns: list[str]) -> str:
        evidence_sources = len(documents) + len(corpus_chunks) + len(context.fundamentals) + len(context.macro)
        if evidence_sources >= 3 and len(patterns) >= 2:
            return "strong"
        if evidence_sources > 0 or patterns:
            return "partial"
        return "weak"

    def _position_bias(self, bullish_score: int, risk_score: int, data_quality: str) -> str:
        if data_quality == "weak":
            return "insufficient_data"
        if bullish_score > risk_score + 1:
            return "bullish"
        if risk_score > bullish_score + 1:
            return "bearish"
        return "neutral"

    def _confidence(self, patterns: list[str], documents, corpus_chunks, context: MarketContext) -> float:
        source_count = len(documents) + len(corpus_chunks) + len(context.fundamentals) + len(context.macro)
        return clamp(0.2 + min(len(patterns), 6) * 0.07 + min(source_count, 5) * 0.06, 0.0, 0.9)

    def _thesis(self, patterns: list[str], position_bias: str, data_quality: str) -> str:
        if data_quality == "weak":
            return "Insufficient research material for a specialist thesis."
        if not patterns:
            return "Research material is available, but no durable pattern is visible yet."
        pattern_text = ", ".join(patterns[:4])
        if position_bias == "bullish":
            return f"Specialist research suggests a constructive setup driven by {pattern_text}."
        if position_bias == "bearish":
            return f"Specialist research flags a risk-heavy setup driven by {pattern_text}."
        return f"Specialist research is mixed; key patterns are {pattern_text}."

    def _catalysts(self, patterns: list[str]) -> list[str]:
        catalyst_patterns = {"defense_rearmament", "policy_easing", "ai_compute_cycle", "supply_chain_reshoring"}
        return [pattern for pattern in patterns if pattern in catalyst_patterns]

    def _risks(self, data_quality: str, risk_score: int) -> list[str]:
        risks = ["Rule-based specialist synthesis can miss nuance that an LLM or human analyst would catch"]
        if data_quality != "strong":
            risks.append("Research evidence is not yet broad enough for high conviction")
        if risk_score:
            risks.append(f"Detected {risk_score} risk-pattern references")
        return risks

    def _verdict(self, position_bias: str, data_quality: str) -> str:
        if data_quality == "weak":
            return "needs_more_data"
        if position_bias == "bullish":
            return "bullish"
        if position_bias == "bearish":
            return "bearish"
        return "neutral"

    def _signal_strength(self, position_bias: str, confidence: float) -> float:
        if position_bias == "bullish":
            return confidence
        if position_bias == "bearish":
            return -confidence
        return 0.0

    def _watchlist_score(self, patterns: list[str], confidence: float, data_quality: str) -> float:
        if data_quality == "weak":
            return 0.0
        return clamp(0.25 + len(patterns) * 0.08 + confidence * 0.25, 0.0, 1.0)


def material_documents(context: MarketContext) -> list[ResearchDocument]:
    raw_documents = list(context.research_documents)
    if context.as_of:
        try:
            document_audit = audit_research_documents(
                raw_documents,
                as_of=context.as_of,
            )
        except ValueError:
            document_audit = {
                "contract": "dean_context_evidence_point_in_time_v1",
                "status": "blocked_context_as_of_invalid",
                "input_count": len(raw_documents),
                "accepted_count": 0,
                "excluded_count": len(raw_documents),
                "accepted": [],
                "exclusions": [
                    {
                        "index": index,
                        "status": "excluded",
                        "reasons": ["context_as_of_invalid"],
                    }
                    for index in range(len(raw_documents))
                ],
            }
    else:
        document_audit = {
            "contract": "dean_context_evidence_point_in_time_v1",
            "status": "blocked_context_as_of_missing",
            "input_count": len(raw_documents),
            "accepted_count": 0,
            "excluded_count": len(raw_documents),
            "accepted": [],
            "exclusions": [
                {
                    "index": index,
                    "status": "excluded",
                    "reasons": ["context_as_of_missing"],
                }
                for index in range(len(raw_documents))
            ],
        }
    documents = list(document_audit["accepted"])
    context.metadata["research_document_point_in_time_audit"] = {
        key: value
        for key, value in document_audit.items()
        if key != "accepted"
    }
    news_audit = audit_market_context_news(context)
    context.metadata["news_point_in_time_audit"] = {
        key: value
        for key, value in news_audit.items()
        if key != "accepted"
    }
    for index, item in enumerate(news_audit["accepted"]):
        if isinstance(item, ResearchDocument):
            documents.append(item)
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("headline") or f"News item {index}")
            text = " ".join(
                str(item.get(key, ""))
                for key in ("title", "headline", "summary", "description", "content", "text")
                if item.get(key)
            )
            if text:
                documents.append(
                    ResearchDocument(
                        title=title,
                        source_type="news",
                        text=text,
                        uri=item.get("url") or item.get("uri"),
                        published_at=item.get("published_at") or item.get("timestamp"),
                        tickers=item.get("tickers") or context.tickers,
                        sectors=item.get("sectors") or [],
                        tags=item.get("tags") or [],
                        metadata={key: value for key, value in item.items() if key not in {"text", "content"}},
                    )
                )
    return documents


def extract_pattern_counts(texts: list[str]) -> Counter:
    counts: Counter = Counter()
    for pattern, terms in PATTERN_TERMS.items():
        counts[pattern] = sum(text.count(term) for text in texts for term in terms)
    return counts


def metric_patterns_from_context(context: MarketContext) -> Counter:
    counts: Counter = Counter()
    for metrics in context.fundamentals.values():
        pe = _as_float(metrics.get("pe") or metrics.get("price_to_earnings"))
        pb = _as_float(metrics.get("pb") or metrics.get("price_to_book"))
        fcf_yield = _as_float(metrics.get("fcf_yield"))
        roe = _as_float(metrics.get("roe"))
        debt_to_equity = _as_float(metrics.get("debt_to_equity"))
        if (pe is not None and 0 < pe <= 15) or (pb is not None and 0 < pb <= 1.5) or (fcf_yield is not None and fcf_yield >= 0.05):
            counts["value_margin_safety"] += 1
        if (roe is not None and roe >= 0.12) and (debt_to_equity is None or debt_to_equity <= 1.0):
            counts["pricing_power"] += 1
        if debt_to_equity is not None and debt_to_equity > 2.0:
            counts["balance_sheet_stress"] += 1
    return counts


def collect_pattern_citations(documents: list[ResearchDocument], patterns: list[str]) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    for document in documents:
        lowered = document.text.lower()
        if any(term in lowered for pattern in patterns for term in PATTERN_TERMS.get(pattern, ())):
            citations.append(citation_for_document(document))
        if len(citations) >= 10:
            break
    return citations


def citation_for_document(document: ResearchDocument) -> SourceCitation:
    return SourceCitation(
        source_id=document.document_id,
        source_type=document.source_type,
        title=document.title,
        uri=document.uri,
        locator="document",
        excerpt=document.text[:280],
        timestamp=document.published_at,
    )


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def memory_context_warnings(memory_snapshot: dict[str, Any]) -> tuple[list[str], list[str]]:
    if not memory_snapshot or not memory_snapshot.get("relevant_count"):
        return [], ["No relevant recommendation memory was available for this context"]
    risks = []
    blind_spots = []
    miss_count = int(memory_snapshot.get("miss_count") or 0)
    hit_count = int(memory_snapshot.get("hit_count") or 0)
    if miss_count:
        risks.append(f"Recommendation memory has {miss_count} miss case(s) in similar context; review lessons before promotion")
    if hit_count:
        blind_spots.append(f"Recommendation memory also has {hit_count} hit case(s); compare conditions before overcorrecting")
    lessons = [lesson for lesson in memory_snapshot.get("lessons", []) if lesson]
    if lessons:
        blind_spots.append(f"Memory lessons to review: {' | '.join(lessons[:3])}")
    return risks, blind_spots
