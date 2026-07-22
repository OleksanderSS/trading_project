from __future__ import annotations

from collections import Counter

from dean_os.schemas import EvidenceItem, FinancialNLPResult, ResearchNote, SourceCitation
from dean_os.utils import clamp


class EvidenceBoundSynthesizer:
    """Creates a thesis only from structured notes and citations."""

    def synthesize(
        self,
        agent_name: str,
        notes: list[ResearchNote],
        nlp_results: list[FinancialNLPResult],
        topic: str = "evidence_synthesis",
    ) -> ResearchNote:
        patterns = Counter(pattern for note in notes for pattern in note.patterns)
        event_types = Counter(event for result in nlp_results for event in result.event_types)
        citations = collect_citations(notes, nlp_results)
        avg_confidence = sum(note.confidence for note in notes) / len(notes) if notes else 0.0
        avg_sentiment = sum(result.sentiment_score for result in nlp_results) / len(nlp_results) if nlp_results else 0.0
        max_risk = max((result.risk_score for result in nlp_results), default=0.0)
        top_patterns = [pattern for pattern, _ in patterns.most_common(8)]
        top_events = [event for event, _ in event_types.most_common(6)]
        data_quality = self._data_quality(notes, citations)

        return ResearchNote(
            agent_name=agent_name,
            topic=topic,
            thesis=self._thesis(top_patterns, top_events, avg_sentiment, max_risk, data_quality),
            patterns=[*top_patterns, *[event for event in top_events if event not in top_patterns]],
            catalysts=[pattern for pattern in top_patterns if pattern in {"defense_rearmament", "ai_compute_cycle", "policy_easing", "capital_cycle"}],
            tailwinds=self._tailwinds(top_patterns, avg_sentiment),
            headwinds=self._headwinds(top_patterns, max_risk),
            tickers=sorted({ticker for note in notes for ticker in note.tickers}),
            sectors=sorted({sector for note in notes for sector in note.sectors}),
            horizon_days=self._horizon(notes),
            confidence=clamp(avg_confidence + min(len(citations), 8) * 0.03, 0.0, 0.9),
            data_quality=data_quality,
            evidence=[
                EvidenceItem(source_type="research_note", source="context.research_notes", key="note_count", value=len(notes)),
                EvidenceItem(source_type="research_note", source="context.nlp_results", key="nlp_result_count", value=len(nlp_results)),
                EvidenceItem(source_type="metric", source="financial_nlp", key="avg_sentiment", value=round(avg_sentiment, 4)),
                EvidenceItem(source_type="metric", source="financial_nlp", key="max_risk", value=round(max_risk, 4)),
                EvidenceItem(source_type="pattern", source="evidence_synthesis", key="top_patterns", value=top_patterns),
            ],
            citations=citations[:12],
            risks=self._risks(data_quality, max_risk),
            blind_spots=[
                "Synthesis is deterministic in this MVP; GPT can be plugged in later with the same evidence package",
                "No claim should be trusted without reviewing citations",
            ],
        )

    def _data_quality(self, notes: list[ResearchNote], citations: list[SourceCitation]) -> str:
        if len(notes) >= 3 and len(citations) >= 3:
            return "strong"
        if notes and citations:
            return "partial"
        return "weak"

    def _thesis(
        self,
        patterns: list[str],
        events: list[str],
        avg_sentiment: float,
        max_risk: float,
        data_quality: str,
    ) -> str:
        if data_quality == "weak":
            return "Insufficient cited evidence for an evidence-bound thesis."
        pattern_text = ", ".join(patterns[:4]) if patterns else "no dominant pattern"
        event_text = ", ".join(events[:3]) if events else "no dominant event class"
        if avg_sentiment > 0.25 and max_risk < 0.65:
            return f"Cited research supports a constructive thesis around {pattern_text}, with events: {event_text}."
        if avg_sentiment < -0.25 or max_risk >= 0.75:
            return f"Cited research supports a risk-aware thesis around {pattern_text}, with events: {event_text}."
        return f"Cited research is mixed; dominant patterns are {pattern_text}, with events: {event_text}."

    def _tailwinds(self, patterns: list[str], avg_sentiment: float) -> list[str]:
        tailwinds = [pattern for pattern in patterns if pattern in {"defense_rearmament", "ai_compute_cycle", "value_margin_safety", "pricing_power"}]
        if avg_sentiment > 0.25:
            tailwinds.append("positive_financial_nlp_tone")
        return tailwinds

    def _headwinds(self, patterns: list[str], max_risk: float) -> list[str]:
        headwinds = [pattern for pattern in patterns if pattern in {"regulatory_risk", "balance_sheet_stress", "capacity_pressure"}]
        if max_risk >= 0.65:
            headwinds.append("high_risk_tone")
        return headwinds

    def _horizon(self, notes: list[ResearchNote]) -> int | None:
        horizons = [note.horizon_days for note in notes if note.horizon_days is not None]
        if not horizons:
            return None
        return int(sum(horizons) / len(horizons))

    def _risks(self, data_quality: str, max_risk: float) -> list[str]:
        risks = ["Evidence-bound synthesis is still a research note, not a trade signal"]
        if data_quality != "strong":
            risks.append("Cited evidence is partial")
        if max_risk >= 0.65:
            risks.append("High risk tone is present in supporting materials")
        return risks


def collect_citations(notes: list[ResearchNote], nlp_results: list[FinancialNLPResult]) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    seen: set[str] = set()
    for citation in [*(citation for note in notes for citation in note.citations), *(citation for result in nlp_results for citation in result.citations)]:
        key = f"{citation.source_id}:{citation.locator}:{citation.title}"
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
    return citations
