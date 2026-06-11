from __future__ import annotations

from dean_os.agents.research_agents import material_documents
from dean_os.base import AnalyticalAgent
from dean_os.financial_nlp import OptionalLocalFinBERT, RuleBasedFinancialNLP
from dean_os.schemas import AnalyticalReport, MarketContext, ResearchNote
from dean_os.utils import clamp


class FinancialNLPAgent(AnalyticalAgent):
    version = "0.1.0"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        documents = material_documents(context)
        analyzer = self._analyzer()
        results = [analyzer.analyze_document(document, self.name) for document in documents]
        context.nlp_results.extend(results)

        if not results:
            note = ResearchNote(
                agent_name=self.name,
                topic="financial_nlp",
                thesis="No research material available for financial NLP analysis.",
                confidence=0.1,
                data_quality="weak",
                risks=["No documents were supplied"],
                blind_spots=["Financial NLP agent needs documents, news, transcripts, or reports"],
            )
            context.research_notes.append(note)
            return self._report(context, note, "needs_more_data", "insufficient_data", 0.0, [])

        avg_sentiment = sum(result.sentiment_score for result in results) / len(results)
        max_risk = max(result.risk_score for result in results)
        event_types = sorted({event for result in results for event in result.event_types})
        key_terms = sorted({term for result in results for term in result.key_terms})
        data_quality = "strong" if len(results) >= 3 else "partial"
        position_bias = self._position_bias(avg_sentiment, max_risk)
        confidence = clamp(0.3 + min(len(results), 5) * 0.08 + abs(avg_sentiment) * 0.2 + max_risk * 0.1, 0.0, 0.9)
        thesis = self._thesis(avg_sentiment, max_risk, event_types, position_bias)

        note = ResearchNote(
            agent_name=self.name,
            topic="financial_nlp",
            thesis=thesis,
            patterns=event_types,
            catalysts=[event for event in event_types if event in {"earnings", "contract", "policy", "capital_cycle"}],
            headwinds=["high_risk_tone"] if max_risk >= 0.65 else [],
            tickers=context.tickers,
            confidence=confidence,
            data_quality=data_quality,
            evidence=[
                self.evidence("research_note", "context.nlp_results", "document_count", len(results)),
                self.evidence("metric", "financial_nlp", "avg_sentiment", round(avg_sentiment, 4)),
                self.evidence("metric", "financial_nlp", "max_risk", round(max_risk, 4)),
                self.evidence("pattern", "financial_nlp", "event_types", event_types),
            ],
            citations=[citation for result in results for citation in result.citations[:1]][:10],
            risks=self._risks(max_risk),
            blind_spots=[
                "Rule-based fallback is active unless a local FinBERT model is configured",
                "No GPT synthesis is used in this MVP NLP agent",
            ],
        )
        context.research_notes.append(note)
        return self._report(context, note, self._verdict(position_bias), position_bias, avg_sentiment, key_terms)

    def _analyzer(self):
        model_name = self.config.get("finbert_model")
        if model_name:
            return OptionalLocalFinBERT(model_name=model_name)
        return RuleBasedFinancialNLP()

    def _position_bias(self, avg_sentiment: float, max_risk: float) -> str:
        if avg_sentiment >= 0.25 and max_risk < 0.65:
            return "bullish"
        if avg_sentiment <= -0.25 or max_risk >= 0.75:
            return "bearish"
        return "neutral"

    def _thesis(self, avg_sentiment: float, max_risk: float, event_types: list[str], position_bias: str) -> str:
        events = ", ".join(event_types[:5]) if event_types else "no dominant event class"
        if position_bias == "bullish":
            return f"Financial NLP tone is constructive across supplied materials, with events: {events}."
        if position_bias == "bearish":
            return f"Financial NLP flags a risk-heavy or negative setup, with events: {events}."
        return f"Financial NLP tone is mixed or neutral; sentiment={avg_sentiment:.2f}, risk={max_risk:.2f}, events: {events}."

    def _risks(self, max_risk: float) -> list[str]:
        risks = ["NLP tone is supporting evidence only, not a trade signal"]
        if max_risk >= 0.65:
            risks.append("High risk tone detected in at least one source")
        return risks

    def _verdict(self, position_bias: str) -> str:
        if position_bias == "bullish":
            return "bullish"
        if position_bias == "bearish":
            return "bearish"
        return "neutral"

    def _report(
        self,
        context: MarketContext,
        note: ResearchNote,
        verdict: str,
        position_bias: str,
        avg_sentiment: float,
        key_terms: list[str],
    ) -> AnalyticalReport:
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=note.confidence,
            data_quality_score={"strong": 0.85, "partial": 0.6, "weak": 0.2}[note.data_quality],
            signal_strength=avg_sentiment if position_bias != "insufficient_data" else 0.0,
            ticker=",".join(context.tickers) if context.tickers else None,
            asset_or_sector="financial_nlp",
            horizon_years=0.25,
            thesis=note.thesis,
            data_quality=note.data_quality,
            position_bias=position_bias,
            catalysts=note.catalysts,
            tailwinds=key_terms[:8] if position_bias == "bullish" else [],
            headwinds=note.headwinds,
            watchlist_score=clamp(0.25 + note.confidence * 0.5 + abs(avg_sentiment) * 0.2, 0.0, 1.0),
            reasons=[note.thesis],
            risks=note.risks,
            blind_spots=note.blind_spots,
            evidence=note.evidence,
            input_hash=self.context_hash(context),
        )
