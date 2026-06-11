from __future__ import annotations

from dean_os.base import AnalyticalAgent
from dean_os.agents.research_agents import memory_context_warnings
from dean_os.schemas import AnalyticalReport, MarketContext, ResearchNote
from dean_os.synthesis import EvidenceBoundSynthesizer
from dean_os.utils import clamp


class EvidenceSynthesisAgent(AnalyticalAgent):
    version = "0.1.0"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        notes = [note for note in context.research_notes if note.agent_name != self.name]
        synthesis_note = EvidenceBoundSynthesizer().synthesize(
            agent_name=self.name,
            notes=notes,
            nlp_results=context.nlp_results,
            topic=self.config.get("topic", "evidence_synthesis"),
        )
        memory_snapshot = context.metadata.get("recommendation_memory", {})
        memory_risks, memory_blind_spots = memory_context_warnings(memory_snapshot)
        if memory_snapshot:
            synthesis_note.evidence.append(
                self.evidence(
                    "memory",
                    "context.metadata.recommendation_memory",
                    "relevant_count",
                    memory_snapshot.get("relevant_count", 0),
                )
            )
        synthesis_note.risks.extend(memory_risks)
        synthesis_note.blind_spots.extend(memory_blind_spots)
        context.research_notes.append(synthesis_note)
        position_bias = self._position_bias(synthesis_note)
        return self._report(context, synthesis_note, position_bias)

    def _position_bias(self, note: ResearchNote) -> str:
        if note.data_quality == "weak":
            return "insufficient_data"
        bullish = len(note.tailwinds)
        bearish = len(note.headwinds)
        if bullish > bearish:
            return "bullish"
        if bearish > bullish:
            return "bearish"
        return "neutral"

    def _report(self, context: MarketContext, note: ResearchNote, position_bias: str) -> AnalyticalReport:
        if note.data_quality == "weak":
            verdict = "needs_more_data"
        elif position_bias == "bullish":
            verdict = "bullish"
        elif position_bias == "bearish":
            verdict = "bearish"
        else:
            verdict = "neutral"
        signal_strength = 0.0
        if position_bias == "bullish":
            signal_strength = note.confidence
        elif position_bias == "bearish":
            signal_strength = -note.confidence
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=note.confidence,
            data_quality_score={"strong": 0.9, "partial": 0.6, "weak": 0.2}[note.data_quality],
            signal_strength=signal_strength,
            ticker=",".join(context.tickers) if context.tickers else None,
            asset_or_sector=note.topic,
            horizon_years=(note.horizon_days or 365) / 365,
            thesis=note.thesis,
            data_quality=note.data_quality,
            position_bias=position_bias,
            catalysts=note.catalysts,
            tailwinds=note.tailwinds,
            headwinds=note.headwinds,
            watchlist_score=clamp(0.25 + note.confidence * 0.5 + len(note.citations) * 0.03, 0.0, 1.0),
            reasons=[note.thesis],
            risks=note.risks,
            blind_spots=note.blind_spots,
            evidence=note.evidence,
            input_hash=self.context_hash(context),
        )
