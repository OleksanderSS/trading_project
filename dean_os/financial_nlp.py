from __future__ import annotations

from collections import Counter
from typing import Protocol

from dean_os.schemas import FinancialNLPResult, ResearchDocument, SourceCitation
from dean_os.utils import clamp


POSITIVE_TERMS = (
    "beat",
    "raise guidance",
    "contract win",
    "approval",
    "buyback",
    "pricing power",
    "margin expansion",
    "free cash flow",
    "backlog",
    "demand growth",
    "soft landing",
)

NEGATIVE_TERMS = (
    "miss",
    "cut guidance",
    "lawsuit",
    "probe",
    "recall",
    "delay",
    "margin pressure",
    "downgrade",
    "recession",
    "liquidity crunch",
    "debt burden",
)

RISK_TERMS = (
    "risk",
    "uncertainty",
    "sanction",
    "export control",
    "regulatory pressure",
    "tariff",
    "shortage",
    "inventory glut",
    "capacity constraint",
    "refinancing",
    "covenant",
)

EVENT_TERMS: dict[str, tuple[str, ...]] = {
    "earnings": ("earnings", "beat", "miss", "guidance"),
    "contract": ("contract", "award", "backlog", "order"),
    "policy": ("fed", "rate cut", "rate hike", "stimulus", "tariff", "sanction"),
    "regulatory": ("approval", "probe", "lawsuit", "antitrust", "export control"),
    "capital_cycle": ("capex", "capacity", "inventory", "cycle", "backlog"),
    "valuation": ("undervalued", "discount", "margin of safety", "free cash flow", "book value"),
}


class FinancialNLPAnalyzer(Protocol):
    def analyze_document(self, document: ResearchDocument, agent_name: str) -> FinancialNLPResult:
        ...


class RuleBasedFinancialNLP:
    """Deterministic financial NLP fallback.

    This is intentionally simple and auditable. It gives the agent lab a stable
    contract before we plug in local FinBERT or an evidence-bound LLM.
    """

    def analyze_document(self, document: ResearchDocument, agent_name: str) -> FinancialNLPResult:
        text = document.text.lower()
        positive_hits = _term_counts(text, POSITIVE_TERMS)
        negative_hits = _term_counts(text, NEGATIVE_TERMS)
        risk_hits = _term_counts(text, RISK_TERMS)
        event_hits = {
            event_type: sum(text.count(term) for term in terms)
            for event_type, terms in EVENT_TERMS.items()
        }
        event_types = [event_type for event_type, count in event_hits.items() if count > 0]
        sentiment_score = _sentiment_score(sum(positive_hits.values()), sum(negative_hits.values()))
        risk_score = clamp(sum(risk_hits.values()) / 8, 0.0, 1.0)
        tone = _tone(sentiment_score, risk_score)
        key_terms = [term for term, count in Counter({**positive_hits, **negative_hits, **risk_hits}).most_common() if count > 0]

        return FinancialNLPResult(
            agent_name=agent_name,
            document_id=document.document_id,
            title=document.title,
            tone=tone,
            sentiment_score=sentiment_score,
            risk_score=risk_score,
            event_types=event_types,
            key_terms=key_terms[:12],
            summary=_summary(document.title, tone, sentiment_score, risk_score, event_types),
            citations=[_citation(document)],
            metadata={
                "method": "rule_based",
                "positive_hits": positive_hits,
                "negative_hits": negative_hits,
                "risk_hits": risk_hits,
                "event_hits": event_hits,
            },
        )


class OptionalLocalFinBERT:
    """Optional local FinBERT wrapper.

    It never downloads models. Pass a local model path/name available in the
    environment and keep `local_files_only=True`.
    """

    def __init__(self, model_name: str, fallback: FinancialNLPAnalyzer | None = None):
        self.model_name = model_name
        self.fallback = fallback or RuleBasedFinancialNLP()
        self._pipeline = None

    def analyze_document(self, document: ResearchDocument, agent_name: str) -> FinancialNLPResult:
        try:
            pipeline = self._load_pipeline()
            text = document.text[:4000]
            raw = pipeline(text)
            result = self.fallback.analyze_document(document, agent_name)
            result.metadata["finbert_raw"] = raw
            result.metadata["method"] = "local_finbert_with_rule_fallback"
            return result
        except Exception as exc:
            result = self.fallback.analyze_document(document, agent_name)
            result.metadata["finbert_error"] = repr(exc)
            return result

    def _load_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                local_files_only=True,
            )
        return self._pipeline


def _term_counts(text: str, terms: tuple[str, ...]) -> dict[str, int]:
    return {term: text.count(term) for term in terms if text.count(term) > 0}


def _sentiment_score(positive_count: int, negative_count: int) -> float:
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    return clamp((positive_count - negative_count) / total, -1.0, 1.0)


def _tone(sentiment_score: float, risk_score: float) -> str:
    if risk_score >= 0.65 and abs(sentiment_score) < 0.35:
        return "mixed"
    if sentiment_score >= 0.25:
        return "positive"
    if sentiment_score <= -0.25:
        return "negative"
    return "neutral"


def _summary(title: str, tone: str, sentiment_score: float, risk_score: float, event_types: list[str]) -> str:
    events = ", ".join(event_types) if event_types else "no explicit event type"
    return f"{title}: tone={tone}, sentiment={sentiment_score:.2f}, risk={risk_score:.2f}, events={events}."


def _citation(document: ResearchDocument) -> SourceCitation:
    return SourceCitation(
        source_id=document.document_id,
        source_type=document.source_type,
        title=document.title,
        uri=document.uri,
        locator="document",
        excerpt=document.text[:280],
        timestamp=document.published_at,
    )
