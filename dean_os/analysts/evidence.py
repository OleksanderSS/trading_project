from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .schemas import AnalystEvidenceItem, DomainProfile


def required_evidence_missing(profile: DomainProfile, evidence: Iterable[AnalystEvidenceItem]) -> list[str]:
    available = {
        item.evidence_type
        for item in evidence
        if item.provenance.get("required_lane_eligible") is not False
    }
    return [item for item in profile.required_evidence_types if item not in available]


def evidence_quality_score(evidence: Iterable[AnalystEvidenceItem]) -> float:
    items = list(evidence)
    if not items:
        return 0.0

    scores = [
        (item.strength * 0.4) + (item.reliability_score * 0.4) + (item.freshness_score * 0.2)
        for item in items
    ]
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def stance_from_evidence(evidence: Iterable[AnalystEvidenceItem]) -> tuple[str, str]:
    counts = Counter(item.stance_hint for item in evidence)
    positive = counts["positive"]
    negative = counts["negative"]
    mixed = counts["mixed"]

    if positive == 0 and negative == 0 and mixed == 0:
        return "insufficient_data", "neutral"
    if mixed > 0 or (positive > 0 and negative > 0):
        return "mixed", "mixed"
    if positive > negative:
        return "constructive", "positive"
    if negative > positive:
        return "risk_heavy", "negative"
    return "neutral", "neutral"


def ticker_specific_evidence(evidence: Iterable[AnalystEvidenceItem], ticker: str) -> list[AnalystEvidenceItem]:
    normalized = ticker.upper()
    return [
        item
        for item in evidence
        if item.directness == "ticker"
        and normalized in item.tickers
        and not (
            item.source_type == "fundamental"
            and item.provenance.get("ticker_thesis_eligible") is not True
        )
        and item.provenance.get("ticker_thesis_eligible") is not False
    ]


def sector_only_evidence(evidence: Iterable[AnalystEvidenceItem]) -> list[AnalystEvidenceItem]:
    return [
        item
        for item in evidence
        if item.directness in {"domain", "sector", "macro", "policy", "market", "geopolitical"}
    ]


def blocked_windows_for_ticker(evidence: Iterable[AnalystEvidenceItem], ticker: str) -> list[str]:
    windows: list[str] = []
    for item in ticker_specific_evidence(evidence, ticker):
        windows.extend(item.blocked_windows)
    return sorted(set(windows))
