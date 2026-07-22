from __future__ import annotations

from .schemas import AnalystEvidenceItem, DomainProfile, TickerBasketReport


def build_quality_gates(
    profile: DomainProfile,
    evidence: list[AnalystEvidenceItem],
    missing_required_evidence: list[str],
    evidence_quality_score: float,
    ticker_basket: TickerBasketReport,
) -> dict:
    return {
        "domain_id": profile.domain_id,
        "missing_required_evidence": missing_required_evidence,
        "required_evidence_complete": not missing_required_evidence,
        "evidence_count": len(evidence),
        "evidence_quality_score": evidence_quality_score,
        "basket_status": ticker_basket.basket_status,
        "direct_ticker_guardrail_enabled": True,
        "review_required": True,
        "live_execution_allowed": False,
        "broker_access_allowed": False,
        "production_config_write_allowed": False,
        "learning_memory_write_allowed": False,
    }
