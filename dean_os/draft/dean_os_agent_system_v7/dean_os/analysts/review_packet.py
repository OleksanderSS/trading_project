from __future__ import annotations

from .schemas import DomainProfile, DomainThesis, TickerBasketReport


def build_review_packet(profile: DomainProfile, thesis: DomainThesis, basket: TickerBasketReport) -> dict:
    return {
        "review_required": True,
        "domain_id": profile.domain_id,
        "domain_questions": profile.core_questions,
        "blocked_if_missing": profile.blocked_if_missing,
        "thesis_id": thesis.thesis_id,
        "thesis_stance": thesis.stance,
        "thesis_expected_direction": thesis.expected_direction,
        "basket_status": basket.basket_status,
        "ticker_candidates": [
            {
                "ticker": candidate.ticker,
                "candidate_status": candidate.candidate_status,
                "confidence": candidate.confidence,
                "blocked_reasons": candidate.blocked_reasons,
                "blocked_windows": candidate.blocked_windows,
            }
            for candidate in basket.candidates
        ],
        "operator_note": "Review-only analyst output. This does not authorize live execution.",
    }
