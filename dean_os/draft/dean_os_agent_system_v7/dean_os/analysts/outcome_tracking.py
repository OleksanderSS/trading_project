from __future__ import annotations

from .schemas import DomainProfile, DomainThesis, TickerBasketReport


def build_outcome_tracking_plan(profile: DomainProfile, thesis: DomainThesis, basket: TickerBasketReport) -> dict:
    return {
        "track_domain_thesis": True,
        "track_ticker_candidates": True,
        "domain_id": profile.domain_id,
        "thesis_id": thesis.thesis_id,
        "horizon_days": thesis.horizon_days,
        "candidate_tickers": [candidate.ticker for candidate in basket.candidates],
        "requires_future_outcome_evaluation": True,
        "calibration_tags": [
            profile.domain_id,
            f"horizon_{thesis.horizon_days}d",
            thesis.stance,
            basket.basket_status,
        ],
    }
