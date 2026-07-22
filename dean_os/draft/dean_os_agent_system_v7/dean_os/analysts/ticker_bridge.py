from __future__ import annotations

from .evidence import blocked_windows_for_ticker, evidence_quality_score, sector_only_evidence, ticker_specific_evidence
from .schemas import AnalystEvidenceItem, DomainThesis, TickerBasketReport, TickerCandidateThesis


class DomainTickerBridge:
    """Converts a domain thesis into ticker candidates with direct-evidence guardrails."""

    def build(
        self,
        domain_id: str,
        thesis: DomainThesis,
        evidence: list[AnalystEvidenceItem],
        tickers: list[str],
    ) -> TickerBasketReport:
        sector_evidence = sector_only_evidence(evidence)
        sector_ids = [item.evidence_id for item in sector_evidence]
        candidates: list[TickerCandidateThesis] = []

        for ticker in sorted({item.upper().strip() for item in tickers if item.strip()}):
            direct = ticker_specific_evidence(evidence, ticker)
            blocked_windows = blocked_windows_for_ticker(evidence, ticker)

            if direct:
                status = "direct_ticker_thesis"
                confidence = min(thesis.confidence, evidence_quality_score(direct))
                blocked_reasons = []
                missing = []
                if blocked_windows:
                    blocked_reasons.append("Some ticker-specific evidence windows are blocked or incomplete.")
            elif sector_evidence:
                status = "basket_candidate"
                confidence = min(thesis.confidence, 0.35)
                blocked_reasons = ["Only sector/domain evidence is available; direct ticker thesis is not allowed."]
                missing = ["ticker_specific_evidence"]
            else:
                status = "blocked_missing_evidence"
                confidence = 0.0
                blocked_reasons = ["No direct ticker evidence and no sector evidence available."]
                missing = ["ticker_specific_evidence", "sector_evidence"]

            candidates.append(
                TickerCandidateThesis(
                    ticker=ticker,
                    domain_id=domain_id,
                    source_thesis_id=thesis.thesis_id,
                    candidate_status=status,
                    expected_direction=thesis.expected_direction,
                    confidence=confidence,
                    ticker_specific_evidence_ids=[item.evidence_id for item in direct],
                    sector_only_evidence_ids=sector_ids,
                    blocked_reasons=blocked_reasons,
                    required_missing_evidence=missing,
                    blocked_windows=blocked_windows,
                    review_required=True,
                )
            )

        return self._basket_report(domain_id, thesis, candidates)

    def _basket_report(
        self,
        domain_id: str,
        thesis: DomainThesis,
        candidates: list[TickerCandidateThesis],
    ) -> TickerBasketReport:
        direct_ready = [item for item in candidates if item.candidate_status == "direct_ticker_thesis" and not item.blocked_windows]
        direct_with_blocked_windows = [
            item for item in candidates if item.candidate_status == "direct_ticker_thesis" and item.blocked_windows
        ]
        basket_only = [item for item in candidates if item.candidate_status == "basket_candidate"]
        blocked = [item for item in candidates if item.candidate_status == "blocked_missing_evidence"]

        if not candidates:
            status = "needs_more_data"
            reasons = ["No ticker universe was provided."]
        elif len(direct_ready) == len(candidates):
            status = "basket_ready_for_review"
            reasons = ["All ticker candidates have direct ticker evidence and no blocked windows."]
        elif direct_ready or direct_with_blocked_windows:
            status = "partial_basket_ready"
            reasons = ["At least one ticker has direct evidence, but the basket is incomplete or partially blocked."]
        elif basket_only:
            status = "partial_basket_ready"
            reasons = ["Only sector/domain evidence exists; candidates are basket-level, not direct ticker theses."]
        else:
            status = "basket_blocked"
            reasons = ["No ticker candidate has enough evidence."]

        return TickerBasketReport(
            domain_id=domain_id,
            source_thesis_id=thesis.thesis_id,
            basket_status=status,
            candidates=candidates,
            direct_ready_count=len(direct_ready) + len(direct_with_blocked_windows),
            basket_candidate_count=len(basket_only),
            blocked_count=len(blocked),
            reasons=reasons,
            review_required=True,
        )
