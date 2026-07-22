from __future__ import annotations

from .schemas import DomainThesis, TickerBasketReport, TickerCandidateThesis


class SectorBridgePayloadAdapter:
    """Converts existing SectorThesisToTickerBasketBridge payloads into analyst schemas.

    Existing bridge statuses:
    - direct_ticker_thesis_ready
    - ticker_context_ready
    - blocked_missing_ticker_evidence
    - sector_context_only

    Analyst system statuses:
    - direct_ticker_thesis
    - basket_candidate
    - blocked_missing_evidence
    """

    @staticmethod
    def from_payload(payload: dict, thesis: DomainThesis | None = None) -> TickerBasketReport:
        summary = payload.get("summary", {})
        sector_thesis = payload.get("sector_thesis", {})
        domain_id = str(payload.get("inputs", {}).get("domain_profile") or sector_thesis.get("domain_profile") or "unknown_domain")
        source_thesis_id = thesis.thesis_id if thesis is not None else str(sector_thesis.get("thesis_id") or payload.get("run_id") or "sector_bridge_payload")
        expected_direction = thesis.expected_direction if thesis is not None else _direction_from_sector(sector_thesis)

        candidates = [
            _candidate_from_existing(domain_id, source_thesis_id, expected_direction, item)
            for item in payload.get("ticker_candidates", [])
        ]

        return TickerBasketReport(
            domain_id=domain_id,
            source_thesis_id=source_thesis_id,
            basket_status=_basket_status_from_existing(str(summary.get("bridge_status") or "")),
            candidates=candidates,
            direct_ready_count=sum(1 for item in candidates if item.candidate_status == "direct_ticker_thesis"),
            basket_candidate_count=sum(1 for item in candidates if item.candidate_status == "basket_candidate"),
            blocked_count=sum(1 for item in candidates if item.candidate_status == "blocked_missing_evidence"),
            reasons=[
                f"Converted from SectorThesisToTickerBasketBridge status: {summary.get('bridge_status')}",
                str(summary.get("next_action") or ""),
            ],
            review_required=True,
        )


def _candidate_from_existing(
    domain_id: str,
    source_thesis_id: str,
    expected_direction: str,
    item: dict,
) -> TickerCandidateThesis:
    status = _candidate_status_from_existing(str(item.get("candidate_status") or ""))
    blocked_windows = [str(value) for value in item.get("blocked_as_of", []) if value]
    blocked_reasons = list(item.get("limitations", []))
    if blocked_windows and "some_windows_blocked_by_weak_direct_evidence" not in blocked_reasons:
        blocked_reasons.append("some_windows_blocked_by_weak_direct_evidence")

    return TickerCandidateThesis(
        ticker=str(item.get("ticker") or "UNKNOWN"),
        domain_id=domain_id,
        source_thesis_id=source_thesis_id,
        candidate_status=status,
        expected_direction=expected_direction,  # type: ignore[arg-type]
        confidence=_confidence_from_existing(item),
        ticker_specific_evidence_ids=[f"sector_bridge:{item.get('ticker')}"] if status == "direct_ticker_thesis" else [],
        sector_only_evidence_ids=[],
        blocked_reasons=blocked_reasons,
        required_missing_evidence=[] if status == "direct_ticker_thesis" else ["ticker_specific_evidence"],
        blocked_windows=blocked_windows,
        review_required=True,
    )


def _candidate_status_from_existing(status: str) -> str:
    if status == "direct_ticker_thesis_ready":
        return "direct_ticker_thesis"
    if status in {"ticker_context_ready", "sector_context_only"}:
        return "basket_candidate"
    return "blocked_missing_evidence"


def _basket_status_from_existing(status: str) -> str:
    if status == "ticker_basket_ready_for_review":
        return "basket_ready_for_review"
    if status == "partial_basket_ready":
        return "partial_basket_ready"
    if status == "no_runs":
        return "needs_more_data"
    if status == "sector_context_only":
        return "partial_basket_ready"
    return "basket_blocked"


def _direction_from_sector(sector_thesis: dict) -> str:
    stance = str(sector_thesis.get("sector_stance") or "")
    if stance == "constructive":
        return "positive"
    if stance == "risk_heavy":
        return "negative"
    if stance == "mixed":
        return "mixed"
    return "neutral"


def _confidence_from_existing(item: dict) -> float:
    ready = float(item.get("overlay_ready_runs") or 0)
    total = float(item.get("runs") or 0)
    if total <= 0:
        return 0.0
    base = ready / total
    if int(item.get("blocked_runs") or 0) > 0:
        base = min(base, 0.65)
    return max(0.0, min(1.0, base))
