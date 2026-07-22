from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from .evidence import evidence_quality_score, required_evidence_missing, stance_from_evidence
from .outcome_tracking import build_outcome_tracking_plan
from .profiles import get_domain_profile
from .quality_gates import build_quality_gates
from .review_packet import build_review_packet
from .schemas import AnalystEvidenceItem, AnalystReport, DomainProfile, DomainThesis
from .ticker_bridge import DomainTickerBridge


class BaseAnalystAgent:
    """Review-only base class for DEAN-OS domain analysts.

    This class intentionally has no execute/trade/live methods.
    """

    def __init__(self, domain_id: str, agent_name: str | None = None, ticker_bridge: DomainTickerBridge | None = None):
        self.profile: DomainProfile = get_domain_profile(domain_id)
        self.agent_name = agent_name or f"{domain_id}_analyst"
        self.ticker_bridge = ticker_bridge or DomainTickerBridge()

    def run(
        self,
        evidence: Iterable[AnalystEvidenceItem | dict],
        tickers: list[str] | None = None,
        horizon_days: int | None = None,
        as_of: str | None = None,
    ) -> AnalystReport:
        as_of = as_of or datetime.now(UTC).isoformat()
        horizon_days = horizon_days or self.profile.horizon_days_default
        normalized_evidence = self.normalize_evidence(evidence, as_of=as_of)

        missing = required_evidence_missing(self.profile, normalized_evidence)
        quality_score = evidence_quality_score(normalized_evidence)

        thesis = self.build_domain_thesis(
            evidence=normalized_evidence,
            missing_required_evidence=missing,
            quality_score=quality_score,
            horizon_days=horizon_days,
            as_of=as_of,
        )

        basket = self.ticker_bridge.build(
            domain_id=self.profile.domain_id,
            thesis=thesis,
            evidence=normalized_evidence,
            tickers=tickers or self.profile.ticker_universe_hint,
        )

        quality_gates = build_quality_gates(
            profile=self.profile,
            evidence=normalized_evidence,
            missing_required_evidence=missing,
            evidence_quality_score=quality_score,
            ticker_basket=basket,
        )

        recommendation = self.recommendation(missing, basket.basket_status, quality_score)
        return AnalystReport(
            agent_name=self.agent_name,
            domain_id=self.profile.domain_id,
            as_of=as_of,
            horizon_days=horizon_days,
            domain_profile_version=self.profile.version,
            thesis=thesis,
            ticker_basket=basket,
            evidence=normalized_evidence,
            quality_gates=quality_gates,
            review_packet=build_review_packet(self.profile, thesis, basket),
            outcome_tracking_plan=build_outcome_tracking_plan(self.profile, thesis, basket),
            recommendation=recommendation,
            review_required=True,
            live_execution_allowed=False,
        )

    def normalize_evidence(self, evidence: Iterable[AnalystEvidenceItem | dict], as_of: str) -> list[AnalystEvidenceItem]:
        items: list[AnalystEvidenceItem] = []
        for raw in evidence:
            if isinstance(raw, AnalystEvidenceItem):
                item = raw
            else:
                payload = dict(raw)
                payload.setdefault("as_of", as_of)
                payload.setdefault("domain_id", self.profile.domain_id)
                item = AnalystEvidenceItem(**payload)

            if item.domain_id != self.profile.domain_id:
                raise ValueError(f"Evidence domain mismatch: {item.domain_id} != {self.profile.domain_id}")
            items.append(item)
        return items

    def build_domain_thesis(
        self,
        evidence: list[AnalystEvidenceItem],
        missing_required_evidence: list[str],
        quality_score: float,
        horizon_days: int,
        as_of: str,
    ) -> DomainThesis:
        if missing_required_evidence:
            stance = "insufficient_data"
            expected_direction = "neutral"
            thesis_text = (
                f"{self.profile.display_name} thesis is blocked by missing required evidence: "
                + ", ".join(missing_required_evidence)
            )
            data_quality = "weak"
            confidence = 0.0
        else:
            stance, expected_direction = stance_from_evidence(evidence)
            thesis_text = self._compose_thesis_text(stance, evidence)
            data_quality = "strong" if quality_score >= 0.75 else "medium" if quality_score >= 0.45 else "weak"
            confidence = quality_score

        supporting = [item.evidence_id for item in evidence if item.stance_hint == "positive"]
        contradicting = [item.evidence_id for item in evidence if item.stance_hint == "negative"]

        return DomainThesis(
            domain_id=self.profile.domain_id,
            as_of=as_of,
            horizon_days=horizon_days,
            stance=stance,
            expected_direction=expected_direction,
            confidence=confidence,
            thesis=thesis_text,
            key_drivers=self._ranked_evidence_summaries(evidence),
            supporting_evidence_ids=supporting,
            contradicting_evidence_ids=contradicting,
            assumptions=[
                "Evidence is interpreted as review-only domain analysis.",
                "Sector thesis is not a ticker forecast.",
            ],
            risks=self.profile.contradiction_rules[:5],
            blind_spots=missing_required_evidence,
            data_quality=data_quality,
            review_required=True,
        )

    def recommendation(self, missing_required_evidence: list[str], basket_status: str, quality_score: float) -> str:
        if missing_required_evidence:
            return "needs_more_data"
        if basket_status == "basket_ready_for_review" and quality_score >= 0.45:
            return "ready_for_review"
        if basket_status == "partial_basket_ready":
            return "partial_ready_for_review"
        if basket_status == "needs_more_data":
            return "needs_more_data"
        return "blocked"

    def _compose_thesis_text(self, stance: str, evidence: list[AnalystEvidenceItem]) -> str:
        summaries = self._ranked_evidence_summaries(
            evidence,
            limit=3,
        )
        joined = "; ".join(summaries) if summaries else "No evidence summaries available."
        return f"{self.profile.display_name} stance is {stance}. Evidence highlights: {joined}"

    @staticmethod
    def _ranked_evidence_summaries(
        evidence: list[AnalystEvidenceItem],
        *,
        limit: int = 5,
    ) -> list[str]:
        ranked = sorted(
            evidence,
            key=lambda item: (
                item.provenance.get("required_lane_eligible") is True,
                item.reliability_score,
                item.strength,
                item.freshness_score,
            ),
            reverse=True,
        )
        summaries: list[str] = []
        seen: set[str] = set()
        for item in ranked:
            normalized = " ".join(item.summary.lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            summaries.append(item.summary)
            if len(summaries) >= limit:
                break
        return summaries
