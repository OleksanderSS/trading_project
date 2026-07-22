from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dean_os.analysts.profiles import get_domain_profile


SOURCE_TIER_SCORES: dict[str, float] = {
    "tier_1_core_evidence": 0.95,
    "tier_2_strong_context": 0.80,
    "tier_3_event_context": 0.62,
    "tier_4_weak_or_unverified": 0.30,
    "unknown": 0.40,
}


class SourceCredibilityAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_name: str
    source_type: str
    source_tier: str
    credibility_score: float = Field(ge=0.0, le=1.0)
    decision_use: Literal["core_evidence", "strong_context", "event_context", "lead_only", "quarantined"]
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SourceCredibilityRegistry:
    """Deterministic, domain-aware source credibility evaluator.

    This is deliberately not a truth oracle. It scores provenance quality and
    permitted analytical use. A high score does not make a claim true.
    """

    domain_id: str
    source_tiers: dict[str, list[str]]
    default_tier: str = "unknown"

    @classmethod
    def from_domain_profile(cls, domain_id: str) -> "SourceCredibilityRegistry":
        profile = get_domain_profile(domain_id)
        policy = dict(profile.source_registry_policy or {})
        configured = policy.get("source_tiers") or {}
        normalized: dict[str, list[str]] = {}
        for tier, values in configured.items():
            if isinstance(values, str):
                values = [values]
            if isinstance(values, (list, tuple, set)):
                normalized[str(tier)] = [_normalize_source(value) for value in values]
        return cls(
            domain_id=domain_id,
            source_tiers=normalized,
            default_tier=str(policy.get("default_tier") or "unknown"),
        )

    def assess(self, payload: dict[str, Any], *, point_in_time_status: str = "review_only") -> SourceCredibilityAssessment:
        source_name = str(payload.get("source") or payload.get("source_name") or "unknown")
        source_type = str(payload.get("source_type") or "unknown").strip().lower()
        explicit_tier = str(payload.get("source_tier") or "").strip()
        tier = explicit_tier if explicit_tier in SOURCE_TIER_SCORES else self._lookup_tier(source_name)
        base = SOURCE_TIER_SCORES.get(tier, SOURCE_TIER_SCORES["unknown"])
        reasons = [f"source tier resolved as {tier}"]
        flags: list[str] = []

        if source_type in {"filing", "dataset", "metric"} and tier == "unknown":
            base = max(base, 0.65)
            reasons.append("structured source type raises provenance floor but does not verify content")
        if payload.get("quarantine_flags"):
            base = min(base, 0.20)
            flags.extend(str(item) for item in payload.get("quarantine_flags") or [])
            reasons.append("quarantine flags cap source credibility")
        if point_in_time_status == "invalid":
            base = 0.0
            flags.append("future_evidence")
            reasons.append("evidence was unavailable at the decision cutoff")
        elif point_in_time_status == "review_only":
            base = min(base, 0.70)
            reasons.append("point-in-time status is review-only")

        score = round(max(0.0, min(1.0, base)), 4)
        if score >= 0.90:
            use = "core_evidence"
        elif score >= 0.72:
            use = "strong_context"
        elif score >= 0.50:
            use = "event_context"
        elif score > 0.0:
            use = "lead_only"
        else:
            use = "quarantined"
        return SourceCredibilityAssessment(
            source_name=source_name,
            source_type=source_type,
            source_tier=tier,
            credibility_score=score,
            decision_use=use,
            reasons=reasons,
            flags=sorted(set(flags)),
        )

    def _lookup_tier(self, source_name: str) -> str:
        normalized = _normalize_source(source_name)
        for tier, names in self.source_tiers.items():
            if normalized in names:
                return tier
        return self.default_tier if self.default_tier in SOURCE_TIER_SCORES else "unknown"


def _normalize_source(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


__all__ = ["SourceCredibilityAssessment", "SourceCredibilityRegistry", "SOURCE_TIER_SCORES"]
