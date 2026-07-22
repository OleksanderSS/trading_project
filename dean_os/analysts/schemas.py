from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

Stance = Literal["constructive", "risk_heavy", "neutral", "mixed", "insufficient_data"]
ExpectedDirection = Literal["positive", "negative", "neutral", "mixed"]
EvidenceDirectness = Literal["domain", "sector", "ticker", "macro", "policy", "market", "geopolitical"]
EvidenceStanceHint = Literal["positive", "negative", "neutral", "mixed", "unknown"]
DataQuality = Literal["strong", "medium", "weak"]
CandidateStatus = Literal["direct_ticker_thesis", "basket_candidate", "blocked_missing_evidence"]
BasketStatus = Literal["basket_ready_for_review", "partial_basket_ready", "basket_blocked", "needs_more_data"]
Recommendation = Literal["ready_for_review", "partial_ready_for_review", "needs_more_data", "blocked"]


def _default_source_registry_policy() -> dict[str, Any]:
    return {
        "policy_id": "default_domain_source_registry_policy_v1",
        "trust_tiers": {
            "tier_1_core_evidence": "Official filings, audited reports, regulators, official statistics, and recognized public data providers.",
            "tier_2_strong_context": "Company disclosures, earnings transcripts, industry-body reports, and methodology-backed research.",
            "tier_3_event_context": "Reputable financial news, trade press, and official announcements for event detection and context.",
            "tier_4_weak_or_unverified": "Blogs, social posts, reposted commentary, unsourced tables, and low-attribution material.",
        },
        "minimum_source_rules": {
            "numeric_claim": {
                "min_trust_tier": "tier_2_strong_context",
                "require_unit": True,
                "require_period": True,
                "require_source_anchor": True,
            },
            "material_event_claim": {
                "min_sources": 2,
                "allow_single_source_if": ["company_press_release", "regulator_announcement", "official_government_notice"],
            },
            "final_domain_conclusion": {
                "min_trust_tier": "tier_2_strong_context",
                "weak_source_allowed": False,
            },
        },
        "weak_source_policy": "Weak or unverified sources can create lead-generation or evidence-gap tasks only.",
    }


def _default_ingestion_filter_policy() -> dict[str, Any]:
    return {
        "policy_id": "default_domain_ingestion_filter_policy_v1",
        "required_metadata": [
            "source_id",
            "source_type",
            "source_hash_or_stable_id",
            "publication_date_or_period",
            "as_of",
        ],
        "date_fields_to_distinguish": [
            "publication_date",
            "period_covered",
            "filing_date",
            "event_date",
            "ingestion_date",
        ],
        "fail_closed_rules": [
            "silently_accept_missing_period",
            "silently_accept_missing_unit_for_numeric_claim",
            "treat_company_presentation_as_audited_source",
            "overwrite_existing_source_without_versioning",
            "use_live_fetch_without_explicit_permission",
            "future_data_detected_in_as_of_analysis",
        ],
        "table_numeric_rules": {
            "require_header_confidence_min": 0.80,
            "require_numeric_parse_confidence_min": 0.85,
            "require_unit_detection_for_numeric_tables": True,
        },
    }


def _default_evidence_scoring_policy() -> dict[str, Any]:
    return {
        "policy_id": "default_domain_evidence_scoring_policy_v1",
        "score_range": {"min": 0.0, "max": 1.0},
        "weights": {
            "source_trust": 0.30,
            "directness": 0.20,
            "period_match": 0.12,
            "entity_match": 0.10,
            "numeric_or_table_quality": 0.10,
            "corroboration": 0.08,
            "contradiction_status": 0.06,
            "recency_when_relevant": 0.04,
        },
        "minimum_use_thresholds": {
            "final_numeric_claim": 0.75,
            "final_materiality_assessment": 0.70,
            "context_only": 0.45,
            "inquiry_trigger": 0.30,
        },
        "fail_closed_rules": [
            "numeric_claim_missing_unit",
            "numeric_claim_missing_period",
            "weak_source_only_material_conclusion",
            "table_confidence_below_threshold",
            "source_conflict_unresolved",
        ],
    }


def _default_review_output_policy() -> dict[str, Any]:
    return {
        "policy_id": "default_domain_review_output_policy_v1",
        "allowed_review_outputs": [
            "analyst_summary",
            "event_interpretation",
            "mechanism_hypothesis",
            "value_chain_mapping",
            "watch_metric_request",
            "contradiction_review",
            "data_quality_note",
            "review_queue_item",
            "research_recommendation",
            "scenario_priority",
            "evidence_request",
            "causal_postmortem",
            "self_improvement_proposal",
        ],
        "blocked_outputs": [
            "buy_sell_hold",
            "position_sizing",
            "portfolio_allocation",
            "order_creation",
            "broker_routing",
            "paper_trade",
            "live_trade",
            "production_config_write",
            "learning_memory_write_without_review",
        ],
        "recommendation_boundary": "Research recommendations are review-only; execution and investment recommendations are separate gated lifecycles.",
    }


def _default_feedback_label_policy() -> dict[str, Any]:
    return {
        "policy_id": "default_domain_feedback_label_policy_v1",
        "review_types": ["evidence", "extraction", "reasoning", "safety", "style", "routing"],
        "severity_labels": ["low", "medium", "high", "blocker"],
        "issue_types": [
            "wrong_source",
            "missing_source",
            "weak_source_overweighted",
            "wrong_unit",
            "wrong_period",
            "entity_mismatch",
            "time_leakage",
            "unsupported_inference",
            "forbidden_execution_recommendation",
            "bad_routing",
            "unclear_output",
        ],
        "learning_update_flags": [
            "source_registry",
            "ingestion_filter",
            "evidence_scoring",
            "eval_pack",
            "fine_tuning_dataset",
        ],
        "promotion_rule": "Feedback can become a learning candidate only after human review and balanced outcome checks.",
    }


class DomainProfile(BaseModel):
    """Configuration-only profile for one economic domain."""

    domain_id: str
    display_name: str
    description: str
    horizon_days_default: int = 180
    allowed_horizons: list[int] = Field(default_factory=lambda: [30, 90, 180, 365])
    core_questions: list[str] = Field(default_factory=list)
    required_evidence_types: list[str] = Field(default_factory=list)
    useful_evidence_types: list[str] = Field(default_factory=list)
    sector_keywords: list[str] = Field(default_factory=list)
    ticker_universe_hint: list[str] = Field(default_factory=list)
    contradiction_rules: list[str] = Field(default_factory=list)
    direct_ticker_evidence_rules: list[str] = Field(default_factory=list)
    blocked_if_missing: list[str] = Field(default_factory=list)
    sector_label: str | None = None
    macro_evidence_type: str | None = None
    trusted_sources: dict[str, list[str]] = Field(default_factory=dict)
    source_registry_policy: dict[str, Any] = Field(default_factory=_default_source_registry_policy)
    ingestion_filter_policy: dict[str, Any] = Field(default_factory=_default_ingestion_filter_policy)
    evidence_scoring_policy: dict[str, Any] = Field(default_factory=_default_evidence_scoring_policy)
    review_output_policy: dict[str, Any] = Field(default_factory=_default_review_output_policy)
    feedback_label_policy: dict[str, Any] = Field(default_factory=_default_feedback_label_policy)
    analyst_lifecycle_profile: dict[str, Any] = Field(default_factory=dict)
    version: str = "0.1.0"

    @field_validator("domain_id")
    @classmethod
    def _domain_id_not_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("domain_id cannot be empty")
        return normalized

    @field_validator("allowed_horizons")
    @classmethod
    def _allowed_horizons_not_empty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("allowed_horizons cannot be empty")
        if any(item <= 0 for item in value):
            raise ValueError("allowed_horizons must be positive")
        return sorted(set(value))

    @field_validator("horizon_days_default")
    @classmethod
    def _default_horizon_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("horizon_days_default must be positive")
        return value


class AnalystEvidenceItem(BaseModel):
    """One normalized evidence item used by a domain analyst."""

    evidence_id: str = Field(default_factory=lambda: f"evidence_{uuid4().hex}")
    source_type: str
    source: str
    published_at: str | None = None
    as_of: str
    domain_id: str
    tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    evidence_type: str
    summary: str
    stance_hint: EvidenceStanceHint = "unknown"
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    directness: EvidenceDirectness
    reliability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
    blocked_windows: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    point_in_time: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, value: list[str]) -> list[str]:
        return sorted({str(item).upper().strip() for item in value if str(item).strip()})


class DomainThesis(BaseModel):
    """Broad domain/sector thesis. This is not a ticker forecast."""

    thesis_id: str = Field(default_factory=lambda: f"thesis_{uuid4().hex}")
    domain_id: str
    as_of: str
    horizon_days: int
    stance: Stance
    expected_direction: ExpectedDirection
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    key_drivers: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    blind_spots: list[str] = Field(default_factory=list)
    data_quality: DataQuality = "weak"
    review_required: bool = True


class TickerCandidateThesis(BaseModel):
    """Ticker-level candidate created from a domain thesis with guardrails."""

    ticker: str
    domain_id: str
    source_thesis_id: str
    candidate_status: CandidateStatus
    expected_direction: ExpectedDirection
    confidence: float = Field(ge=0.0, le=1.0)
    ticker_specific_evidence_ids: list[str] = Field(default_factory=list)
    sector_only_evidence_ids: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    required_missing_evidence: list[str] = Field(default_factory=list)
    blocked_windows: list[str] = Field(default_factory=list)
    review_required: bool = True

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        normalized = str(value).upper().strip()
        if not normalized:
            raise ValueError("ticker cannot be empty")
        return normalized


class TickerBasketReport(BaseModel):
    """Evidence-gated basket status for ticker candidates."""

    domain_id: str
    source_thesis_id: str
    basket_status: BasketStatus
    candidates: list[TickerCandidateThesis] = Field(default_factory=list)
    direct_ready_count: int = 0
    basket_candidate_count: int = 0
    blocked_count: int = 0
    reasons: list[str] = Field(default_factory=list)
    review_required: bool = True


class AnalystReport(BaseModel):
    """Review-only output from a domain analyst."""

    report_id: str = Field(default_factory=lambda: f"analyst_report_{uuid4().hex}")
    agent_name: str
    domain_id: str
    as_of: str
    horizon_days: int
    domain_profile_version: str
    thesis: DomainThesis
    ticker_basket: TickerBasketReport
    evidence: list[AnalystEvidenceItem] = Field(default_factory=list)
    quality_gates: dict[str, Any] = Field(default_factory=dict)
    review_packet: dict[str, Any] = Field(default_factory=dict)
    outcome_tracking_plan: dict[str, Any] = Field(default_factory=dict)
    recommendation: Recommendation
    review_required: bool = True
    live_execution_allowed: bool = False

    @field_validator("live_execution_allowed")
    @classmethod
    def _never_live_execution(cls, value: bool) -> bool:
        if value:
            raise ValueError("AnalystReport cannot authorize live execution")
        return value
