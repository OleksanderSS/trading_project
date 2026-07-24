from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal[
    "clear",
    "caution",
    "blocked",
    "bullish",
    "bearish",
    "neutral",
    "undervalued",
    "overvalued",
    "needs_more_data",
]

DecisionType = Literal[
    "blocked",
    "no_trade",
    "watchlist",
    "paper_trade_only",
    "candidate_long",
    "candidate_short",
    "reduce_position",
    "exit_position",
    "needs_more_data",
]

Branch = Literal["pipeline", "analytical"]
DataQuality = Literal["strong", "partial", "weak"]
PositionBias = Literal["bullish", "bearish", "neutral", "insufficient_data"]
ContextPhase = Literal["pre_pipeline", "post_pipeline", "pre_trade"]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class EvidenceItem(BaseModel):
    source_type: Literal[
        "metric",
        "file",
        "news",
        "article",
        "book",
        "report",
        "filing",
        "transcript",
        "document",
        "research_note",
        "pattern",
        "operation",
        "memory",
        "mlflow",
        "audit_finding",
        "dataframe_check",
        "config",
        "fundamental",
        "macro",
        "sector",
        "news_event",
        "news_analysis",
        "world_state",
        "dependency_graph",
        "unknown",
        "historical_match",
        "data_loader",
        "outcome_tracker",
        "calibration",
    ]
    source: str
    key: str
    value: Any
    timestamp: str | None = None


class AgentCapabilities(BaseModel):
    can_veto: bool = False
    can_modify_pipeline: bool = False
    can_generate_trade_signal: bool = False
    can_access_network: bool = False
    can_use_llm: bool = False
    requires_human_review: bool = True
    timeout_seconds: int = 10
    error_behavior: Literal["block", "skip", "warn"] = "skip"
    proposal_only: bool = False


class MarketContext(BaseModel):
    """Runtime snapshot shared with agents.

    DataFrames, model outputs, and pipeline internals are intentionally typed as
    Any so DEAN-OS can wrap the current project without forcing a refactor.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    phase: ContextPhase = "pre_pipeline"
    as_of: str | None = None
    tickers: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    dataframes: dict[str, Any] = Field(default_factory=dict)
    returns: Any | None = None
    positions: dict[str, float] = Field(default_factory=dict)
    news: list[Any] = Field(default_factory=list)
    fundamentals: dict[str, dict[str, Any]] = Field(default_factory=dict)
    macro: dict[str, Any] = Field(default_factory=dict)
    sector_data: dict[str, Any] = Field(default_factory=dict)
    research_documents: list[ResearchDocument] = Field(default_factory=list)
    research_notes: list[ResearchNote] = Field(default_factory=list)
    nlp_results: list[FinancialNLPResult] = Field(default_factory=list)
    action_proposals: list[PipelineActionProposal] = Field(default_factory=list)
    pipeline_result: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarketRegimeSnapshot(BaseModel):
    """Normalized market regime context for agent memory and review."""

    regime: str = "UNKNOWN"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    context_tags: list[str] = Field(default_factory=list)
    source: str = "unknown"
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class BaseAgentReport(BaseModel):
    agent_name: str
    agent_version: str
    branch: Branch
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    data_quality_score: float = Field(..., ge=0.0, le=1.0)
    signal_strength: float | None = Field(None, ge=-1.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    blind_spots: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    input_hash: str | None = None
    config_hash: str | None = None
    timestamp: str = Field(default_factory=utc_now_iso)


class PipelineReport(BaseAgentReport):
    branch: Literal["pipeline"] = "pipeline"
    metrics_snapshot: dict[str, Any] = Field(default_factory=dict)
    risk_context: dict[str, Any] | None = None

    @model_validator(mode="after")
    def blocked_verdict_needs_evidence(self) -> PipelineReport:
        if self.verdict == "blocked" and not self.evidence:
            raise ValueError("Blocked verdict requires at least one evidence item")
        return self


class AnalyticalReport(BaseAgentReport):
    branch: Literal["analytical"] = "analytical"
    ticker: str | None = None
    asset_or_sector: str | None = None
    horizon_years: float = Field(default=1.0, ge=0.0)
    thesis: str = ""
    data_quality: DataQuality = "weak"
    position_bias: PositionBias = "insufficient_data"
    catalysts: list[str] = Field(default_factory=list)
    tailwinds: list[str] = Field(default_factory=list)
    headwinds: list[str] = Field(default_factory=list)
    valuation_gap: str | None = None
    watchlist_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ConsensusDecision(BaseModel):
    decision_id: str
    decision: DecisionType
    requires_human_approval: bool = True
    final_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    blocking_agents: list[str] = Field(default_factory=list)
    supporting_agents: list[str] = Field(default_factory=list)
    opposing_agents: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    blind_spots: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    risk_context: dict[str, Any] | None = None
    world_state: dict[str, Any] | None = None
    agent_report_hashes: dict[str, str] = Field(default_factory=dict)
    # Count of reports that actually influenced final_score (decision_influence
    # is not False), unlike len(agent_report_hashes) which includes every
    # report that ran -- review-only domain analysts always run and would
    # otherwise mask a real "too few decision-relevant agents responded" state.
    decision_influencing_agent_count: int = 0
    config_hash: str = ""
    narrative: str = ""
    timestamp: str = Field(default_factory=utc_now_iso)
    # Anxiety Kill-Switch fields
    anxiety_kill_switch_triggered: bool = False
    kill_switch_reasons: list[str] = Field(default_factory=list)

    @property
    def trade_allowed(self) -> bool:
        return self.decision in {"candidate_long", "candidate_short", "paper_trade_only"}


class ExecutionOutcome(BaseModel):
    status: Literal[
        "blocked",
        "paper_trade_preview",
        "paper_trade_logged",
        "queued_for_review",
        "blocked_no_adapter",
        "executed",
    ]
    decision_id: str
    decision: DecisionType
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utc_now_iso)


class SourceCitation(BaseModel):
    citation_id: str = Field(default_factory=lambda: uuid4().hex)
    source_id: str
    source_type: Literal["news", "article", "book", "report", "filing", "transcript", "metric", "note"]
    title: str
    locator: str | None = None
    uri: str | None = None
    excerpt: str | None = None
    timestamp: str | None = None


class ResearchDocument(BaseModel):
    document_id: str = Field(default_factory=lambda: uuid4().hex)
    title: str
    source_type: Literal["news", "article", "book", "report", "filing", "transcript"]
    text: str
    uri: str | None = None
    authors: list[str] = Field(default_factory=list)
    published_at: str | None = None
    tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    quarantine_flags: list[str] = Field(default_factory=list)
    quality_precheck: str | None = None
    ingested_at: str = Field(default_factory=utc_now_iso)


class ResearchChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    chunk_index: int
    text: str
    token_estimate: int = 0
    citations: list[SourceCitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    quarantine_flags: list[str] = Field(default_factory=list)
    quality_precheck: str | None = None


class ResearchNote(BaseModel):
    note_id: str = Field(default_factory=lambda: uuid4().hex)
    agent_name: str
    topic: str
    thesis: str
    patterns: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    tailwinds: list[str] = Field(default_factory=list)
    headwinds: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    horizon_days: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data_quality: DataQuality = "weak"
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    blind_spots: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class FinancialNLPResult(BaseModel):
    result_id: str = Field(default_factory=lambda: uuid4().hex)
    agent_name: str
    document_id: str
    title: str
    tone: Literal["positive", "negative", "neutral", "mixed"]
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    event_types: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    extracted_events: list[ExtractedEvent] = Field(default_factory=list)
    summary: str = ""
    citations: list[SourceCitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class AgentLearningRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: uuid4().hex)
    agent_name: str
    note_id: str
    expected_direction: Literal["bullish", "bearish", "neutral"]
    horizon_days: int
    created_at: str = Field(default_factory=utc_now_iso)
    outcome_at: str | None = None
    realized_return: float | None = None
    outcome_label: Literal["hit", "miss", "inconclusive"] | None = None
    calibration_delta: float | None = None
    lifecycle_status: Literal[
        "draft", "validated", "rejected", "superseded", "human-corrected"
    ] = "draft"
    lifecycle_updated_at: str | None = None
    lifecycle_actor: str | None = None
    lifecycle_reason: str = ""
    supersedes_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperTradeRecord(BaseModel):
    trade_id: str = Field(default_factory=lambda: uuid4().hex)
    source_type: Literal["chief_review", "consensus_decision", "operation_proposal", "manual"] = "manual"
    source_id: str = ""
    agent_name: str = "chief_review"
    action: Literal["watchlist", "paper_trade_only", "candidate_long", "candidate_short", "no_trade"]
    tickers: list[str] = Field(default_factory=list)
    expected_direction: Literal["bullish", "bearish", "neutral"]
    horizon_days: int = Field(default=30, ge=1)
    thesis: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    context_tags: list[str] = Field(default_factory=list)
    regime_tags: list[str] = Field(default_factory=list)
    status: Literal["pending", "evaluated", "voided"] = "pending"
    created_at: str = Field(default_factory=utc_now_iso)
    outcome_at: str | None = None
    realized_return: float | None = None
    outcome_label: Literal["hit", "miss", "inconclusive"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineActionProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: uuid4().hex)
    agent_name: str
    action_type: Literal["parse", "enrich", "accumulate", "validate", "run_stage", "train", "tune", "report"]
    target: str
    reason: str
    status: Literal["proposed", "approved", "rejected", "expired", "executed"] = "proposed"
    dry_run: bool = True
    requires_human_approval: bool = True
    command_preview: str | None = None
    expected_effect: str | None = None
    risks: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class ReviewActionRecord(BaseModel):
    action_id: str = Field(default_factory=lambda: uuid4().hex)
    source_type: Literal["agent_lab_report", "learning_record", "operation_proposal", "review_snapshot"]
    source_id: str
    action_type: Literal["mark_reviewed", "needs_more_data", "promote_to_watchlist_proposal"]
    status: Literal["recorded", "queued", "completed", "voided"] = "recorded"
    reviewer: str = "human"
    notes: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    linked_proposal_id: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class RecommendationMemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: uuid4().hex)
    source_type: Literal["agent_lab_report", "learning_record", "operation_proposal", "manual_case"]
    source_id: str
    agent_name: str
    topic: str
    thesis: str
    context_tags: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    expected_direction: Literal["bullish", "bearish", "neutral"]
    outcome_label: Literal["hit", "miss", "inconclusive", "pending"] = "pending"
    realized_return: float | None = None
    lesson: str = ""
    confidence_before: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_after: float | None = Field(default=None, ge=0.0, le=1.0)
    lifecycle_status: Literal[
        "draft", "validated", "rejected", "superseded", "human-corrected"
    ] = "draft"
    lifecycle_updated_at: str | None = None
    lifecycle_actor: str | None = None
    lifecycle_reason: str = ""
    supersedes_id: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    outcome_at: str | None = None


class AgentLabRunReport(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    corpus_path: str
    document_count: int = 0
    chunk_count: int = 0
    note_count: int = 0
    reports: list[BaseAgentReport] = Field(default_factory=list)
    research_notes: list[ResearchNote] = Field(default_factory=list)
    learning_records: list[AgentLearningRecord] = Field(default_factory=list)
    action_proposals: list[PipelineActionProposal] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class ApprovalReceipt(BaseModel):
    """Receipt for approving a transition (learning/config/paper/apply).

    Provides audit trail for authority boundaries and ensures that
    sensitive transitions require explicit human approval with evidence.
    """
    receipt_id: str = Field(default_factory=lambda: uuid4().hex)
    transition_type: Literal["learning", "config", "paper", "apply", "operation"]
    source_id: str
    source_type: Literal["agent_lab_report", "learning_record", "operation_proposal", "config_change", "paper_trade"]
    reviewer: str = "human"
    reviewer_role: Literal["human", "chief_review_agent", "system"] = "human"
    approved: bool = True
    reason: str = ""
    evidence_ref: str | None = None
    conditions: list[str] = Field(default_factory=list)
    expires_at: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def approval_requires_reason(self) -> ApprovalReceipt:
        if self.approved and not self.reason:
            raise ValueError("Approval requires a reason")
        return self


class ExtractedFact(BaseModel):
    """A strictly typed fact extracted from a ResearchChunk. Replaces generic NLP sentiment."""
    fact_id: str = Field(default_factory=lambda: uuid4().hex)
    chunk_id: str
    fact_type: Literal["claim", "entity_resolution"]
    description: str
    is_trading_signal: bool = False  # Always False from extraction
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_citation: SourceCitation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_at: str = Field(default_factory=utc_now_iso)


class ExtractedEvent(BaseModel):
    """A strictly typed event extracted from a ResearchChunk."""
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    chunk_id: str
    event_type: str
    description: str
    date: str | None = None
    is_trading_signal: bool = False  # Always False from extraction
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_citation: SourceCitation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_at: str = Field(default_factory=utc_now_iso)


class ExtractedFundamentalMetric(BaseModel):
    """Normalized financial statement value or ratio for Graham/Buffett agents."""
    metric_id: str = Field(default_factory=lambda: uuid4().hex)
    ticker: str
    metric_name: str
    value: float
    unit: Literal["USD", "ratio", "percent", "shares"]
    period: str
    is_trading_signal: bool = False  # Always False
    source_citation: SourceCitation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_at: str = Field(default_factory=utc_now_iso)
