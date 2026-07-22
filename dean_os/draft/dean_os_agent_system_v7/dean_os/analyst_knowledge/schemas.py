from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from dean_os.schemas import utc_now_iso
except Exception:  # pragma: no cover
    from datetime import datetime

    def utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()


KnowledgeItemType = Literal[
    "concept",
    "driver",
    "risk",
    "metric",
    "ticker",
    "pattern",
    "glossary",
    "source_note",
    "question",
    "thesis_rule",
]

KnowledgeStance = Literal["positive", "negative", "neutral", "mixed", "unknown"]

KnowledgeQuality = Literal["high", "medium", "low", "unverified"]


class KnowledgeSource(BaseModel):
    """Reference to a source behind a knowledge item.

    This may point to a local note, filing, article, report, transcript, or
    user-created research note. URLs are allowed but not fetched here.
    """

    source_id: str = Field(default_factory=lambda: f"source_{uuid4().hex}")
    title: str
    source_type: str = "note"
    reference: str | None = None
    publisher: str | None = None
    published_at: str | None = None
    event_at: str | None = None
    retrieved_at: str | None = None
    content_sha256: str | None = None
    raw_storage_path: str | None = None
    anchor: str | None = None
    allowed_uses: list[str] = Field(default_factory=lambda: ["context", "evidence", "review"])
    known_limitations: list[str] = Field(default_factory=list)
    reliability: KnowledgeQuality = "unverified"

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("KnowledgeSource.content_sha256 must be a 64-character hexadecimal SHA-256")
        return normalized

    @model_validator(mode="after")
    def normalize(self) -> KnowledgeSource:
        self.source_id = self.source_id.strip()
        self.title = self.title.strip()
        self.allowed_uses = sorted(
            {str(value).strip().lower() for value in self.allowed_uses if str(value).strip()}
        )
        self.known_limitations = sorted(
            {str(value).strip() for value in self.known_limitations if str(value).strip()}
        )
        if not self.source_id:
            raise ValueError("KnowledgeSource.source_id cannot be empty")
        if not self.title:
            raise ValueError("KnowledgeSource.title cannot be empty")
        return self


class KnowledgeItem(BaseModel):
    """Atomic piece of analyst knowledge.

    Keep items small enough to retrieve independently.
    """

    item_id: str = Field(default_factory=lambda: f"knowledge_item_{uuid4().hex}")
    domain_id: str
    item_type: KnowledgeItemType
    title: str
    body: str
    stance_hint: KnowledgeStance = "unknown"
    tags: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: int = Field(default=3, ge=1, le=5)
    updated_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize(self) -> KnowledgeItem:
        self.domain_id = self.domain_id.strip()
        self.title = self.title.strip()
        self.body = self.body.strip()
        self.tags = sorted({tag.strip().lower() for tag in self.tags if tag and tag.strip()})
        self.tickers = sorted({ticker.strip().upper() for ticker in self.tickers if ticker and ticker.strip()})
        self.sectors = sorted({sector.strip().lower() for sector in self.sectors if sector and sector.strip()})
        self.metrics = sorted({metric.strip().lower() for metric in self.metrics if metric and metric.strip()})
        if not self.domain_id:
            raise ValueError("KnowledgeItem.domain_id cannot be empty")
        if not self.title:
            raise ValueError("KnowledgeItem.title cannot be empty")
        if not self.body:
            raise ValueError("KnowledgeItem.body cannot be empty")
        return self


class KnowledgePack(BaseModel):
    """Reusable domain knowledge pack for an analyst.

    This is the first real 'feed knowledge to the analyst' layer. It is local,
    deterministic, and does not call an LLM or vector DB.
    """

    pack_id: str
    domain_id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    sources: list[KnowledgeSource] = Field(default_factory=list)
    items: list[KnowledgeItem] = Field(default_factory=list)
    safety: dict[str, bool] = Field(
        default_factory=lambda: {
            "review_only": True,
            "no_live_execution": True,
            "no_broker_access": True,
            "no_production_config_write": True,
            "no_learning_memory_write": True,
            "no_model_promotion": True,
        }
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pack(self) -> KnowledgePack:
        self.domain_id = self.domain_id.strip()
        self.tags = sorted({tag.strip().lower() for tag in self.tags if tag and tag.strip()})
        self.tickers = sorted({ticker.strip().upper() for ticker in self.tickers if ticker and ticker.strip()})
        if not self.pack_id.strip():
            raise ValueError("KnowledgePack.pack_id cannot be empty")
        if not self.domain_id:
            raise ValueError("KnowledgePack.domain_id cannot be empty")
        for item in self.items:
            if item.domain_id != self.domain_id:
                raise ValueError(f"Item {item.item_id} has domain_id={item.domain_id!r}, expected {self.domain_id!r}")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("KnowledgePack source_id values must be unique")
        known_source_ids = set(source_ids)
        for item in self.items:
            missing_source_ids = sorted(set(item.source_ids).difference(known_source_ids))
            if missing_source_ids:
                raise ValueError(
                    f"Item {item.item_id} references unknown source_ids: {', '.join(missing_source_ids)}"
                )
        for flag in [
            "no_live_execution",
            "no_broker_access",
            "no_production_config_write",
            "no_learning_memory_write",
            "no_model_promotion",
        ]:
            if self.safety.get(flag) is not True:
                raise ValueError(f"KnowledgePack safety flag {flag} must be true")
        return self


class KnowledgeQuery(BaseModel):
    query: str
    domain_id: str | None = None
    tickers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    item_types: list[KnowledgeItemType] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=50)
    as_of: str | None = None
    require_point_in_time: bool = False
    require_source_provenance: bool = False
    intended_use: str = "evidence"

    @model_validator(mode="after")
    def normalize(self) -> KnowledgeQuery:
        self.query = self.query.strip()
        self.tickers = sorted({ticker.strip().upper() for ticker in self.tickers if ticker and ticker.strip()})
        self.tags = sorted({tag.strip().lower() for tag in self.tags if tag and tag.strip()})
        if self.domain_id:
            self.domain_id = self.domain_id.strip()
        if self.as_of:
            self.as_of = self.as_of.strip()
        self.intended_use = self.intended_use.strip().lower()
        if self.require_point_in_time and not self.as_of:
            raise ValueError("KnowledgeQuery.as_of is required when require_point_in_time=true")
        if self.require_point_in_time and self.as_of:
            try:
                parsed_as_of = datetime.fromisoformat(self.as_of.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("KnowledgeQuery.as_of must be a valid ISO-8601 timestamp") from exc
            if parsed_as_of.tzinfo is None:
                raise ValueError("KnowledgeQuery.as_of must include a timezone")
        if not self.intended_use:
            raise ValueError("KnowledgeQuery.intended_use cannot be empty")
        return self


class KnowledgeRetrievalHit(BaseModel):
    item: KnowledgeItem
    score: float
    matched_terms: list[str] = Field(default_factory=list)
    match_reasons: list[str] = Field(default_factory=list)
    sources: list[KnowledgeSource] = Field(default_factory=list)
    lineage: dict[str, Any] = Field(default_factory=dict)
    point_in_time: dict[str, Any] = Field(default_factory=dict)


class KnowledgeRetrievalExclusion(BaseModel):
    item_id: str
    title: str
    reasons: list[str] = Field(default_factory=list)
    point_in_time: dict[str, Any] = Field(default_factory=dict)


class KnowledgeRetrievalResult(BaseModel):
    query: KnowledgeQuery
    hits: list[KnowledgeRetrievalHit] = Field(default_factory=list)
    exclusions: list[KnowledgeRetrievalExclusion] = Field(default_factory=list)
    audit: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    safety: dict[str, bool] = Field(
        default_factory=lambda: {
            "review_only": True,
            "no_live_execution": True,
            "no_broker_access": True,
            "no_production_config_write": True,
            "no_learning_memory_write": True,
            "no_model_promotion": True,
        }
    )
