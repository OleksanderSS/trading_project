"""
dean_os/observability/daily_run_audit_log.py

Детальний аудит-лог щоденного запуску пайплайну, що відповідає схемі
DAILY_RUN_AUDIT_LOG_SCHEMA з Codex Phase 6 (Eval/Observability Kit).
"""
from __future__ import annotations

import datetime
import json
from typing import Any

from pydantic import BaseModel, Field


class CollectorRunStats(BaseModel):
    collector_id: str
    source_count_attempted: int = 0
    source_count_succeeded: int = 0
    source_count_failed: int = 0
    documents_collected: int = 0
    duplicates_removed: int = 0
    errors: list[str] = Field(default_factory=list)


class NormalizationStats(BaseModel):
    entities_normalized: int = 0
    dates_normalized: int = 0
    units_normalized: int = 0
    low_confidence_items: int = 0


class EventExtractionStats(BaseModel):
    events_created: int = 0
    claims_created: int = 0
    numeric_claims_created: int = 0
    items_requiring_review: int = 0


class CausalPatternStats(BaseModel):
    candidate_patterns_matched: int = 0
    high_materiality_watchlist_items: int = 0
    false_positive_suspected: int = 0


class StorageUpdateFlags(BaseModel):
    document_store_updated: bool = False
    structured_fact_store_updated: bool = False
    evidence_graph_updated: bool = False
    vector_index_updated: bool = False


class SafetyCounterSnapshot(BaseModel):
    """Знімок лічильників безпеки. Всі мають дорівнювати 0."""
    buy_sell_hold_generated: int = 0
    price_target_generated: int = 0
    trade_signal_generated: int = 0
    broker_call_attempted: int = 0
    production_config_mutation_attempted: int = 0
    model_promotion_attempted: int = 0

    def is_clean(self) -> bool:
        return all(v == 0 for v in self.model_dump().values())


class ReviewQueueStats(BaseModel):
    items_created: int = 0
    high_priority_items: int = 0
    blocked_items: int = 0


class DailyRunAuditLog(BaseModel):
    """
    Повний аудит-журнал одного щоденного запуску системи.
    Сумісний зі схемою DAILY_RUN_AUDIT_LOG_SCHEMA (Codex after_385_v1).
    """
    run_id: str
    scheduled_at: str | None = None
    started_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    finished_at: str | None = None
    mode: str = "data_accumulation_and_review_only"

    source_snapshot_id: str | None = None
    collectors: list[CollectorRunStats] = Field(default_factory=list)
    normalization: NormalizationStats = Field(default_factory=NormalizationStats)
    event_extraction: EventExtractionStats = Field(default_factory=EventExtractionStats)
    causal_pattern_matching: CausalPatternStats = Field(default_factory=CausalPatternStats)
    storage_updates: StorageUpdateFlags = Field(default_factory=StorageUpdateFlags)
    safety_counters: SafetyCounterSnapshot = Field(default_factory=SafetyCounterSnapshot)
    review_queue: ReviewQueueStats = Field(default_factory=ReviewQueueStats)

    # Поля якості виходів аналітика
    quality_metrics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def start(cls, run_id: str, scheduled_at: str | None = None) -> "DailyRunAuditLog":
        return cls(run_id=run_id, scheduled_at=scheduled_at)

    def finish(self) -> None:
        self.finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def add_collector(self, stats: CollectorRunStats) -> None:
        self.collectors.append(stats)

    def populate_from_event_result(self, event_result: dict) -> None:
        """Заповнює поля з результату WorldModelEventLearningPacket."""
        inputs = event_result.get("inputs", {})
        self.event_extraction.events_created = len(event_result.get("event_records", []))
        self.event_extraction.items_requiring_review = event_result.get("review_items_count", 0)
        self.review_queue.items_created = len(event_result.get("hypotheses", []))
        self.review_queue.high_priority_items = sum(
            1 for h in event_result.get("hypotheses", [])
            if h.get("priority") in ("critical", "high")
        )

    def populate_from_gate_result(self, gate_result: dict) -> None:
        """Заповнює поля зі стану Review Gate."""
        summary = gate_result.get("summary", gate_result)
        self.review_queue.blocked_items = 1 if not summary.get("can_register_replay_tasks") else 0

    def set_quality_metric(self, name: str, value: Any) -> None:
        self.quality_metrics[name] = value

    def as_json(self) -> str:
        return self.model_dump_json(indent=2)
