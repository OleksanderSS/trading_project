from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dean_os.learning import LearningStore
from dean_os.recommendation_memory import RecommendationMemoryStore
from dean_os.regime_context import normalize_context_tags
from dean_os.utils import clamp


COMPLETED_OUTCOMES = {"hit", "miss", "inconclusive"}
REGIME_CONTEXT_TAGS = {
    "calm_market",
    "rising_market",
    "falling_market",
    "crisis",
    "volatility_spike",
    "range_bound",
    "momentum",
    "breakout",
    "mean_reversion",
    "volume_expansion",
}


class AgentPerformanceByContext:
    """Aggregates agent outcomes by market/theme context."""

    def __init__(
        self,
        learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
        memory_path: str | Path = "data/dean_os/recommendation_memory.sqlite",
    ):
        self.learning_path = Path(learning_path)
        self.memory_path = Path(memory_path)

    def build_summary(
        self,
        agent_name: str | None = None,
        context_tag: str | None = None,
        min_completed: int = 1,
        limit: int = 10,
    ) -> dict[str, Any]:
        records = self._records()
        normalized_context_tag = normalize_context_tags([context_tag])[0] if context_tag else None
        if agent_name:
            records = [record for record in records if record["agent_name"] == agent_name]
        if normalized_context_tag:
            records = [
                record
                for record in records
                if normalized_context_tag in record["context_tags"] or normalized_context_tag in record["regime_tags"]
            ]

        by_agent = self._bucket_by(records, ("agent_name",), limit=limit)
        by_context_tag = self._bucket_by_tags(records, "context_tags", "context_tag", limit=limit)
        by_regime_tag = self._bucket_by_tags(records, "regime_tags", "regime_tag", limit=limit, empty_tag="no_regime")
        by_agent_context = self._bucket_by_agent_and_tags(
            records,
            "context_tags",
            "context_tag",
            limit=limit,
        )
        by_agent_regime = self._bucket_by_agent_and_tags(
            records,
            "regime_tags",
            "regime_tag",
            limit=limit,
            empty_tag="no_regime",
        )

        weak_contexts = self._weak_contexts([*by_agent_context, *by_agent_regime], min_completed=min_completed, limit=limit)
        strengths = self._strengths([*by_agent_context, *by_agent_regime], min_completed=min_completed, limit=limit)

        return {
            "query": {
                "agent_name": agent_name,
                "context_tag": normalized_context_tag,
                "min_completed": min_completed,
                "limit": limit,
            },
            "overall": self._bucket(records, {}),
            "records_by_source": dict(sorted(Counter(record["source"] for record in records).items())),
            "by_agent": by_agent,
            "by_context_tag": by_context_tag,
            "by_regime_tag": by_regime_tag,
            "by_agent_context": by_agent_context,
            "by_agent_regime": by_agent_regime,
            "weak_contexts": weak_contexts,
            "strengths": strengths,
            "recent_miss_lessons": self._recent_miss_lessons(records, limit=limit),
            "recommendations": self._recommendations(records, weak_contexts, strengths),
        }

    def _records(self) -> list[dict[str, Any]]:
        return [*self._learning_records(), *self._memory_records()]

    def _learning_records(self) -> list[dict[str, Any]]:
        records = []
        for record in LearningStore(self.learning_path).list_records():
            metadata = record.metadata or {}
            context_tags = normalize_context_tags(metadata.get("context_tags", []))
            regime_tags = normalize_context_tags(metadata.get("regime_tags", []))
            inferred_regime_tags = [tag for tag in context_tags if tag in REGIME_CONTEXT_TAGS]
            regime_tags = normalize_context_tags([*regime_tags, *inferred_regime_tags])
            theme_tags = [tag for tag in context_tags if tag not in REGIME_CONTEXT_TAGS]
            records.append(
                {
                    "source": "learning_record",
                    "record_id": record.record_id,
                    "agent_name": record.agent_name,
                    "topic": metadata.get("topic", ""),
                    "expected_direction": record.expected_direction,
                    "outcome_label": record.outcome_label or "pending",
                    "realized_return": record.realized_return,
                    "context_tags": normalize_context_tags([*theme_tags, *regime_tags]),
                    "theme_tags": theme_tags,
                    "regime_tags": regime_tags,
                    "tickers": metadata.get("tickers", []),
                    "sectors": metadata.get("sectors", []),
                    "patterns": metadata.get("patterns", []),
                    "lesson": metadata.get("lesson", ""),
                    "created_at": record.created_at,
                    "outcome_at": record.outcome_at,
                }
            )
        return records

    def _memory_records(self) -> list[dict[str, Any]]:
        records = []
        for record in RecommendationMemoryStore(self.memory_path).list_records():
            context_tags = normalize_context_tags(record.context_tags)
            regime_tags = [tag for tag in context_tags if tag in REGIME_CONTEXT_TAGS]
            theme_tags = [tag for tag in context_tags if tag not in REGIME_CONTEXT_TAGS]
            records.append(
                {
                    "source": "recommendation_memory",
                    "record_id": record.memory_id,
                    "agent_name": record.agent_name,
                    "topic": record.topic,
                    "expected_direction": record.expected_direction,
                    "outcome_label": record.outcome_label,
                    "realized_return": record.realized_return,
                    "context_tags": context_tags,
                    "theme_tags": theme_tags,
                    "regime_tags": regime_tags,
                    "tickers": record.tickers,
                    "sectors": record.sectors,
                    "patterns": [],
                    "lesson": record.lesson,
                    "created_at": record.created_at,
                    "outcome_at": record.outcome_at,
                }
            )
        return records

    def _bucket_by(self, records: list[dict[str, Any]], keys: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[tuple(record.get(key) for key in keys)].append(record)
        buckets = []
        for key_values, group in grouped.items():
            label = dict(zip(keys, key_values, strict=False))
            buckets.append(self._bucket(group, label))
        return self._sort_buckets(buckets)[:limit]

    def _bucket_by_tags(
        self,
        records: list[dict[str, Any]],
        tags_field: str,
        label_name: str,
        limit: int,
        empty_tag: str = "untagged",
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            tags = record.get(tags_field) or [empty_tag]
            for tag in tags:
                grouped[tag].append(record)
        buckets = [self._bucket(group, {label_name: tag}) for tag, group in grouped.items()]
        return self._sort_buckets(buckets)[:limit]

    def _bucket_by_agent_and_tags(
        self,
        records: list[dict[str, Any]],
        tags_field: str,
        label_name: str,
        limit: int,
        empty_tag: str = "untagged",
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            tags = record.get(tags_field) or [empty_tag]
            for tag in tags:
                grouped[(record["agent_name"], tag)].append(record)
        buckets = [
            self._bucket(group, {"agent_name": agent_name, label_name: tag})
            for (agent_name, tag), group in grouped.items()
        ]
        return self._sort_buckets(buckets)[:limit]

    def _bucket(self, records: list[dict[str, Any]], label: dict[str, Any]) -> dict[str, Any]:
        completed = [record for record in records if record["outcome_label"] in COMPLETED_OUTCOMES]
        hits = sum(1 for record in completed if record["outcome_label"] == "hit")
        misses = sum(1 for record in completed if record["outcome_label"] == "miss")
        inconclusive = sum(1 for record in completed if record["outcome_label"] == "inconclusive")
        pending = len(records) - len(completed)
        completed_count = len(completed)
        hit_rate = hits / completed_count if completed_count else None
        miss_rate = misses / completed_count if completed_count else None
        return {
            **label,
            "record_count": len(records),
            "completed_count": completed_count,
            "pending_count": pending,
            "hit_count": hits,
            "miss_count": misses,
            "inconclusive_count": inconclusive,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "suggested_weight": _suggested_weight(hit_rate, miss_rate),
            "suggested_attention": _suggested_attention(hits, misses, miss_rate),
            "latest_lessons": [
                record["lesson"]
                for record in records[-3:]
                if record["outcome_label"] == "miss" and record.get("lesson")
            ],
        }

    def _sort_buckets(self, buckets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            buckets,
            key=lambda bucket: (
                bucket["completed_count"],
                bucket["record_count"],
                bucket["miss_count"],
                bucket["hit_count"],
            ),
            reverse=True,
        )

    def _weak_contexts(
        self,
        buckets: list[dict[str, Any]],
        min_completed: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        weak = [
            bucket
            for bucket in buckets
            if bucket["completed_count"] >= min_completed and bucket["miss_count"] > bucket["hit_count"]
        ]
        weak.sort(key=lambda bucket: (bucket["miss_rate"] or 0.0, bucket["miss_count"]), reverse=True)
        return weak[:limit]

    def _strengths(
        self,
        buckets: list[dict[str, Any]],
        min_completed: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        strong = [
            bucket
            for bucket in buckets
            if bucket["completed_count"] >= min_completed and bucket["hit_count"] > bucket["miss_count"]
        ]
        strong.sort(key=lambda bucket: (bucket["hit_rate"] or 0.0, bucket["hit_count"]), reverse=True)
        return strong[:limit]

    def _recent_miss_lessons(self, records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        lessons = [
            {
                "source": record["source"],
                "record_id": record["record_id"],
                "agent_name": record["agent_name"],
                "topic": record["topic"],
                "context_tags": record["context_tags"],
                "lesson": record["lesson"],
            }
            for record in records
            if record["outcome_label"] == "miss" and record.get("lesson")
        ]
        return lessons[-limit:]

    def _recommendations(
        self,
        records: list[dict[str, Any]],
        weak_contexts: list[dict[str, Any]],
        strengths: list[dict[str, Any]],
    ) -> list[str]:
        if not records:
            return ["Add recommendation memory cases or completed learning outcomes before calibrating agents by context."]
        completed = [record for record in records if record["outcome_label"] in COMPLETED_OUTCOMES]
        recommendations = []
        if not completed:
            recommendations.append("Current records are pending only; keep collecting outcomes before changing agent weights.")
        if not any(record["regime_tags"] for record in records):
            recommendations.append("Add regime tags such as calm_market, rising_market, crisis, or volatility_spike to future cases.")
        if weak_contexts:
            recommendations.append("Use weak_contexts as review guardrails; require extra evidence before accepting similar theses.")
        if strengths:
            recommendations.append("Use strengths as candidates for higher review confidence, not automatic trade approval.")
        if not recommendations:
            recommendations.append("No context-specific weakness detected yet; continue collecting completed outcomes.")
        return recommendations


def _suggested_weight(hit_rate: float | None, miss_rate: float | None) -> float:
    if hit_rate is None or miss_rate is None:
        return 0.5
    return clamp(0.35 + hit_rate * 0.55 - miss_rate * 0.25, 0.1, 1.0)


def _suggested_attention(hits: int, misses: int, miss_rate: float | None) -> float:
    if miss_rate is None:
        return 0.5
    return clamp(0.5 + misses * 0.12 + miss_rate * 0.25 - hits * 0.04, 0.1, 1.0)
