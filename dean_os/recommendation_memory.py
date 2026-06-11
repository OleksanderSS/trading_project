from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dean_os.schemas import RecommendationMemoryRecord
from dean_os.utils import clamp, json_ready


class RecommendationMemoryStore:
    """Stores recommendation cases, outcomes, and lessons for calibration."""

    def __init__(self, db_path: str | Path = "data/dean_os/recommendation_memory.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_record(self, record: RecommendationMemoryRecord) -> str:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO recommendation_memory
                (memory_id, source_type, source_id, agent_name, topic, expected_direction,
                 outcome_label, created_at, outcome_at, context_tags, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.source_type,
                    record.source_id,
                    record.agent_name,
                    record.topic,
                    record.expected_direction,
                    record.outcome_label,
                    record.created_at,
                    record.outcome_at,
                    json.dumps(record.context_tags, ensure_ascii=True),
                    json.dumps(json_ready(record), ensure_ascii=True),
                ),
            )
        return record.memory_id

    def add_case(
        self,
        source_type: str,
        source_id: str,
        agent_name: str,
        topic: str,
        thesis: str,
        expected_direction: str,
        context_tags: list[str] | None = None,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        outcome_label: str = "pending",
        realized_return: float | None = None,
        lesson: str = "",
        confidence_before: float | None = None,
        confidence_after: float | None = None,
        outcome_at: str | None = None,
    ) -> RecommendationMemoryRecord:
        record = RecommendationMemoryRecord(
            source_type=source_type,
            source_id=source_id,
            agent_name=agent_name,
            topic=topic,
            thesis=thesis,
            context_tags=context_tags or [],
            tickers=tickers or [],
            sectors=sectors or [],
            expected_direction=expected_direction,
            outcome_label=outcome_label,
            realized_return=realized_return,
            lesson=lesson,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            outcome_at=outcome_at,
        )
        self.add_record(record)
        return record

    def get_record(self, memory_id: str) -> RecommendationMemoryRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM recommendation_memory WHERE memory_id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return RecommendationMemoryRecord(**json.loads(row["payload"]))

    def list_records(
        self,
        agent_name: str | None = None,
        context_tag: str | None = None,
        outcome_label: str | None = None,
    ) -> list[RecommendationMemoryRecord]:
        clauses = []
        params: list[Any] = []
        if agent_name:
            clauses.append("agent_name = ?")
            params.append(agent_name)
        if outcome_label:
            clauses.append("outcome_label = ?")
            params.append(outcome_label)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT payload FROM recommendation_memory{where} ORDER BY rowid", params).fetchall()
        records = [RecommendationMemoryRecord(**json.loads(row["payload"])) for row in rows]
        if context_tag:
            records = [record for record in records if context_tag in record.context_tags]
        return records

    def update_outcome(
        self,
        memory_id: str,
        outcome_label: str,
        realized_return: float | None = None,
        lesson: str | None = None,
        confidence_after: float | None = None,
        outcome_at: str | None = None,
    ) -> RecommendationMemoryRecord:
        record = self.get_record(memory_id)
        if record is None:
            raise KeyError(f"Recommendation memory record not found: {memory_id}")
        record.outcome_label = outcome_label
        record.realized_return = realized_return
        record.outcome_at = outcome_at
        if lesson is not None:
            record.lesson = lesson
        if confidence_after is not None:
            record.confidence_after = confidence_after
        self.add_record(record)
        return record

    def summary(self) -> dict[str, Any]:
        records = self.list_records()
        completed = [record for record in records if record.outcome_label in {"hit", "miss", "inconclusive"}]
        hits = sum(1 for record in completed if record.outcome_label == "hit")
        misses = sum(1 for record in completed if record.outcome_label == "miss")
        hit_rate = hits / len(completed) if completed else None

        tag_counts: dict[str, Counter] = defaultdict(Counter)
        for record in records:
            for tag in record.context_tags:
                tag_counts[tag][record.outcome_label] += 1

        tag_stats = {}
        for tag, counts in sorted(tag_counts.items()):
            completed_count = counts["hit"] + counts["miss"] + counts["inconclusive"]
            tag_stats[tag] = {
                "total": sum(counts.values()),
                "hit": counts["hit"],
                "miss": counts["miss"],
                "pending": counts["pending"],
                "hit_rate": counts["hit"] / completed_count if completed_count else None,
                "suggested_attention": clamp(0.5 + counts["miss"] * 0.15 - counts["hit"] * 0.05, 0.1, 1.0),
            }

        miss_lessons = [
            {
                "memory_id": record.memory_id,
                "agent_name": record.agent_name,
                "topic": record.topic,
                "context_tags": record.context_tags,
                "lesson": record.lesson,
            }
            for record in records
            if record.outcome_label == "miss" and record.lesson
        ]

        return {
            "record_count": len(records),
            "completed_count": len(completed),
            "pending_count": sum(1 for record in records if record.outcome_label == "pending"),
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate": hit_rate,
            "records_by_agent": dict(sorted(Counter(record.agent_name for record in records).items())),
            "records_by_outcome": dict(sorted(Counter(record.outcome_label for record in records).items())),
            "tag_stats": tag_stats,
            "recent_lessons": miss_lessons[-5:],
        }

    def relevant_records(
        self,
        context_tags: list[str] | None = None,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        limit: int = 5,
    ) -> list[RecommendationMemoryRecord]:
        context_tags = [item.lower() for item in context_tags or []]
        tickers = [item.upper() for item in tickers or []]
        sectors = [item.lower() for item in sectors or []]
        scored = []
        for record in self.list_records():
            score = 0
            score += _overlap_count(context_tags, [tag.lower() for tag in record.context_tags]) * 3
            score += _overlap_count(tickers, [ticker.upper() for ticker in record.tickers]) * 2
            score += _overlap_count(sectors, [sector.lower() for sector in record.sectors]) * 2
            if score:
                scored.append((score, record.created_at, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:limit]]

    def context_snapshot(
        self,
        context_tags: list[str] | None = None,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        records = self.relevant_records(
            context_tags=context_tags,
            tickers=tickers,
            sectors=sectors,
            limit=limit,
        )
        hits = [record for record in records if record.outcome_label == "hit"]
        misses = [record for record in records if record.outcome_label == "miss"]
        completed = [record for record in records if record.outcome_label in {"hit", "miss", "inconclusive"}]
        return {
            "query": {
                "context_tags": context_tags or [],
                "tickers": tickers or [],
                "sectors": sectors or [],
            },
            "relevant_count": len(records),
            "hit_count": len(hits),
            "miss_count": len(misses),
            "hit_rate": len(hits) / len(completed) if completed else None,
            "records": [
                {
                    "memory_id": record.memory_id,
                    "source_id": record.source_id,
                    "agent_name": record.agent_name,
                    "topic": record.topic,
                    "context_tags": record.context_tags,
                    "tickers": record.tickers,
                    "sectors": record.sectors,
                    "expected_direction": record.expected_direction,
                    "outcome_label": record.outcome_label,
                    "lesson": record.lesson,
                }
                for record in records
            ],
            "lessons": [record.lesson for record in records if record.lesson],
        }

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_memory (
                    memory_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    expected_direction TEXT NOT NULL,
                    outcome_label TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    outcome_at TEXT,
                    context_tags TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _overlap_count(left: list[str], right: list[str]) -> int:
    return len(set(left).intersection(right))
