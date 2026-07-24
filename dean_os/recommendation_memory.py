from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from dean_os.schemas import RecommendationMemoryRecord, utc_now_iso
from dean_os.utils import clamp, json_ready


class RecommendationMemoryStore:
    """Stores recommendation cases, outcomes, and lessons for calibration."""

    def __init__(self, db_path: str | Path = "data/dean_os/recommendation_memory.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_record(self, record: RecommendationMemoryRecord) -> str:
        payload = json.dumps(json_ready(record), ensure_ascii=True)
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM recommendation_memory WHERE memory_id = ?",
                (record.memory_id,),
            )
            next_revision = (cur.fetchone()[0] or 0) + 1
            conn.execute(
                """
                INSERT INTO recommendation_memory
                (memory_id, revision, source_type, source_id, agent_name, topic, expected_direction,
                 outcome_label, created_at, outcome_at, context_tags, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    next_revision,
                    record.source_type,
                    record.source_id,
                    record.agent_name,
                    record.topic,
                    record.expected_direction,
                    record.outcome_label,
                    record.created_at,
                    record.outcome_at,
                    json.dumps(record.context_tags, ensure_ascii=True),
                    payload,
                ),
            )
            conn.execute(
                """
                INSERT INTO recommendation_memory_events
                (event_id, memory_id, revision, actor, reason, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    record.memory_id,
                    next_revision,
                    record.lifecycle_actor or "system",
                    record.lifecycle_reason or "memory write",
                    payload,
                    utc_now_iso(),
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
        lifecycle_status: str = "draft",
        lifecycle_actor: str | None = None,
        lifecycle_reason: str = "",
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
            lifecycle_status=lifecycle_status,
            lifecycle_updated_at=utc_now_iso(),
            lifecycle_actor=lifecycle_actor,
            lifecycle_reason=lifecycle_reason,
        )
        self.add_record(record)
        return record

    def get_record(self, memory_id: str) -> RecommendationMemoryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM recommendation_memory WHERE memory_id = ? ORDER BY revision DESC LIMIT 1",
                (memory_id,),
            ).fetchone()
        if row is None:
            return None
        return RecommendationMemoryRecord(**json.loads(row["payload"]))

    def list_records(
        self,
        agent_name: str | None = None,
        context_tag: str | None = None,
        outcome_label: str | None = None,
    ) -> list[RecommendationMemoryRecord]:
        latest_sql = """
            SELECT rm.payload
            FROM recommendation_memory rm
            INNER JOIN (
                SELECT memory_id, MAX(revision) AS max_rev
                FROM recommendation_memory
                GROUP BY memory_id
            ) latest ON rm.memory_id = latest.memory_id AND rm.revision = latest.max_rev
        """
        clauses = []
        params: list[Any] = []
        if agent_name:
            clauses.append("rm.agent_name = ?")
            params.append(agent_name)
        if outcome_label:
            clauses.append("rm.outcome_label = ?")
            params.append(outcome_label)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(latest_sql + where + " ORDER BY rm.rowid", params).fetchall()
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
        all_records = self.list_records()
        records = [
            record
            for record in all_records
            if record.lifecycle_status in {"validated", "human-corrected"}
        ]
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
            "total_record_count": len(all_records),
            "eligible_record_count": len(records),
            "lifecycle_excluded_count": len(all_records) - len(records),
            "records_by_lifecycle_status": dict(
                sorted(Counter(record.lifecycle_status for record in all_records).items())
            ),
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
        for record in self.eligible_records():
            score = 0
            score += _overlap_count(context_tags, [tag.lower() for tag in record.context_tags]) * 3
            score += _overlap_count(tickers, [ticker.upper() for ticker in record.tickers]) * 2
            score += _overlap_count(sectors, [sector.lower() for sector in record.sectors]) * 2
            if score:
                scored.append((score, record.created_at, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:limit]]

    def eligible_records(self) -> list[RecommendationMemoryRecord]:
        return [
            record
            for record in self.list_records()
            if record.lifecycle_status in {"validated", "human-corrected"}
        ]

    def transition_lifecycle(
        self,
        memory_id: str,
        status: str,
        *,
        actor: str,
        reason: str,
        supersedes_id: str | None = None,
    ) -> RecommendationMemoryRecord:
        record = self.get_record(memory_id)
        if record is None:
            raise KeyError(f"Recommendation memory record not found: {memory_id}")
        _validate_lifecycle_transition(
            record.lifecycle_status, status, actor=actor, reason=reason
        )
        record.lifecycle_status = status
        record.lifecycle_updated_at = utc_now_iso()
        record.lifecycle_actor = actor.strip()
        record.lifecycle_reason = reason.strip()
        record.supersedes_id = supersedes_id
        self.add_record(record)
        return record

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

    @staticmethod
    def _migrate_legacy_table_without_revision(conn: sqlite3.Connection) -> None:
        """One-time migration for recommendation_memory.sqlite files created
        before the memory_id/revision versioning scheme existed (a single
        row per memory_id, memory_id as PRIMARY KEY). `CREATE TABLE IF NOT
        EXISTS` below is a no-op against such a file -- the missing `revision`
        column would otherwise persist forever, breaking every query that
        reads it (e.g. RecommendationMemoryStore.list_records's ORDER BY
        rm.rowid / JOIN on MAX(revision))."""
        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recommendation_memory'"
        ).fetchone()
        if existing is None:
            return
        columns = {row[1] for row in conn.execute("PRAGMA table_info(recommendation_memory)")}
        if "revision" in columns:
            return
        conn.execute("ALTER TABLE recommendation_memory RENAME TO recommendation_memory_legacy")
        conn.execute(
            """
            CREATE TABLE recommendation_memory (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
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
        conn.execute(
            """
            INSERT INTO recommendation_memory (
                memory_id, revision, source_type, source_id, agent_name, topic,
                expected_direction, outcome_label, created_at, outcome_at,
                context_tags, payload
            )
            SELECT memory_id, 1, source_type, source_id, agent_name, topic,
                expected_direction, outcome_label, created_at, outcome_at,
                context_tags, payload
            FROM recommendation_memory_legacy
            """
        )
        conn.execute("DROP TABLE recommendation_memory_legacy")

    def _init_db(self) -> None:
        with self._connect() as conn:
            self._migrate_legacy_table_without_revision(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_memory (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recommendation_memory_id ON recommendation_memory(memory_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_memory_events (
                    event_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recommendation_memory_events_id ON recommendation_memory_events(memory_id)"
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


def _validate_lifecycle_transition(
    current: str, target: str, *, actor: str, reason: str
) -> None:
    allowed = {
        "draft": {"validated", "rejected", "superseded", "human-corrected"},
        "validated": {"rejected", "superseded", "human-corrected"},
        "rejected": {"human-corrected", "superseded"},
        "human-corrected": {"validated", "rejected", "superseded"},
        "superseded": set(),
    }
    if target not in allowed.get(current, set()):
        raise ValueError(f"Invalid memory lifecycle transition: {current} -> {target}")
    if not actor.strip() or not reason.strip():
        raise ValueError("Memory lifecycle transition requires actor and reason")
