from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from dean_os.schemas import AgentLearningRecord, ResearchNote
from dean_os.utils import clamp
from dean_os.schemas import utc_now_iso

Direction = Literal["bullish", "bearish", "neutral"]


class LearningStore:
    """Stores agent thesis/outcome records for later calibration."""

    def __init__(self, db_path: str | Path = "data/dean_os/agent_learning.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_record_from_note(
        self,
        note: ResearchNote,
        expected_direction: Direction,
        horizon_days: int | None = None,
        metadata: dict[str, Any] | None = None,
        lifecycle_status: str = "draft",
        lifecycle_actor: str | None = None,
        lifecycle_reason: str = "",
    ) -> AgentLearningRecord:
        record = AgentLearningRecord(
            agent_name=note.agent_name,
            note_id=note.note_id,
            expected_direction=expected_direction,
            horizon_days=horizon_days or note.horizon_days or 365,
            lifecycle_status=lifecycle_status,
            lifecycle_updated_at=utc_now_iso(),
            lifecycle_actor=lifecycle_actor,
            lifecycle_reason=lifecycle_reason,
            metadata={
                "topic": note.topic,
                "confidence": note.confidence,
                "patterns": note.patterns,
                **(metadata or {}),
            },
        )
        self.add_record(record)
        return record

    def add_record(self, record: AgentLearningRecord) -> str:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO learning_records
                (record_id, agent_name, note_id, expected_direction, horizon_days, created_at,
                 outcome_at, realized_return, outcome_label, calibration_delta, metadata, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.agent_name,
                    record.note_id,
                    record.expected_direction,
                    record.horizon_days,
                    record.created_at,
                    record.outcome_at,
                    record.realized_return,
                    record.outcome_label,
                    record.calibration_delta,
                    json.dumps(record.metadata, ensure_ascii=True),
                    json.dumps(record.model_dump(mode="json"), ensure_ascii=True),
                ),
            )
        return record.record_id

    def update_outcome(
        self,
        record_id: str,
        realized_return: float,
        outcome_at: str | None = None,
        neutral_band: float = 0.01,
    ) -> AgentLearningRecord:
        record = self.get_record(record_id)
        if record is None:
            raise KeyError(f"Learning record not found: {record_id}")
        label = classify_outcome(record.expected_direction, realized_return, neutral_band=neutral_band)
        record.realized_return = realized_return
        record.outcome_label = label
        record.outcome_at = outcome_at
        record.calibration_delta = calibration_delta(record.expected_direction, realized_return)
        self.add_record(record)
        return record

    def get_record(self, record_id: str) -> AgentLearningRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM learning_records WHERE record_id = ?", (record_id,)).fetchone()
        if row is None:
            return None
        return AgentLearningRecord(**json.loads(row["payload"]))

    def list_records(self, agent_name: str | None = None) -> list[AgentLearningRecord]:
        if agent_name:
            sql = "SELECT payload FROM learning_records WHERE agent_name = ? ORDER BY rowid"
            params: tuple[Any, ...] = (agent_name,)
        else:
            sql = "SELECT payload FROM learning_records ORDER BY rowid"
            params = ()
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [AgentLearningRecord(**json.loads(row["payload"])) for row in rows]

    def score_agent(self, agent_name: str) -> dict[str, Any]:
        raw_records = self.list_records(agent_name)
        all_records = [
            record
            for record in raw_records
            if record.lifecycle_status in {"validated", "human-corrected"}
        ]
        records = [record for record in all_records if record.outcome_label is not None]
        pending_count = len(all_records) - len(records)
        if not records:
            return {
                "agent_name": agent_name,
                "record_count": 0,
                "total_record_count": len(raw_records),
                "eligible_record_count": len(all_records),
                "lifecycle_excluded_count": len(raw_records) - len(all_records),
                "pending_record_count": pending_count,
                "hit_rate": None,
                "miss_rate": None,
                "inconclusive_rate": None,
                "suggested_weight": 0.5,
            }
        hits = sum(1 for record in records if record.outcome_label == "hit")
        misses = sum(1 for record in records if record.outcome_label == "miss")
        inconclusive = sum(1 for record in records if record.outcome_label == "inconclusive")
        count = len(records)
        hit_rate = hits / count
        miss_rate = misses / count
        inconclusive_rate = inconclusive / count
        suggested_weight = clamp(0.25 + hit_rate * 0.65 - miss_rate * 0.20, 0.1, 1.0)
        return {
            "agent_name": agent_name,
            "record_count": count,
            "total_record_count": len(raw_records),
            "eligible_record_count": len(all_records),
            "lifecycle_excluded_count": len(raw_records) - len(all_records),
            "pending_record_count": pending_count,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "inconclusive_rate": inconclusive_rate,
            "suggested_weight": suggested_weight,
        }

    def eligible_records(
        self, agent_name: str | None = None
    ) -> list[AgentLearningRecord]:
        return [
            record
            for record in self.list_records(agent_name)
            if record.lifecycle_status in {"validated", "human-corrected"}
        ]

    def transition_lifecycle(
        self,
        record_id: str,
        status: str,
        *,
        actor: str,
        reason: str,
        supersedes_id: str | None = None,
    ) -> AgentLearningRecord:
        record = self.get_record(record_id)
        if record is None:
            raise KeyError(f"Learning record not found: {record_id}")
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

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_records (
                    record_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    note_id TEXT NOT NULL,
                    expected_direction TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    outcome_at TEXT,
                    realized_return REAL,
                    outcome_label TEXT,
                    calibration_delta REAL,
                    metadata TEXT NOT NULL,
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


def direction_from_note(note: ResearchNote) -> Direction:
    bullish = len(note.tailwinds) + len([pattern for pattern in note.patterns if pattern in {"defense_rearmament", "ai_compute_cycle", "value_margin_safety", "pricing_power"}])
    bearish = len(note.headwinds) + len([pattern for pattern in note.patterns if pattern in {"regulatory_risk", "balance_sheet_stress", "capacity_pressure"}])
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "neutral"


def classify_outcome(expected_direction: Direction, realized_return: float, neutral_band: float = 0.01) -> str:
    if abs(realized_return) <= neutral_band:
        return "hit" if expected_direction == "neutral" else "inconclusive"
    if expected_direction == "bullish":
        return "hit" if realized_return > neutral_band else "miss"
    if expected_direction == "bearish":
        return "hit" if realized_return < -neutral_band else "miss"
    return "miss"


def calibration_delta(expected_direction: Direction, realized_return: float) -> float:
    if expected_direction == "bullish":
        return clamp(realized_return, -1.0, 1.0)
    if expected_direction == "bearish":
        return clamp(-realized_return, -1.0, 1.0)
    return clamp(-abs(realized_return), -1.0, 0.0)


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
