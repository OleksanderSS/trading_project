from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReviewInboxItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str = Field(default_factory=lambda: f"review_item_{uuid4().hex}")
    daily_run_id: str
    domain_id: str
    as_of: str
    item_type: Literal["daily_briefing", "evidence_gap_plan", "due_replay", "credibility_exception"]
    priority: Literal["critical", "high", "medium", "low"]
    title: str
    summary: str
    artifact_ref: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    status: Literal["pending_review", "acknowledged", "needs_more_data", "approved", "rejected"] = "pending_review"
    review_only: bool = True
    content_hash: str

    @model_validator(mode="after")
    def _validate_hash(self) -> "ReviewInboxItem":
        payload = self.model_dump(mode="json", exclude={"item_id", "content_hash"})
        if _sha256(payload) != self.content_hash:
            raise ValueError("review inbox content hash mismatch")
        if not self.review_only:
            raise ValueError("review inbox item must remain review-only")
        return self


class ReviewInboxProtocol(Protocol):
    def append(self, item: ReviewInboxItem) -> str: ...
    def list_pending(self, *, domain_id: str | None = None, limit: int = 100) -> list[ReviewInboxItem]: ...


class SQLiteOperatorReviewInbox:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS operator_review_inbox (
                    item_id TEXT PRIMARY KEY,
                    domain_id TEXT NOT NULL,
                    priority_rank INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_operator_review_pending
                    ON operator_review_inbox(status, priority_rank, domain_id);
                CREATE TRIGGER IF NOT EXISTS operator_review_inbox_no_update
                    BEFORE UPDATE ON operator_review_inbox
                    BEGIN SELECT RAISE(ABORT, 'operator_review_inbox is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS operator_review_inbox_no_delete
                    BEFORE DELETE ON operator_review_inbox
                    BEGIN SELECT RAISE(ABORT, 'operator_review_inbox is append-only'); END;
            """)

    def append(self, item: ReviewInboxItem) -> str:
        payload = item.model_dump(mode="json")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT item_id, payload_json FROM operator_review_inbox WHERE item_id = ? OR content_hash = ?",
                (item.item_id, item.content_hash),
            ).fetchone()
            if existing:
                if json.loads(existing["payload_json"]) != payload:
                    raise ValueError("review inbox identity/content conflict")
                return "already_exists"
            connection.execute(
                "INSERT INTO operator_review_inbox(item_id, domain_id, priority_rank, status, content_hash, payload_json) VALUES(?,?,?,?,?,?)",
                (item.item_id, item.domain_id, _priority_rank(item.priority), item.status, item.content_hash, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
        return "stored"

    def list_pending(self, *, domain_id: str | None = None, limit: int = 100) -> list[ReviewInboxItem]:
        query = "SELECT payload_json FROM operator_review_inbox WHERE status = 'pending_review'"
        values: list[Any] = []
        if domain_id:
            query += " AND domain_id = ?"
            values.append(domain_id)
        query += " ORDER BY priority_rank ASC LIMIT ?"
        values.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [ReviewInboxItem.model_validate(json.loads(row["payload_json"])) for row in rows]


class OperatorReviewInboxBuilder:
    def build(self, daily_result: Any, *, evidence_gap_plan: Any) -> list[ReviewInboxItem]:
        result = daily_result.model_dump(mode="json") if hasattr(daily_result, "model_dump") else dict(daily_result)
        plan = evidence_gap_plan.model_dump(mode="json") if hasattr(evidence_gap_plan, "model_dump") else dict(evidence_gap_plan)
        items: list[ReviewInboxItem] = []
        items.append(self._item(
            result,
            item_type="daily_briefing",
            priority="high" if result.get("status") != "completed" else "medium",
            title="Review daily analytical briefing",
            summary=f"Review briefing {result.get('briefing', {}).get('briefing_id')} and scenario assumptions.",
            required_actions=["review coverage gate", "review scenario invalidation conditions", "record decision"],
        ))
        if plan.get("tasks"):
            items.append(self._item(
                result,
                item_type="evidence_gap_plan",
                priority="high",
                title="Resolve evidence gaps",
                summary=f"{len(plan.get('tasks', []))} bounded evidence-acquisition tasks require review.",
                required_actions=["approve or edit collector routes", "do not auto-execute network collectors"],
            ))
        if result.get("due_replay_tasks"):
            items.append(self._item(
                result,
                item_type="due_replay",
                priority="high",
                title="Evaluate due replay outcomes",
                summary=f"{len(result.get('due_replay_tasks', []))} replay tasks are due for outcome evidence.",
                required_actions=["attach point-in-time outcome evidence", "run outcome evaluation", "review calibration proposal"],
            ))
        weak = [record for record in result.get("evidence_records", []) or [] if float(record.get("credibility_score", 0.0)) < 0.45]
        if weak:
            items.append(self._item(
                result,
                item_type="credibility_exception",
                priority="medium",
                title="Review weak-source evidence",
                summary=f"{len(weak)} retained records are lead-only or quarantined.",
                evidence_ids=[str(record.get("evidence_id")) for record in weak],
                required_actions=["confirm source tier", "replace weak source before material conclusion"],
            ))
        return items

    def _item(self, result: dict[str, Any], *, item_type: str, priority: str, title: str, summary: str, required_actions: list[str], evidence_ids: list[str] | None = None) -> ReviewInboxItem:
        payload = {
            "daily_run_id": str(result.get("daily_run_id")),
            "domain_id": str(result.get("domain_id")),
            "as_of": str(result.get("as_of")),
            "item_type": item_type,
            "priority": priority,
            "title": title,
            "summary": summary,
            "artifact_ref": None,
            "evidence_ids": evidence_ids or [],
            "required_actions": required_actions,
            "status": "pending_review",
            "review_only": True,
        }
        return ReviewInboxItem(**payload, content_hash=_sha256(payload))


def _sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _priority_rank(priority: str) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(priority, 9)


__all__ = ["ReviewInboxItem", "ReviewInboxProtocol", "SQLiteOperatorReviewInbox", "OperatorReviewInboxBuilder"]
