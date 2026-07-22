from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from dean_os.event_log import EventLog
from dean_os.operation_queue import OperationQueue
from dean_os.schemas import EvidenceItem, PipelineActionProposal, ReviewActionRecord, utc_now_iso
from dean_os.utils import json_ready

PLACEHOLDER_SOURCE_IDS = {
    "RUN_ID_HERE",
    "RECORD_ID_HERE",
    "PROPOSAL_ID_HERE",
    "SOURCE_ID_HERE",
    "<RUN_ID>",
    "<RECORD_ID>",
    "<PROPOSAL_ID>",
    "<SOURCE_ID>",
}


def is_placeholder_source_id(source_id: str) -> bool:
    normalized = source_id.strip().upper()
    return normalized in PLACEHOLDER_SOURCE_IDS or normalized.endswith("_HERE") or normalized.startswith("<")


def validate_review_source(
    source_type: str,
    source_id: str,
    reports_dir: str | Path = "reports/dean_os/agent_lab",
    learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
    operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
) -> None:
    if is_placeholder_source_id(source_id):
        raise ValueError(f"Placeholder source_id is not allowed: {source_id}")
    if source_type == "agent_lab_report":
        report_path = Path(reports_dir) / f"{source_id}.json"
        if not report_path.exists():
            raise ValueError(f"Agent Lab report not found for source_id: {source_id}")
    elif source_type == "learning_record":
        from dean_os.learning import LearningStore

        if LearningStore(learning_path).get_record(source_id) is None:
            raise ValueError(f"Learning record not found for source_id: {source_id}")
    elif source_type == "operation_proposal":
        if OperationQueue(operations_path).get_proposal(source_id) is None:
            raise ValueError(f"Operation proposal not found for source_id: {source_id}")


class ReviewActionStore:
    """Durable store for human/agent review lifecycle decisions."""

    def __init__(
        self,
        db_path: str | Path = "data/dean_os/review_actions.sqlite",
        operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
        event_log_path: str | Path | None = "logs/dean_os/events.jsonl",
    ):
        self.db_path = Path(db_path)
        self.operations_path = Path(operations_path)
        self.event_log = EventLog(event_log_path) if event_log_path else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_action(self, action: ReviewActionRecord) -> str:
        payload = json.dumps(json_ready(action), ensure_ascii=True)
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM review_actions WHERE action_id = ?",
                (action.action_id,),
            )
            next_revision = (cur.fetchone()[0] or 0) + 1
            conn.execute(
                """
                INSERT INTO review_actions
                (action_id, revision, source_type, source_id, action_type, status, reviewer,
                 linked_proposal_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.action_id,
                    next_revision,
                    action.source_type,
                    action.source_id,
                    action.action_type,
                    action.status,
                    action.reviewer,
                    action.linked_proposal_id,
                    action.created_at,
                    payload,
                ),
            )
            conn.execute(
                """
                INSERT INTO review_action_events
                (event_id, action_id, revision, actor, reason, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    action.action_id,
                    next_revision,
                    action.reviewer,
                    action.notes or "review action write",
                    payload,
                    utc_now_iso(),
                ),
            )
        self._log("review_action_recorded", action.model_dump(mode="json"))
        return action.action_id

    def list_actions(
        self,
        source_type: str | None = None,
        action_type: str | None = None,
    ) -> list[ReviewActionRecord]:
        latest_sql = """
            SELECT ra.payload
            FROM review_actions ra
            INNER JOIN (
                SELECT action_id, MAX(revision) AS max_rev
                FROM review_actions
                GROUP BY action_id
            ) latest ON ra.action_id = latest.action_id AND ra.revision = latest.max_rev
        """
        clauses = []
        params: list[Any] = []
        if source_type:
            clauses.append("ra.source_type = ?")
            params.append(source_type)
        if action_type:
            clauses.append("ra.action_type = ?")
            params.append(action_type)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(latest_sql + where + " ORDER BY ra.rowid", params).fetchall()
        return [ReviewActionRecord(**json.loads(row["payload"])) for row in rows]

    def get_action(self, action_id: str) -> ReviewActionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM review_actions WHERE action_id = ? ORDER BY revision DESC LIMIT 1",
                (action_id,),
            ).fetchone()
        if row is None:
            return None
        return ReviewActionRecord(**json.loads(row["payload"]))

    def mark_reviewed(
        self,
        source_type: str,
        source_id: str,
        notes: str = "",
        reviewer: str = "human",
    ) -> ReviewActionRecord:
        action = ReviewActionRecord(
            source_type=source_type,
            source_id=source_id,
            action_type="mark_reviewed",
            reviewer=reviewer,
            notes=notes,
        )
        self.add_action(action)
        return action

    def void_action(self, action_id: str, reason: str = "") -> ReviewActionRecord:
        action = self.get_action(action_id)
        if action is None:
            raise KeyError(f"Review action not found: {action_id}")
        action.status = "voided"
        reason_text = f"Voided: {reason}" if reason else "Voided."
        action.notes = f"{action.notes}\n{reason_text}".strip()
        if action.linked_proposal_id:
            OperationQueue(
                self.operations_path,
                event_log_path=self.event_log.log_path if self.event_log else None,
            ).reject(action.linked_proposal_id)
        self.add_action(action)
        self._log("review_action_voided", action.model_dump(mode="json"))
        return action

    def needs_more_data(
        self,
        source_type: str,
        source_id: str,
        data_request: str,
        notes: str = "",
        reviewer: str = "human",
    ) -> ReviewActionRecord:
        action = ReviewActionRecord(
            source_type=source_type,
            source_id=source_id,
            action_type="needs_more_data",
            reviewer=reviewer,
            notes=notes,
            payload={"data_request": data_request},
        )
        self.add_action(action)
        return action

    def promote_to_watchlist_proposal(
        self,
        source_type: str,
        source_id: str,
        tickers: list[str],
        thesis: str,
        reason: str,
        notes: str = "",
        reviewer: str = "human",
    ) -> ReviewActionRecord:
        action = ReviewActionRecord(
            source_type=source_type,
            source_id=source_id,
            action_type="promote_to_watchlist_proposal",
            status="queued",
            reviewer=reviewer,
            notes=notes,
            payload={"tickers": tickers, "thesis": thesis, "reason": reason},
        )
        proposal = PipelineActionProposal(
            agent_name="review_lifecycle",
            action_type="report",
            target=f"watchlist_candidate:{','.join(tickers) if tickers else source_id}",
            reason=reason,
            command_preview="review watchlist candidate in DEAN-OS; no trading pipeline stage is executed",
            expected_effect="Create a reviewable watchlist candidate for later consensus or paper-trade evaluation",
            risks=[
                "Watchlist promotion is not a trade signal",
                "Requires real evidence and future consensus review before any trading decision",
            ],
            evidence=[
                EvidenceItem(
                    source_type="operation",
                    source=f"review_action:{action.action_id}",
                    key="source_id",
                    value=source_id,
                )
            ],
        )
        proposal_id = OperationQueue(self.operations_path, event_log_path=self.event_log.log_path if self.event_log else None).add_proposal(proposal)
        action.linked_proposal_id = proposal_id
        self.add_action(action)
        return action

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_actions (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    linked_proposal_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_actions_id ON review_actions(action_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_action_events (
                    event_id TEXT PRIMARY KEY,
                    action_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_action_events_id ON review_action_events(action_id)"
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

    def _log(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_log:
            self.event_log.write(event_type=event_type, source="review_actions", payload=payload)
