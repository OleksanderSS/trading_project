from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from dean_os.event_log import EventLog
from dean_os.schemas import AgentLabRunReport, PipelineActionProposal
from dean_os.utils import json_ready


ProposalStatus = Literal["proposed", "approved", "rejected", "expired", "executed"]


class OperationQueue:
    """Durable review queue for agent-proposed operational actions."""

    def __init__(
        self,
        db_path: str | Path = "data/dean_os/operation_queue.sqlite",
        event_log_path: str | Path | None = None,
    ):
        self.db_path = Path(db_path)
        self.event_log = EventLog(event_log_path) if event_log_path else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_proposal(self, proposal: PipelineActionProposal) -> str:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO proposals
                (proposal_id, agent_name, action_type, target, status, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.proposal_id,
                    proposal.agent_name,
                    proposal.action_type,
                    proposal.target,
                    proposal.status,
                    proposal.created_at,
                    json.dumps(json_ready(proposal), ensure_ascii=True),
                ),
            )
        self._log(
            "operation_proposal_saved",
            {
                "proposal_id": proposal.proposal_id,
                "action_type": proposal.action_type,
                "target": proposal.target,
                "status": proposal.status,
            },
        )
        return proposal.proposal_id

    def add_many(self, proposals: list[PipelineActionProposal]) -> list[str]:
        return [self.add_proposal(proposal) for proposal in proposals]

    def import_agent_lab_report(self, report_path: str | Path) -> list[str]:
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        report = AgentLabRunReport(**payload)
        proposal_ids = self.add_many(report.action_proposals)
        self._log(
            "operation_proposals_imported",
            {
                "report_path": str(report_path),
                "run_id": report.run_id,
                "imported_count": len(proposal_ids),
                "proposal_ids": proposal_ids,
            },
        )
        return proposal_ids

    def list_proposals(
        self,
        status: ProposalStatus | None = None,
        action_type: str | None = None,
    ) -> list[PipelineActionProposal]:
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if action_type:
            clauses.append("action_type = ?")
            params.append(action_type)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT payload FROM proposals{where} ORDER BY rowid", params).fetchall()
        return [PipelineActionProposal(**json.loads(row["payload"])) for row in rows]

    def get_proposal(self, proposal_id: str) -> PipelineActionProposal | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
        if row is None:
            return None
        return PipelineActionProposal(**json.loads(row["payload"]))

    def set_status(self, proposal_id: str, status: ProposalStatus) -> PipelineActionProposal:
        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Operation proposal not found: {proposal_id}")
        updated = PipelineActionProposal(**{**proposal.model_dump(mode="json"), "status": status})
        self.add_proposal(updated)
        self._log(
            "operation_proposal_status_changed",
            {"proposal_id": proposal_id, "old_status": proposal.status, "new_status": status},
        )
        return updated

    def approve(self, proposal_id: str) -> PipelineActionProposal:
        return self.set_status(proposal_id, "approved")

    def reject(self, proposal_id: str) -> PipelineActionProposal:
        return self.set_status(proposal_id, "rejected")

    def dry_run(self, proposal_id: str) -> dict[str, Any]:
        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Operation proposal not found: {proposal_id}")
        preview = {
            "proposal_id": proposal.proposal_id,
            "agent_name": proposal.agent_name,
            "action_type": proposal.action_type,
            "target": proposal.target,
            "status": proposal.status,
            "dry_run": True,
            "requires_human_approval": proposal.requires_human_approval,
            "ready_for_manual_execution": proposal.status == "approved",
            "command_preview": proposal.command_preview,
            "reason": proposal.reason,
            "expected_effect": proposal.expected_effect,
            "risks": proposal.risks,
            "message": "This is a preview only. No pipeline stage was executed.",
        }
        self._log("operation_proposal_dry_run_previewed", preview)
        return preview

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proposals (
                    proposal_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
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

    def _log(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_log:
            self.event_log.write(event_type=event_type, source="operation_queue", payload=payload)
