from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from dean_os.draft.dean_os_agent_system_v7.dean_os.event_log import EventLog
from dean_os.schemas import AgentLabRunReport, PipelineActionProposal, utc_now_iso
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
                INSERT INTO proposal_transitions
                (transition_id, proposal_id, agent_name, action_type, target, status, created_at, payload, reviewer, reason, evidence_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._generate_transition_id(),
                    proposal.proposal_id,
                    proposal.agent_name,
                    proposal.action_type,
                    proposal.target,
                    proposal.status,
                    proposal.created_at,
                    json.dumps(json_ready(proposal), ensure_ascii=True),
                    None,  # reviewer
                    None,  # reason
                    None,  # evidence_ref
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
        from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

        payload = DeanPaths.load_json(report_path)
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
            clauses.append("current_status.status = ?")
            params.append(status)
        if action_type:
            clauses.append("current_status.action_type = ?")
            params.append(action_type)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT current_status.payload 
                FROM proposal_transitions AS current_status
                INNER JOIN (
                    SELECT proposal_id, MAX(created_at) as latest_created_at
                    FROM proposal_transitions
                    GROUP BY proposal_id
                ) AS latest 
                ON current_status.proposal_id = latest.proposal_id 
                AND current_status.created_at = latest.latest_created_at
                {where}
                ORDER BY current_status.created_at DESC
                """,
                params
            ).fetchall()
        return [PipelineActionProposal(**json.loads(row["payload"])) for row in rows]

    def get_proposal(self, proposal_id: str) -> PipelineActionProposal | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload 
                FROM proposal_transitions 
                WHERE proposal_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
                """,
                (proposal_id,)
            ).fetchone()
        if row is None:
            return None
        return PipelineActionProposal(**json.loads(row["payload"]))

    def set_status(self, proposal_id: str, status: ProposalStatus, reviewer: str | None = None, reason: str | None = None, evidence_ref: str | None = None) -> PipelineActionProposal:
        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Operation proposal not found: {proposal_id}")
        updated = PipelineActionProposal(**{**proposal.model_dump(mode="json"), "status": status, "created_at": utc_now_iso()})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO proposal_transitions
                (transition_id, proposal_id, agent_name, action_type, target, status, created_at, payload, reviewer, reason, evidence_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._generate_transition_id(),
                    updated.proposal_id,
                    updated.agent_name,
                    updated.action_type,
                    updated.target,
                    updated.status,
                    updated.created_at,
                    json.dumps(json_ready(updated), ensure_ascii=True),
                    reviewer,
                    reason,
                    evidence_ref,
                ),
            )
        self._log(
            "operation_proposal_status_changed",
            {"proposal_id": proposal_id, "old_status": proposal.status, "new_status": status, "reviewer": reviewer, "reason": reason},
        )
        return updated

    def approve(self, proposal_id: str, reviewer: str, reason: str, evidence_ref: str | None = None) -> PipelineActionProposal:
        if not reviewer:
            raise ValueError("Approval requires reviewer")
        if not reason:
            raise ValueError("Approval requires reason")
        return self.set_status(proposal_id, "approved", reviewer=reviewer, reason=reason, evidence_ref=evidence_ref)

    def reject(self, proposal_id: str, reviewer: str, reason: str, evidence_ref: str | None = None) -> PipelineActionProposal:
        if not reviewer:
            raise ValueError("Rejection requires reviewer")
        if not reason:
            raise ValueError("Rejection requires reason")
        return self.set_status(proposal_id, "rejected", reviewer=reviewer, reason=reason, evidence_ref=evidence_ref)

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
                CREATE TABLE IF NOT EXISTS proposal_transitions (
                    transition_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    reviewer TEXT,
                    reason TEXT,
                    evidence_ref TEXT
                )
                """
            )
            # Create index for faster queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_proposal_id 
                ON proposal_transitions(proposal_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON proposal_transitions(created_at)
                """
            )

    def _generate_transition_id(self) -> str:
        """Generate a unique transition ID."""
        from uuid import uuid4
        return uuid4().hex

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
