from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.context_performance import AgentPerformanceByContext
from dean_os.event_log import EventLog
from dean_os.learning import LearningStore
from dean_os.operation_queue import OperationQueue
from dean_os.recommendation_memory import RecommendationMemoryStore
from dean_os.review_actions import ReviewActionStore
from dean_os.schemas import AgentLabRunReport
from dean_os.utils import json_ready


class AgentReviewBuilder:
    """Builds a human-review snapshot from lab reports, learning, queue, and logs."""

    def __init__(
        self,
        reports_dir: str | Path = "reports/dean_os/agent_lab",
        learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
        operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
        review_actions_path: str | Path = "data/dean_os/review_actions.sqlite",
        memory_path: str | Path = "data/dean_os/recommendation_memory.sqlite",
        log_path: str | Path = "logs/dean_os/events.jsonl",
        output_dir: str | Path = "reports/dean_os/review",
    ):
        self.reports_dir = Path(reports_dir)
        self.learning_path = Path(learning_path)
        self.operations_path = Path(operations_path)
        self.review_actions_path = Path(review_actions_path)
        self.memory_path = Path(memory_path)
        self.log_path = Path(log_path)
        self.output_dir = Path(output_dir)

    def build(self, report_path: str | Path | None = None, event_limit: int = 10) -> dict[str, Any]:
        report = self._load_report(report_path)
        learning_records = LearningStore(self.learning_path).list_records()
        proposals = OperationQueue(self.operations_path).list_proposals()
        review_actions = ReviewActionStore(
            db_path=self.review_actions_path,
            operations_path=self.operations_path,
            event_log_path=None,
        ).list_actions()
        memory_summary = RecommendationMemoryStore(self.memory_path).summary()
        context_performance = AgentPerformanceByContext(
            learning_path=self.learning_path,
            memory_path=self.memory_path,
        ).build_summary(limit=5)
        events = EventLog(self.log_path).read(limit=event_limit)

        pending_records = [record for record in learning_records if record.outcome_label is None]
        completed_records = [record for record in learning_records if record.outcome_label is not None]
        proposed = [proposal for proposal in proposals if proposal.status == "proposed"]
        approved = [proposal for proposal in proposals if proposal.status == "approved"]
        open_data_request_actions = [
            action
            for action in review_actions
            if action.action_type == "needs_more_data" and action.status != "voided"
        ]

        snapshot = {
            "report": self._report_summary(report),
            "learning": {
                "total_record_count": len(learning_records),
                "pending_record_count": len(pending_records),
                "completed_record_count": len(completed_records),
                "records_by_agent": dict(sorted(Counter(record.agent_name for record in learning_records).items())),
                "pending_by_agent": dict(sorted(Counter(record.agent_name for record in pending_records).items())),
                "latest_pending_records": [
                    {
                        "record_id": record.record_id,
                        "agent_name": record.agent_name,
                        "expected_direction": record.expected_direction,
                        "horizon_days": record.horizon_days,
                        "topic": record.metadata.get("topic"),
                        "patterns": record.metadata.get("patterns", []),
                    }
                    for record in pending_records[-5:]
                ],
            },
            "operations": {
                "proposal_count": len(proposals),
                "proposed_count": len(proposed),
                "approved_count": len(approved),
                "proposals_by_status": dict(sorted(Counter(proposal.status for proposal in proposals).items())),
                "latest_proposals": [
                    {
                        "proposal_id": proposal.proposal_id,
                        "status": proposal.status,
                        "action_type": proposal.action_type,
                        "target": proposal.target,
                        "reason": proposal.reason,
                        "command_preview": proposal.command_preview,
                    }
                    for proposal in proposals[-5:]
                ],
            },
            "review_actions": {
                "action_count": len(review_actions),
                "actions_by_type": dict(sorted(Counter(action.action_type for action in review_actions).items())),
                "actions_by_status": dict(sorted(Counter(action.status for action in review_actions).items())),
                "actions_by_source_type": dict(sorted(Counter(action.source_type for action in review_actions).items())),
                "open_data_requests": [
                    {
                        "action_id": action.action_id,
                        "source_type": action.source_type,
                        "source_id": action.source_id,
                        "data_request": action.payload.get("data_request", ""),
                        "notes": action.notes,
                    }
                    for action in open_data_request_actions[-5:]
                ],
                "latest_actions": [
                    {
                        "action_id": action.action_id,
                        "source_type": action.source_type,
                        "source_id": action.source_id,
                        "action_type": action.action_type,
                        "status": action.status,
                        "linked_proposal_id": action.linked_proposal_id,
                        "notes": action.notes,
                    }
                    for action in review_actions[-5:]
                ],
            },
            "memory": memory_summary,
            "context_performance": context_performance,
            "logs": {
                "log_path": str(self.log_path),
                "event_count_in_tail": len(events),
                "latest_events": [
                    {
                        "event_type": event.get("event_type"),
                        "source": event.get("source"),
                        "run_id": event.get("run_id"),
                        "timestamp": event.get("timestamp"),
                        "payload": event.get("payload", {}),
                    }
                    for event in events
                ],
            },
        }
        snapshot["next_actions"] = self._next_actions(snapshot)
        return snapshot

    def save(self, snapshot: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = snapshot.get("report", {}).get("run_id") or "no-run"
        json_path = self.output_dir / f"review_{run_id}.json"
        md_path = self.output_dir / f"review_{run_id}.md"
        json_path.write_text(json.dumps(json_ready(snapshot), indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(render_review_markdown(snapshot), encoding="utf-8")
        return json_path, md_path

    def _load_report(self, report_path: str | Path | None) -> AgentLabRunReport | None:
        path = Path(report_path) if report_path else latest_agent_lab_report(self.reports_dir)
        if path is None:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AgentLabRunReport(**payload)

    def _report_summary(self, report: AgentLabRunReport | None) -> dict[str, Any]:
        if report is None:
            return {
                "available": False,
                "run_id": None,
                "document_count": 0,
                "note_count": 0,
                "top_patterns": [],
                "latest_thesis": "",
                "load_error_count": None,
            }
        return {
            "available": True,
            "run_id": report.run_id,
            "created_at": report.created_at,
            "corpus_path": report.corpus_path,
            "document_count": report.document_count,
            "note_count": report.note_count,
            "top_patterns": report.summary.get("top_patterns", []),
            "latest_thesis": report.summary.get("latest_thesis", ""),
            "load_error_count": report.summary.get("load_error_count", 0),
            "learning_record_count": report.summary.get("learning_record_count", 0),
            "proposal_count": report.summary.get("proposal_count", 0),
            "queued_proposal_count": report.summary.get("queued_proposal_count", 0),
        }

    def _next_actions(self, snapshot: dict[str, Any]) -> list[str]:
        actions: list[str] = []
        report = snapshot["report"]
        learning = snapshot["learning"]
        operations = snapshot["operations"]
        review_actions = snapshot["review_actions"]
        memory = snapshot["memory"]
        context_performance = snapshot.get("context_performance", {})

        if not report["available"]:
            actions.append("Run Agent Lab with real materials or --sample to create the first reviewable report.")
            return actions
        if report.get("load_error_count"):
            actions.append("Create docs/research or pass a valid materials path, then rerun Agent Lab.")
        if report.get("document_count", 0) == 0:
            actions.append("Add research materials before treating specialist theses as meaningful.")
        if operations["proposed_count"] > 0:
            actions.append("Review proposed operation actions with run_agent_ops.py list/dry-run before approval.")
        if review_actions["action_count"] == 0:
            actions.append("Record a review lifecycle action: mark-reviewed, needs-more-data, or promote-watchlist.")
        if review_actions["open_data_requests"]:
            actions.append("Fulfill latest needs-more-data requests before promoting theses or watchlist candidates.")
        if memory["recent_lessons"]:
            actions.append("Check recommendation memory lessons for matching context tags before accepting a new thesis.")
        if context_performance.get("weak_contexts"):
            actions.append("Review weak agent/context buckets and require stronger evidence in matching future regimes.")
        if learning["pending_record_count"] > 0:
            actions.append("Keep pending learning records until their horizon/outcome can be evaluated.")
        if report.get("top_patterns"):
            actions.append("Use top_patterns to decide which specialist branch to deepen next.")
        if not actions:
            actions.append("No urgent review action detected; continue adding higher-quality research materials.")
        return actions


def latest_agent_lab_report(reports_dir: str | Path) -> Path | None:
    root = Path(reports_dir)
    if not root.exists():
        return None
    reports = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime)
    return reports[-1] if reports else None


def render_review_markdown(snapshot: dict[str, Any]) -> str:
    report = snapshot["report"]
    learning = snapshot["learning"]
    operations = snapshot["operations"]
    review_actions = snapshot["review_actions"]
    memory = snapshot["memory"]
    context_performance = snapshot.get("context_performance", {})
    lines = [
        "# DEAN-OS Review Summary",
        "",
        "## Latest Agent Lab",
        "",
        f"- Report available: {report.get('available')}",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Documents: {report.get('document_count', 0)}",
        f"- Research notes: {report.get('note_count', 0)}",
        f"- Top patterns: {', '.join(report.get('top_patterns', [])) or 'none'}",
        f"- Latest thesis: {report.get('latest_thesis') or 'none'}",
        "",
        "## Learning",
        "",
        f"- Total records: {learning['total_record_count']}",
        f"- Pending records: {learning['pending_record_count']}",
        f"- Completed records: {learning['completed_record_count']}",
        f"- Pending by agent: {learning['pending_by_agent']}",
        "",
        "## Operations",
        "",
        f"- Proposals: {operations['proposal_count']}",
        f"- Proposed: {operations['proposed_count']}",
        f"- Approved: {operations['approved_count']}",
        f"- By status: {operations['proposals_by_status']}",
        "",
        "## Review Actions",
        "",
        f"- Actions: {review_actions['action_count']}",
        f"- By type: {review_actions['actions_by_type']}",
        f"- Open data requests: {len(review_actions['open_data_requests'])}",
        "",
        "## Recommendation Memory",
        "",
        f"- Memory records: {memory['record_count']}",
        f"- Hit rate: {memory['hit_rate']}",
        f"- Records by outcome: {memory['records_by_outcome']}",
        "",
        "## Context Performance",
        "",
        f"- Completed outcomes: {context_performance.get('overall', {}).get('completed_count', 0)}",
        f"- Weak contexts: {len(context_performance.get('weak_contexts', []))}",
        f"- Strengths: {len(context_performance.get('strengths', []))}",
        "",
        "## Next Actions",
        "",
    ]
    for action in snapshot["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines).strip() + "\n"
