from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.calibration_review_lifecycle import CALIBRATION_AGENT_NAME, CALIBRATION_TARGET_PREFIX
from dean_os.draft.dean_os_agent_system_v7.dean_os.operation_queue import OperationQueue
from dean_os.schemas import EvidenceItem, PipelineActionProposal, utc_now_iso
from dean_os.utils import json_ready


class ManualImplementationBacklog:
    """Read-only backlog for approved agent proposals awaiting manual work.

    This report is the explicit boundary between "approved recommendation" and
    "implemented system change". It never writes config, code, queue status, or
    consensus weights.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/manual_implementation_backlog"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
        include_proposed: bool = False,
        include_rejected: bool = False,
        include_non_calibration: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        proposals = OperationQueue(operations_path, event_log_path=None).list_proposals()
        scoped = [
            proposal
            for proposal in proposals
            if _include_proposal(
                proposal=proposal,
                include_proposed=include_proposed,
                include_rejected=include_rejected,
                include_non_calibration=include_non_calibration,
            )
        ]
        tasks = [_task_from_proposal(proposal) for proposal in scoped]
        payload = {
            "run_id": _run_id("manual_implementation_backlog"),
            "created_at": utc_now_iso(),
            "mode": "manual_implementation_backlog",
            "inputs": {
                "operations_path": str(operations_path),
                "include_proposed": include_proposed,
                "include_rejected": include_rejected,
                "include_non_calibration": include_non_calibration,
            },
            "backlog_gate": _backlog_gate(tasks, proposals),
            "status_counts": dict(sorted(Counter(task["proposal_status"] for task in tasks).items())),
            "tasks": tasks,
            "recommendations": _recommendations(tasks),
        }
        if save:
            self.save(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_manual_implementation_backlog_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_manual_implementation_backlog_markdown(payload: dict[str, Any]) -> str:
    gate = payload.get("backlog_gate", {})
    lines = [
        "# DEAN-OS Manual Implementation Backlog",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{gate.get('status')}`",
        f"- Manual tasks: {gate.get('task_count', 0)}",
        f"- Approved tasks: {gate.get('approved_task_count', 0)}",
        "",
        "## Tasks",
        "",
    ]
    for task in payload.get("tasks", []):
        lines.extend(
            [
                f"### {task.get('task_id')}",
                "",
                f"- Status: `{task.get('proposal_status')}`",
                f"- Target: `{task.get('target')}`",
                f"- Profile: `{task.get('profile')}`",
                f"- Suggested weight delta: {task.get('suggested_weight_delta')}",
                f"- Config write performed: {task.get('config_write_performed')}",
                "",
            ]
        )
    lines.extend(["## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _include_proposal(
    proposal: PipelineActionProposal,
    include_proposed: bool,
    include_rejected: bool,
    include_non_calibration: bool,
) -> bool:
    if not include_non_calibration and not _is_calibration_proposal(proposal):
        return False
    if proposal.status == "approved":
        return True
    if include_proposed and proposal.status == "proposed":
        return True
    if include_rejected and proposal.status == "rejected":
        return True
    return False


def _task_from_proposal(proposal: PipelineActionProposal) -> dict[str, Any]:
    profile = _profile_from_target(proposal.target)
    suggested_delta = _evidence_value(proposal.evidence, "suggested_weight_delta")
    return {
        "task_id": f"manual_impl_{proposal.proposal_id}",
        "proposal_id": proposal.proposal_id,
        "proposal_status": proposal.status,
        "manual_status": _manual_status(proposal),
        "target": proposal.target,
        "profile": profile,
        "suggested_weight_delta": suggested_delta,
        "reason": proposal.reason,
        "expected_effect": proposal.expected_effect,
        "command_preview": proposal.command_preview,
        "risks": proposal.risks,
        "evidence": [item.model_dump(mode="json") for item in proposal.evidence],
        "implementation_checklist": _implementation_checklist(proposal, profile, suggested_delta),
        "config_write_performed": False,
        "requires_separate_pr": proposal.status == "approved",
        "requires_human_owner": True,
    }


def _manual_status(proposal: PipelineActionProposal) -> str:
    if proposal.status == "approved":
        return "waiting_manual_implementation"
    if proposal.status == "proposed":
        return "waiting_review"
    if proposal.status == "rejected":
        return "rejected_no_implementation"
    return "not_actionable"


def _implementation_checklist(
    proposal: PipelineActionProposal,
    profile: str,
    suggested_delta: Any,
) -> list[str]:
    return [
        "Open a separate branch or PR for any config/default/weight change.",
        f"Re-read the source proposal `{proposal.proposal_id}` and calibration gate evidence.",
        f"Confirm target profile `{profile}` and suggested delta `{suggested_delta}` are still valid.",
        "Verify completed outcomes are mature and not historical-diagnostic-only.",
        "Run relevant DEAN-OS tests and config validation after any manual edit.",
        "Document rollback instructions and reason for the final human decision.",
        "Do not mark implementation complete just because the OperationQueue proposal is approved.",
    ]


def _is_calibration_proposal(proposal: PipelineActionProposal) -> bool:
    return proposal.agent_name == CALIBRATION_AGENT_NAME and proposal.target.startswith(CALIBRATION_TARGET_PREFIX)


def _profile_from_target(target: str) -> str:
    if target.startswith(CALIBRATION_TARGET_PREFIX):
        return target[len(CALIBRATION_TARGET_PREFIX):]
    return target


def _evidence_value(evidence: list[EvidenceItem], key: str) -> Any:
    for item in evidence:
        if item.key == key:
            return item.value
    return None


def _backlog_gate(tasks: list[dict[str, Any]], proposals: list[PipelineActionProposal]) -> dict[str, Any]:
    approved = [task for task in tasks if task["proposal_status"] == "approved"]
    if approved:
        status = "manual_implementation_required"
    elif tasks:
        status = "review_items_visible"
    elif proposals:
        status = "no_manual_tasks_in_scope"
    else:
        status = "operation_queue_empty"
    return {
        "status": status,
        "task_count": len(tasks),
        "approved_task_count": len(approved),
        "config_write_performed": False,
    }


def _recommendations(tasks: list[dict[str, Any]]) -> list[str]:
    if not tasks:
        return ["No manual implementation tasks are currently in scope."]
    approved = [task for task in tasks if task["proposal_status"] == "approved"]
    if approved:
        return [
            "Approved proposals require a separate manual implementation PR or config change.",
            "Do not treat OperationQueue approval as implementation completion.",
            "Run tests and document rollback after any manual config edit.",
        ]
    return ["Backlog has review-visible items, but no approved manual implementation task yet."]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
