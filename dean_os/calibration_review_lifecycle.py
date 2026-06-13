from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.operation_queue import OperationQueue
from dean_os.schemas import PipelineActionProposal, utc_now_iso
from dean_os.utils import json_ready


CALIBRATION_TARGET_PREFIX = "analyst_calibration:"
CALIBRATION_AGENT_NAME = "calibration_proposal_agent"


class CalibrationReviewLifecycle:
    """Review-only lifecycle manager for calibration operation proposals.

    Approval in this lifecycle only changes an OperationQueue proposal status.
    It never writes production config, analyst defaults, or consensus weights.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/calibration_review_lifecycle"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
        log_path: str | Path | None = "logs/dean_os/events.jsonl",
        proposal_ids: list[str] | None = None,
        dry_run_proposals: bool = False,
        approve_ids: list[str] | None = None,
        reject_ids: list[str] | None = None,
        include_non_calibration: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        queue = OperationQueue(operations_path, event_log_path=log_path)
        selected_ids = {proposal_id for proposal_id in proposal_ids or [] if proposal_id}
        approve_set = {proposal_id for proposal_id in approve_ids or [] if proposal_id}
        reject_set = {proposal_id for proposal_id in reject_ids or [] if proposal_id}
        if approve_set & reject_set:
            raise ValueError("The same proposal_id cannot be both approved and rejected.")

        initial = _select_proposals(queue.list_proposals(), selected_ids, include_non_calibration)
        action_results = _apply_status_actions(
            queue=queue,
            approve_ids=approve_set,
            reject_ids=reject_set,
            include_non_calibration=include_non_calibration,
        )
        after_actions = _select_proposals(queue.list_proposals(), selected_ids, include_non_calibration)
        dry_run_previews = (
            _dry_run_selected(queue, after_actions)
            if dry_run_proposals
            else []
        )

        payload = {
            "run_id": _run_id("calibration_review_lifecycle"),
            "created_at": utc_now_iso(),
            "mode": "calibration_review_lifecycle",
            "inputs": {
                "operations_path": str(operations_path),
                "log_path": str(log_path) if log_path else None,
                "proposal_ids": sorted(selected_ids),
                "dry_run_proposals": dry_run_proposals,
                "approve_ids": sorted(approve_set),
                "reject_ids": sorted(reject_set),
                "include_non_calibration": include_non_calibration,
            },
            "lifecycle_gate": _lifecycle_gate(after_actions, action_results, dry_run_previews),
            "initial_status_counts": _status_counts(initial),
            "final_status_counts": _status_counts(after_actions),
            "action_results": action_results,
            "dry_run_previews": dry_run_previews,
            "calibration_proposals": [_proposal_summary(proposal) for proposal in after_actions],
            "approved_waiting_manual_implementation": [
                _proposal_summary(proposal)
                for proposal in after_actions
                if proposal.status == "approved"
            ],
            "recommendations": _recommendations(after_actions, action_results, dry_run_previews),
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
        rendered_md = render_calibration_review_lifecycle_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_calibration_review_lifecycle_markdown(payload: dict[str, Any]) -> str:
    gate = payload.get("lifecycle_gate", {})
    lines = [
        "# DEAN-OS Calibration Review Lifecycle",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{gate.get('status')}`",
        f"- Proposals: {gate.get('proposal_count', 0)}",
        f"- Dry-run previews: {gate.get('dry_run_count', 0)}",
        f"- Approved waiting manual implementation: {gate.get('approved_waiting_manual_implementation_count', 0)}",
        "",
        "## Calibration Proposals",
        "",
    ]
    for proposal in payload.get("calibration_proposals", []):
        lines.append(f"- `{proposal.get('proposal_id')}` {proposal.get('status')} -> {proposal.get('target')}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _select_proposals(
    proposals: list[PipelineActionProposal],
    selected_ids: set[str],
    include_non_calibration: bool,
) -> list[PipelineActionProposal]:
    selected = []
    for proposal in proposals:
        if selected_ids and proposal.proposal_id not in selected_ids:
            continue
        if not include_non_calibration and not _is_calibration_proposal(proposal):
            continue
        selected.append(proposal)
    return selected


def _apply_status_actions(
    queue: OperationQueue,
    approve_ids: set[str],
    reject_ids: set[str],
    include_non_calibration: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for proposal_id in sorted(approve_ids):
        results.append(_set_status(queue, proposal_id, "approved", include_non_calibration))
    for proposal_id in sorted(reject_ids):
        results.append(_set_status(queue, proposal_id, "rejected", include_non_calibration))
    return results


def _set_status(
    queue: OperationQueue,
    proposal_id: str,
    status: str,
    include_non_calibration: bool,
) -> dict[str, Any]:
    proposal = queue.get_proposal(proposal_id)
    if proposal is None:
        return {"proposal_id": proposal_id, "requested_status": status, "status": "not_found"}
    if not include_non_calibration and not _is_calibration_proposal(proposal):
        return {
            "proposal_id": proposal_id,
            "requested_status": status,
            "status": "skipped_non_calibration",
            "target": proposal.target,
        }
    updated = queue.approve(proposal_id) if status == "approved" else queue.reject(proposal_id)
    return {
        "proposal_id": proposal_id,
        "requested_status": status,
        "status": "updated",
        "target": updated.target,
        "proposal_status": updated.status,
        "manual_implementation_required": True,
    }


def _dry_run_selected(queue: OperationQueue, proposals: list[PipelineActionProposal]) -> list[dict[str, Any]]:
    previews = []
    for proposal in proposals:
        preview = queue.dry_run(proposal.proposal_id)
        preview["manual_implementation_required"] = proposal.status == "approved"
        preview["config_write_performed"] = False
        previews.append(preview)
    return previews


def _is_calibration_proposal(proposal: PipelineActionProposal) -> bool:
    return proposal.agent_name == CALIBRATION_AGENT_NAME and proposal.target.startswith(CALIBRATION_TARGET_PREFIX)


def _proposal_summary(proposal: PipelineActionProposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "agent_name": proposal.agent_name,
        "action_type": proposal.action_type,
        "target": proposal.target,
        "status": proposal.status,
        "dry_run": proposal.dry_run,
        "requires_human_approval": proposal.requires_human_approval,
        "reason": proposal.reason,
        "command_preview": proposal.command_preview,
        "expected_effect": proposal.expected_effect,
        "risk_count": len(proposal.risks),
        "evidence_count": len(proposal.evidence),
    }


def _status_counts(proposals: list[PipelineActionProposal]) -> dict[str, int]:
    return dict(sorted(Counter(proposal.status for proposal in proposals).items()))


def _lifecycle_gate(
    proposals: list[PipelineActionProposal],
    action_results: list[dict[str, Any]],
    dry_run_previews: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = _status_counts(proposals)
    successful_actions = [item for item in action_results if item.get("status") == "updated"]
    if not proposals:
        status = "no_calibration_proposals"
    elif status_counts.get("approved"):
        status = "approved_waiting_manual_implementation"
    elif successful_actions:
        status = "actions_applied"
    elif dry_run_previews:
        status = "dry_run_reviewed"
    elif status_counts.get("proposed"):
        status = "awaiting_review"
    else:
        status = "review_complete"
    return {
        "status": status,
        "proposal_count": len(proposals),
        "dry_run_count": len(dry_run_previews),
        "action_count": len(action_results),
        "successful_action_count": len(successful_actions),
        "approved_waiting_manual_implementation_count": status_counts.get("approved", 0),
        "status_counts": status_counts,
    }


def _recommendations(
    proposals: list[PipelineActionProposal],
    action_results: list[dict[str, Any]],
    dry_run_previews: list[dict[str, Any]],
) -> list[str]:
    if not proposals:
        return ["No calibration proposals are in scope. Run the calibration proposal agent when a profile is ready for review."]
    status_counts = _status_counts(proposals)
    recommendations: list[str] = []
    if status_counts.get("proposed"):
        recommendations.append("Dry-run proposed calibration items before approving or rejecting them.")
    if dry_run_previews:
        recommendations.append("Dry-run previews performed no config writes; inspect risks before approval.")
    if status_counts.get("approved"):
        recommendations.append("Approved calibration proposals still require a separate manual implementation task; do not treat approval as config mutation.")
    if any(item.get("status") in {"not_found", "skipped_non_calibration"} for item in action_results):
        recommendations.append("Some requested actions were skipped; inspect action_results before proceeding.")
    if not recommendations:
        recommendations.append("No further calibration review action is required right now.")
    return recommendations


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
