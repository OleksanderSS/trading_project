from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.operation_queue import OperationQueue
from dean_os.schemas import EvidenceItem, PipelineActionProposal, utc_now_iso
from dean_os.utils import json_ready


READY_STATUSES = {"ready_for_review"}
CAUTION_STATUSES = {"ready_with_caution"}


class CalibrationProposalAgent:
    """Turns calibration-gate readiness into reviewable operation proposals.

    The agent is proposal-only. It never edits profile defaults, consensus
    weights, or production configuration.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/calibration_proposals"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        calibration_gate_path: str | Path,
        operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
        log_path: str | Path | None = "logs/dean_os/events.jsonl",
        include_caution: bool = False,
        enqueue: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        gate_path = Path(calibration_gate_path)
        gate = _load_gate(gate_path)
        proposals = _proposals_from_gate(gate, include_caution=include_caution)
        enqueued_ids: list[str] = []
        if enqueue and proposals:
            queue = OperationQueue(operations_path, event_log_path=log_path)
            enqueued_ids = queue.add_many(proposals)

        payload = {
            "run_id": _run_id("calibration_proposals"),
            "created_at": utc_now_iso(),
            "mode": "calibration_proposal_agent",
            "inputs": {
                "calibration_gate_path": str(gate_path),
                "operations_path": str(operations_path),
                "log_path": str(log_path) if log_path else None,
                "include_caution": include_caution,
                "enqueue": enqueue,
            },
            "proposal_gate": _proposal_gate(gate, proposals, enqueue, enqueued_ids),
            "calibration_gate": {
                "run_id": gate.get("run_id"),
                "summary": gate.get("summary", {}),
                "saved_paths": gate.get("saved_paths", {}),
            },
            "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
            "enqueued_proposal_ids": enqueued_ids,
            "recommendations": _recommendations(proposals, enqueue, enqueued_ids),
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
        rendered_md = render_calibration_proposals_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_calibration_proposals_markdown(payload: dict[str, Any]) -> str:
    gate = payload.get("proposal_gate", {})
    lines = [
        "# DEAN-OS Calibration Proposals",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{gate.get('status')}`",
        f"- Proposals: {gate.get('proposal_count', 0)}",
        f"- Enqueued: {gate.get('enqueued_count', 0)}",
        "",
        "## Proposals",
        "",
    ]
    for proposal in payload.get("proposals", []):
        lines.extend(
            [
                f"- `{proposal.get('proposal_id')}` target={proposal.get('target')} status={proposal.get('status')}",
                f"  Reason: {proposal.get('reason')}",
            ]
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _load_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration gate JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration gate payload must be a JSON object: {path}")
    if payload.get("mode") != "analyst_calibration_gate":
        raise ValueError(f"Expected analyst_calibration_gate payload, got: {payload.get('mode')}")
    return payload


def _proposals_from_gate(gate: dict[str, Any], include_caution: bool) -> list[PipelineActionProposal]:
    allowed_statuses = READY_STATUSES | (CAUTION_STATUSES if include_caution else set())
    proposals: list[PipelineActionProposal] = []
    for profile, card in sorted(gate.get("profiles", {}).items()):
        if card.get("calibration_status") not in allowed_statuses:
            continue
        proposals.append(_proposal_for_profile(gate, profile, card))
    return proposals


def _proposal_for_profile(gate: dict[str, Any], profile: str, card: dict[str, Any]) -> PipelineActionProposal:
    outcomes = card.get("outcomes", {})
    scorecard = card.get("scorecard", {})
    delta = float(card.get("suggested_weight_delta") or 0.0)
    status = card.get("calibration_status")
    reason = (
        f"Analyst calibration gate marked {profile} as {status}; "
        f"completed_outcomes={outcomes.get('completed_count')}, "
        f"hit_rate={outcomes.get('hit_rate')}, miss_rate={outcomes.get('miss_rate')}, "
        f"suggested_weight_delta={delta}."
    )
    return PipelineActionProposal(
        agent_name="calibration_proposal_agent",
        action_type="report",
        target=f"analyst_calibration:{profile}",
        reason=reason,
        command_preview=(
            "Manual review only: inspect calibration gate, review evidence/outcomes, "
            f"then decide whether to adjust analyst profile `{profile}` by delta {delta}."
        ),
        expected_effect=(
            "Create a human-reviewed calibration proposal. No config, defaults, "
            "or consensus weights are changed by this proposal."
        ),
        risks=[
            "Completed outcomes may be too few or regime-specific.",
            "Historical diagnostics are not production learning truth.",
            "Increasing profile weight can amplify correlated analyst errors.",
            "Approval must not bypass normal config review and validation.",
        ],
        evidence=[
            EvidenceItem(
                source_type="operation",
                source=f"analyst_calibration_gate:{gate.get('run_id')}",
                key="calibration_status",
                value=status,
            ),
            EvidenceItem(
                source_type="metric",
                source=f"profile:{profile}",
                key="suggested_weight_delta",
                value=delta,
            ),
            EvidenceItem(
                source_type="metric",
                source=f"profile:{profile}",
                key="completed_outcomes",
                value=outcomes.get("completed_count"),
            ),
            EvidenceItem(
                source_type="metric",
                source=f"profile:{profile}",
                key="hit_rate",
                value=outcomes.get("hit_rate"),
            ),
            EvidenceItem(
                source_type="metric",
                source=f"profile:{profile}",
                key="scorecard_activation_status",
                value=scorecard.get("activation_status"),
            ),
        ],
    )


def _proposal_gate(
    gate: dict[str, Any],
    proposals: list[PipelineActionProposal],
    enqueue: bool,
    enqueued_ids: list[str],
) -> dict[str, Any]:
    if proposals and enqueue:
        status = "enqueued"
    elif proposals:
        status = "dry_run_ready"
    elif gate.get("summary", {}).get("profile_count", 0) == 0:
        status = "no_profiles"
    else:
        status = "no_ready_profiles"
    return {
        "status": status,
        "proposal_count": len(proposals),
        "enqueue_requested": enqueue,
        "enqueued_count": len(enqueued_ids),
        "enqueued_proposal_ids": enqueued_ids,
    }


def _recommendations(
    proposals: list[PipelineActionProposal],
    enqueue: bool,
    enqueued_ids: list[str],
) -> list[str]:
    if not proposals:
        return ["No calibration proposals were created; keep collecting completed outcomes and profile evidence."]
    if not enqueue:
        return ["Dry-run proposals are ready. Inspect risks and evidence before rerunning with --enqueue."]
    return [
        f"Enqueued {len(enqueued_ids)} calibration proposal(s) for review.",
        "Use run_agent_ops.py list/dry-run before approving any calibration change.",
        "Approval still must not directly edit production config without a separate reviewed implementation step.",
    ]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
