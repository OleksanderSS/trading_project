from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from dean_os.world_model.world_model_event_learning import WORLD_MODEL_EVENT_LEARNING_CONTRACT

WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT = "dean_world_model_replay_review_gate_v1"


class WorldModelReplayReviewGate:
    """Manual review gate for registering world-model replay tasks.

    The gate can produce an approved registration bundle, but it intentionally
    does not write a replay queue, learning memory, model config, or trading
    state. A separate operator-controlled step must consume the bundle.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/world_model_replay_review_gate",
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        packet_json: str | Path | dict[str, Any],
        *,
        approve: bool = False,
        reviewer: str | None = None,
        review_notes: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        packet, packet_source_path = _load_packet(packet_json)
        replay_tasks = list(packet.get("replay_tasks", []) or [])
        issues = _gate_issues(packet, replay_tasks, approve=approve, reviewer=reviewer)
        status = _gate_status(issues, approve=approve)
        can_register = status == "replay_tasks_approved_for_registration"
        registration_bundle = (
            _registration_bundle(
                packet,
                replay_tasks,
                reviewer=str(reviewer).strip(),
                review_notes=review_notes,
            )
            if can_register
            else None
        )
        payload = {
            "run_id": _run_id("world_model_replay_review_gate"),
            "created_at": utc_now_iso(),
            "mode": "world_model_replay_review_gate",
            "contract": WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
            "source_packet": {
                "path": str(packet_source_path) if packet_source_path else None,
                "run_id": packet.get("run_id"),
                "contract": packet.get("contract"),
                "packet_status": packet.get("summary", {}).get("packet_status"),
                "domain_id": packet.get("summary", {}).get("domain_id"),
                "as_of": packet.get("summary", {}).get("as_of"),
            },
            "summary": {
                "gate_status": status,
                "issue_count": len(issues),
                "issues": issues,
                "replay_task_count": len(replay_tasks),
                "approved": can_register,
                "manual_review_required_before_registration": not can_register,
                "can_register_replay_tasks": can_register,
                "registration_bundle_created": registration_bundle is not None,
                "replay_task_registration_performed": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "operator_decision": {
                "requested_approval": approve,
                "reviewer": str(reviewer).strip() if reviewer else None,
                "review_notes": review_notes,
            },
            "registration_bundle": registration_bundle,
            "source_replay_tasks_preview": replay_tasks[:20],
            "operator_next_steps": _operator_next_steps(
                status,
                can_register=can_register,
            ),
            "safety": _safety(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_world_model_replay_review_gate_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_world_model_replay_review_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    source = payload.get("source_packet", {})
    lines = [
        "# DEAN-OS World Model Replay Review Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Gate status: `{summary.get('gate_status')}`",
        f"- Source packet: `{source.get('run_id')}`",
        f"- Source status: `{source.get('packet_status')}`",
        f"- Domain: `{source.get('domain_id')}`",
        f"- As-of: `{source.get('as_of')}`",
        f"- Replay tasks: {summary.get('replay_task_count')}",
        f"- Can register replay tasks: {summary.get('can_register_replay_tasks')}",
        f"- Registration performed now: {summary.get('replay_task_registration_performed')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Issues",
        "",
    ]
    issues = summary.get("issues") or []
    lines.extend(f"- {issue}" for issue in issues)
    if not issues:
        lines.append("- none")
    bundle = payload.get("registration_bundle")
    lines.extend(["", "## Registration Bundle", ""])
    if bundle:
        lines.extend(
            [
                f"- Bundle ID: `{bundle.get('bundle_id')}`",
                f"- Approved by: `{bundle.get('approved_by')}`",
                f"- Task count: {bundle.get('task_count')}",
                "- Status: approved candidate bundle; not yet written to a replay queue.",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _load_packet(
    packet_json: str | Path | dict[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(packet_json, dict):
        return dict(packet_json), None
    path = Path(packet_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"World-model packet must be a JSON object: {path}")
    return payload, path


def _gate_issues(
    packet: dict[str, Any],
    replay_tasks: list[dict[str, Any]],
    *,
    approve: bool,
    reviewer: str | None,
) -> list[str]:
    issues: list[str] = []
    if packet.get("contract") != WORLD_MODEL_EVENT_LEARNING_CONTRACT:
        issues.append("source_packet_contract_mismatch")
    if not packet.get("run_id"):
        issues.append("source_packet_run_id_missing")
    if not replay_tasks:
        issues.append("source_packet_has_no_replay_tasks")
    if packet.get("summary", {}).get("can_trade") is not False:
        issues.append("source_packet_trade_boundary_not_false")
    if packet.get("summary", {}).get("can_write_learning_memory") is not False:
        issues.append("source_packet_learning_write_boundary_not_false")
    if approve and not str(reviewer or "").strip():
        issues.append("reviewer_required_for_replay_registration_approval")
    if approve:
        invalid_tasks = [
            str(task.get("task_id") or index)
            for index, task in enumerate(replay_tasks)
            if task.get("manual_review_gate_required") is not True
        ]
        if invalid_tasks:
            issues.append(
                "replay_tasks_missing_manual_review_gate_required:"
                + ",".join(invalid_tasks[:10])
            )
    return issues


def _gate_status(issues: list[str], *, approve: bool) -> str:
    if issues:
        if "reviewer_required_for_replay_registration_approval" in issues:
            return "blocked_missing_reviewer_for_replay_approval"
        return "blocked_replay_registration_gate"
    if approve:
        return "replay_tasks_approved_for_registration"
    return "manual_review_required_for_replay_registration"


def _registration_bundle(
    packet: dict[str, Any],
    replay_tasks: list[dict[str, Any]],
    *,
    reviewer: str,
    review_notes: str | None,
) -> dict[str, Any]:
    return {
        "bundle_id": _run_id("world_model_replay_registration_candidate"),
        "source_packet_id": packet.get("run_id"),
        "source_packet_contract": packet.get("contract"),
        "approved_by": reviewer,
        "approved_at": utc_now_iso(),
        "review_notes": review_notes,
        "task_count": len(replay_tasks),
        "tasks": replay_tasks,
        "allowed_next_step": "operator_may_register_replay_tasks_in_replay_queue",
        "forbidden_next_steps": [
            "trade_signal",
            "position_sizing",
            "learning_memory_write_without_outcome_review",
            "model_promotion_without_calibration_gate",
        ],
        "registration_note": (
            "This bundle authorizes only replay-task registration after manual "
            "review. Outcome scoring and learning-memory writes require later "
            "evidence and review gates."
        ),
    }


def _operator_next_steps(status: str, *, can_register: bool) -> list[str]:
    if can_register:
        return [
            "Register only the approved replay tasks in the replay queue.",
            "Wait for fixed-horizon outcomes before scoring hypotheses.",
            "Run a separate outcome/calibration gate before any learning-memory write.",
        ]
    if status == "manual_review_required_for_replay_registration":
        return [
            "Review hypotheses, scenario graph, evidence gaps, and context tags.",
            "If acceptable, rerun this gate with --approve and --reviewer.",
            "Do not register replay tasks directly from the event packet.",
        ]
    return [
        "Resolve gate issues before approval.",
        "Do not register replay tasks from a blocked packet.",
    ]


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "live_execution_performed": False,
        "broker_access_performed": False,
        "production_config_write_performed": False,
        "model_promotion_performed": False,
        "learning_memory_write_performed": False,
        "outcome_registration_performed": False,
        "replay_task_registration_performed": False,
    }


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT",
    "WorldModelReplayReviewGate",
    "render_world_model_replay_review_gate_markdown",
]
