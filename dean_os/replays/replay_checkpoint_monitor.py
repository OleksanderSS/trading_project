from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


class ReplayCheckpointMonitorBuilder:
    contract = "dean_replay_checkpoint_monitor_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/replay_checkpoint_monitor_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        evidence_plan_path: str | Path,
        *,
        as_of: str,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = parse_timezone_aware(as_of)
        if cutoff is None:
            raise ValueError("monitor as_of must be timezone-aware")
        plan_path = Path(evidence_plan_path)
        plan = _load(plan_path)
        if plan.get("contract") != "dean_replay_outcome_evidence_plan_v1":
            raise ValueError("unsupported replay evidence plan contract")

        tasks = [_task_status(task, cutoff) for task in plan.get("task_plans") or []]
        counts = Counter(task["checkpoint_status"] for task in tasks)
        action_items = [
            {
                "task_id": task["task_id"],
                "checkpoint_status": task["checkpoint_status"],
                "next_checkpoint_at": task["next_checkpoint_at"],
                "actions": task["actions"],
            }
            for task in tasks
            if task["actions"]
        ]
        created_at = utc_now_iso()
        run_id = "replay_checkpoint_monitor_" + created_at.replace(":", "").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "replay_checkpoint_monitor",
            "contract": self.contract,
            "inputs": {
                "as_of": cutoff.isoformat(),
                "evidence_plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
            },
            "summary": {
                "task_count": len(tasks),
                "checkpoint_status_counts": dict(sorted(counts.items())),
                "collecting_count": counts.get("collecting", 0),
                "pre_due_review_due_count": counts.get("pre_due_source_review_due", 0),
                "outcome_review_due_count": counts.get("outcome_review_due", 0),
                "early_outcome_evaluation_allowed": False,
                "automatic_collection_allowed": False,
                "can_trade": False,
            },
            "tasks": tasks,
            "action_items": action_items,
            "monitoring_policy": {
                "before_pre_due": "Collect source evidence and preserve point-in-time lineage; do not score outcomes.",
                "pre_due": "Review whether required source lanes are covered before the outcome date.",
                "due": "Route matured task to outcome evidence review; maturity is not proof or learning approval.",
                "missed_checkpoint": "Keep the task visible and record missing coverage; never backfill with future-known evidence as if contemporaneous.",
            },
            "safety": {
                "review_only": True,
                "outcome_evaluation_performed": False,
                "collector_execution_performed": False,
                "replay_registration_performed": False,
                "learning_write_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload, markdown=_markdown(payload), run_id=run_id
            )
        return payload


def _task_status(task: dict[str, Any], cutoff: Any) -> dict[str, Any]:
    checkpoints = task.get("checkpoints") or {}
    start = parse_timezone_aware(str(checkpoints.get("collection_start") or ""))
    pre_due = parse_timezone_aware(str(checkpoints.get("pre_due_source_review") or ""))
    due = parse_timezone_aware(str(checkpoints.get("due_outcome_review") or ""))
    if start is None or pre_due is None or due is None:
        raise ValueError(f"task {task.get('task_id')} has invalid checkpoint timestamps")
    if not start <= pre_due <= due:
        raise ValueError(f"task {task.get('task_id')} checkpoint order invalid")

    if cutoff < start:
        status = "scheduled_not_started"
        next_checkpoint = start
        actions = []
    elif cutoff < pre_due:
        status = "collecting"
        next_checkpoint = pre_due
        actions = _collection_actions(task)
    elif cutoff < due:
        status = "pre_due_source_review_due"
        next_checkpoint = due
        actions = _pre_due_actions(task)
    else:
        status = "outcome_review_due"
        next_checkpoint = None
        actions = _due_actions(task)

    return {
        "task_id": task.get("task_id"),
        "hypothesis_id": task.get("hypothesis_id"),
        "horizon_days": task.get("horizon_days"),
        "checkpoint_status": status,
        "collection_start": start.isoformat(),
        "pre_due_source_review": pre_due.isoformat(),
        "due_outcome_review": due.isoformat(),
        "next_checkpoint_at": next_checkpoint.isoformat() if next_checkpoint else None,
        "days_until_due": max(0, (due - cutoff).days),
        "can_evaluate_outcome": status == "outcome_review_due",
        "evidence_lane_count": len(task.get("evidence_lanes") or []),
        "unresolved_lane_count": sum(
            lane.get("resolution_status") != "resolved"
            for lane in task.get("evidence_lanes") or []
        ),
        "actions": actions,
    }


def _collection_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    unique = {}
    for lane in task.get("evidence_lanes") or []:
        route = lane.get("collection_route") or {}
        unique.setdefault(
            str(lane.get("gap_id")),
            {
                "action_type": "continue_evidence_collection",
                "gap_id": lane.get("gap_id"),
                "description": lane.get("description"),
                "route_status": route.get("status"),
                "next_action": route.get("next_action"),
                "automatic_execution_allowed": False,
            },
        )
    return list(unique.values())


def _pre_due_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **action,
            "action_type": "pre_due_source_coverage_review",
        }
        for action in _collection_actions(task)
    ]


def _due_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "action_type": "route_to_outcome_evidence_review",
        "task_id": task.get("task_id"),
        "expected_observations": task.get("expected_observations") or [],
        "invalidation_signals": task.get("invalidation_signals") or [],
        "automatic_learning_allowed": False,
    }]


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    counts = payload["summary"]["checkpoint_status_counts"]
    return (
        "# Replay Checkpoint Monitor\n\n"
        f"- Tasks: `{payload['summary']['task_count']}`\n"
        f"- Statuses: `{json.dumps(counts, sort_keys=True)}`\n"
        "- Early outcome evaluation allowed: `false`\n"
        "- Automatic collection allowed: `false`\n"
        "- Can trade: `false`\n"
    )


__all__ = ["ReplayCheckpointMonitorBuilder"]
