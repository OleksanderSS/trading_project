from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.outcome_tracker import OutcomeTracker
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from dean_os.world_model.world_model_event_learning import WORLD_MODEL_EVENT_LEARNING_CONTRACT
from dean_os.world_model.world_model_replay_review_gate import (
    WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
)

WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT = "dean_world_model_replay_registration_bridge_v1"
DEFAULT_OUTCOME_TRACKER_DB = Path("data/dean_os/outcome_tracker.sqlite")


class WorldModelReplayRegistrationBridge:
    """Consumes approved world-model replay bundles and registers tracker events.

    This bridge is deliberately narrower than a learning loop. It can register
    approved replay tasks in OutcomeTracker, but it does not score outcomes,
    write learning memory, promote models, recommend trades, or touch broker
    state. Without ``apply=True`` it produces a dry-run plan only.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/world_model_replay_registration",
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        gate_json: str | Path | dict[str, Any],
        *,
        source_packet_json: str | Path | dict[str, Any] | None = None,
        tracker_db_path: str | Path = DEFAULT_OUTCOME_TRACKER_DB,
        apply: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        gate, gate_source_path = _load_json_object(gate_json, "replay review gate")
        source_packet, source_packet_path = _load_source_packet(
            gate,
            source_packet_json=source_packet_json,
        )
        bundle = gate.get("registration_bundle") or {}
        tasks = list(bundle.get("tasks") or [])
        issues = _bridge_issues(gate, bundle, tasks)
        issues.extend(
            _source_packet_binding_issues(
                gate,
                bundle,
                source_packet_path,
            )
        )
        plan = _registration_plan(gate, bundle, tasks, source_packet)
        tracker_path = Path(tracker_db_path)

        registered: list[dict[str, Any]] = []
        skipped_existing: list[dict[str, Any]] = []
        deferred_historical: list[dict[str, Any]] = []
        apply_attempted = bool(apply)
        if apply and not issues:
            tracker = OutcomeTracker(tracker_path)
            for item in plan:
                if item.get("outcome_evaluation_mode") == (
                    "historical_point_in_time_outcome_review_required"
                ):
                    deferred_historical.append(
                        {
                            "task_id": item["task_id"],
                            "source": item["source"],
                            "due_at": item.get("due_at"),
                            "status": "deferred_to_historical_point_in_time_outcome_review",
                        }
                    )
                    continue
                existing_event_id = _existing_event_id(tracker_path, item["source"])
                if existing_event_id:
                    skipped_existing.append(
                        {
                            "task_id": item["task_id"],
                            "source": item["source"],
                            "event_id": existing_event_id,
                            "status": "skipped_existing_outcome_tracker_event",
                        }
                    )
                    continue
                event_id = tracker.register(
                    headline=item["headline"],
                    event_type=item["event_type"],
                    shock=item["shock"],
                    impact_estimate=item["impact_estimate"],
                    confidence=item["confidence"],
                    sectors=item["sectors"],
                    source=item["source"],
                    directions=item["directions"],
                    intervals=item["tracker_intervals"],
                    registered_at=item["event_anchor_at"],
                )
                registered.append(
                    {
                        "task_id": item["task_id"],
                        "source": item["source"],
                        "event_id": event_id,
                        "status": "registered_in_outcome_tracker",
                    }
                )

        status = _bridge_status(
            issues,
            apply=apply,
            registered_count=len(registered),
            skipped_existing_count=len(skipped_existing),
            deferred_historical_count=len(deferred_historical),
        )
        payload = {
            "run_id": _run_id("world_model_replay_registration"),
            "created_at": utc_now_iso(),
            "mode": "world_model_replay_registration_bridge",
            "contract": WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT,
            "source_gate": {
                "path": str(gate_source_path) if gate_source_path else None,
                "sha256": (
                    _sha256(gate_source_path)
                    if gate_source_path is not None and gate_source_path.is_file()
                    else None
                ),
                "run_id": gate.get("run_id"),
                "contract": gate.get("contract"),
                "gate_status": gate.get("summary", {}).get("gate_status"),
            },
            "source_packet": {
                "path": (
                    str(source_packet_path)
                    if source_packet_path
                    else gate.get("source_packet", {}).get("path")
                ),
                "run_id": (
                    source_packet.get("run_id")
                    if source_packet
                    else gate.get("source_packet", {}).get("run_id")
                ),
                "contract": (
                    source_packet.get("contract")
                    if source_packet
                    else gate.get("source_packet", {}).get("contract")
                ),
                "domain_id": gate.get("source_packet", {}).get("domain_id"),
                "as_of": gate.get("source_packet", {}).get("as_of"),
                "loaded_for_enrichment": source_packet is not None,
                "sha256": (
                    _sha256(source_packet_path)
                    if source_packet_path is not None and source_packet_path.is_file()
                    else None
                ),
            },
            "registration_bundle": {
                "bundle_id": bundle.get("bundle_id"),
                "approved_by": bundle.get("approved_by"),
                "approved_at": bundle.get("approved_at"),
                "task_count": bundle.get("task_count"),
            },
            "summary": {
                "bridge_status": status,
                "issue_count": len(issues),
                "issues": issues,
                "warning_count": len(_plan_warnings(plan)),
                "warnings": _plan_warnings(plan),
                "apply_requested": apply_attempted,
                "dry_run": not apply_attempted,
                "planned_registration_count": len(plan),
                "prospective_registration_count": sum(
                    item.get("outcome_evaluation_mode")
                    == "prospective_checkpoint_review"
                    for item in plan
                ),
                "historical_review_required_count": sum(
                    item.get("outcome_evaluation_mode")
                    == "historical_point_in_time_outcome_review_required"
                    for item in plan
                ),
                "registered_count": len(registered),
                "skipped_existing_count": len(skipped_existing),
                "registered_or_existing_count": len(registered)
                + len(skipped_existing),
                "deferred_historical_count": len(deferred_historical),
                "outcome_tracker_path": str(tracker_path),
                "outcome_tracker_registration_performed": bool(registered),
                "outcome_scoring_performed": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "registration_plan": plan,
            "registered_events": registered,
            "skipped_existing_events": skipped_existing,
            "deferred_historical_tasks": deferred_historical,
            "operator_next_steps": _operator_next_steps(status, apply=apply),
            "safety": _safety(outcome_tracker_write_performed=bool(registered)),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_world_model_replay_registration_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_world_model_replay_registration_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    gate = payload.get("source_gate", {})
    bundle = payload.get("registration_bundle", {})
    lines = [
        "# DEAN-OS World Model Replay Registration Bridge",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Bridge status: `{summary.get('bridge_status')}`",
        f"- Source gate: `{gate.get('run_id')}`",
        f"- Gate status: `{gate.get('gate_status')}`",
        f"- Bundle ID: `{bundle.get('bundle_id')}`",
        f"- Planned registrations: {summary.get('planned_registration_count')}",
        f"- Apply requested: {summary.get('apply_requested')}",
        f"- Registered now: {summary.get('registered_count')}",
        f"- Skipped existing: {summary.get('skipped_existing_count')}",
        f"- Historical review required: {summary.get('historical_review_required_count')}",
        f"- Deferred from live tracker: {summary.get('deferred_historical_count')}",
        f"- Outcome scoring performed: {summary.get('outcome_scoring_performed')}",
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

    lines.extend(["", "## Warnings", ""])
    warnings = summary.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- none")

    lines.extend(["", "## Registration Plan Preview", ""])
    for item in payload.get("registration_plan", [])[:10]:
        lines.extend(
            [
                f"- `{item.get('task_id')}` → `{item.get('event_type')}`",
                f"  - horizon: {item.get('horizon_days')}d",
                f"  - projected direction: {item.get('predicted_direction')}",
                f"  - source: `{item.get('source')}`",
            ]
        )
    if not payload.get("registration_plan"):
        lines.append("- none")

    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _load_json_object(
    value: str | Path | dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, dict):
        return dict(value), None
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object: {path}")
    return payload, path


def _load_source_packet(
    gate: dict[str, Any],
    *,
    source_packet_json: str | Path | dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if source_packet_json is not None:
        return _load_json_object(source_packet_json, "world-model source packet")
    source_path = gate.get("source_packet", {}).get("path")
    if not source_path:
        return None, None
    path = Path(source_path)
    if not path.exists():
        return None, path
    return _load_json_object(path, "world-model source packet")


def _bridge_issues(
    gate: dict[str, Any],
    bundle: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    if gate.get("contract") != WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT:
        issues.append("source_gate_contract_mismatch")
    summary = gate.get("summary", {})
    if summary.get("gate_status") != "replay_tasks_approved_for_registration":
        issues.append("source_gate_not_approved_for_registration")
    if summary.get("can_register_replay_tasks") is not True:
        issues.append("source_gate_can_register_replay_tasks_not_true")
    if not bundle:
        issues.append("registration_bundle_missing")
    if bundle and bundle.get("source_packet_contract") != WORLD_MODEL_EVENT_LEARNING_CONTRACT:
        issues.append("registration_bundle_source_packet_contract_mismatch")
    if not tasks:
        issues.append("registration_bundle_has_no_tasks")
    declared_count = bundle.get("task_count")
    if declared_count is not None and int(declared_count) != len(tasks):
        issues.append("registration_bundle_task_count_mismatch")
    invalid_manual_gate = [
        str(task.get("task_id") or index)
        for index, task in enumerate(tasks)
        if task.get("manual_review_gate_required") is not True
    ]
    if invalid_manual_gate:
        issues.append(
            "tasks_missing_manual_review_gate_required:"
            + ",".join(invalid_manual_gate[:10])
        )
    allowed_candidate_statuses = {
        "candidate_pending_manual_review",
        "candidate_pending_new_manual_review",
    }
    invalid_status = [
        str(task.get("task_id") or index)
        for index, task in enumerate(tasks)
        if task.get("registration_status") not in allowed_candidate_statuses
    ]
    if invalid_status:
        issues.append(
            "tasks_not_candidate_pending_manual_review:"
            + ",".join(invalid_status[:10])
        )
    invalid_horizon = [
        str(task.get("task_id") or index)
        for index, task in enumerate(tasks)
        if _positive_int(task.get("horizon_days")) is None
    ]
    if invalid_horizon:
        issues.append("tasks_missing_positive_horizon:" + ",".join(invalid_horizon[:10]))
    return issues


def _source_packet_binding_issues(
    gate: dict[str, Any],
    bundle: dict[str, Any],
    source_packet_path: Path | None,
) -> list[str]:
    expected = (
        bundle.get("source_packet_sha256")
        or (gate.get("source_packet") or {}).get("sha256")
    )
    if not expected:
        return []
    if source_packet_path is None or not source_packet_path.is_file():
        return ["hash_bound_source_packet_missing"]
    if _sha256(source_packet_path) != expected:
        return ["source_packet_sha256_mismatch"]
    return []


def _registration_plan(
    gate: dict[str, Any],
    bundle: dict[str, Any],
    tasks: list[dict[str, Any]],
    source_packet: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    hypotheses = _hypotheses_by_id(source_packet)
    domain_id = (
        gate.get("source_packet", {}).get("domain_id")
        or (source_packet or {}).get("summary", {}).get("domain_id")
        or "global"
    )
    plan: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        hypothesis_id = str(task.get("hypothesis_id") or "")
        hypothesis = hypotheses.get(hypothesis_id, {})
        direction, direction_source = _predicted_direction(task, hypothesis)
        horizon = _positive_int(task.get("horizon_days")) or 0
        confidence = _bounded_float(hypothesis.get("confidence"), default=0.5)
        source = _tracker_source(bundle, task)
        plan.append(
            {
                "task_id": task_id,
                "hypothesis_id": hypothesis_id,
                "bundle_id": bundle.get("bundle_id"),
                "source_packet_id": bundle.get("source_packet_id"),
                "scenario_graph_id": task.get("scenario_graph_id"),
                "as_of": task.get("as_of") or gate.get("source_packet", {}).get("as_of"),
                "packet_as_of": task.get("packet_as_of")
                or gate.get("source_packet", {}).get("as_of"),
                "event_anchor_at": task.get("trigger_event_at")
                or task.get("as_of"),
                "horizon_days": horizon,
                "due_at": task.get("due_at"),
                "checkpoint_state_at_packet": task.get(
                    "checkpoint_state_at_packet"
                ),
                "outcome_evaluation_mode": (
                    "historical_point_in_time_outcome_review_required"
                    if task.get("checkpoint_state_at_packet") == "matured"
                    else "prospective_checkpoint_review"
                ),
                "event_type": "world_model_replay_task",
                "headline": _headline(task, hypothesis, horizon),
                "shock": _shock_from_direction(direction),
                "impact_estimate": 0.0,
                "confidence": confidence,
                "sectors": _sectors(task, domain_id),
                "source": source,
                "directions": {horizon: direction},
                "tracker_intervals": [horizon],
                "predicted_direction": direction,
                "predicted_direction_source": direction_source,
                "projection_limitation": _projection_limitation(direction_source),
                "outcome_tracker_semantics": (
                    "This checkpoint registers exactly its requested horizon from the "
                    "trigger-event timestamp. World-model replay hypotheses may be "
                    "non-directional, so neutral projection is used unless an explicit "
                    "direction is present."
                ),
                "pipeline_context_snapshot": task.get("pipeline_context_snapshot") or {},
                "allowed_update_after_review": task.get("allowed_update_after_review") or [],
                "forbidden_update": task.get("forbidden_update") or [],
            }
        )
    return plan


def _hypotheses_by_id(source_packet: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not source_packet:
        return {}
    return {
        str(item.get("hypothesis_id")): item
        for item in source_packet.get("hypotheses", []) or []
        if isinstance(item, dict) and item.get("hypothesis_id")
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predicted_direction(
    task: dict[str, Any],
    hypothesis: dict[str, Any],
) -> tuple[str, str]:
    for source_name, source in (("task", task), ("hypothesis", hypothesis)):
        for key in (
            "expected_direction",
            "predicted_direction",
            "market_direction",
            "direction",
        ):
            value = _normalize_direction(source.get(key))
            if value:
                return value, f"explicit_{source_name}_{key}"
    return "neutral", "neutral_projection_no_explicit_direction"


def _normalize_direction(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"bullish", "up", "positive", "long", "constructive"}:
        return "bullish"
    if text in {"bearish", "down", "negative", "short", "risk_off"}:
        return "bearish"
    if text in {"neutral", "mixed", "flat", "unclear", "insufficient_data"}:
        return "neutral"
    return None


def _shock_from_direction(direction: str) -> str:
    if direction == "bullish":
        return "positive"
    if direction == "bearish":
        return "negative"
    return "neutral"


def _projection_limitation(direction_source: str) -> str:
    if direction_source.startswith("explicit_"):
        return "explicit_direction_available"
    return (
        "non_directional_hypothesis_registered_as_neutral_projection; use later "
        "manual outcome review before treating calibration as signal quality"
    )


def _headline(
    task: dict[str, Any],
    hypothesis: dict[str, Any],
    horizon: int,
) -> str:
    hypothesis_text = str(hypothesis.get("hypothesis") or "").strip()
    if hypothesis_text:
        base = hypothesis_text
    else:
        base = str(task.get("hypothesis_id") or task.get("task_id") or "world_model_replay")
    headline = f"World model replay {horizon}d: {base}"
    return headline[:500]


def _sectors(task: dict[str, Any], domain_id: str) -> list[str]:
    sectors: list[str] = []
    context = task.get("pipeline_context_snapshot") or {}
    for tag in context.get("context_tags") or []:
        text = str(tag)
        if ":" in text:
            prefix, value = text.split(":", 1)
            if prefix in {"sector", "domain"} and value:
                sectors.append(value)
    if domain_id:
        sectors.append(str(domain_id))
    cleaned: list[str] = []
    for sector in sectors or ["global"]:
        normalized = sector.strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned[:8]


def _tracker_source(bundle: dict[str, Any], task: dict[str, Any]) -> str:
    bundle_id = str(bundle.get("bundle_id") or "unknown_bundle")
    task_id = str(task.get("task_id") or "unknown_task")
    return f"world_model_replay|bundle={bundle_id}|task={task_id}"


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _existing_event_id(db_path: Path, source: str) -> str | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(str(db_path)) as con:
        try:
            row = con.execute(
                "SELECT event_id FROM tracked_events WHERE source = ? ORDER BY registered_at DESC LIMIT 1",
                (source,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return str(row[0]) if row else None


def _plan_warnings(plan: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if any(
        item.get("predicted_direction_source")
        == "neutral_projection_no_explicit_direction"
        for item in plan
    ):
        warnings.append("some_replay_tasks_registered_as_neutral_projection")
    if any(
        item.get("outcome_evaluation_mode")
        == "historical_point_in_time_outcome_review_required"
        for item in plan
    ):
        warnings.append(
            "matured_checkpoints_require_historical_point_in_time_outcome_review"
        )
    return warnings


def _bridge_status(
    issues: list[str],
    *,
    apply: bool,
    registered_count: int,
    skipped_existing_count: int,
    deferred_historical_count: int,
) -> str:
    if issues:
        return "blocked_world_model_replay_registration_bridge"
    if not apply:
        return "dry_run_ready_for_outcome_tracker_registration"
    if registered_count and deferred_historical_count:
        return "outcome_tracker_registration_partially_applied_historical_review_required"
    if registered_count:
        return "outcome_tracker_registration_applied"
    if skipped_existing_count and deferred_historical_count:
        return "outcome_tracker_registration_already_applied_historical_review_required"
    if deferred_historical_count:
        return "historical_outcome_review_required_no_prospective_registration"
    if skipped_existing_count:
        return "outcome_tracker_registration_already_applied"
    return "outcome_tracker_registration_noop"


def _operator_next_steps(status: str, *, apply: bool) -> list[str]:
    if status == "dry_run_ready_for_outcome_tracker_registration":
        return [
            "Review the registration plan and neutral-projection warnings.",
            "If acceptable, rerun with --apply and an explicit tracker DB path.",
            "Do not treat tracker registration as outcome scoring or learning promotion.",
        ]
    if status == "outcome_tracker_registration_applied":
        return [
            "Wait until fixed OutcomeTracker horizons become due.",
            "Run a separate outcome-check/calibration review before learning-memory writes.",
            "Use registered event sources to trace each outcome back to bundle_id and task_id.",
        ]
    if status == (
        "outcome_tracker_registration_partially_applied_historical_review_required"
    ):
        return [
            "Monitor only the newly registered future checkpoints from their trigger timestamps.",
            "Route matured checkpoints to a point-in-time historical outcome review.",
            "Do not score matured checkpoints from the current market stance.",
        ]
    if status == "historical_outcome_review_required_no_prospective_registration":
        return [
            "Route every matured checkpoint to a point-in-time historical outcome review.",
            "Do not backdate a live tracker event and score it from the current stance.",
        ]
    if status == "outcome_tracker_registration_already_applied":
        return [
            "No new tracker records were needed; matching bundle/task sources already exist.",
            "Continue with due-outcome checks only after the relevant horizons elapse.",
        ]
    return [
        "Resolve bridge issues before applying registration.",
        "Do not bypass the approved replay review gate.",
    ]


def _safety(*, outcome_tracker_write_performed: bool) -> dict[str, bool]:
    return {
        "network_access_performed": False,
        "live_execution_performed": False,
        "broker_access_performed": False,
        "production_config_write_performed": False,
        "model_promotion_performed": False,
        "learning_memory_write_performed": False,
        "outcome_scoring_performed": False,
        "outcome_tracker_write_performed": outcome_tracker_write_performed,
        "trade_signal_created": False,
        "paper_trade_created": False,
    }


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "DEFAULT_OUTCOME_TRACKER_DB",
    "WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT",
    "WorldModelReplayRegistrationBridge",
    "render_world_model_replay_registration_markdown",
]
