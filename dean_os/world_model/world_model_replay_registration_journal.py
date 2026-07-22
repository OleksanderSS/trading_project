from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.world_model.world_model_replay_registration import (
    WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT,
)
from dean_os.world_model.world_model_replay_review_gate import (
    WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
)


def append_world_model_replay_registration_journal(
    *,
    review_gate_json: str | Path,
    registration_json: str | Path,
    closure_json: str | Path,
    journal_path: str | Path = "data/dean_os/system_journal.jsonl",
) -> dict[str, Any]:
    gate_path = Path(review_gate_json)
    registration_path = Path(registration_json)
    closure_path = Path(closure_json)
    gate = _load(gate_path)
    registration = _load(registration_path)
    closure = _load(closure_path)
    gate_binding = artifact_binding(gate_path, gate)
    registration_binding = artifact_binding(registration_path, registration)
    closure_binding = artifact_binding(closure_path, closure)
    _verify(
        gate=gate,
        gate_binding=gate_binding,
        registration=registration,
        registration_binding=registration_binding,
        closure=closure,
    )
    domain_id = str(gate.get("source_packet", {}).get("domain_id") or "unknown")
    created_at = str(registration.get("created_at"))
    events: list[dict[str, Any]] = []

    for role, payload, binding in (
        ("approved_replay_review_gate", gate, gate_binding),
        ("applied_replay_registration", registration, registration_binding),
        ("post_registration_cycle_closure", closure, closure_binding),
    ):
        events.append(
            _event(
                "source_snapshot_recorded",
                effective_at=str(payload.get("created_at")),
                actor="world_model_replay_registration_journal",
                domain_id=domain_id,
                entity_type="source_snapshot",
                entity_id=str(payload.get("run_id")),
                source_artifact=binding,
                context={"artifact_role": role},
                payload={"artifact_role": role, "contract": payload.get("contract")},
            )
        )

    bundle = gate.get("registration_bundle") or {}
    events.append(
        _event(
            "action_reviewed",
            effective_at=str(bundle.get("approved_at")),
            actor=str(bundle.get("approved_by")),
            domain_id=domain_id,
            entity_type="replay_registration_bundle",
            entity_id=str(bundle.get("bundle_id")),
            source_artifact=gate_binding,
            context={"review_gate_run_id": gate.get("run_id")},
            payload={
                "action": "approve_observation_only_replay_registration",
                "approved_task_count": bundle.get("task_count"),
                "review_notes": bundle.get("review_notes"),
                "trading_authorized": False,
                "learning_memory_write_authorized": False,
            },
        )
    )

    for item in registration.get("registered_events", []) or []:
        events.append(_registered_event(item, created_at, domain_id, registration_binding))
    for item in registration.get("skipped_existing_events", []) or []:
        events.append(_registered_event(item, created_at, domain_id, registration_binding))
    for item in registration.get("deferred_historical_tasks", []) or []:
        events.append(
            _event(
                "action_proposed",
                effective_at=created_at,
                actor="world_model_replay_registration_bridge",
                domain_id=domain_id,
                entity_type="historical_point_in_time_review",
                entity_id=str(item.get("task_id")),
                source_artifact=registration_binding,
                context={"registration_run_id": registration.get("run_id")},
                payload={
                    "action": "prepare_historical_point_in_time_outcome_review",
                    "task_id": item.get("task_id"),
                    "due_at": item.get("due_at"),
                    "status": item.get("status"),
                    "execution_status": "not_scored_pending_verified_historical_evidence",
                },
            )
        )

    events.append(
        _event(
            "governance_closure_recorded",
            effective_at=str(closure.get("created_at")),
            actor="full_system_cycle_closure",
            domain_id=domain_id,
            entity_type="cycle_closure",
            entity_id=str(closure.get("run_id")),
            source_artifact=closure_binding,
            context={
                "review_gate_run_id": gate.get("run_id"),
                "registration_run_id": registration.get("run_id"),
            },
            payload={
                "closure_status": closure.get("summary", {}).get("closure_status"),
                "registered_or_existing_replay_task_count": closure.get("summary", {}).get(
                    "registered_or_existing_replay_task_count"
                ),
                "historical_review_required_replay_task_count": closure.get("summary", {}).get(
                    "historical_review_required_replay_task_count"
                ),
                "outcome_scoring_performed": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
        )
    )
    journal = SystemJournal(journal_path)
    write_result = journal.append_many(events)
    return {
        "contract": "dean_world_model_replay_registration_journal_append_v1",
        "requested_event_count": len(events),
        "write_result": write_result,
        "journal_status": journal.status(),
        "replay_registration_observed": True,
        "outcome_scoring_performed": False,
        "learning_memory_write_performed": False,
        "production_rule_update_performed": False,
        "can_trade": False,
    }


def _registered_event(
    item: dict[str, Any],
    effective_at: str,
    domain_id: str,
    source_artifact: dict[str, Any],
) -> dict[str, Any]:
    return _event(
        "action_executed",
        effective_at=effective_at,
        actor="world_model_replay_registration_bridge",
        domain_id=domain_id,
        entity_type="outcome_tracker_replay_task",
        entity_id=str(item.get("task_id")),
        source_artifact=source_artifact,
        context={"outcome_tracker_event_id": item.get("event_id")},
        payload={
            "action": "register_observation_only_replay_task",
            "task_id": item.get("task_id"),
            "event_id": item.get("event_id"),
            "source": item.get("source"),
            "status": item.get("status"),
            "outcome_scoring_performed": False,
            "trading_performed": False,
        },
    )


def _verify(
    *,
    gate: dict[str, Any],
    gate_binding: dict[str, Any],
    registration: dict[str, Any],
    registration_binding: dict[str, Any],
    closure: dict[str, Any],
) -> None:
    if gate.get("contract") != WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT:
        raise ValueError("unsupported replay review gate contract")
    if gate.get("summary", {}).get("can_register_replay_tasks") is not True:
        raise ValueError("replay review gate did not authorize registration")
    if registration.get("contract") != WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT:
        raise ValueError("unsupported replay registration contract")
    source_gate = registration.get("source_gate") or {}
    if source_gate.get("run_id") != gate.get("run_id"):
        raise ValueError("registration points to a different review gate")
    if source_gate.get("sha256") != gate_binding.get("sha256"):
        raise ValueError("review gate changed after registration")
    summary = registration.get("summary") or {}
    if summary.get("apply_requested") is not True or summary.get("issue_count") != 0:
        raise ValueError("registration artifact is not a clean applied result")
    inputs = closure.get("inputs") or {}
    for label, recorded, actual in (
        ("review gate", inputs.get("replay_review_gate") or {}, gate_binding),
        ("registration", inputs.get("replay_registration") or {}, registration_binding),
    ):
        if recorded.get("sha256") != actual.get("sha256"):
            raise ValueError(f"closure {label} hash binding mismatch")
    if closure.get("summary", {}).get("replay_registration_observed") is not True:
        raise ValueError("closure does not observe applied replay registration")


def _event(
    event_type: str,
    *,
    effective_at: str,
    actor: str,
    domain_id: str,
    entity_type: str,
    entity_id: str,
    source_artifact: dict[str, Any],
    context: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "effective_at": effective_at,
        "actor": actor,
        "domain_id": domain_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "source_artifact": source_artifact,
        "context": context,
        "payload": payload,
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


__all__ = ["append_world_model_replay_registration_journal"]
