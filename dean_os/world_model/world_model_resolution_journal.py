from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.world_model.world_model_review_resolution import (
    WORLD_MODEL_REVIEW_RESOLUTION_CONTRACT,
)


def append_world_model_resolution_journal(
    *,
    resolution_packet_json: str | Path,
    review_gate_json: str | Path,
    closure_json: str | Path,
    journal_path: str | Path = "data/dean_os/system_journal.jsonl",
) -> dict[str, Any]:
    packet_path = Path(resolution_packet_json)
    gate_path = Path(review_gate_json)
    closure_path = Path(closure_json)
    packet = _load(packet_path)
    gate = _load(gate_path)
    closure = _load(closure_path)
    packet_binding = artifact_binding(packet_path, packet)
    gate_binding = artifact_binding(gate_path, gate)
    closure_binding = artifact_binding(closure_path, closure)
    _verify(packet, packet_binding, gate, gate_binding, closure)
    domain_id = str(packet.get("summary", {}).get("domain_id") or "unknown")
    events: list[dict[str, Any]] = []

    for label, payload, binding in (
        ("review_resolution_packet", packet, packet_binding),
        ("resolved_hypothesis_review_gate", gate, gate_binding),
        ("resolved_cycle_closure", closure, closure_binding),
    ):
        events.append(
            _event(
                "source_snapshot_recorded",
                effective_at=str(payload.get("created_at")),
                actor="world_model_resolution_journal",
                domain_id=domain_id,
                entity_type="source_snapshot",
                entity_id=str(payload.get("run_id")),
                source_artifact=binding,
                context={"artifact_role": label},
                payload={"artifact_role": label, "contract": payload.get("contract")},
            )
        )

    hypotheses = {
        str(item.get("hypothesis_id")): item
        for item in packet.get("hypotheses", []) or []
        if item.get("hypothesis_id")
    }
    for hypothesis_id, hypothesis in hypotheses.items():
        lineage = hypothesis.get("resolution_lineage") or {}
        events.append(
            _event(
                "hypothesis_created",
                effective_at=str(packet.get("summary", {}).get("as_of")),
                actor="world_model_review_resolution",
                domain_id=domain_id,
                entity_type="hypothesis_version",
                entity_id=hypothesis_id,
                source_artifact=packet_binding,
                context={
                    "resolution_packet_run_id": packet.get("run_id"),
                    "original_hypothesis_id": hypothesis.get(
                        "original_hypothesis_id"
                    ),
                },
                payload={
                    "hypothesis": hypothesis.get("hypothesis"),
                    "claim_version": hypothesis.get("claim_version"),
                    "resolution_action": hypothesis.get("resolution_action"),
                    "resolution_status": hypothesis.get("resolution_status"),
                    "trigger_evidence_ids": hypothesis.get("trigger_evidence_ids")
                    or [],
                    "expected_observations": hypothesis.get(
                        "expected_observations"
                    )
                    or [],
                    "invalidation_signals": hypothesis.get("invalidation_signals")
                    or [],
                    "measurement_spec": hypothesis.get("measurement_spec"),
                    "registration_blockers": hypothesis.get(
                        "registration_blockers"
                    )
                    or [],
                    "resolution_lineage": lineage,
                },
            )
        )
        events.append(
            _event(
                "action_reviewed",
                effective_at=str(packet.get("created_at")),
                actor=str(
                    packet.get("source_review_resolution", {}).get("reviewer")
                    or "manual_reviewer"
                ),
                domain_id=domain_id,
                entity_type="hypothesis_resolution",
                entity_id=str(lineage.get("resolution_id") or hypothesis_id),
                source_artifact=packet_binding,
                context={
                    "original_hypothesis_id": hypothesis.get(
                        "original_hypothesis_id"
                    ),
                    "resolved_hypothesis_id": hypothesis_id,
                },
                payload={
                    "action": hypothesis.get("resolution_action"),
                    "claim_version": hypothesis.get("claim_version"),
                    "execution_scope": "new_review_artifact_only",
                    "production_update_performed": False,
                    "replay_registration_performed": False,
                },
            )
        )

    for review in gate.get("hypothesis_review", []) or []:
        hypothesis_id = str(review.get("hypothesis_id"))
        events.append(
            _event(
                "hypothesis_reviewed",
                effective_at=str(gate.get("created_at")),
                actor=str(
                    gate.get("operator_decision", {}).get("reviewer")
                    or "manual_reviewer"
                ),
                domain_id=domain_id,
                entity_type="hypothesis",
                entity_id=hypothesis_id,
                source_artifact=gate_binding,
                context={
                    "resolution_packet_run_id": packet.get("run_id"),
                    "review_gate_run_id": gate.get("run_id"),
                },
                payload={
                    "hypothesis": review.get("hypothesis"),
                    "disposition": review.get("disposition"),
                    "rationale": review.get("rationale"),
                    "source_assessment": review.get("source_assessment"),
                    "claim_version": review.get("claim_version"),
                    "registration_blockers": review.get("registration_blockers")
                    or [],
                    "quality_assessment": review.get("quality_assessment"),
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
                "resolution_packet_run_id": packet.get("run_id"),
                "review_gate_run_id": gate.get("run_id"),
            },
            payload={
                "closure_status": closure.get("summary", {}).get("closure_status"),
                "decision_state": closure.get("summary", {}).get(
                    "current_cycle_decision_state"
                ),
                "can_register_new_replay_tasks": closure.get("summary", {}).get(
                    "can_register_new_replay_tasks"
                ),
                "can_write_learning_memory": closure.get("summary", {}).get(
                    "can_write_learning_memory"
                ),
                "can_trade": closure.get("summary", {}).get("can_trade"),
            },
        )
    )
    journal = SystemJournal(journal_path)
    write_result = journal.append_many(events)
    return {
        "contract": "dean_world_model_resolution_journal_append_v1",
        "resolution_packet_run_id": packet.get("run_id"),
        "requested_event_count": len(events),
        "write_result": write_result,
        "journal_status": journal.status(),
        "action_execution_performed": False,
        "learning_memory_write_performed": False,
        "production_rule_update_performed": False,
        "can_trade": False,
    }


def _verify(
    packet: dict[str, Any],
    packet_binding: dict[str, Any],
    gate: dict[str, Any],
    gate_binding: dict[str, Any],
    closure: dict[str, Any],
) -> None:
    if packet.get("review_resolution_contract") != WORLD_MODEL_REVIEW_RESOLUTION_CONTRACT:
        raise ValueError("packet is not a world-model review resolution")
    gate_source = gate.get("source_packet") or {}
    if gate_source.get("run_id") != packet.get("run_id"):
        raise ValueError("resolved gate points to a different resolution packet")
    if gate_source.get("sha256") != packet_binding.get("sha256"):
        raise ValueError("resolution packet changed after resolved review")
    closure_inputs = closure.get("inputs") or {}
    for label, recorded, actual in (
        ("world model", closure_inputs.get("world_model") or {}, packet_binding),
        ("review gate", closure_inputs.get("replay_review_gate") or {}, gate_binding),
    ):
        if recorded.get("sha256") != actual.get("sha256"):
            raise ValueError(f"resolved closure {label} hash binding mismatch")
    if closure.get("summary", {}).get("can_register_new_replay_tasks") is not False:
        raise ValueError("resolution journal expects a non-registering closure")


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


__all__ = ["append_world_model_resolution_journal"]
