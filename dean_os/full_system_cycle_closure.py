from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.accumulation_authorization_ledger import AccumulationAuthorizationLedger
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.world_model_cycle_binding import verify_world_model_cycle_binding
from dean_os.world_model.world_model_replay_registration import (
    WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT,
)


class FullSystemCycleClosureBuilder:
    contract = "dean_full_system_cycle_closure_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/full_system_cycle_closure_current",
        authorization_ledger_path: str | Path = "data/dean_os/accumulation_authorization_ledger.jsonl",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.authorization_ledger_path = Path(authorization_ledger_path)

    def build(
        self,
        *,
        cycle_path: str | Path,
        world_model_path: str | Path,
        prior_checkpoint_monitor_path: str | Path,
        replay_review_gate_path: str | Path | None = None,
        replay_registration_path: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        cycle_file = Path(cycle_path)
        world_file = Path(world_model_path)
        monitor_file = Path(prior_checkpoint_monitor_path)
        cycle = _load(cycle_file)
        world = _load(world_file)
        monitor = _load(monitor_file)
        verify_world_model_cycle_binding(cycle_file, cycle, world_file, world)
        review_gate_file = (
            Path(replay_review_gate_path)
            if replay_review_gate_path is not None
            else None
        )
        review_gate = (
            _load(review_gate_file) if review_gate_file is not None else None
        )
        review_gate_status = _verify_replay_review_gate_binding(
            world_file,
            world,
            review_gate_file,
            review_gate,
        )
        registration_file = (
            Path(replay_registration_path)
            if replay_registration_path is not None
            else None
        )
        registration = (
            _load(registration_file) if registration_file is not None else None
        )
        registration_summary = _verify_replay_registration_binding(
            review_gate_file,
            review_gate,
            registration_file,
            registration,
        )
        if monitor.get("contract") != "dean_replay_checkpoint_monitor_v1":
            raise ValueError("unsupported prior checkpoint monitor contract")
        new_hypotheses = int(world.get("summary", {}).get("hypothesis_count") or 0)
        new_replay_tasks = int(world.get("summary", {}).get("replay_task_count") or 0)
        existing_tasks = int(monitor.get("summary", {}).get("task_count") or 0)
        ledger = AccumulationAuthorizationLedger(self.authorization_ledger_path).status()
        registration_observed = int(
            registration_summary.get("registered_or_existing_count") or 0
        )
        deferred_historical = int(
            registration_summary.get("deferred_historical_count") or 0
        )
        decision_state = (
            "partially_registered_historical_review_required"
            if registration_observed and deferred_historical
            else _decision_state(new_hypotheses, review_gate_status)
        )
        closure_status = (
            "current_cycle_replay_partially_registered_historical_review_required"
            if registration_observed and deferred_historical
            else _closure_status(new_replay_tasks, review_gate_status)
        )
        can_register = (
            new_replay_tasks > 0
            and review_gate_status == "replay_tasks_approved_for_registration"
            and registration_observed == 0
        )
        created_at = utc_now_iso()
        payload = {
            "run_id": "full_system_cycle_closure_" + created_at.replace(":", "").replace("+00:00", "Z"),
            "created_at": created_at,
            "mode": "full_system_cycle_closure",
            "contract": self.contract,
            "inputs": {
                "cycle": _binding(cycle_file),
                "world_model": _binding(world_file),
                "prior_checkpoint_monitor": _binding(monitor_file),
                "authorization_ledger": {
                    "path": str(self.authorization_ledger_path),
                    "record_count": ledger["record_count"],
                    "chain_valid": ledger["chain_valid"],
                },
                "replay_review_gate": (
                    _binding(review_gate_file)
                    if review_gate_file is not None
                    else None
                ),
                "replay_registration": (
                    _binding(registration_file)
                    if registration_file is not None
                    else None
                ),
            },
            "summary": {
                "closure_status": closure_status,
                "current_cycle_decision_state": decision_state,
                "current_cycle_world_model_hash_bound": True,
                "current_cycle_hypothesis_count": new_hypotheses,
                "current_cycle_new_replay_task_count": new_replay_tasks,
                "prior_lineage_monitoring_task_count": existing_tasks,
                "prior_tasks_promoted_to_current_cycle": False,
                "authorization_ledger_record_count": ledger["record_count"],
                "can_submit_new_replay_tasks_for_manual_review": (
                    new_replay_tasks > 0
                    and review_gate_status
                    in {
                        "not_supplied",
                        "manual_review_required_for_replay_registration",
                    }
                ),
                "manual_hypothesis_review_complete": review_gate_status.startswith(
                    "hypothesis_review_complete_"
                )
                or review_gate_status == "replay_tasks_approved_for_registration",
                "replay_review_gate_status": review_gate_status,
                "can_register_new_replay_tasks": can_register,
                "approved_replay_task_count": int(
                    registration_summary.get("planned_registration_count") or 0
                ),
                "registered_or_existing_replay_task_count": registration_observed,
                "historical_review_required_replay_task_count": deferred_historical,
                "replay_registration_artifact_status": registration_summary.get(
                    "bridge_status"
                ),
                "replay_registration_observed": registration_observed > 0,
                "outcome_scoring_performed": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "branch_updates": [
                {
                    "branch_id": "world_model",
                    "status": "completed_current_cycle_hash_bound",
                    "artifact_sha256": _sha256(world_file),
                },
                {
                    "branch_id": "replay_evaluation",
                    "status": _replay_branch_status(
                        new_replay_tasks,
                        review_gate_status,
                        registration_observed=registration_observed,
                        deferred_historical=deferred_historical,
                    ),
                    "new_task_count": new_replay_tasks,
                    "prior_task_count": existing_tasks,
                },
                {
                    "branch_id": "governance_review",
                    "status": decision_state,
                    "reason": _governance_reason(
                        new_hypotheses,
                        review_gate_status,
                    ),
                },
                {
                    "branch_id": "operations_authorization",
                    "status": "ledger_observed",
                    "record_count": ledger["record_count"],
                    "chain_valid": ledger["chain_valid"],
                },
                {
                    "branch_id": "system_audit",
                    "status": "closure_manifest_assembled",
                },
            ],
            "lineage_policy": {
                "prior_replay_tasks": "continue monitoring under their original hashes",
                "current_cycle": "must not inherit prior hypotheses or replay tasks without an explicit reviewed bridge",
                "zero_hypotheses": "valid needs_more_data outcome, not a runtime failure",
            },
            "safety": {
                "review_only": True,
                "prior_replay_relabel_performed": False,
                "replay_registration_performed": registration_observed > 0,
                "authorization_write_performed": False,
                "learning_write_performed": False,
                "broker_access_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=payload["run_id"],
            )
        return payload


def _decision_state(new_hypotheses: int, review_gate_status: str) -> str:
    if new_hypotheses == 0:
        return "needs_more_data"
    return {
        "hypothesis_review_complete_reformulation_required": "reformulation_required",
        "hypothesis_review_complete_deferred": "deferred_pending_evidence",
        "hypothesis_review_complete_no_replay": "no_action",
        "hypothesis_review_complete_registration_not_requested": (
            "reviewed_pending_operator_approval"
        ),
        "replay_tasks_approved_for_registration": "approved_for_registration",
    }.get(review_gate_status, "ready_for_hypothesis_review")


def _closure_status(new_replay_tasks: int, review_gate_status: str) -> str:
    if new_replay_tasks == 0:
        return "current_cycle_closed_no_new_replay_prior_tasks_monitoring"
    return {
        "hypothesis_review_complete_reformulation_required": (
            "current_cycle_hypothesis_review_complete_reformulation_required"
        ),
        "hypothesis_review_complete_deferred": (
            "current_cycle_hypothesis_review_complete_deferred"
        ),
        "hypothesis_review_complete_no_replay": (
            "current_cycle_hypothesis_review_closed_no_replay"
        ),
        "hypothesis_review_complete_registration_not_requested": (
            "current_cycle_hypothesis_review_complete_registration_not_requested"
        ),
        "replay_tasks_approved_for_registration": (
            "current_cycle_replay_registration_authorized_not_performed"
        ),
    }.get(review_gate_status, "current_cycle_requires_new_replay_review")


def _replay_branch_status(
    new_replay_tasks: int,
    review_gate_status: str,
    *,
    registration_observed: int = 0,
    deferred_historical: int = 0,
) -> str:
    if new_replay_tasks == 0:
        return "no_new_tasks_prior_lineage_monitoring_continues"
    if registration_observed and deferred_historical:
        return "prospective_tasks_registered_historical_tasks_require_point_in_time_review"
    if registration_observed:
        return "approved_tasks_registered_for_prospective_observation"
    if review_gate_status == "hypothesis_review_complete_reformulation_required":
        return "new_tasks_not_registerable_claim_reformulation_required"
    if review_gate_status == "hypothesis_review_complete_deferred":
        return "new_tasks_not_registerable_pending_evidence"
    if review_gate_status == "hypothesis_review_complete_no_replay":
        return "new_tasks_rejected_no_registration"
    if review_gate_status == "replay_tasks_approved_for_registration":
        return "new_tasks_approved_registration_not_performed"
    return "new_tasks_pending_manual_review"


def _verify_replay_registration_binding(
    review_gate_path: Path | None,
    review_gate: dict[str, Any] | None,
    registration_path: Path | None,
    registration: dict[str, Any] | None,
) -> dict[str, Any]:
    if registration_path is None or registration is None:
        return {}
    if review_gate_path is None or review_gate is None:
        raise ValueError("replay registration requires the reviewed gate artifact")
    if registration.get("contract") != WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT:
        raise ValueError("unsupported replay registration contract")
    source_gate = registration.get("source_gate") or {}
    if source_gate.get("run_id") != review_gate.get("run_id"):
        raise ValueError("replay registration points to a different review gate")
    recorded_sha = source_gate.get("sha256")
    if not recorded_sha:
        raise ValueError("replay registration is missing review-gate SHA-256 binding")
    if recorded_sha != _sha256(review_gate_path):
        raise ValueError("replay review gate changed after replay registration")
    summary = registration.get("summary") or {}
    if summary.get("apply_requested") is not True:
        raise ValueError("replay registration artifact is a dry-run plan")
    if summary.get("issue_count") != 0:
        raise ValueError("replay registration artifact contains blocking issues")
    return summary


def _governance_reason(new_hypotheses: int, review_gate_status: str) -> str:
    if new_hypotheses == 0:
        return "No evidence-backed current-cycle hypotheses were produced."
    return {
        "hypothesis_review_complete_reformulation_required": (
            "Content review is complete; one or more generated claims must be "
            "reformulated before any replay registration."
        ),
        "hypothesis_review_complete_deferred": (
            "Content review is complete; hypotheses are deferred pending named evidence."
        ),
        "hypothesis_review_complete_no_replay": (
            "Content review rejected the current replay candidate set."
        ),
        "hypothesis_review_complete_registration_not_requested": (
            "Content review is complete; operator registration approval was not requested."
        ),
        "replay_tasks_approved_for_registration": (
            "Manual review approved a bounded registration bundle; registration is not performed by closure."
        ),
    }.get(review_gate_status, "Current-cycle hypotheses require manual review.")


def _verify_replay_review_gate_binding(
    world_path: Path,
    world: dict[str, Any],
    gate_path: Path | None,
    gate: dict[str, Any] | None,
) -> str:
    if gate_path is None or gate is None:
        return "not_supplied"
    if gate.get("contract") != "dean_world_model_replay_review_gate_v1":
        raise ValueError("unsupported replay review gate contract")
    source = gate.get("source_packet") or {}
    if source.get("run_id") != world.get("run_id"):
        raise ValueError("replay review gate world-model run_id mismatch")
    if source.get("sha256") != _sha256(world_path):
        raise ValueError("replay review gate world-model SHA-256 mismatch")
    return str(
        (gate.get("summary") or {}).get("gate_status") or "status_missing"
    )


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return (
        "# Full System Cycle Closure\n\n"
        f"- Status: `{summary['closure_status']}`\n"
        f"- Decision state: `{summary['current_cycle_decision_state']}`\n"
        f"- Current hypotheses: `{summary['current_cycle_hypothesis_count']}`\n"
        f"- New replay tasks: `{summary['current_cycle_new_replay_task_count']}`\n"
        f"- Approved replay tasks: `{summary['approved_replay_task_count']}`\n"
        f"- Registered or already existing replay tasks: `{summary['registered_or_existing_replay_task_count']}`\n"
        f"- Historical point-in-time review required: `{summary['historical_review_required_replay_task_count']}`\n"
        f"- Outcome scoring performed: `{summary['outcome_scoring_performed']}`\n"
        f"- Prior monitoring tasks: `{summary['prior_lineage_monitoring_task_count']}`\n"
        f"- Authorization ledger records: `{summary['authorization_ledger_record_count']}`\n"
        "- Can trade: `false`\n"
    )


__all__ = ["FullSystemCycleClosureBuilder", "verify_world_model_cycle_binding"]
