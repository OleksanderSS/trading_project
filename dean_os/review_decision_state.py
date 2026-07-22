from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso


ReviewDecisionState = Literal[
    "blocked", "needs_more_data", "partial_ready", "ready_for_review", "no_action"
]

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "blocked": {"blocked", "needs_more_data", "no_action"},
    "needs_more_data": {"blocked", "needs_more_data", "partial_ready", "no_action"},
    "partial_ready": {"blocked", "needs_more_data", "partial_ready", "ready_for_review", "no_action"},
    "ready_for_review": {"blocked", "needs_more_data", "partial_ready", "ready_for_review", "no_action"},
    "no_action": {"no_action", "needs_more_data"},
}


class DecisionStateTransition(BaseModel):
    previous_state: ReviewDecisionState
    next_state: ReviewDecisionState
    reasons: list[str] = Field(min_length=1)
    actor: str
    decided_at: str
    input_hashes: dict[str, str] = Field(default_factory=dict)
    false_ready_loss: float = Field(default=1.0, ge=0.0)
    false_block_loss: float = Field(default=0.25, ge=0.0)
    human_approval_required: bool = True
    automatic_execution_allowed: bool = False

    def validate_transition(self) -> "DecisionStateTransition":
        if self.next_state not in ALLOWED_TRANSITIONS[self.previous_state]:
            raise ValueError(
                f"invalid review decision transition: {self.previous_state} -> {self.next_state}"
            )
        if self.false_ready_loss < self.false_block_loss:
            raise ValueError("review safety policy requires false-ready loss >= false-block loss")
        if not self.actor.strip():
            raise ValueError("decision transition requires actor")
        if self.automatic_execution_allowed:
            raise ValueError("review decision states cannot authorize automatic execution")
        return self


class ReviewDecisionStateBuilder:
    contract = "dean_review_decision_state_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/review_decision_state_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        evidence_plan_path: str | Path,
        voi_review_path: str | Path,
        *,
        previous_state: ReviewDecisionState = "blocked",
        actor: str = "dean_os_policy",
        save: bool = True,
    ) -> dict[str, Any]:
        plan_path = Path(evidence_plan_path)
        voi_path = Path(voi_review_path)
        plan = _load(plan_path)
        voi = _load(voi_path)
        if plan.get("contract") != "dean_replay_outcome_evidence_plan_v1":
            raise ValueError("unsupported evidence plan contract")
        if voi.get("contract") != "dean_unknown_voi_review_v1":
            raise ValueError("unsupported VoI review contract")
        voi_plan_ref = (voi.get("inputs") or {}).get("evidence_plan") or {}
        if voi_plan_ref.get("sha256") != _sha256(plan_path):
            raise ValueError("VoI review is not bound to current evidence plan hash")

        next_state, reasons = _derive_state(plan, voi)
        transition = DecisionStateTransition(
            previous_state=previous_state,
            next_state=next_state,
            reasons=reasons,
            actor=actor,
            decided_at=utc_now_iso(),
            input_hashes={
                "evidence_plan": _sha256(plan_path),
                "voi_review": _sha256(voi_path),
            },
        ).validate_transition()
        run_id = "review_decision_state_" + transition.decided_at.replace(":", "").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": transition.decided_at,
            "mode": "review_decision_state",
            "contract": self.contract,
            "state": next_state,
            "transition": transition.model_dump(mode="json"),
            "decision_inputs": {
                "task_plan_count": (plan.get("summary") or {}).get("task_plan_count", 0),
                "unresolved_lane_reference_count": (plan.get("summary") or {}).get("unresolved_lane_reference_count", 0),
                "outcome_evaluation_can_run": bool((plan.get("summary") or {}).get("outcome_evaluation_can_run")),
                "voi_validated_scored_count": (voi.get("summary") or {}).get("validated_scored_count", 0),
                "voi_unscored_count": (voi.get("summary") or {}).get("unscored_count", 0),
            },
            "state_meaning": _state_meaning(next_state),
            "next_allowed_states": sorted(ALLOWED_TRANSITIONS[next_state]),
            "safety": {
                "review_only": True,
                "human_approval_required": True,
                "automatic_execution_allowed": False,
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


def _derive_state(plan: dict[str, Any], voi: dict[str, Any]) -> tuple[ReviewDecisionState, list[str]]:
    summary = plan.get("summary") or {}
    voi_summary = voi.get("summary") or {}
    task_count = int(summary.get("task_plan_count") or 0)
    unresolved = int(summary.get("unresolved_lane_reference_count") or 0)
    can_evaluate = bool(summary.get("outcome_evaluation_can_run"))
    unscored = int(voi_summary.get("unscored_count") or 0)
    if task_count == 0:
        return "no_action", ["no_replay_tasks_in_scope"]
    if not can_evaluate:
        return "needs_more_data", [
            "prospective_outcomes_not_matured",
            f"unresolved_evidence_lane_references:{unresolved}",
            f"unscored_value_of_information_gaps:{unscored}",
        ]
    if unresolved > 0:
        return "partial_ready", [
            "outcome_review_available_but_evidence_gaps_remain",
            f"unresolved_evidence_lane_references:{unresolved}",
        ]
    return "ready_for_review", ["outcomes_matured_and_no_unresolved_evidence_lanes"]


def _state_meaning(state: ReviewDecisionState) -> str:
    return {
        "blocked": "A contract, safety, or lineage failure prevents review progress.",
        "needs_more_data": "Wait for evidence or outcomes; no forecast/action escalation is justified.",
        "partial_ready": "Some review is possible, but unresolved evidence must remain explicit.",
        "ready_for_review": "Inputs are reviewable by a human; this is not approval or execution authority.",
        "no_action": "No task is currently in scope; preserve state and do not manufacture work.",
    }[state]


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    return (
        "# Review Decision State\n\n"
        f"- State: `{payload['state']}`\n"
        f"- Meaning: {payload['state_meaning']}\n"
        "- Human approval required: `true`\n"
        "- Automatic execution allowed: `false`\n"
        "- Can trade: `false`\n"
    )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "DecisionStateTransition",
    "ReviewDecisionStateBuilder",
    "ReviewDecisionState",
]
