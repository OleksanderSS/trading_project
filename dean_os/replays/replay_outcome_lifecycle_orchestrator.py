from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.replays.historical_replay_outcome_review import HistoricalReplayOutcomeReview
from dean_os.research_corpus.hypothesis_learning_review import HypothesisLearningReview
from dean_os.replays.replay_checkpoint_due_router import ReplayCheckpointDueRouter
from dean_os.schemas import utc_now_iso


class ReplayOutcomeLifecycleOrchestrator:
    """Compose due routing, scoped outcome review, and reverse analysis safely."""

    contract = "dean_replay_outcome_lifecycle_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/replay_outcome_lifecycle_current",
        router_output_dir: str | Path = (
            "reports/dean_os/replay_checkpoint_due_router_current"
        ),
        outcome_output_dir: str | Path = (
            "reports/dean_os/replay_matured_outcome_review_current"
        ),
        learning_output_dir: str | Path = (
            "reports/dean_os/hypothesis_learning_review_post_outcome_current"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.router_output_dir = Path(router_output_dir)
        self.outcome_output_dir = Path(outcome_output_dir)
        self.learning_output_dir = Path(learning_output_dir)

    def build(
        self,
        *,
        registration_json: str | Path,
        review_gate_json: str | Path,
        as_of: str,
        verified_price_paths: list[str | Path],
        pipeline_paths: list[str | Path],
        prior_outcome_json_paths: list[str | Path],
        packet_json: str | Path | None = None,
        journal_path: str | Path = "data/dean_os/system_journal.jsonl",
        save: bool = True,
    ) -> dict[str, Any]:
        registration_path = Path(registration_json)
        gate_path = Path(review_gate_json)
        gate = _load(gate_path)
        packet_path = Path(packet_json) if packet_json is not None else _source_packet_path(gate)

        initial_router = ReplayCheckpointDueRouter(self.router_output_dir).build(
            registration_path,
            gate_path,
            as_of=as_of,
            verified_price_paths=verified_price_paths,
            pipeline_paths=pipeline_paths,
            outcome_json_paths=prior_outcome_json_paths,
            save=save,
        )
        initial_inbox = initial_router.get("chief_review_inbox") or {}
        matured = list(initial_inbox.get("matured_checkpoints") or [])
        waiting = list(initial_inbox.get("data_accrual_actions") or [])
        outcome_review: dict[str, Any] | None = None
        learning_review: dict[str, Any] | None = None
        final_router = initial_router

        if matured:
            task_ids = [str(item.get("task_id")) for item in matured]
            outcome_review = HistoricalReplayOutcomeReview(
                self.outcome_output_dir
            ).build(
                review_gate_json=gate_path,
                registration_json=registration_path,
                price_paths=verified_price_paths,
                pipeline_paths=pipeline_paths,
                task_ids=task_ids,
                save=save,
            )
            primary_outcomes = list(outcome_review.get("outcomes") or [])
            if primary_outcomes:
                if not packet_path.is_file():
                    raise FileNotFoundError(
                        f"hash-bound world-model packet is missing: {packet_path}"
                    )
                outcome_input: str | Path | dict[str, Any] = outcome_review
                if save:
                    saved = outcome_review.get("saved_paths") or {}
                    outcome_input = saved.get("latest_json") or saved.get("json")
                learning_review = HypothesisLearningReview(
                    self.learning_output_dir
                ).build(
                    packet_path,
                    gate_path,
                    outcome_json=outcome_input,
                    journal_path=journal_path,
                    save=save,
                )

            # Re-run the router with the newly SHA-bound outcome artifact so the
            # same checkpoint cannot remain in the due queue.
            final_outcome_paths = list(prior_outcome_json_paths)
            if save:
                saved = outcome_review.get("saved_paths") or {}
                new_path = saved.get("latest_json") or saved.get("json")
                if new_path:
                    final_outcome_paths.append(new_path)
            else:
                # In-memory smoke runs cannot establish file SHA lineage. Keep
                # the initial route and expose this explicitly in the payload.
                final_outcome_paths = list(prior_outcome_json_paths)
            if save:
                final_router = ReplayCheckpointDueRouter(
                    self.router_output_dir
                ).build(
                    registration_path,
                    gate_path,
                    as_of=as_of,
                    verified_price_paths=verified_price_paths,
                    pipeline_paths=pipeline_paths,
                    outcome_json_paths=final_outcome_paths,
                    save=True,
                )

        status, inbox = _lifecycle_status(
            waiting=waiting,
            matured=matured,
            outcome_review=outcome_review,
            learning_review=learning_review,
        )
        created_at = utc_now_iso()
        run_id = "replay_outcome_lifecycle_" + created_at.replace(":", "").replace(
            "+00:00", "Z"
        )
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "replay_outcome_lifecycle",
            "contract": self.contract,
            "inputs": {
                "as_of": as_of,
                "registration_json": str(registration_path),
                "review_gate_json": str(gate_path),
                "packet_json": str(packet_path),
                "verified_price_paths": [str(item) for item in verified_price_paths],
                "pipeline_paths": [str(item) for item in pipeline_paths],
                "prior_outcome_json_paths": [
                    str(item) for item in prior_outcome_json_paths
                ],
                "journal_path": str(journal_path),
            },
            "summary": {
                "status": status,
                "initial_matured_checkpoint_count": len(matured),
                "waiting_for_verified_data_count": len(waiting),
                "outcome_packet_created": outcome_review is not None,
                "primary_outcome_count": len(
                    (outcome_review or {}).get("outcomes") or []
                ),
                "reverse_analysis_created": learning_review is not None,
                "learning_proposal_count": int(
                    ((learning_review or {}).get("summary") or {}).get(
                        "learning_proposal_count"
                    )
                    or 0
                ),
                "automatic_rule_update_allowed": False,
                "can_trade": False,
            },
            "initial_due_router": _artifact_summary(initial_router),
            "outcome_review": _artifact_summary(outcome_review),
            "learning_review": _artifact_summary(learning_review),
            "final_due_router": _artifact_summary(final_router),
            "review_inbox": inbox,
            "system_recommendations": _system_recommendations(
                waiting=waiting,
                matured=matured,
                outcome_review=outcome_review,
            ),
            "safety": {
                "review_only": True,
                "network_collection_performed": False,
                "outcome_scoring_performed": False,
                "causal_attribution_approved": False,
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "registration_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=run_id,
            )
        return payload


def _lifecycle_status(
    *,
    waiting: list[dict[str, Any]],
    matured: list[dict[str, Any]],
    outcome_review: dict[str, Any] | None,
    learning_review: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if matured and outcome_review is not None:
        outcomes = list(outcome_review.get("outcomes") or [])
        if outcomes:
            decisions = [
                {
                    "decision_type": "primary_checkpoint_causal_disposition",
                    "task_id": item.get("task_id"),
                    "hypothesis_id": item.get("hypothesis_id"),
                    "result_label": item.get("result_label"),
                    "observable": item.get("observable"),
                    "allowed_decisions": [
                        "accept_machine_diagnosis_for_case_log",
                        "amend_causal_attribution",
                        "defer_for_missing_evidence",
                    ],
                }
                for item in outcomes
            ]
            return "primary_outcome_packet_pending_causal_review", {
                "status": "primary_outcome_packet_pending_causal_review",
                "data_actions": [],
                "outcome_packets": decisions,
                "learning_proposals": list(
                    (learning_review or {}).get("learning_proposals") or []
                ),
                "pending_decisions": decisions,
            }
        return "intermediate_checkpoint_packet_recorded", {
            "status": "intermediate_checkpoint_packet_recorded",
            "data_actions": [],
            "outcome_packets": [
                {
                    "task_id": item.get("task_id"),
                    "hypothesis_id": item.get("hypothesis_id"),
                    "checkpoint_role": item.get("checkpoint_role"),
                    "review_status": item.get("review_status"),
                    "result_label": item.get("result_label"),
                }
                for item in outcome_review.get("checkpoint_reviews") or []
            ],
            "learning_proposals": [],
            "pending_decisions": [],
        }
    if waiting:
        return "waiting_for_verified_checkpoint_data", {
            "status": "waiting_for_verified_checkpoint_data",
            "data_actions": waiting,
            "outcome_packets": [],
            "learning_proposals": [],
            "pending_decisions": [],
        }
    return "no_matured_checkpoint_action", {
        "status": "no_matured_checkpoint_action",
        "data_actions": [],
        "outcome_packets": [],
        "learning_proposals": [],
        "pending_decisions": [],
    }


def _system_recommendations(
    *,
    waiting: list[dict[str, Any]],
    matured: list[dict[str, Any]],
    outcome_review: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    recommendations = []
    for item in waiting:
        recommendations.append(
            {
                "action_type": "refresh_verified_checkpoint_evidence",
                "task_id": item.get("task_id"),
                "reason": "Due checkpoint lacks a verified post-close market session.",
                "execute_automatically": False,
                "rerun_after": "verified snapshot is refreshed and immutable lineage is recorded",
            }
        )
    if matured and outcome_review is not None and not outcome_review.get("outcomes"):
        recommendations.append(
            {
                "action_type": "retain_intermediate_checkpoint_as_evidence",
                "reason": "Intermediate evidence cannot rewrite the primary hypothesis outcome.",
                "execute_automatically": True,
            }
        )
    return recommendations


def _artifact_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "run_id": payload.get("run_id"),
        "contract": payload.get("contract"),
        "summary": payload.get("summary"),
        "saved_paths": payload.get("saved_paths"),
    }


def _source_packet_path(gate: dict[str, Any]) -> Path:
    value = (gate.get("source_packet") or {}).get("path")
    if not value:
        raise ValueError("review gate does not declare its hash-bound source packet path")
    return Path(str(value))


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    inbox = payload["review_inbox"]
    lines = [
        "# Replay Outcome Lifecycle",
        "",
        f"- Status: `{summary['status']}`",
        f"- Initial matured checkpoints: `{summary['initial_matured_checkpoint_count']}`",
        f"- Waiting for verified data: `{summary['waiting_for_verified_data_count']}`",
        f"- Outcome packet created: `{summary['outcome_packet_created']}`",
        f"- Reverse analysis created: `{summary['reverse_analysis_created']}`",
        f"- Learning proposals: `{summary['learning_proposal_count']}`",
        "",
        "## Machine Recommendations",
        "",
    ]
    for item in payload["system_recommendations"]:
        lines.append(
            f"- `{item.get('action_type')}` task={item.get('task_id')} — {item.get('reason')}"
        )
    if not payload["system_recommendations"]:
        lines.append("- No system action is currently required.")
    lines.extend(["", "## Human Review Inbox", ""])
    lines.append(f"- Status: `{inbox['status']}`")
    lines.append(f"- Pending decisions: `{len(inbox['pending_decisions'])}`")
    lines.extend(
        [
            "",
            "No automatic causal approval, learning-rule update, registration, or trading was performed.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


__all__ = ["ReplayOutcomeLifecycleOrchestrator"]
