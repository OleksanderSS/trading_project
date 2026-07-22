from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


class ReplayEvaluationRoute(BaseModel):
    task_id: str
    route: Literal[
        "event_study",
        "hypothesis_outcome_replay",
        "event_and_hypothesis_replay",
        "blocked_unroutable",
    ]
    maturity_status: Literal["waiting_for_horizon", "matured", "not_applicable"]
    evaluation_status: Literal[
        "waiting",
        "ready_for_event_study_eligibility",
        "ready_for_outcome_evidence_review",
        "blocked",
    ]
    due_at: str | None = None
    event_study_eligible_to_check: bool = False
    hypothesis_outcome_eligible_to_check: bool = False
    primary_outcomes: list[str] = Field(default_factory=list)
    secondary_outcomes: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReplayEvaluationRouter:
    """Route replay tasks without confusing hypotheses with timestamped events."""

    contract = "dean_replay_evaluation_routing_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/replay_evaluation_routing_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        packet_path: str | Path,
        *,
        evaluation_as_of: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        path = Path(packet_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("contract") != "dean_world_model_event_learning_packet_v1":
            raise ValueError("unsupported replay packet contract")
        safety = payload.get("safety") or {}
        if safety.get("review_only") is not True:
            raise ValueError("replay packet must be review-only")

        evaluated_at = parse_timezone_aware(evaluation_as_of or utc_now_iso())
        if evaluated_at is None:
            raise ValueError("evaluation_as_of must be timezone-aware")
        routes = [
            self.route_task(task, evaluation_as_of=evaluated_at)
            for task in payload.get("replay_tasks") or []
        ]
        counts: dict[str, int] = {}
        for route in routes:
            counts[route.route] = counts.get(route.route, 0) + 1
        status_counts: dict[str, int] = {}
        for route in routes:
            status_counts[route.evaluation_status] = (
                status_counts.get(route.evaluation_status, 0) + 1
            )

        created_at = utc_now_iso()
        run_id = "replay_evaluation_routing_" + created_at.replace(":", "").replace(
            "+00:00", "Z"
        )
        result: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "replay_evaluation_routing",
            "contract": self.contract,
            "source_packet": {
                "path": str(path),
                "run_id": payload.get("run_id"),
                "contract": payload.get("contract"),
            },
            "summary": {
                "task_count": len(routes),
                "route_counts": dict(sorted(counts.items())),
                "evaluation_status_counts": dict(sorted(status_counts.items())),
                "event_study_task_count": sum(
                    route.event_study_eligible_to_check for route in routes
                ),
                "hypothesis_outcome_task_count": sum(
                    route.hypothesis_outcome_eligible_to_check for route in routes
                ),
                "waiting_task_count": sum(
                    route.evaluation_status == "waiting" for route in routes
                ),
                "evaluation_as_of": evaluated_at.isoformat(),
                "registration_performed": False,
                "outcome_write_performed": False,
                "learning_write_performed": False,
                "can_trade": False,
            },
            "routes": [route.model_dump(mode="json") for route in routes],
            "routing_policy": {
                "event_study_requires": [
                    "event_id",
                    "event_timestamp",
                    "release_timestamp_verified=true",
                ],
                "hypothesis_replay_requires": ["hypothesis_id", "due_at"],
                "rules": [
                    "Task as_of is not automatically an event release timestamp.",
                    "Hypothesis replay evaluates expected observations and invalidation signals.",
                    "Price reaction is secondary unless the task is a timestamped event study.",
                    "No outcome is evaluated before due_at.",
                ],
            },
            "safety": {
                "review_only": True,
                "task_registration_performed": False,
                "outcome_write_performed": False,
                "learning_memory_write_performed": False,
                "model_promotion_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "can_trade": False,
            },
        }
        if save:
            result["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=result,
                markdown=_render_markdown(result),
                run_id=run_id,
            )
        return result

    def route_task(
        self,
        task: dict[str, Any],
        *,
        evaluation_as_of: datetime,
    ) -> ReplayEvaluationRoute:
        task_id = str(task.get("task_id") or "")
        hypothesis = bool(task.get("hypothesis_id"))
        event = bool(task.get("event_id"))
        event_timestamp = parse_timezone_aware(str(task.get("event_timestamp") or ""))
        release_verified = task.get("release_timestamp_verified") is True
        event_ready = event and event_timestamp is not None and release_verified
        due_at = parse_timezone_aware(str(task.get("due_at") or ""))
        hypothesis_ready = hypothesis and due_at is not None

        if event_ready and hypothesis_ready:
            route = "event_and_hypothesis_replay"
        elif event_ready:
            route = "event_study"
        elif hypothesis_ready:
            route = "hypothesis_outcome_replay"
        else:
            route = "blocked_unroutable"

        blockers: list[str] = []
        warnings: list[str] = []
        if event and not event_ready:
            blockers.append("event_release_timestamp_not_verified")
        if hypothesis and due_at is None:
            blockers.append("hypothesis_due_at_missing_or_invalid")
        if not event and task.get("as_of"):
            warnings.append("task_as_of_not_treated_as_event_timestamp")

        maturity = (
            "waiting_for_horizon"
            if hypothesis_ready and evaluation_as_of < due_at
            else "matured"
            if hypothesis_ready
            else "not_applicable"
        )
        if route == "blocked_unroutable":
            status = "blocked"
        elif maturity == "waiting_for_horizon":
            status = "waiting"
        elif event_ready:
            status = "ready_for_event_study_eligibility"
        else:
            status = "ready_for_outcome_evidence_review"

        primary = list(task.get("expected_observations") or [])
        primary.extend(
            f"invalidation:{item}"
            for item in task.get("invalidation_signals") or []
        )
        secondary = (
            ["abnormal_return", "cumulative_abnormal_return", "post_event_drift"]
            if event_ready
            else ["market_price_response_context_only"]
            if hypothesis
            else []
        )
        return ReplayEvaluationRoute(
            task_id=task_id,
            route=route,
            maturity_status=maturity,
            evaluation_status=status,
            due_at=due_at.isoformat() if due_at else None,
            event_study_eligible_to_check=event_ready and maturity != "waiting_for_horizon",
            hypothesis_outcome_eligible_to_check=hypothesis_ready and maturity == "matured",
            primary_outcomes=primary,
            secondary_outcomes=secondary,
            blockers=blockers,
            warnings=warnings,
        )


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return (
        "# Replay Evaluation Routing\n\n"
        f"- Tasks: `{summary['task_count']}`\n"
        f"- Routes: `{summary['route_counts']}`\n"
        f"- Statuses: `{summary['evaluation_status_counts']}`\n"
        f"- Waiting: `{summary['waiting_task_count']}`\n"
        "- Registration performed: `false`\n"
        "- Outcome write performed: `false`\n"
        "- Can trade: `false`\n"
    )


__all__ = ["ReplayEvaluationRoute", "ReplayEvaluationRouter"]
