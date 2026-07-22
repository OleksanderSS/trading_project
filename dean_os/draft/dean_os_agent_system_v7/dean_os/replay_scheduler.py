from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dean_os.analyst_core import OUTCOME_HORIZONS


def _parse(value: str, *, field_name: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


class ReplayScheduleItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    world_state_snapshot_id: str
    domain_id: str
    scenario_graph_id: str | None = None
    hypothesis_id: str | None = None
    horizon_days: int
    created_as_of: str
    due_at: str
    status: str = "pending_not_due"
    manual_review_required: bool = True
    can_write_learning_memory: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "ReplayScheduleItem":
        created = _parse(self.created_as_of, field_name="created_as_of")
        due = _parse(self.due_at, field_name="due_at")
        if self.horizon_days not in OUTCOME_HORIZONS:
            raise ValueError(f"horizon_days must be one of {OUTCOME_HORIZONS}")
        if due != created + timedelta(days=self.horizon_days):
            raise ValueError("due_at must equal created_as_of + horizon_days")
        if not self.manual_review_required or self.can_write_learning_memory:
            raise ValueError("replay schedule v1 must remain review-gated")
        return self


class ReplayScheduler:
    def build_from_run_result(self, run_result: Any) -> list[ReplayScheduleItem]:
        result = run_result.model_dump(mode="json") if hasattr(run_result, "model_dump") else dict(run_result)
        snapshot = result.get("world_state_snapshot", {}) or {}
        world_model = result.get("world_model_event_learning", {}) or {}
        tasks = list(world_model.get("replay_tasks", []) or [])
        items: list[ReplayScheduleItem] = []
        for task in tasks:
            items.append(ReplayScheduleItem(
                task_id=str(task.get("task_id")),
                world_state_snapshot_id=str(snapshot.get("snapshot_id") or "unpersisted_world_state"),
                domain_id=str(result.get("domain_id") or snapshot.get("domain_id") or "unknown_domain"),
                scenario_graph_id=task.get("scenario_graph_id"),
                hypothesis_id=task.get("hypothesis_id"),
                horizon_days=int(task.get("horizon_days")),
                created_as_of=str(task.get("as_of") or snapshot.get("as_of")),
                due_at=str(task.get("due_at")),
            ))
        return items

    def due(self, tasks: Iterable[ReplayScheduleItem | dict[str, Any]], *, as_of: str) -> list[ReplayScheduleItem]:
        as_of_dt = _parse(as_of, field_name="as_of")
        due_items: list[ReplayScheduleItem] = []
        for raw in tasks:
            item = raw if isinstance(raw, ReplayScheduleItem) else ReplayScheduleItem.model_validate(raw)
            if _parse(item.due_at, field_name="due_at") <= as_of_dt:
                due_items.append(item.model_copy(update={"status": "due_pending_evidence"}))
        due_items.sort(key=lambda item: (item.due_at, item.task_id))
        return due_items
