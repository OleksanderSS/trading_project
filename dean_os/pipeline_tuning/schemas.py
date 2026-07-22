from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

PlaneStatus = Literal["candidate", "blocked", "not_needed", "needs_validation"]
PlanStatus = Literal["tuning_candidate", "validate_first", "blocked", "no_action"]


class TuningPlaneProfile(BaseModel):
    plane_id: str
    display_name: str
    description: str
    allowed_parameters: list[str] = Field(default_factory=list)
    required_preconditions: list[str] = Field(default_factory=list)
    blocked_if: list[str] = Field(default_factory=list)
    max_change_pct: float = Field(default=0.15, ge=0.0, le=1.0)
    review_required: bool = True


class TuningPlaneDecision(BaseModel):
    plane_id: str
    status: PlaneStatus
    reasons: list[str] = Field(default_factory=list)
    proposed_bounds: dict[str, Any] = Field(default_factory=dict)
    guardrails: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class PipelineTuningPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: f"pipeline_tuning_plan_{uuid4().hex}")
    status: PlanStatus
    target: str = "walk_forward_tuning_experiment"
    planes: list[TuningPlaneDecision] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    command_preview: str
    review_required: bool = True
    live_execution_allowed: bool = False
    production_config_write_allowed: bool = False
