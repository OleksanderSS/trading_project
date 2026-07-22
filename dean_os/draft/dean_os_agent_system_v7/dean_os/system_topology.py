from __future__ import annotations

from collections import defaultdict, deque
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dean_os.utils import sha256_json


TOPOLOGY_SCHEMA_VERSION = "dean_system_topology_v1"
RUN_MANIFEST_SCHEMA_VERSION = "dean_system_run_manifest_v1"


class BranchId(StrEnum):
    PIPELINE_STAGE03_INTAKE = "pipeline_stage03_intake"
    PIPELINE_CONTROL = "pipeline_control"
    EVIDENCE_INTELLIGENCE = "evidence_intelligence"
    DOMAIN_ANALYSIS = "domain_analysis"
    WORLD_MODEL = "world_model"
    REPLAY_EVALUATION = "replay_evaluation"
    GOVERNANCE_REVIEW = "governance_review"
    DAILY_AUDIT = "daily_audit"


class BranchPlane(StrEnum):
    DATA = "data_plane"
    CONTROL = "control_plane"
    ANALYSIS = "analysis_plane"
    EVALUATION = "evaluation_plane"
    REVIEW = "review_plane"


class BranchRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


class BranchSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    branch_id: str
    title: str
    plane: BranchPlane
    parent_branch: Literal["pipeline", "analytical", "governance"]
    depends_on: list[str] = Field(default_factory=list)
    required: bool = True
    enabled: bool = True
    execution_mode: Literal["sequential", "parallel", "projection"] = "sequential"
    input_contracts: list[str] = Field(default_factory=list)
    output_contracts: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)


class SystemTopology(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = TOPOLOGY_SCHEMA_VERSION
    topology_id: str = "dean_default_topology"
    root_orchestrator: str = "dean_os.draft.dean_os_agent_system_v7.dean_os.full_system_orchestrator:DEANAgentSystemOrchestrator"
    branches: list[BranchSpec]
    topology_hash: str = ""

    @model_validator(mode="after")
    def validate_graph(self) -> "SystemTopology":
        branch_ids = [item.branch_id for item in self.branches]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("System topology contains duplicate branch IDs")
        known = set(branch_ids)
        for item in self.branches:
            unknown = set(item.depends_on) - known
            if unknown:
                raise ValueError(
                    f"Branch {item.branch_id!r} depends on unknown branches: {sorted(unknown)}"
                )
            if item.branch_id in item.depends_on:
                raise ValueError(f"Branch {item.branch_id!r} cannot depend on itself")
        _assert_acyclic(self.branches)
        expected = sha256_json(
            {
                "schema_version": self.schema_version,
                "topology_id": self.topology_id,
                "root_orchestrator": self.root_orchestrator,
                "branches": [item.model_dump(mode="json") for item in self.branches],
            }
        )
        if self.topology_hash and self.topology_hash != expected:
            raise ValueError("System topology hash mismatch")
        object.__setattr__(self, "topology_hash", expected)
        return self

    def enabled_branches(self) -> list[BranchSpec]:
        return [item for item in self.branches if item.enabled]

    def execution_order(self) -> list[BranchSpec]:
        by_id = {item.branch_id: item for item in self.enabled_branches()}
        indegree = {branch_id: 0 for branch_id in by_id}
        children: dict[str, list[str]] = defaultdict(list)
        for branch_id, spec in by_id.items():
            for dependency in spec.depends_on:
                if dependency not in by_id:
                    continue
                indegree[branch_id] += 1
                children[dependency].append(branch_id)
        queue = deque(sorted(branch_id for branch_id, degree in indegree.items() if degree == 0))
        order: list[BranchSpec] = []
        while queue:
            branch_id = queue.popleft()
            order.append(by_id[branch_id])
            for child in sorted(children[branch_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(by_id):
            raise ValueError("Enabled system topology contains a dependency cycle")
        return order


class BranchExecutionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    branch_id: str
    plane: BranchPlane
    parent_branch: str
    status: BranchRunStatus
    started_at: str
    finished_at: str
    duration_ms: float = Field(ge=0.0)
    required: bool = True
    input_hash: str
    output_hash: str
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    safety: dict[str, bool] = Field(default_factory=dict)


class SystemRunManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION
    system_run_id: str = Field(default_factory=lambda: f"agent_system_{uuid4().hex}")
    topology_id: str
    topology_hash: str
    domain_id: str
    as_of: str
    knowledge_cutoff: str
    status: BranchRunStatus
    started_at: str
    finished_at: str
    branch_records: list[BranchExecutionRecord] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    safety: dict[str, bool] = Field(
        default_factory=lambda: {
            "can_trade": False,
            "can_write_production_config": False,
            "can_promote_model": False,
            "can_write_learning_memory": False,
            "human_review_required": True,
        }
    )
    content_hash: str = ""

    @model_validator(mode="after")
    def validate_manifest(self) -> "SystemRunManifest":
        started = datetime.fromisoformat(self.started_at)
        finished = datetime.fromisoformat(self.finished_at)
        if started.tzinfo is None or finished.tzinfo is None:
            raise ValueError("System run timestamps must be timezone-aware")
        if finished < started:
            raise ValueError("finished_at cannot be earlier than started_at")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        expected = sha256_json(payload)
        if self.content_hash and self.content_hash != expected:
            raise ValueError("System run manifest hash mismatch")
        object.__setattr__(self, "content_hash", expected)
        return self


class BranchTimer:
    def __init__(self) -> None:
        self.started = datetime.now(UTC)

    def finish(
        self,
        *,
        spec: BranchSpec,
        status: BranchRunStatus,
        input_payload: Any,
        output_payload: Any,
        summary: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        error: str | None = None,
        safety: dict[str, bool] | None = None,
    ) -> BranchExecutionRecord:
        finished = datetime.now(UTC)
        return BranchExecutionRecord(
            branch_id=spec.branch_id,
            plane=spec.plane,
            parent_branch=spec.parent_branch,
            status=status,
            started_at=self.started.isoformat(),
            finished_at=finished.isoformat(),
            duration_ms=max(0.0, (finished - self.started).total_seconds() * 1000.0),
            required=spec.required,
            input_hash=sha256_json(input_payload),
            output_hash=sha256_json(output_payload),
            summary=dict(summary or {}),
            warnings=list(warnings or []),
            error=error,
            safety=dict(safety or {}),
        )


def load_system_topology(path: str | Path) -> SystemTopology:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return SystemTopology(**raw)


def default_system_topology_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "system_topology.yaml"


def load_default_system_topology() -> SystemTopology:
    return load_system_topology(default_system_topology_path())


def _assert_acyclic(branches: list[BranchSpec]) -> None:
    graph = {item.branch_id: list(item.depends_on) for item in branches}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError(f"System topology cycle detected at {node!r}")
        visiting.add(node)
        for dependency in graph.get(node, []):
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for branch_id in graph:
        visit(branch_id)


__all__ = [
    "BranchExecutionRecord",
    "BranchId",
    "BranchPlane",
    "BranchRunStatus",
    "BranchSpec",
    "BranchTimer",
    "SystemRunManifest",
    "SystemTopology",
    "default_system_topology_path",
    "load_default_system_topology",
    "load_system_topology",
]
