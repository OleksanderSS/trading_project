from __future__ import annotations

from collections import defaultdict, deque
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dean_os.utils import sha256_json


class SystemBranchId(StrEnum):
    ARTIFACT_INTAKE = "artifact_intake"
    PIPELINE_CONTROL = "pipeline_control"
    EVIDENCE_INTELLIGENCE = "evidence_intelligence"
    DOMAIN_ANALYSIS = "domain_analysis"
    WORLD_MODEL = "world_model"
    REPLAY_EVALUATION = "replay_evaluation"
    GOVERNANCE_REVIEW = "governance_review"
    OPERATIONS_AUTHORIZATION = "operations_authorization"
    SYSTEM_AUDIT = "system_audit"


class SystemOperatingProfile(StrEnum):
    DATA_ONLY = "data_only"
    ANALYSIS_READY = "analysis_ready"
    PREDICTION_READY = "prediction_ready"
    REPLAY_READY = "replay_ready"
    PAPER_READY = "paper_ready"


class BranchSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    branch_id: str
    title: str
    plane: Literal["data", "control", "analysis", "evaluation", "governance", "audit"]
    depends_on: list[str] = Field(default_factory=list)
    required: bool = True
    enabled: bool = True
    minimum_profile: SystemOperatingProfile = SystemOperatingProfile.DATA_ONLY
    input_contracts: list[str] = Field(default_factory=list)
    output_contracts: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)


class SystemTopology(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: str = "dean_system_topology_v2"
    topology_id: str
    branches: list[BranchSpec]
    topology_sha256: str = ""

    @model_validator(mode="after")
    def validate_topology(self) -> "SystemTopology":
        ids = [branch.branch_id for branch in self.branches]
        if len(ids) != len(set(ids)):
            raise ValueError("system topology has duplicate branch IDs")
        known = set(ids)
        for branch in self.branches:
            unknown = set(branch.depends_on) - known
            if unknown:
                raise ValueError(f"branch {branch.branch_id} has unknown dependencies: {sorted(unknown)}")
            if branch.branch_id in branch.depends_on:
                raise ValueError(f"branch {branch.branch_id} depends on itself")
        _assert_acyclic(self.branches)
        expected = sha256_json(
            self.model_dump(mode="json", exclude={"topology_sha256"})
        )
        if self.topology_sha256 and self.topology_sha256 != expected:
            raise ValueError("system topology hash mismatch")
        object.__setattr__(self, "topology_sha256", expected)
        return self

    def execution_order(self) -> list[BranchSpec]:
        enabled = {branch.branch_id: branch for branch in self.branches if branch.enabled}
        for branch in enabled.values():
            disabled_dependencies = [dep for dep in branch.depends_on if dep not in enabled]
            if disabled_dependencies:
                raise ValueError(
                    f"enabled branch {branch.branch_id} depends on disabled branches: {disabled_dependencies}"
                )
        indegree = {branch_id: 0 for branch_id in enabled}
        children: dict[str, list[str]] = defaultdict(list)
        for branch in enabled.values():
            for dependency in branch.depends_on:
                indegree[branch.branch_id] += 1
                children[dependency].append(branch.branch_id)
        queue = deque(sorted(branch_id for branch_id, degree in indegree.items() if degree == 0))
        ordered: list[BranchSpec] = []
        while queue:
            branch_id = queue.popleft()
            ordered.append(enabled[branch_id])
            for child in sorted(children[branch_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(enabled):
            raise ValueError("enabled system topology contains a dependency cycle")
        return ordered


def load_system_topology(path: str | Path = "dean_os/config/system_topology.yaml") -> SystemTopology:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SystemTopology(**payload)


def _assert_acyclic(branches: list[BranchSpec]) -> None:
    graph = {branch.branch_id: branch.depends_on for branch in branches}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError(f"system topology dependency cycle at {node}")
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for branch_id in graph:
        visit(branch_id)


__all__ = [
    "BranchSpec",
    "SystemBranchId",
    "SystemOperatingProfile",
    "SystemTopology",
    "load_system_topology",
]
