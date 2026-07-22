from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ReadinessCheck(BaseModel):
    check_id: str
    area: str
    status: str
    structural_weight: float = Field(ge=0.0)
    operational_weight: float = Field(ge=0.0)
    structural_credit: float = Field(ge=0.0, le=1.0)
    operational_credit: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class BranchReadiness(BaseModel):
    branch: str
    status: str
    structural_score: float = Field(ge=0.0, le=1.0)
    operational_score: float = Field(ge=0.0, le=1.0)
    checks: list[ReadinessCheck] = Field(default_factory=list)


class AgentSystemReadinessReport(BaseModel):
    schema_version: str = "dean_agent_system_readiness_v1"
    domain_id: str
    mode: str
    structural_readiness: float = Field(ge=0.0, le=1.0)
    operational_readiness: float = Field(ge=0.0, le=1.0)
    overall_status: str
    branches: list[BranchReadiness] = Field(default_factory=list)
    cross_cutting_checks: list[ReadinessCheck] = Field(default_factory=list)
    interpretation: dict[str, Any] = Field(default_factory=dict)


class AgentSystemReadinessAssessor:
    """Transparent readiness assessment over the current package.

    This is not a production certification. It distinguishes whether a
    component is structurally present from whether it is operationally proven
    on real data, repeated runs, and reviewed outcomes.
    """

    def assess(
        self,
        *,
        package_root: str | Path,
        domain_id: str = "semiconductor_ai_infrastructure",
        pipeline_deferred: bool = True,
    ) -> AgentSystemReadinessReport:
        root = Path(package_root).resolve()
        dean_root = root / "dean_os"
        registry_path = dean_root / "config" / "minimal_system_registry.yaml"
        domain_profile_path = dean_root / "config" / "domain_profiles" / f"{domain_id}.yaml"
        registry = self._load_yaml(registry_path)
        agents = registry.get("agents", {}) if isinstance(registry, dict) else {}

        pipeline_checks = [
            self._file_check(
                root,
                "pipeline_orchestrator_boundary",
                "pipeline",
                ["dean_os/pipeline_adapter.py", "dean_os/agents/pipeline_control.py"],
                structural_weight=2.0,
                operational_weight=1.0,
                operational_credit=0.25 if pipeline_deferred else 0.5,
                structural_cap=0.85,
                gaps=(
                    ["heavy pipeline execution intentionally deferred"]
                    if pipeline_deferred
                    else ["full live pipeline and analyzer chain still needs environment verification"]
                ),
            ),
            self._registry_check(
                "pipeline_registry",
                "pipeline",
                agents,
                required=["pipeline_control", "data_quality", "risk", "tuning"],
                structural_weight=1.5,
                operational_weight=1.0,
                operational_credit=0.25 if pipeline_deferred else 0.5,
                gaps=["pipeline branch is a prepared boundary, not the current build focus"],
            ),
        ]

        analytical_checks = [
            self._registry_check(
                "analytical_registry",
                "analytical",
                agents,
                required=["domain_analyst"],
                structural_weight=1.5,
                operational_weight=1.0,
                operational_credit=0.65,
                structural_cap=0.9,
                gaps=["only one canonical domain profile is integrated end-to-end"],
            ),
            self._file_check(
                root,
                "domain_analyst_runtime",
                "analytical",
                [
                    "dean_os/agents/domain_analytical.py",
                    "dean_os/analyst_core/domain_analyst_runtime.py",
                    "dean_os/world_model_event_learning.py",
                ],
                structural_weight=2.0,
                operational_weight=2.0,
                operational_credit=0.65,
                structural_cap=0.8,
                gaps=[
                    "real recurring source ingestion and domain evaluation corpus remain incomplete",
                    "LLM/model provider behavior is not validated here as a production service",
                ],
            ),
            self._file_check(
                root,
                "domain_profile",
                "analytical",
                [str(domain_profile_path.relative_to(root))],
                structural_weight=1.0,
                operational_weight=1.0,
                operational_credit=0.75,
                structural_cap=0.75,
                gaps=["portability to a second domain has not yet been demonstrated end-to-end"],
            ),
            self._manual_check(
                "second_domain_replication",
                "analytical",
                structural_credit=0.25,
                operational_credit=0.1,
                structural_weight=1.0,
                operational_weight=1.0,
                evidence=["domain-neutral registry override and profile contract exist"],
                gaps=["no second domain has passed the same end-to-end tests yet"],
            ),
            self._manual_check(
                "recurring_evidence_acquisition",
                "analytical",
                structural_credit=0.55,
                operational_credit=0.25,
                structural_weight=1.5,
                operational_weight=2.0,
                evidence=["material loaders and domain feeder scaffolding exist"],
                gaps=["no canonical scheduler/source refresh loop is part of the minimal runtime"],
            ),
        ]

        world_model_checks = [
            self._file_check(
                root,
                "context_indicator_contracts",
                "world_model",
                ["dean_os/context_grids.py", "dean_os/world_state_store.py"],
                structural_weight=2.0,
                operational_weight=1.5,
                operational_credit=0.7,
                structural_cap=0.85,
                gaps=["historical corpus is not yet populated at useful scale"],
            ),
            self._file_check(
                root,
                "scenario_graph",
                "world_model",
                ["dean_os/analyst_core/schemas.py", "dean_os/world_model_event_learning.py"],
                structural_weight=2.0,
                operational_weight=1.5,
                operational_credit=0.6,
                structural_cap=0.8,
                gaps=["scenario probabilities are review priors, not calibrated probabilities"],
            ),
            self._file_check(
                root,
                "historical_analog_retrieval",
                "world_model",
                ["dean_os/world_state_store.py"],
                structural_weight=1.5,
                operational_weight=1.5,
                operational_credit=0.45,
                structural_cap=0.55,
                gaps=[
                    "current retrieval is a deterministic baseline rather than learned clustering/KNN",
                    "analog quality requires replay-confirmed false-analogy statistics",
                ],
            ),
        ]

        replay_checks = [
            self._file_check(
                root,
                "fixed_horizon_outcomes",
                "replay_learning",
                ["dean_os/world_state_outcomes.py", "dean_os/world_state_outcome_cli.py"],
                structural_weight=2.0,
                operational_weight=2.0,
                operational_credit=0.55,
                structural_cap=0.8,
                gaps=["real outcome evidence adapters and scheduled due-task execution are not connected"],
            ),
            self._file_check(
                root,
                "review_gated_calibration",
                "replay_learning",
                ["dean_os/world_state_outcomes.py"],
                structural_weight=1.5,
                operational_weight=2.0,
                operational_credit=0.3,
                structural_cap=0.45,
                gaps=[
                    "no approved calibration sample exists yet",
                    "no shadow-trained calibration artifact exists yet",
                    "learning promotion remains proposal-only by design",
                ],
            ),
            self._manual_check(
                "due_task_scheduler",
                "replay_learning",
                structural_credit=0.25,
                operational_credit=0.05,
                structural_weight=1.0,
                operational_weight=1.5,
                evidence=["replay task due_at contracts exist"],
                gaps=["no recurring task runner automatically evaluates due horizons"],
            ),
        ]

        governance_checks = [
            self._file_check(
                root,
                "review_only_authority_boundaries",
                "governance",
                [
                    "dean_os/minimal_system.py",
                    "dean_os/world_state_store.py",
                    "dean_os/world_state_outcomes.py",
                ],
                structural_weight=2.0,
                operational_weight=1.0,
                operational_credit=0.8,
                structural_cap=0.9,
                gaps=["production identity/access and external approval workflow are outside this package"],
            ),
            self._file_check(
                root,
                "contract_tests",
                "governance",
                [
                    "tests/test_minimal_system.py",
                    "tests/test_world_state_store.py",
                    "tests/test_world_state_outcomes.py",
                ],
                structural_weight=1.5,
                operational_weight=1.5,
                operational_credit=0.75,
                structural_cap=0.7,
                gaps=["tests are contract/integration fixtures, not full production-system tests"],
            ),
        ]

        branches = [
            self._branch("pipeline", pipeline_checks, deferred=pipeline_deferred),
            self._branch("analytical", analytical_checks),
            self._branch("world_model", world_model_checks),
            self._branch("replay_learning", replay_checks),
        ]
        cross_cutting = governance_checks
        all_checks = [check for branch in branches for check in branch.checks] + cross_cutting
        structural = self._weighted_score(all_checks, "structural")
        operational = self._weighted_score(all_checks, "operational")
        if operational >= 0.75:
            status = "operational_mvp"
        elif structural >= 0.70 and operational >= 0.45:
            status = "runnable_structural_mvp_with_major_operational_gaps"
        elif structural >= 0.55:
            status = "integrated_scaffold"
        else:
            status = "early_scaffold"
        return AgentSystemReadinessReport(
            domain_id=domain_id,
            mode="agent_first_pipeline_deferred" if pipeline_deferred else "hybrid",
            structural_readiness=round(structural, 4),
            operational_readiness=round(operational, 4),
            overall_status=status,
            branches=branches,
            cross_cutting_checks=cross_cutting,
            interpretation={
                "structural_readiness": (
                    "How much of the intended architecture has executable contracts, composition, "
                    "persistence and tests."
                ),
                "operational_readiness": (
                    "How much has been demonstrated on real recurring data, reviewed outcomes, "
                    "calibrated probabilities and production operations."
                ),
                "not_a_certification": True,
            },
        )

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _file_check(
        root: Path,
        check_id: str,
        area: str,
        relative_paths: list[str],
        *,
        structural_weight: float,
        operational_weight: float,
        operational_credit: float,
        gaps: list[str],
        structural_cap: float = 1.0,
    ) -> ReadinessCheck:
        existing = [path for path in relative_paths if (root / path).exists()]
        missing = [path for path in relative_paths if not (root / path).exists()]
        structural_credit = (
            (len(existing) / len(relative_paths)) * structural_cap
            if relative_paths
            else 0.0
        )
        status = "implemented" if not missing else "partial" if existing else "missing"
        return ReadinessCheck(
            check_id=check_id,
            area=area,
            status=status,
            structural_weight=structural_weight,
            operational_weight=operational_weight,
            structural_credit=structural_credit,
            operational_credit=operational_credit if not missing else operational_credit * structural_credit,
            evidence=existing,
            gaps=sorted(set(gaps + [f"missing:{path}" for path in missing])),
        )

    @staticmethod
    def _manual_check(
        check_id: str,
        area: str,
        *,
        structural_credit: float,
        operational_credit: float,
        structural_weight: float,
        operational_weight: float,
        evidence: list[str],
        gaps: list[str],
    ) -> ReadinessCheck:
        if structural_credit >= 0.85:
            status = "implemented"
        elif structural_credit >= 0.4:
            status = "partial"
        else:
            status = "early"
        return ReadinessCheck(
            check_id=check_id,
            area=area,
            status=status,
            structural_weight=structural_weight,
            operational_weight=operational_weight,
            structural_credit=structural_credit,
            operational_credit=operational_credit,
            evidence=evidence,
            gaps=gaps,
        )

    @staticmethod
    def _registry_check(
        check_id: str,
        area: str,
        agents: dict[str, Any],
        *,
        required: list[str],
        structural_weight: float,
        operational_weight: float,
        operational_credit: float,
        gaps: list[str],
        structural_cap: float = 1.0,
    ) -> ReadinessCheck:
        enabled = [
            name
            for name in required
            if isinstance(agents.get(name), dict) and agents[name].get("enabled") is True
        ]
        missing = [name for name in required if name not in enabled]
        structural_credit = len(enabled) / len(required) if required else 0.0
        status = "implemented" if not missing else "partial" if enabled else "missing"
        return ReadinessCheck(
            check_id=check_id,
            area=area,
            status=status,
            structural_weight=structural_weight,
            operational_weight=operational_weight,
            structural_credit=structural_credit,
            operational_credit=operational_credit * structural_credit,
            evidence=[f"enabled_agent:{name}" for name in enabled],
            gaps=sorted(set(gaps + [f"disabled_or_missing_agent:{name}" for name in missing])),
        )

    @classmethod
    def _branch(
        cls,
        branch: str,
        checks: list[ReadinessCheck],
        *,
        deferred: bool = False,
    ) -> BranchReadiness:
        structural = cls._weighted_score(checks, "structural")
        operational = cls._weighted_score(checks, "operational")
        if deferred:
            status = "prepared_boundary_deferred"
        elif structural >= 0.9 and operational >= 0.65:
            status = "runnable_mvp"
        elif structural >= 0.75:
            status = "integrated_with_operational_gaps"
        elif structural >= 0.5:
            status = "partial"
        else:
            status = "early"
        return BranchReadiness(
            branch=branch,
            status=status,
            structural_score=round(structural, 4),
            operational_score=round(operational, 4),
            checks=checks,
        )

    @staticmethod
    def _weighted_score(checks: list[ReadinessCheck], kind: str) -> float:
        if kind == "structural":
            weights = [item.structural_weight for item in checks]
            credits = [item.structural_credit for item in checks]
        else:
            weights = [item.operational_weight for item in checks]
            credits = [item.operational_credit for item in checks]
        denominator = sum(weights)
        if denominator <= 0:
            return 0.0
        return sum(weight * credit for weight, credit in zip(weights, credits)) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess DEAN-OS agent-system readiness.")
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--domain", default="semiconductor_ai_infrastructure")
    parser.add_argument("--pipeline-active", action="store_true")
    args = parser.parse_args()
    report = AgentSystemReadinessAssessor().assess(
        package_root=args.package_root,
        domain_id=args.domain,
        pipeline_deferred=not args.pipeline_active,
    )
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
