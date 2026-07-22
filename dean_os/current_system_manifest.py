from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dean_os.accumulation_authorization_ledger import AccumulationAuthorizationLedger
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.system_topology import SystemOperatingProfile, SystemTopology, load_system_topology
from dean_os.utils import sha256_json


class ManifestArtifactReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    path: str
    required: bool
    exists: bool
    sha256: str | None = None
    contract: str | None = None
    created_at: str | None = None


class SystemBranchRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    branch_id: str
    status: Literal["present", "partial", "missing", "blocked"]
    measurement_mode: Literal["artifact_observed", "ledger_observed", "manifest_assembled"]
    artifacts: list[ManifestArtifactReference] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    record_sha256: str = ""

    @model_validator(mode="after")
    def bind_hash(self) -> "SystemBranchRecord":
        expected = sha256_json(self.model_dump(mode="json", exclude={"record_sha256"}))
        if self.record_sha256 and self.record_sha256 != expected:
            raise ValueError("system branch record hash mismatch")
        object.__setattr__(self, "record_sha256", expected)
        return self


class CurrentSystemManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: str = "dean_current_system_manifest_v1"
    run_id: str
    created_at: str
    as_of: str
    domain_id: str
    operating_profile: SystemOperatingProfile
    topology_id: str
    topology_sha256: str
    status: Literal["observed_complete", "observed_partial", "observed_blocked"]
    branch_records: list[SystemBranchRecord]
    safety: dict[str, bool]
    manifest_sha256: str = ""

    @model_validator(mode="after")
    def bind_hash(self) -> "CurrentSystemManifest":
        expected = sha256_json(self.model_dump(mode="json", exclude={"manifest_sha256"}))
        if self.manifest_sha256 and self.manifest_sha256 != expected:
            raise ValueError("current system manifest hash mismatch")
        object.__setattr__(self, "manifest_sha256", expected)
        return self


BRANCH_ARTIFACTS: dict[str, list[tuple[str, str, bool]]] = {
    "artifact_intake": [
        ("clean_market_15m_60m_1d", "reports/dean_os/clean_market_snapshot_current/latest.json", True),
        ("saved_news", "reports/dean_os/saved_semiconductor_news_evidence_producer_current/latest.json", False),
        ("saved_macro", "reports/dean_os/saved_macro_evidence_producer_current/latest.json", False),
        ("saved_sec", "reports/dean_os/saved_sec_fundamental_evidence_merger_current/latest.json", False),
    ],
    "pipeline_control": [
        ("timeframe_lane_readiness", "reports/dean_os/pipeline_timeframe_lane_readiness_clean_current/latest.json", True),
        ("world_model_pipeline_context", "reports/dean_os/world_model_pipeline_context_clean_current/latest.json", True),
    ],
    "evidence_intelligence": [
        ("saved_news", "reports/dean_os/saved_semiconductor_news_evidence_producer_current/latest.json", True),
        ("saved_macro", "reports/dean_os/saved_macro_evidence_producer_current/latest.json", True),
        ("saved_sec", "reports/dean_os/saved_sec_fundamental_evidence_merger_current/latest.json", True),
    ],
    "domain_analysis": [
        ("semiconductor_domain_report", "reports/dean_os/domain_analyst_review_clean_current/latest.json", True),
    ],
    "world_model": [
        ("world_model_event_learning", "reports/dean_os/world_model_event_learning_cycle_current/latest.json", True),
    ],
    "replay_evaluation": [
        ("replay_evidence_plan", "reports/dean_os/replay_outcome_evidence_plan_current/latest.json", True),
        ("checkpoint_monitor", "reports/dean_os/replay_checkpoint_monitor_current/latest.json", True),
    ],
    "governance_review": [
        ("current_cycle_closure", "reports/dean_os/full_system_cycle_closure_current/latest.json", True),
        ("review_decision_state", "reports/dean_os/review_decision_state_current/latest.json", True),
        ("unknown_voi_review", "reports/dean_os/unknown_voi_review_current/latest.json", True),
    ],
}


class CurrentSystemManifestBuilder:
    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/current_system_manifest",
        topology_path: str | Path = "dean_os/config/system_topology.yaml",
        authorization_ledger_path: str | Path = "data/dean_os/accumulation_authorization_ledger.jsonl",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.topology_path = Path(topology_path)
        self.authorization_ledger_path = Path(authorization_ledger_path)

    def build(
        self,
        *,
        as_of: str,
        domain_id: str = "semiconductor_ai_infrastructure",
        operating_profile: SystemOperatingProfile = SystemOperatingProfile.REPLAY_READY,
        save: bool = True,
    ) -> dict[str, Any]:
        if parse_timezone_aware(as_of) is None:
            raise ValueError("manifest as_of must be timezone-aware")
        topology = load_system_topology(self.topology_path)
        records: list[SystemBranchRecord] = []
        by_id: dict[str, SystemBranchRecord] = {}
        for spec in topology.execution_order():
            if spec.branch_id == "operations_authorization":
                record = _authorization_record(spec.branch_id, self.authorization_ledger_path)
            elif spec.branch_id == "system_audit":
                record = _audit_record(spec.branch_id, records)
            else:
                record = _artifact_record(spec.branch_id, BRANCH_ARTIFACTS.get(spec.branch_id, []))
            dependency_blocks = [
                dependency
                for dependency in spec.depends_on
                if by_id.get(dependency) and by_id[dependency].status in {"missing", "blocked"}
            ]
            if dependency_blocks and spec.required:
                record = record.model_copy(
                    update={
                        "status": "blocked",
                        "warnings": record.warnings + [f"blocked by dependencies: {dependency_blocks}"],
                        "record_sha256": "",
                    }
                )
                record = SystemBranchRecord(**record.model_dump(mode="json"))
            records.append(record)
            by_id[spec.branch_id] = record
        required_statuses = [
            by_id[spec.branch_id].status for spec in topology.execution_order() if spec.required
        ]
        status = "observed_blocked" if "blocked" in required_statuses or "missing" in required_statuses else (
            "observed_partial" if "partial" in required_statuses else "observed_complete"
        )
        created_at = utc_now_iso()
        manifest = CurrentSystemManifest(
            run_id="current_system_manifest_" + created_at.replace(":", "").replace("+00:00", "Z"),
            created_at=created_at,
            as_of=parse_timezone_aware(as_of).isoformat(),
            domain_id=domain_id,
            operating_profile=operating_profile,
            topology_id=topology.topology_id,
            topology_sha256=topology.topology_sha256,
            status=status,
            branch_records=records,
            safety={
                "artifact_observation_only": True,
                "independent_branch_execution_claimed": False,
                "operational_readiness_claimed": False,
                "authorization_ledger_registered": True,
                "collector_execution_performed": False,
                "pipeline_execution_performed": False,
                "learning_write_performed": False,
                "can_trade": False,
            },
        )
        payload = manifest.model_dump(mode="json")
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=manifest.run_id,
            )
        return payload


def _artifact_record(branch_id: str, specs: list[tuple[str, str, bool]]) -> SystemBranchRecord:
    artifacts = [_inspect_artifact(role, path, required) for role, path, required in specs]
    missing_required = [item.role for item in artifacts if item.required and not item.exists]
    missing_optional = [item.role for item in artifacts if not item.required and not item.exists]
    status: Literal["present", "partial", "missing", "blocked"] = (
        "missing" if missing_required and len(missing_required) == sum(item.required for item in artifacts)
        else "partial" if missing_required or missing_optional else "present"
    )
    return SystemBranchRecord(
        branch_id=branch_id,
        status=status,
        measurement_mode="artifact_observed",
        artifacts=artifacts,
        summary={"artifact_count": len(artifacts), "present_count": sum(item.exists for item in artifacts)},
        warnings=[f"missing required artifact: {role}" for role in missing_required]
        + [f"missing optional artifact: {role}" for role in missing_optional],
    )


def _authorization_record(branch_id: str, ledger_path: Path) -> SystemBranchRecord:
    status = AccumulationAuthorizationLedger(ledger_path).status()
    return SystemBranchRecord(
        branch_id=branch_id,
        status="present",
        measurement_mode="ledger_observed",
        artifacts=[
            ManifestArtifactReference(
                role="accumulation_authorization_ledger",
                path=str(ledger_path),
                required=True,
                exists=ledger_path.exists(),
                sha256=_sha256_file(ledger_path) if ledger_path.exists() else sha256_json(status),
                contract=status["contract"],
            )
        ],
        summary={
            "record_count": status["record_count"],
            "chain_valid": status["chain_valid"],
            "empty_ledger_is_valid": True,
        },
        warnings=[] if ledger_path.exists() else ["authorization ledger has no records yet"],
    )


def _audit_record(branch_id: str, prior: list[SystemBranchRecord]) -> SystemBranchRecord:
    return SystemBranchRecord(
        branch_id=branch_id,
        status="present",
        measurement_mode="manifest_assembled",
        summary={
            "linked_branch_count": len(prior),
            "linked_branch_hashes": {item.branch_id: item.record_sha256 for item in prior},
        },
    )


def _inspect_artifact(role: str, path_text: str, required: bool) -> ManifestArtifactReference:
    path = Path(path_text)
    contract = None
    created_at = None
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                contract = payload.get("contract") or payload.get("producer_contract") or payload.get("mode")
                created_at = payload.get("created_at")
        except (OSError, json.JSONDecodeError):
            pass
    return ManifestArtifactReference(
        role=role,
        path=str(path),
        required=required,
        exists=path.exists(),
        sha256=_sha256_file(path) if path.exists() else None,
        contract=contract,
        created_at=created_at,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Current DEAN-OS System Manifest",
        "",
        f"- Status: `{payload['status']}`",
        f"- Operating profile: `{payload['operating_profile']}`",
        f"- Topology SHA-256: `{payload['topology_sha256']}`",
        f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
        "- Authorization ledger registered: `true`",
        "- Independent branch execution claimed: `false`",
        "- Operational readiness claimed: `false`",
        "",
        "## Branches",
        "",
    ]
    lines.extend(
        f"- `{record['branch_id']}`: `{record['status']}` ({record['measurement_mode']})"
        for record in payload["branch_records"]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "CurrentSystemManifest",
    "CurrentSystemManifestBuilder",
    "ManifestArtifactReference",
    "SystemBranchRecord",
]
