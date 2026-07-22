from __future__ import annotations

import json
from pathlib import Path

import pytest

from dean_os.current_system_manifest import CurrentSystemManifestBuilder
from dean_os.system_topology import SystemTopology, load_system_topology


def test_default_topology_is_acyclic_and_registers_authorization_ledger() -> None:
    topology = load_system_topology()
    order = [branch.branch_id for branch in topology.execution_order()]
    assert order[0] == "artifact_intake"
    assert order.index("domain_analysis") > order.index("evidence_intelligence")
    assert order.index("operations_authorization") > order.index("governance_review")
    assert order[-1] == "system_audit"
    assert len(topology.topology_sha256) == 64


def test_topology_rejects_cycle() -> None:
    payload = load_system_topology().model_dump(mode="json")
    payload["topology_sha256"] = ""
    payload["branches"][0]["depends_on"] = ["system_audit"]
    with pytest.raises(ValueError, match="cycle"):
        SystemTopology(**payload)


def test_manifest_is_honest_artifact_observation_and_registers_empty_ledger(tmp_path: Path) -> None:
    topology = Path("dean_os/config/system_topology.yaml")
    output = tmp_path / "out"
    payload = CurrentSystemManifestBuilder(
        output_dir=output,
        topology_path=topology,
        authorization_ledger_path=tmp_path / "authorization.jsonl",
    ).build(as_of="2026-07-12T12:00:00+00:00", save=False)
    assert payload["safety"]["artifact_observation_only"] is True
    assert payload["safety"]["independent_branch_execution_claimed"] is False
    assert payload["safety"]["operational_readiness_claimed"] is False
    assert payload["safety"]["authorization_ledger_registered"] is True
    auth = next(item for item in payload["branch_records"] if item["branch_id"] == "operations_authorization")
    assert auth["summary"]["record_count"] == 0
    assert auth["summary"]["chain_valid"] is True
    assert auth["summary"]["empty_ledger_is_valid"] is True
    assert len(payload["manifest_sha256"]) == 64


def test_missing_required_artifact_propagates_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dean_os import current_system_manifest as module

    monkeypatch.setitem(
        module.BRANCH_ARTIFACTS,
        "artifact_intake",
        [("missing", str(tmp_path / "missing.json"), True)],
    )
    payload = CurrentSystemManifestBuilder(
        output_dir=tmp_path / "out",
        authorization_ledger_path=tmp_path / "ledger.jsonl",
    ).build(as_of="2026-07-12T12:00:00+00:00", save=False)
    intake = next(item for item in payload["branch_records"] if item["branch_id"] == "artifact_intake")
    control = next(item for item in payload["branch_records"] if item["branch_id"] == "pipeline_control")
    assert intake["status"] == "missing"
    assert control["status"] == "blocked"
    assert payload["status"] == "observed_blocked"
