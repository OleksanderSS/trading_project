from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dean_os.context_acquisition_state_machine import (
    ContextAcquisitionStateMachine,
    ContextAcquisitionTransitionLedger,
    DEFAULT_REGISTRY_PATH,
)
from dean_os.system_journal import SystemJournal


AS_OF = "2026-07-14T12:00:00+00:00"
DOMAIN = "energy"
ACQUISITION = "context_energy_macro_cycle_1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _base(contract: str, mode: str) -> dict:
    return {
        "run_id": f"fixture_{mode}",
        "created_at": AS_OF,
        "contract": contract,
        "mode": mode,
        "domain_id": DOMAIN,
    }


def _macro_artifacts(tmp_path: Path) -> dict[str, Path]:
    gap = _write(
        tmp_path / "01_gap.json",
        {
            **_base(
                "dean_domain_macro_binding_quality_review_v1",
                "domain_macro_binding_quality_review",
            ),
            "summary": {
                "status": "quality_review_ready_recommendation_only",
                "recommendation": "replace_candidate",
                "structural_blockers": [],
                "decision_recorded": False,
                "binding_accepted": False,
                "can_invoke_domain_analysis": False,
                "can_trade": False,
            },
        },
    )
    request = _write(
        tmp_path / "02_request.json",
        {
            **_base(
                "dean_domain_macro_collection_request_v1",
                "domain_macro_collection_request",
            ),
            "inputs": {"quality_review_sha256": _sha(gap)},
            "summary": {
                "status": "macro_collection_request_ready",
                "structural_blockers": [],
                "request_required": True,
                "execution_authorized": False,
                "collector_run_performed": False,
                "binding_accepted": False,
                "can_trade": False,
            },
        },
    )
    gate = _write(
        tmp_path / "03_gate.json",
        {
            **_base(
                "dean_domain_macro_collection_execution_gate_v1",
                "domain_macro_collection_execution_gate",
            ),
            "inputs": {"request_sha256": _sha(request)},
            "summary": {
                "status": "macro_collection_execution_ready_single_run",
                "structural_blockers": [],
                "single_run_authorized": True,
                "execution_ticket_issued": True,
                "maximum_collection_runs": 1,
                "automatic_retry_allowed": False,
                "collector_run_performed": False,
                "binding_accepted": False,
                "can_trade": False,
            },
            "execution_ticket": {
                "maximum_collection_runs": 1,
                "automatic_retry_allowed": False,
                "consumed": False,
            },
        },
    )
    executor = _write(
        tmp_path / "04_executor.json",
        {
            **_base(
                "dean_domain_macro_collection_executor_v1",
                "domain_macro_collection_executor",
            ),
            "inputs": {"gate_sha256": _sha(gate)},
            "summary": {
                "status": "macro_collection_execution_completed_candidate_ready",
                "structural_blockers": [],
                "ticket_claimed": True,
                "ticket_consumed": True,
                "collector_run_performed": True,
                "snapshot_written": True,
                "second_run_allowed": False,
                "automatic_retry_allowed": False,
                "binding_accepted": False,
                "can_trade": False,
            },
        },
    )
    candidate = _write(tmp_path / "macro_candidate.json", {"candidate": "verified"})
    receipt = _write(
        tmp_path / "05_receipt.json",
        {
            **_base(
                "dean_domain_macro_retrieval_receipt_v1",
                "domain_macro_retrieval_receipt",
            ),
            "inputs": {"executor_sha256": _sha(executor)},
            "summary": {
                "status": "macro_retrieval_receipt_completed_candidate_ready",
                "structural_blockers": [],
                "retrieval_timestamp_applied": True,
                "snapshot_written": True,
                "candidate_ready_for_binding_review": True,
                "network_access_performed": False,
                "second_collection_performed": False,
                "binding_accepted": False,
                "can_trade": False,
            },
            "envelope": {"candidate_path": str(candidate)},
        },
    )
    quality = _write(
        tmp_path / "06_quality.json",
        {
            **_base(
                "dean_domain_macro_binding_quality_review_v1",
                "domain_macro_binding_quality_review",
            ),
            "inputs": {
                "candidate_path": str(candidate),
                "candidate_sha256": _sha(candidate),
            },
            "summary": {
                "status": "quality_review_ready_recommendation_only",
                "recommendation": "accept_binding",
                "structural_blockers": [],
                "decision_recorded": False,
                "binding_accepted": False,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
        },
    )
    return {
        "gap_identified": gap,
        "request_prepared": request,
        "execution_authorized": gate,
        "execution_completed": executor,
        "retrieval_verified": receipt,
        "quality_recommended": quality,
    }


def _machine(tmp_path: Path, registry_path: str | Path = DEFAULT_REGISTRY_PATH):
    return ContextAcquisitionStateMachine(
        registry_path=registry_path,
        ledger_path=tmp_path / "transitions.jsonl",
        journal_path=tmp_path / "journal.jsonl",
        output_dir=tmp_path / "report",
    )


def _advance(
    machine: ContextAcquisitionStateMachine,
    stage: str,
    artifact: Path,
    **kwargs,
) -> dict:
    return machine.advance(
        acquisition_id=ACQUISITION,
        domain_id=DOMAIN,
        context_family="macro",
        stage_id=stage,
        artifact_path=artifact,
        evaluated_at=AS_OF,
        save=False,
        **kwargs,
    )


def test_full_macro_chain_uses_one_universal_state_machine(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    states = []
    for stage, artifact in artifacts.items():
        result = _advance(
            machine,
            stage,
            artifact,
            apply_transition=True,
            apply_journal=True,
        )
        assert result["summary"]["status"] == "transition_recorded"
        assert result["summary"]["automatic_next_stage_run"] is False
        states.append(result["summary"]["persisted_state"])

    assert states == [
        "gap_identified",
        "request_prepared",
        "execution_authorized",
        "execution_completed",
        "retrieval_verified",
        "awaiting_binding_decision",
    ]
    reconciliation = machine.reconcile(ACQUISITION)
    assert reconciliation["status"] == "reconciliation_valid"
    assert reconciliation["current_state"] == "awaiting_binding_decision"
    assert reconciliation["next_stage_id"] is None
    assert reconciliation["authority"]["binding_accepted"] is False
    assert reconciliation["authority"]["trade_executed"] is False
    assert SystemJournal(tmp_path / "journal.jsonl").status()["record_count"] == 6


def test_dry_run_does_not_advance_or_append(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    result = _advance(machine, "gap_identified", artifacts["gap_identified"])

    assert result["summary"]["status"] == "transition_ready_not_recorded"
    assert result["summary"]["proposed_state"] == "gap_identified"
    assert result["summary"]["persisted_state"] == "idle"
    assert result["ledger"]["record_count"] == 0


def test_stage_jump_is_blocked_and_not_persisted(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    result = _advance(
        machine,
        "request_prepared",
        artifacts["request_prepared"],
        apply_transition=True,
    )

    assert result["summary"]["status"] == "transition_blocked"
    assert "non_sequential_transition:idle->request_prepared" in result["summary"]["structural_blockers"]
    assert result["ledger"]["record_count"] == 0


def test_request_must_bind_exact_previous_artifact_sha(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    _advance(
        machine,
        "gap_identified",
        artifacts["gap_identified"],
        apply_transition=True,
    )
    request = json.loads(artifacts["request_prepared"].read_text(encoding="utf-8"))
    request["inputs"]["quality_review_sha256"] = "0" * 64
    _write(artifacts["request_prepared"], request)
    result = _advance(
        machine,
        "request_prepared",
        artifacts["request_prepared"],
        apply_transition=True,
    )

    assert result["summary"]["status"] == "transition_blocked"
    assert any(item.startswith("previous_artifact_sha_binding_failed") for item in result["summary"]["structural_blockers"])


def test_changed_previous_artifact_blocks_next_transition(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    _advance(machine, "gap_identified", artifacts["gap_identified"], apply_transition=True)
    artifacts["gap_identified"].write_text("{}", encoding="utf-8")
    result = _advance(
        machine,
        "request_prepared",
        artifacts["request_prepared"],
        apply_transition=True,
    )

    assert "previous_artifact_sha256_changed" in result["summary"]["structural_blockers"]
    assert result["summary"]["ledger_appended"] is False


def test_executor_stage_requires_single_use_evidence(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    for stage in ("gap_identified", "request_prepared", "execution_authorized"):
        _advance(machine, stage, artifacts[stage], apply_transition=True)
    executor = json.loads(artifacts["execution_completed"].read_text(encoding="utf-8"))
    executor["summary"]["ticket_consumed"] = False
    executor["summary"]["automatic_retry_allowed"] = True
    _write(artifacts["execution_completed"], executor)
    result = _advance(
        machine,
        "execution_completed",
        artifacts["execution_completed"],
        apply_transition=True,
    )

    blockers = result["summary"]["structural_blockers"]
    assert "required_true_failed:summary.ticket_consumed" in blockers
    assert "required_false_failed:summary.automatic_retry_allowed" in blockers


def test_final_quality_must_reference_retrieval_candidate(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    for stage in list(artifacts)[:-1]:
        _advance(machine, stage, artifacts[stage], apply_transition=True)
    quality = json.loads(artifacts["quality_recommended"].read_text(encoding="utf-8"))
    other = _write(tmp_path / "other_candidate.json", {"candidate": "other"})
    quality["inputs"]["candidate_path"] = str(other)
    quality["inputs"]["candidate_sha256"] = _sha(other)
    _write(artifacts["quality_recommended"], quality)
    result = _advance(
        machine,
        "quality_recommended",
        artifacts["quality_recommended"],
        apply_transition=True,
    )

    assert "referenced_artifact_path_binding_failed" in result["summary"]["structural_blockers"]
    assert result["summary"]["persisted_state"] == "retrieval_verified"


def test_identical_transition_is_idempotent(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    first = _advance(
        machine,
        "gap_identified",
        artifacts["gap_identified"],
        apply_transition=True,
        apply_journal=True,
    )
    second = _advance(
        machine,
        "gap_identified",
        artifacts["gap_identified"],
        apply_transition=True,
        apply_journal=True,
    )

    assert first["summary"]["ledger_appended"] is True
    assert second["summary"]["status"] == "transition_already_recorded"
    assert second["ledger"]["record_count"] == 1
    assert SystemJournal(tmp_path / "journal.jsonl").status()["record_count"] == 1


def test_reconciliation_detects_artifact_tampering(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    _advance(machine, "gap_identified", artifacts["gap_identified"], apply_transition=True)
    artifacts["gap_identified"].write_text("tampered", encoding="utf-8")

    result = machine.reconcile(ACQUISITION)
    assert result["status"] == "reconciliation_blocked"
    assert "artifact_gap_identified:sha256_changed" in result["blockers"]


def test_registry_authority_change_blocks_every_transition(tmp_path: Path) -> None:
    registry = json.loads(Path(DEFAULT_REGISTRY_PATH).read_text(encoding="utf-8"))
    registry["authority_boundary"]["trading_allowed"] = True
    registry_path = _write(tmp_path / "unsafe_registry.json", registry)
    artifacts = _macro_artifacts(tmp_path)
    result = _advance(
        _machine(tmp_path, registry_path),
        "gap_identified",
        artifacts["gap_identified"],
        apply_transition=True,
    )

    assert "authority_boundary_not_fail_closed:trading_allowed" in result["summary"]["structural_blockers"]
    assert result["summary"]["ledger_appended"] is False


def test_news_family_cannot_reuse_macro_contract_by_label_only(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    result = machine.advance(
        acquisition_id="context_energy_news_cycle_1",
        domain_id=DOMAIN,
        context_family="news",
        stage_id="gap_identified",
        artifact_path=artifacts["gap_identified"],
        evaluated_at=AS_OF,
        apply_transition=True,
        save=False,
    )

    assert "unknown_stage_for_context_family:gap_identified" in result["summary"]["structural_blockers"]
    assert result["summary"]["ledger_appended"] is False


def test_macro_family_supports_direct_saved_candidate_reuse_route(
    tmp_path: Path,
) -> None:
    candidate = _write(
        tmp_path / "macro_domain_candidate.json",
        {
            **_base(
                "dean_domain_scoped_macro_evidence_envelope_v1",
                "domain_scoped_macro_evidence_envelope",
            ),
            "status": "domain_macro_binding_candidate_ready",
            "summary": {
                "status": "domain_macro_binding_candidate_ready",
                "structural_blockers": [],
                "adapter_run_performed": True,
                "source_lineage_verified": True,
                "candidate_ready_for_binding_review": True,
                "binding_accepted": False,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
        },
    )
    result = _advance(_machine(tmp_path), "candidate_verified", candidate)

    assert result["summary"]["status"] == "transition_ready_not_recorded"
    assert result["summary"]["proposed_state"] == "awaiting_binding_decision"
    assert result["summary"]["persisted_state"] == "idle"
    assert result["summary"]["automatic_next_stage_run"] is False


def test_ledger_detects_record_tampering(tmp_path: Path) -> None:
    artifacts = _macro_artifacts(tmp_path)
    machine = _machine(tmp_path)
    _advance(machine, "gap_identified", artifacts["gap_identified"], apply_transition=True)
    ledger_path = tmp_path / "transitions.jsonl"
    record = json.loads(ledger_path.read_text(encoding="utf-8"))
    record["transition_receipt"]["to_state"] = "execution_completed"
    ledger_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ledger hash mismatch"):
        ContextAcquisitionTransitionLedger(ledger_path).read_verified()
