from __future__ import annotations

import json
from pathlib import Path

import pytest

from dean_os.execution.maturity_gates import REPLAY_GATE_CHECKS, run_promotion_pipeline
from dean_os.strategy_maturity_operations import (
    StrategyMaturityDailyReconciler,
    StrategyMaturityDecisionLedger,
    StrategyReplayCandidateAssessment,
    journal_simulated_order_decision,
)
from dean_os.system_journal import SystemJournal


HYPOTHESIS_ID = "hypothesis_real_reviewed_1"


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _review_gate(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "review_gate.json",
        {
            "run_id": "review_gate_fixture",
            "created_at": "2026-07-16T10:00:00+00:00",
            "mode": "world_model_replay_review_gate",
            "contract": "dean_world_model_replay_review_gate_v1",
            "source_packet": {
                "path": "source_packet.json",
                "sha256": "a" * 64,
                "run_id": "source_packet_1",
            },
            "summary": {
                "approved": True,
                "manual_hypothesis_review_complete": True,
                "can_register_replay_tasks": True,
                "replay_task_registration_performed": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "hypothesis_review": [
                {
                    "hypothesis_id": HYPOTHESIS_ID,
                    "hypothesis": "A reviewed event-response hypothesis.",
                    "disposition": "accept_for_replay",
                    "registration_blockers": [],
                    "trigger_event": {"record_sha256": "b" * 64},
                }
            ],
            "registration_bundle": {
                "tasks": [
                    {
                        "task_id": f"replay_{HYPOTHESIS_ID}_20d",
                        "hypothesis_id": HYPOTHESIS_ID,
                        "horizon_days": 20,
                        "packet_as_of": "2026-07-16T09:00:00+00:00",
                        "trigger_event_at": "2026-06-25T07:53:06+00:00",
                        "trigger_evidence_id": "evidence_1",
                        "resolution_lineage": {
                            "source_packet_sha256": "c" * 64,
                            "source_review_gate_sha256": "d" * 64,
                        },
                    }
                ]
            },
        },
    )


def _assessment(tmp_path: Path, *, apply: bool = True) -> dict:
    return StrategyReplayCandidateAssessment(tmp_path / "assessment").build(
        review_gate_path=_review_gate(tmp_path),
        ledger_path=tmp_path / "maturity.jsonl",
        journal_path=tmp_path / "journal.jsonl",
        apply_ledger=apply,
        apply_journal=apply,
        save=True,
    )


def test_real_reviewed_hypothesis_is_evaluated_but_not_promoted(tmp_path: Path) -> None:
    payload = _assessment(tmp_path)

    assert payload["summary"]["gate_decision"] == "blocked"
    assert set(payload["summary"]["failed_checks"]) == {
        "no_future_leakage",
        "model_state_manifest_present",
        "risk_limits_simulated",
        "outcome_review_generated",
    }
    assert payload["summary"]["ledger_appended"] is True
    assert payload["summary"]["strategy_registry_mutated"] is False
    assert payload["summary"]["replay_task_registered"] is False
    assert payload["summary"]["strategy_promoted"] is False
    assert payload["strategy_playbook"]["promotion_policy"]["current_maturity_level"] == "research"
    assert payload["strategy_playbook"]["promotion_policy"]["next_allowed_level"] == "replay"


def test_blocked_gate_receipt_is_hash_chained_and_journaled(tmp_path: Path) -> None:
    payload = _assessment(tmp_path)
    ledger = StrategyMaturityDecisionLedger(tmp_path / "maturity.jsonl")

    assert ledger.status()["record_count"] == 1
    assert ledger.read_verified()[0]["gate_receipt"]["decision"] == "blocked"
    assert payload["journal"]["appended_count"] == 1
    assert SystemJournal(tmp_path / "journal.jsonl").status()["chain_valid"] is True


def test_assessment_without_apply_is_non_mutating(tmp_path: Path) -> None:
    payload = _assessment(tmp_path, apply=False)
    assert payload["summary"]["ledger_appended"] is False
    assert payload["ledger"]["record_count"] == 0
    assert payload["journal"]["appended_count"] == 0


def test_daily_reconciliation_keeps_blocked_candidate_at_research(tmp_path: Path) -> None:
    assessment = _assessment(tmp_path)
    result = StrategyMaturityDailyReconciler(tmp_path / "reconcile").build(
        candidate_assessment_path=assessment["saved_paths"]["latest_json"],
        ledger_path=tmp_path / "maturity.jsonl",
        journal_path=tmp_path / "journal.jsonl",
        apply_journal=True,
        save=False,
    )

    assert result["summary"]["status"] == "maturity_reconciliation_valid"
    assert result["summary"]["registry_maturity_level"] == "research"
    assert result["summary"]["derived_approved_maturity_level"] == "research"
    assert result["summary"]["latest_gate_decision"] == "blocked"
    assert result["summary"]["approved_decision_count"] == 0
    assert result["summary"]["strategy_promoted"] is False
    assert result["summary"]["paper_execution_performed"] is False


def test_reconciliation_detects_changed_gate_evidence(tmp_path: Path) -> None:
    assessment = _assessment(tmp_path)
    gate_path = tmp_path / "review_gate.json"
    gate_path.write_text("changed", encoding="utf-8")
    result = StrategyMaturityDailyReconciler(tmp_path / "reconcile").build(
        candidate_assessment_path=assessment["saved_paths"]["latest_json"],
        ledger_path=tmp_path / "maturity.jsonl",
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert result["summary"]["status"] == "maturity_reconciliation_blocked"
    assert any(
        item.startswith("latest_receipt_invalid:maturity_receipt_evidence_hash_invalid")
        for item in result["summary"]["structural_blockers"]
    )


def test_approved_receipt_cannot_silently_outrun_registry_state(tmp_path: Path) -> None:
    assessment = _assessment(tmp_path, apply=False)
    assessment_path = Path(assessment["saved_paths"]["latest_json"])
    strategy_id = assessment["strategy_id"]
    evidence = _review_gate(tmp_path)
    approved = run_promotion_pipeline(
        strategy_id,
        "replay",
        {key: True for key in REPLAY_GATE_CHECKS},
        approver="operator_a",
        evidence_artifacts={"review": evidence},
    )["receipt"]
    StrategyMaturityDecisionLedger(tmp_path / "maturity.jsonl").append(
        receipt=approved,
        source_artifact_path=evidence,
    )
    result = StrategyMaturityDailyReconciler(tmp_path / "reconcile").build(
        candidate_assessment_path=assessment_path,
        ledger_path=tmp_path / "maturity.jsonl",
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert result["summary"]["derived_approved_maturity_level"] == "replay"
    assert "registry_maturity_does_not_match_approved_receipt" in result["summary"]["structural_blockers"]


def test_replay_maturity_requires_valid_strategy_risk_snapshot(tmp_path: Path) -> None:
    assessment = _assessment(tmp_path, apply=False)
    assessment_path = Path(assessment["saved_paths"]["latest_json"])
    stored = json.loads(assessment_path.read_text(encoding="utf-8"))
    stored["strategy_playbook"]["status"] = "replay"
    stored["strategy_playbook"]["promotion_policy"]["current_maturity_level"] = "replay"
    stored["strategy_playbook"]["promotion_policy"]["next_allowed_level"] = "paper"
    assessment_path.write_text(json.dumps(stored), encoding="utf-8")
    strategy_id = assessment["strategy_id"]
    evidence = _review_gate(tmp_path)
    approved = run_promotion_pipeline(
        strategy_id,
        "replay",
        {key: True for key in REPLAY_GATE_CHECKS},
        approver="operator_a",
        evidence_artifacts={"review": evidence},
    )["receipt"]
    StrategyMaturityDecisionLedger(tmp_path / "maturity.jsonl").append(
        receipt=approved,
        source_artifact_path=evidence,
    )

    missing = StrategyMaturityDailyReconciler(tmp_path / "reconcile").build(
        candidate_assessment_path=assessment_path,
        ledger_path=tmp_path / "maturity.jsonl",
        save=False,
    )
    assert "strategy_risk_snapshot_missing" in missing["summary"]["structural_blockers"]

    risk_path = _write(
        tmp_path / "risk.json",
        {
            "contract": "dean_strategy_risk_snapshot_v1",
            "strategy_id": strategy_id,
            "summary": {"risk_check_passed": True, "kill_switch_active": False},
        },
    )
    valid = StrategyMaturityDailyReconciler(tmp_path / "reconcile").build(
        candidate_assessment_path=assessment_path,
        ledger_path=tmp_path / "maturity.jsonl",
        risk_snapshot_path=risk_path,
        save=False,
    )
    assert valid["summary"]["status"] == "maturity_reconciliation_valid"
    assert valid["summary"]["risk_snapshot_valid"] is True


def test_maturity_ledger_detects_record_tampering(tmp_path: Path) -> None:
    _assessment(tmp_path)
    path = tmp_path / "maturity.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["gate_receipt"]["strategy_id"] = "tampered"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ledger hash mismatch"):
        StrategyMaturityDecisionLedger(path).read_verified()


def test_simulated_order_decision_uses_canonical_journal_and_rejects_live(tmp_path: Path) -> None:
    source = _review_gate(tmp_path)
    order = {
        "order_id": "order_1",
        "strategy_id": "strategy_1",
        "domain_id": "semiconductor_ai_infrastructure",
        "mode": "paper",
        "created_at": "2026-07-16T11:00:00+00:00",
    }
    result = {
        "decision": "approved_simulated",
        "lineage_id": "lineage_1",
        "maturity_receipt_sha256": "a" * 64,
    }
    first = journal_simulated_order_decision(
        order=order,
        result=result,
        source_artifact_path=source,
        journal_path=tmp_path / "journal.jsonl",
        apply=True,
    )
    second = journal_simulated_order_decision(
        order=order,
        result=result,
        source_artifact_path=source,
        journal_path=tmp_path / "journal.jsonl",
        apply=True,
    )
    assert first["appended_count"] == 1
    assert second["appended_count"] == 0

    with pytest.raises(ValueError, match="only paper/shadow"):
        journal_simulated_order_decision(
            order={**order, "mode": "supervised_live"},
            result=result,
            source_artifact_path=source,
        )
