from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dean_os.context_acquisition_state_machine import ContextAcquisitionStateMachine
from dean_os.analyst_core.domain_analyst_binding_planner import DomainAnalystBindingPlanner
from dean_os.domain_scoped_pipeline_context_envelope import (
    CONTRACT,
    DomainScopedPipelineContextEnvelope,
    load_verified_domain_pipeline_context_fragment,
)
from dean_os.system_journal import SystemJournal
from dean_os.world_model_pipeline_context import WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT


AS_OF = "2026-07-10T12:00:00+00:00"
DOMAIN = "semiconductor_ai_infrastructure"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _lineage(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "pipeline_stage23" / "latest.json",
        {
            "run_id": "stage23_fixture",
            "mode": "pipeline_stage23_regeneration",
            "status": "stage23_regeneration_review_ready",
        },
    )


def _source(tmp_path: Path, *, ticker: str = "NVDA") -> Path:
    lineage = _lineage(tmp_path)
    return _write(
        tmp_path / "pipeline_context.json",
        {
            "run_id": "pipeline_context_fixture",
            "created_at": "2026-07-10T10:00:00+00:00",
            "mode": "world_model_pipeline_context_discovery",
            "contract": WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT,
            "requested": {"tickers": [ticker], "timeframes": ["15m", "60m", "1d"]},
            "summary": {
                "status": "pipeline_context_bundle_ready_with_gaps",
                "available_lane_count": 1,
                "exact_context_lane_count": 1,
                "missing_lane_count": 2,
                "can_register_replay_tasks": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "pipeline_context": {"status": "pipeline_context_bundle_ready_with_gaps"},
            "indicator_state_grid": {"status": "indicator_state_grid_ready_with_gaps"},
            "timeframe_lanes": [{"timeframe": "15m", "exact_context_ready": True}],
            "artifact_inventory": {
                "stage23_regeneration": [
                    {
                        "available": True,
                        "path": str(lineage),
                        "sha256": _sha(lineage),
                    }
                ]
            },
            "safety": {
                "review_only": True,
                "pipeline_regeneration_performed": False,
                "stage4_run_performed": False,
                "stage5_run_performed": False,
                "replay_task_registration_performed": False,
                "learning_memory_write_performed": False,
                "production_config_write_performed": False,
                "model_promotion_performed": False,
                "can_trade": False,
            },
        },
    )


def _dispatch(tmp_path: Path, domain_id: str = DOMAIN) -> Path:
    return _write(
        tmp_path / "dispatch.json",
        {
            "run_id": "dispatch_fixture",
            "created_at": AS_OF,
            "mode": "domain_binding_task_dispatch",
            "summary": {"domain_id": domain_id, "structural_blockers": []},
            "task_dispatches": [
                {
                    "task_id": f"bind_{domain_id}_pipeline_context",
                    "domain_id": domain_id,
                    "context_family": "pipeline_context",
                    "recommended_action": "domain_scoped_pipeline_context_envelope",
                }
            ],
        },
    )


def _build(tmp_path: Path, **kwargs) -> dict:
    source_path = kwargs.pop("source_path", None)
    dispatch_path = kwargs.pop("dispatch_path", None)
    return DomainScopedPipelineContextEnvelope(tmp_path / "report").build(
        domain_id=kwargs.pop("domain_id", DOMAIN),
        as_of=kwargs.pop("as_of", AS_OF),
        source_path=source_path or _source(tmp_path),
        dispatch_path=dispatch_path or _dispatch(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=kwargs.pop("save", False),
        **kwargs,
    )


def test_envelope_binds_exact_local_context_without_running_pipeline(tmp_path: Path) -> None:
    payload = _build(tmp_path)

    assert payload["contract"] == CONTRACT
    assert payload["source_producer_contract"] == WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT
    assert payload["summary"]["status"] == "domain_pipeline_context_candidate_ready_with_gaps"
    assert payload["summary"]["candidate_ready_for_binding_review"] is True
    assert payload["summary"]["lineage_verified_count"] == 1
    assert payload["summary"]["pipeline_stage_run_performed"] is False
    assert payload["summary"]["binding_accepted"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["domain_binding"]["may_replace_verified_market_outcome"] is False


def test_cross_domain_ticker_scope_is_blocked(tmp_path: Path) -> None:
    payload = _build(
        tmp_path,
        domain_id="energy",
        source_path=_source(tmp_path, ticker="NVDA"),
        dispatch_path=_dispatch(tmp_path, "energy"),
    )

    assert "pipeline_context_ticker_outside_domain:NVDA" in payload["summary"]["structural_blockers"]
    assert payload["summary"]["candidate_ready_for_binding_review"] is False


def test_future_source_and_unsafe_authority_fail_closed(tmp_path: Path) -> None:
    source_path = _source(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["created_at"] = "2026-07-11T10:00:00+00:00"
    source["summary"]["can_register_replay_tasks"] = True
    source["safety"]["can_trade"] = True
    _write(source_path, source)
    payload = _build(tmp_path, source_path=source_path)

    blockers = payload["summary"]["structural_blockers"]
    assert "pipeline_context_source_after_as_of" in blockers
    assert "pipeline_context_replay_authority_invalid" in blockers
    assert "pipeline_context_safety_invalid:can_trade" in blockers


def test_tampered_lineage_reference_is_rejected(tmp_path: Path) -> None:
    source_path = _source(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    lineage_path = Path(source["artifact_inventory"]["stage23_regeneration"][0]["path"])
    lineage_path.write_text("tampered", encoding="utf-8")
    payload = _build(tmp_path, source_path=source_path)

    assert "pipeline_lineage_sha_mismatch:stage23_regeneration" in payload["summary"]["structural_blockers"]
    assert payload["lineage_verification"]["all_references_verified"] is False


def test_dispatch_must_match_domain_and_adapter(tmp_path: Path) -> None:
    payload = _build(tmp_path, dispatch_path=_dispatch(tmp_path, "energy"))
    assert "binding_dispatch_domain_mismatch" in payload["summary"]["structural_blockers"]


def test_envelope_is_accepted_as_pipeline_context_candidate_by_binding_planner(tmp_path: Path) -> None:
    envelope = _build(tmp_path, save=True)
    candidate_path = Path(envelope["saved_paths"]["latest_json"])
    plan = DomainAnalystBindingPlanner(tmp_path / "binding_plan").build(
        domain_id=DOMAIN,
        candidate_artifacts={"pipeline_context": [candidate_path]},
        as_of="2026-07-10T13:00:00+00:00",
        save=False,
    )
    family = next(item for item in plan["family_plans"] if item["context_family"] == "pipeline_context")

    assert family["status"] == "reuse_candidate_ready_for_review"
    assert family["proposed_candidate"]["validation_status"] == "valid"
    assert family["binding_written"] is False


def test_recursive_loader_rechecks_pipeline_inventory(tmp_path: Path) -> None:
    envelope = _build(tmp_path, save=True)
    fragment = load_verified_domain_pipeline_context_fragment(
        envelope["saved_paths"]["latest_json"],
        expected_domain_id=DOMAIN,
        expected_as_of=AS_OF,
    )

    assert fragment["metadata"]["domain_pipeline_context_envelope_verified"] is True
    assert len(fragment["timeframe_lanes"]) == 1


def test_shared_state_machine_records_pipeline_context_short_route(tmp_path: Path) -> None:
    envelope = _build(tmp_path, save=True)
    machine = ContextAcquisitionStateMachine(
        ledger_path=tmp_path / "transition_ledger.jsonl",
        journal_path=tmp_path / "state_journal.jsonl",
        output_dir=tmp_path / "state_report",
    )
    result = machine.advance(
        acquisition_id="context_semiconductor_pipeline_cycle_1",
        domain_id=DOMAIN,
        context_family="pipeline_context",
        stage_id="candidate_verified",
        artifact_path=envelope["saved_paths"]["latest_json"],
        evaluated_at=AS_OF,
        apply_transition=True,
        apply_journal=True,
        save=False,
    )

    assert result["summary"]["status"] == "transition_recorded"
    assert result["summary"]["state_before"] == "idle"
    assert result["summary"]["persisted_state"] == "awaiting_binding_decision"
    assert result["summary"]["automatic_next_stage_run"] is False
    assert machine.reconcile("context_semiconductor_pipeline_cycle_1")["status"] == "reconciliation_valid"


def test_envelope_journal_is_idempotent(tmp_path: Path) -> None:
    source = _source(tmp_path)
    dispatch = _dispatch(tmp_path)
    first = _build(
        tmp_path,
        source_path=source,
        dispatch_path=dispatch,
        apply_journal=True,
    )
    second = _build(
        tmp_path,
        source_path=source,
        dispatch_path=dispatch,
        apply_journal=True,
    )

    assert first["journal"]["appended_count"] == 1
    assert second["journal"]["appended_count"] == 0
    assert SystemJournal(tmp_path / "journal.jsonl").status()["chain_valid"] is True
