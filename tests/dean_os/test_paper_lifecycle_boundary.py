from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dean_os.paper_lifecycle_contract import (
    PAPER_LIFECYCLE_SCHEMA_VERSION,
    file_sha256,
    object_fingerprint,
)
from dean_os.paper_simulation_plan import PaperSimulationPlanBuilder
from dean_os.paper_simulation_result import PaperSimulationResultRecorder
from dean_os.post_paper_simulation_review import (
    PostPaperSimulationReviewBuilder,
)
from dean_os.review_decision import ReviewDecisionRecorder


def _write_post_dry_run_review(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": "post_dry_review_1",
                "mode": "post_dry_run_review",
                "post_dry_run_review": {
                    "review_id": "post_dry_review_1",
                    "decision": "ready_for_human_review",
                    "verdict": "clear",
                    "confidence": 0.8,
                    "data_quality_score": 0.75,
                },
                "safety": {
                    "live_execution_allowed": False,
                    "broker_access_allowed": False,
                    "production_config_write_allowed": False,
                    "learning_memory_write_allowed": False,
                    "model_promotion_allowed": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_receipt_and_plan(tmp_path: Path) -> tuple[dict, Path, dict, Path]:
    source_path = tmp_path / "post_dry_review.json"
    _write_post_dry_run_review(source_path)
    receipt = ReviewDecisionRecorder(
        tmp_path / "receipts"
    ).create_receipt(
        reviewer="operator",
        decision="approve_paper_only_simulation",
        rationale="Approve one isolated paper-only simulation.",
        source_artifact_path=source_path,
        expires_at="2099-01-01T00:00:00+00:00",
        save=True,
    )
    receipt_path = Path(receipt["saved_paths"]["latest_json"])
    plan = PaperSimulationPlanBuilder(
        receipt_path=receipt_path,
        output_dir=tmp_path / "plans",
    ).build(save=True)
    plan_path = Path(plan["saved_paths"]["latest_json"])
    return receipt, receipt_path, plan, plan_path


def _guardrail_checks() -> dict[str, bool]:
    return {
        "no_live_execution": True,
        "no_broker_access": True,
        "no_production_config_write": True,
        "no_learning_memory_write": True,
        "no_model_promotion": True,
        "review_artifact_written": True,
    }


def _write_external_result(
    path: Path,
    *,
    plan: dict,
    plan_path: Path,
    executor: str,
    summary: str,
    metrics: list[dict],
    guardrail_checks: dict[str, bool],
) -> None:
    output = {
        "executor": executor,
        "source_plan_id": plan["paper_simulation_plan"]["plan_id"],
        "source_plan_sha256": file_sha256(plan_path),
        "status": "completed",
        "summary": summary,
        "metrics": metrics,
        "warnings": [],
        "errors": [],
        "artifacts": [],
        "guardrail_checks": guardrail_checks,
        "safety": {
            "live_execution_performed": False,
            "broker_access_performed": False,
            "production_config_write_performed": False,
            "learning_write_performed": False,
            "model_promotion_performed": False,
        },
    }
    path.write_text(
        json.dumps(
            {
                "mode": "isolated_paper_simulation_output",
                "schema_version": PAPER_LIFECYCLE_SCHEMA_VERSION,
                "output": output,
                "output_fingerprint": object_fingerprint(output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_paper_receipt_requires_post_dry_run_review_source(tmp_path):
    arbitrary = tmp_path / "pipeline_result.json"
    arbitrary.write_text(
        json.dumps(
            {
                "run_id": "pipeline_1",
                "mode": "pipeline_result",
                "decision": {
                    "decision": "ready_for_human_review",
                    "verdict": "clear",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="post_dry_run_review",
    ):
        ReviewDecisionRecorder(
            tmp_path / "receipts"
        ).create_receipt(
            reviewer="operator",
            decision="approve_paper_only_simulation",
            rationale="This source must not enter paper lifecycle.",
            source_artifact_path=arbitrary,
            expires_at="2099-01-01T00:00:00+00:00",
            save=False,
        )


def test_plan_binds_receipt_and_source_hashes(tmp_path):
    receipt, receipt_path, plan, _ = _build_receipt_and_plan(tmp_path)

    assert receipt["schema_version"] == PAPER_LIFECYCLE_SCHEMA_VERSION
    assert len(receipt["receipt"]["source_artifact_sha256"]) == 64
    assert len(receipt["receipt_fingerprint"]) == 64
    assert plan["paper_simulation_plan"]["status"] == (
        "paper_simulation_plan_ready"
    )
    assert plan["paper_simulation_plan"]["lineage_verified"] is True
    assert plan["paper_simulation_plan"]["source_receipt_sha256"] == (
        file_sha256(receipt_path)
    )
    assert len(plan["plan_fingerprint"]) == 64


def test_plan_blocks_when_review_source_changes_after_receipt(tmp_path):
    receipt, receipt_path, _, _ = _build_receipt_and_plan(tmp_path)
    source_path = Path(receipt["receipt"]["source_artifact_path"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["post_dry_run_review"]["verdict"] = "blocked"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    plan = PaperSimulationPlanBuilder(
        receipt_path=receipt_path,
        output_dir=tmp_path / "tampered_plan",
    ).build(save=False)

    assert plan["paper_simulation_plan"]["status"] == (
        "blocked_lineage_mismatch"
    )
    assert "receipt_source_artifact_sha256_mismatch" in (
        plan["paper_simulation_plan"]["reasons"]
    )


def test_result_recorder_blocks_claim_without_external_executor_manifest(
    tmp_path,
):
    _, _, plan, plan_path = _build_receipt_and_plan(tmp_path)

    result = PaperSimulationResultRecorder(
        plan_path,
        output_dir=tmp_path / "results",
    ).record(
        executor="isolated_paper_executor",
        status="completed",
        summary="Claimed completion without immutable output.",
        guardrail_checks=_guardrail_checks(),
        save=False,
    )

    assert result["paper_simulation_result"]["status"] == (
        "blocked_missing_external_evidence"
    )
    assert result["paper_simulation_result"]["lineage_verified"] is False
    assert result["safety"]["paper_simulation_executed_by_this_layer"] is False
    assert plan["safety"]["paper_simulation_executed"] is False


def test_hash_bound_external_result_reaches_human_review_only(tmp_path):
    _, _, plan, plan_path = _build_receipt_and_plan(tmp_path)
    external_path = tmp_path / "isolated_output.json"
    executor = "isolated_paper_executor"
    summary = "Isolated paper simulation completed."
    metrics = [
        {
            "name": "total_return",
            "value": 0.01,
            "unit": "ratio",
            "interpretation": "diagnostic only",
        }
    ]
    checks = _guardrail_checks()
    _write_external_result(
        external_path,
        plan=plan,
        plan_path=plan_path,
        executor=executor,
        summary=summary,
        metrics=metrics,
        guardrail_checks=checks,
    )

    result = PaperSimulationResultRecorder(
        plan_path,
        output_dir=tmp_path / "results",
    ).record(
        executor=executor,
        status="completed",
        summary=summary,
        metrics=metrics,
        guardrail_checks=checks,
        external_result_path=external_path,
        save=True,
    )
    result_path = Path(result["saved_paths"]["latest_json"])
    review = PostPaperSimulationReviewBuilder(
        result_path,
        output_dir=tmp_path / "post_review",
    ).build(save=False)

    assert result["paper_simulation_result"]["status"] == "completed"
    assert result["paper_simulation_result"]["lineage_verified"] is True
    assert result["paper_simulation_result"]["source_plan_sha256"] == (
        file_sha256(plan_path)
    )
    assert result["paper_simulation_result"]["external_result_sha256"] == (
        file_sha256(external_path)
    )
    assert review["post_paper_simulation_review"]["decision"] == (
        "ready_for_human_review"
    )
    assert review["post_paper_simulation_review"]["lineage_verified"] is True
    assert review["post_paper_simulation_review"][
        "live_candidate_allowed"
    ] is False
    assert review["safety"]["approval_performed"] is False


def test_post_review_rejects_changed_result_artifact(tmp_path):
    _, _, plan, plan_path = _build_receipt_and_plan(tmp_path)
    external_path = tmp_path / "isolated_output.json"
    checks = _guardrail_checks()
    _write_external_result(
        external_path,
        plan=plan,
        plan_path=plan_path,
        executor="isolated_paper_executor",
        summary="Completed.",
        metrics=[],
        guardrail_checks=checks,
    )
    result = PaperSimulationResultRecorder(
        plan_path,
        output_dir=tmp_path / "results",
    ).record(
        executor="isolated_paper_executor",
        status="completed",
        summary="Completed.",
        guardrail_checks=checks,
        external_result_path=external_path,
        save=True,
    )
    result_path = Path(result["saved_paths"]["latest_json"])
    changed = json.loads(result_path.read_text(encoding="utf-8"))
    changed["paper_simulation_result"]["summary"] = "Changed after record."
    result_path.write_text(json.dumps(changed), encoding="utf-8")

    review = PostPaperSimulationReviewBuilder(
        result_path,
        output_dir=tmp_path / "post_review",
    ).build(save=False)

    assert review["post_paper_simulation_review"]["decision"] == "reject"
    assert review["post_paper_simulation_review"]["lineage_verified"] is False
    assert "result_fingerprint_invalid" in review[
        "post_paper_simulation_review"
    ]["reasons"]
