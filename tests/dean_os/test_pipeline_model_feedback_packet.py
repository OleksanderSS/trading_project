from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_model_feedback_packet import (
    PipelineModelFeedbackPacket,
)
from dean_os.chief_review_index import classify_review_index
from dean_os.review_index import ReviewIndexBuilder


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_path(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    sources = {
        "real_metric_evidence": _write_json(
            tmp_path / "sources" / "chain.json",
            {"mode": "pipeline_control_real_metric_evidence_run"},
        ),
        "model_evaluation": _write_json(
            tmp_path / "sources" / "model.json",
            {"artifact_class": "locked_model_evaluation"},
        ),
        "feature_stability": _write_json(
            tmp_path / "sources" / "feature.json",
            {"artifact_class": "locked_feature_stability_report"},
        ),
        "metric_readiness": _write_json(
            tmp_path / "sources" / "readiness.json",
            {"mode": "pipeline_metric_input_readiness_gate"},
        ),
    }
    payload = {
        "mode": "pipeline_model_case_packet",
        "inputs": {
            "real_metric_evidence_json": str(
                sources["real_metric_evidence"]
            ),
            "real_metric_evidence_sha256": _sha(
                sources["real_metric_evidence"]
            ),
            "model_evaluation_json": str(
                sources["model_evaluation"]
            ),
            "model_evaluation_sha256": _sha(
                sources["model_evaluation"]
            ),
            "feature_stability_json": str(
                sources["feature_stability"]
            ),
            "feature_stability_sha256": _sha(
                sources["feature_stability"]
            ),
            "metric_readiness_json": str(
                sources["metric_readiness"]
            ),
            "metric_readiness_sha256": _sha(
                sources["metric_readiness"]
            ),
        },
        "summary": {
            "case_id": "pipeline_model_case:fixture",
            "case_status": "evaluation_block_case_ready",
            "case_classification": "negative_evaluation_block_case",
            "result_label": (
                "failed_validation_and_feature_stability"
            ),
            "blocked_metric_planes": [
                "validation",
                "feature_stability",
            ],
            "root_cause_categories": [
                "generalization_gap",
                "feature_instability",
            ],
            "failed_review_checks": [],
        },
        "case": {
            "lineage": {
                "ticker": "AMD",
                "model": "random_forest",
                "target_name": "target_intraday_up_15m",
                "timeframe": "15m",
                "context_fingerprint": "ctx-1",
            },
            "evaluated_at": "2026-06-24T19:30:00+00:00",
        },
        "new_data_requirement": {
            "status": "wait_for_new_forward_development_data",
            "data_after": "2026-06-24T19:30:00+00:00",
        },
    }
    return _write_json(
        tmp_path / "case" / "latest.json",
        payload,
    ), sources


def test_model_feedback_waits_for_manual_feedback_and_blocks_apply(
    tmp_path,
):
    case_path, _ = _case_path(tmp_path)

    payload = PipelineModelFeedbackPacket().build(
        pipeline_model_case_json=case_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == (
        "pipeline_model_feedback_ready_pending_manual_feedback"
    )
    assert payload["summary"]["manual_feedback_record_count"] == 0
    assert payload["summary"][
        "can_route_to_existing_analyst_learning_apply_loop"
    ] is False
    assert payload["summary"]["can_apply_learning"] is False
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_launch_model_variant_now"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["review_label_taxonomy"]["taxonomy_id"] == (
        "dean_review_feedback_taxonomy_v1"
    )
    assert payload["existing_learning_loop_compatibility"][
        "compatible"
    ] is False


def test_model_feedback_creates_proposal_only_candidates(tmp_path):
    case_path, _ = _case_path(tmp_path)
    feedback_path = _write_json(
        tmp_path / "feedback.json",
        [
            {
                "feedback_id": "feedback-1",
                "reviewer": "operator",
                "target_id": "pipeline_model_case:fixture",
                "labels": [
                    "evaluation_block_valid",
                    "generalization_gap_confirmed",
                    "feature_instability_confirmed",
                    "needs_new_forward_data",
                ],
                "proposed_learning_actions": [
                    "create_eval_test_candidate",
                    "propose_model_iteration_after_new_data",
                ],
                "notes": (
                    "Keep the negative case and add future-window "
                    "regression checks."
                ),
            }
        ],
    )

    payload = PipelineModelFeedbackPacket().build(
        pipeline_model_case_json=case_path,
        manual_feedback_json=feedback_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == (
        "pipeline_model_feedback_ready_with_candidates"
    )
    assert payload["summary"]["learning_candidate_proposal_count"] == 2
    actions = {
        item["proposed_action"]
        for item in payload["learning_candidate_proposals"]
    }
    assert actions == {
        "create_eval_test_candidate",
        "propose_model_iteration_after_new_data",
    }
    iteration = next(
        item
        for item in payload["learning_candidate_proposals"]
        if item["proposed_action"]
        == "propose_model_iteration_after_new_data"
    )
    assert iteration["requires_new_forward_data"] is True
    assert iteration["data_after"] == (
        "2026-06-24T19:30:00+00:00"
    )
    assert iteration["can_apply_now"] is False
    assert iteration["can_launch_model_variant_now"] is False


def test_model_feedback_rejects_wrong_family_and_unsafe_requests(
    tmp_path,
):
    case_path, _ = _case_path(tmp_path)
    feedback_path = _write_json(
        tmp_path / "feedback.json",
        {
            "feedback_records": [
                {
                    "feedback_id": "feedback-unsafe",
                    "reviewer": "operator",
                    "target_id": "pipeline_model_case:fixture",
                    "labels": ["hit_correct_reason"],
                    "proposed_learning_actions": [
                        "create_incident_candidate",
                    ],
                    "notes": "Wrong outcome family and unsafe request.",
                    "apply_learning": True,
                    "same_fold_retry": True,
                }
            ]
        },
    )

    payload = PipelineModelFeedbackPacket().build(
        pipeline_model_case_json=case_path,
        manual_feedback_json=feedback_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == (
        "pipeline_model_feedback_blocked"
    )
    record = payload["manual_feedback_records"][0]
    assert any(
        blocker.startswith("unknown_or_wrong_family_labels:")
        for blocker in record["blockers"]
    )
    assert "unsafe_request:apply_learning" in record["blockers"]
    assert "unsafe_request:same_fold_retry" in record["blockers"]
    assert (
        "incident_candidate_requires_issue_label"
        in record["blockers"]
    )
    assert payload["summary"]["learning_candidate_proposal_count"] == 0


def test_model_feedback_rejects_stale_case_binding(tmp_path):
    case_path, sources = _case_path(tmp_path)
    sources["metric_readiness"].write_text(
        json.dumps({"changed": True}),
        encoding="utf-8",
    )

    payload = PipelineModelFeedbackPacket().build(
        pipeline_model_case_json=case_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == (
        "pipeline_model_feedback_blocked"
    )
    assert "metric_readiness_json_binding_current" in (
        payload["summary"]["failed_review_checks"]
    )


def test_model_feedback_cli_writes_review_artifact(tmp_path):
    case_path, _ = _case_path(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(
                repo_root
                / "run_agent_pipeline_model_feedback_packet.py"
            ),
            "--pipeline-model-case-json",
            str(case_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert (
        "Packet status: "
        "pipeline_model_feedback_ready_pending_manual_feedback"
        in result.stdout
    )
    assert "Can route to analyst apply loop: False" in result.stdout
    assert (tmp_path / "reports" / "latest.json").exists()


def test_model_feedback_routes_to_index_without_unblocking_case(
    tmp_path,
):
    case_path, _ = _case_path(tmp_path)
    feedback = PipelineModelFeedbackPacket(
        tmp_path / "feedback_reports"
    ).build(
        pipeline_model_case_json=case_path,
    )
    review_index = ReviewIndexBuilder(
        sources={
            "pipeline_model_case": case_path,
            "pipeline_model_feedback": feedback["saved_paths"][
                "latest_json"
            ],
        }
    ).build(save=False)

    decision = classify_review_index(review_index)
    feedback_entry = next(
        entry
        for entry in review_index["entries"]
        if entry["source_name"] == "pipeline_model_feedback"
    )

    assert feedback_entry["status"] == (
        "pipeline_model_feedback_ready_pending_manual_feedback"
    )
    assert decision["decision"] == "model_candidate_blocked"
    assert decision["model_feedback_state"][
        "pending_manual_feedback"
    ] is True
    assert decision["model_feedback_state"][
        "has_proposal_candidates"
    ] is False
