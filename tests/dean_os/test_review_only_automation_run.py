from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.review_only_automation_run import (
    DeanOSReviewOnlyAutomationRun,
    _discover_bounded_candidate_inputs,
)


def test_review_only_automation_refreshes_safe_status_without_live_actions(tmp_path):
    payload = DeanOSReviewOnlyAutomationRun(tmp_path / "reports" / "review_only_automation_run_current").build(
        run_real_metric_when_ready=False,
    )

    summary = payload["summary"]
    assert summary["automation_status"] in {
        "review_automation_completed_missing_locked_metric_inputs",
        "review_automation_completed_waiting_for_metric_pair",
        "review_automation_completed_metric_pair_ready_real_run_skipped",
    }
    assert summary["real_metric_evidence_invoked"] is False
    assert summary["can_write_learning_memory"] is False
    assert summary["can_write_production_config"] is False
    assert summary["can_trade"] is False
    assert any(step["step_id"] == "pipeline_control_locked_evaluation_assembler" for step in payload["steps"])
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")


def test_review_only_automation_collects_locked_pair_without_invoking_real_metric_when_disabled(tmp_path):
    window = {"start": "2026-01-01", "end": "2026-01-03", "sample_count": 3}
    training = _write_json(
        tmp_path / "training_candidate.json",
        {
            "artifact_class": "pipeline_control_model_evaluation_candidate",
            "evidence_class": "pipeline_training_output",
            "ticker": "AMD",
            "target_name": "target_up_1d",
            "model_type": "random_forest",
            "timeframe": "1d",
            "context_fingerprint": "ctx_semis_2026_01",
            "metrics": {"train_score": 0.75, "validation_score": 0.62, "test_score": 0.62, "sample_count": 123},
            "evaluation_window": window,
        },
    )
    evaluation = _write_json(
        tmp_path / "evaluation_candidate.json",
        {
            "artifact_class": "pipeline_control_evaluation_metric_candidate",
            "evidence_class": "pipeline_stage_7_evaluation_output",
            "ticker": "AMD",
            "target_name": "target_up_1d",
            "model_type": "random_forest",
            "timeframe": "1d",
            "context_fingerprint": "ctx_semis_2026_01",
            "metrics": {"max_drawdown": 0.08, "total_return": 0.12},
            "evaluation_window": window,
        },
    )
    feature = _write_json(
        tmp_path / "feature_candidate.json",
        {
            "artifact_class": "pipeline_control_feature_stability_candidate",
            "evidence_class": "pipeline_training_output",
            "ticker": "AMD",
            "target_name": "target_up_1d",
            "model_type": "random_forest",
            "timeframe": "1d",
            "context_fingerprint": "ctx_semis_2026_01",
            "feature_importance": {"macro_pressure": 0.7, "news_pressure": 0.3},
            "feature_stability_score": 0.82,
            "unstable_feature_count": 1,
            "unstable_features": ["news_pressure"],
            "contract_status": "ready_feature_stability_candidate",
        },
    )

    payload = DeanOSReviewOnlyAutomationRun(tmp_path / "reports" / "review_only_automation_run_current").build(
        training_candidate_json=training,
        evaluation_candidate_json=evaluation,
        feature_stability_candidate_json=feature,
        run_real_metric_when_ready=False,
    )

    assert payload["summary"]["automation_status"] == "review_automation_completed_metric_pair_ready_real_run_skipped"
    assert payload["summary"]["ready_locked_model_evaluation"] is True
    assert payload["summary"]["ready_locked_feature_stability"] is True
    assert payload["summary"]["real_metric_evidence_invoked"] is False
    assert payload["summary"]["can_trade"] is False
    assert Path(payload["next_runner_inputs"]["model_evaluation_json"]).exists()
    assert Path(payload["next_runner_inputs"]["feature_stability_report"]).exists()


def test_review_only_automation_does_not_materialize_walk_forward_as_locked_evidence(tmp_path):
    walk_forward = _write_json(
        tmp_path / "walk_forward.json",
        {
            "mode": "pipeline_control_walk_forward_validation_run",
            "walk_forward_candidate": {
                "artifact_class": "pipeline_control_walk_forward_validation_candidate",
                "evidence_class": "development_train_validation_only",
                "contract_status": "walk_forward_candidate_blocked_by_validation_contract",
                "metrics": {
                    "max_drawdown": 0.08,
                    "train_score": 0.81,
                    "validation_score": 0.52,
                    "sample_count": 480,
                    "contract_passed": False,
                },
                "feature_importance": {"momentum": 0.4},
                "feature_stability_score": 0.53,
                "test_contract": {"eligible_as_locked_test_evidence": False},
            },
        },
    )

    payload = DeanOSReviewOnlyAutomationRun(
        tmp_path / "reports" / "review_only_automation_run_current"
    ).build(
        candidate_paths=[walk_forward],
        run_real_metric_when_ready=False,
    )

    assert payload["summary"]["ready_locked_model_evaluation"] is False
    assert payload["summary"]["ready_locked_feature_stability"] is False
    assert payload["summary"]["real_metric_evidence_invoked"] is False
    inventory_path = Path(payload["report_paths"]["pipeline_control_evidence_inventory"]["json"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    record = inventory["candidate_artifacts"][0]
    assert record["classification"] == "supporting_walk_forward_train_validation"
    assert record["eligible_as_locked_test_evidence"] is False
    materializer_path = Path(
        payload["report_paths"]["pipeline_control_metric_artifact_materializer"]["json"]
    )
    materializer = json.loads(materializer_path.read_text(encoding="utf-8"))
    assert materializer["summary"]["ready_model_candidate_found"] is False
    assert materializer["summary"]["ready_feature_candidate_found"] is False
    assert materializer["next_runner_inputs"]["model_evaluation_json"] is None
    assert materializer["next_runner_inputs"]["feature_stability_report"] is None


def test_review_only_automation_invokes_real_metric_for_matching_locked_pair(tmp_path):
    window = {"start": "2026-01-01", "end": "2026-01-03", "sample_count": 3}
    lineage = {
        "ticker": "AMD",
        "target_name": "target_up_1d",
        "model_type": "random_forest",
        "timeframe": "1d",
        "context_fingerprint": "ctx_semis_2026_01",
    }
    training = _write_json(
        tmp_path / "training_candidate.json",
        {
            "artifact_class": "pipeline_control_model_evaluation_candidate",
            "evidence_class": "pipeline_training_output",
            **lineage,
            "metrics": {"train_score": 0.75, "validation_score": 0.62, "test_score": 0.62, "sample_count": 123},
            "evaluation_window": window,
        },
    )
    evaluation = _write_json(
        tmp_path / "evaluation_candidate.json",
        {
            "artifact_class": "pipeline_control_evaluation_metric_candidate",
            "evidence_class": "pipeline_stage_7_evaluation_output",
            **lineage,
            "metrics": {"max_drawdown": 0.08, "total_return": 0.12},
            "evaluation_window": window,
        },
    )
    feature = _write_json(
        tmp_path / "feature_candidate.json",
        {
            "artifact_class": "pipeline_control_feature_stability_candidate",
            "evidence_class": "pipeline_training_output",
            **lineage,
            "feature_importance": {"macro_pressure": 0.4, "news_pressure": 0.3, "momentum": 0.3},
            "feature_stability_score": 0.82,
            "unstable_feature_count": 0,
            "unstable_features": [],
            "contract_status": "ready_feature_stability_candidate",
        },
    )
    replay = _write_json(
        tmp_path / "replay.json",
        {"summary": {"clear_hit_rate": 0.72, "clear_evaluated_runs": 8, "quality_blocked_runs": 0}},
    )
    quality = _write_json(tmp_path / "quality.json", {"warnings": [], "leakage_flags": []})
    domain = _write_json(
        tmp_path / "domain.json",
        {"summary": {"can_scale_to_other_domains_now": False, "can_trade": False}},
    )

    payload = DeanOSReviewOnlyAutomationRun(tmp_path / "reports" / "review_only_automation_run_current").build(
        training_candidate_json=training,
        evaluation_candidate_json=evaluation,
        feature_stability_candidate_json=feature,
        replay_batch_json=replay,
        data_quality_json=quality,
        domain_instance_contract_json=domain,
    )

    assert payload["summary"]["real_metric_evidence_invoked"] is True
    assert payload["summary"]["can_use_as_metric_evidence"] is True
    assert payload["summary"]["can_trade"] is False
    real_step = next(
        step for step in payload["steps"] if step["step_id"] == "pipeline_control_real_metric_evidence_run"
    )
    assert real_step["status"] == "completed"


def test_review_only_automation_cli_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_dean_os_review_automation.py"),
            "--output-dir",
            str(tmp_path / "cli_reports"),
            "--no-real-metric-run",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        # Without a timeout this call froze the entire dean_os suite: the CLI walked a
        # 66 MB accumulated-results artifact and never returned, so the suite sat at
        # 79% indefinitely with no indication of which test was responsible.
        timeout=300,
    )

    assert "Automation status:" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()


def test_review_automation_discovers_latest_bounded_candidates(tmp_path):
    base_reports = tmp_path / "reports"
    bounded_dir = base_reports / "pipeline_control_bounded_evidence_run_current"
    training = _write_json(tmp_path / "training.json", {"artifact_class": "candidate"})
    evaluation = _write_json(tmp_path / "evaluation.json", {"artifact_class": "candidate"})
    feature = _write_json(tmp_path / "feature.json", {"artifact_class": "candidate"})
    _write_json(
        bounded_dir / "latest.json",
        {
            "artifacts": {
                "training_candidates": {
                    "model_evaluation_json": str(training),
                    "feature_stability_report": str(feature),
                },
                "evaluation_candidate": {
                    "evaluation_metric_candidate": str(evaluation),
                },
            }
        },
    )

    discovered = _discover_bounded_candidate_inputs(base_reports)

    assert discovered == {
        "training_candidate_json": str(training),
        "evaluation_candidate_json": str(evaluation),
        "feature_stability_candidate_json": str(feature),
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
