from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_control_caution_review_packet import PipelineControlCautionReviewPacket


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _axis(name: str, status: str = "clear") -> dict:
    metrics = {}
    reasons = [f"{name} fixture reason"]
    if name == "risk":
        metrics = {"max_drawdown": None}
        reasons = ["No max drawdown metric supplied; downside boundary is not proven."]
    elif name == "validation":
        metrics = {"train_score": None, "validation_score": None, "train_test_gap": None, "sample_count": None}
        reasons = ["No validation/test score supplied.", "Train/test gap cannot be computed."]
    elif name == "feature_stability":
        metrics = {}
        reasons = ["No feature stability report supplied; feature-weight sanity is not proven."]
    return {
        "name": name,
        "status": status,
        "score": 0.8 if status == "clear" else 0.45,
        "metrics": metrics,
        "constraints": {},
        "reasons": reasons,
    }


def _readiness(*, blocked: list[str] | None = None, caution: list[str] | None = None) -> dict:
    blocked = blocked or []
    caution = caution if caution is not None else ["risk", "validation", "feature_stability"]
    axis_names = ["profitability", "risk", "validation", "feature_stability", "data_quality", "replay_repeatability"]
    axes = [_axis(name, "blocked" if name in blocked else "caution" if name in caution else "clear") for name in axis_names]
    status = "blocked_metric_inputs" if blocked else "metric_inputs_ready_with_cautions" if caution else "metric_inputs_ready"
    return {
        "mode": "pipeline_metric_input_readiness_gate",
        "summary": {
            "readiness_status": status,
            "blocked_metric_planes": blocked,
            "caution_metric_planes": caution,
            "can_refresh_pipeline_control_surface_now": True,
            "can_propose_reviewed_tuning_after_surface_and_manual_review": not blocked,
            "can_run_autonomous_tuning_now": False,
            "can_write_production_config": False,
            "can_write_learning_memory": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
        "input_inventory": [
            {
                "input_id": "model_performance",
                "path": "performance_data.json",
                "available": True,
                "status": "available",
                "recognized_metrics": {
                    "profitability_metric_present": False,
                    "max_drawdown_present": False,
                    "validation_score_present": False,
                    "train_score_present": False,
                    "sample_count_present": False,
                },
                "notes": ["Max drawdown is missing.", "Train/validation/sample-count metrics are incomplete."],
            },
            {
                "input_id": "feature_report",
                "path": None,
                "available": False,
                "status": "missing",
                "recognized_metrics": {},
                "notes": ["No path was supplied for this input."],
            },
        ],
        "metric_plane_readiness": axes,
    }


def _instance(*, blocked: list[str] | None = None, caution: list[str] | None = None) -> dict:
    blocked = blocked or []
    caution = caution if caution is not None else ["risk", "validation", "feature_stability"]
    status = "blocked_pipeline_control_instance" if blocked else "pipeline_control_instance_review_ready_with_cautions" if caution else "pipeline_control_instance_review_ready"
    return {
        "mode": "pipeline_control_instance_contract",
        "summary": {
            "instance_status": status,
            "blocked_metric_planes": blocked,
            "caution_metric_planes": caution,
            "can_propose_reviewed_experiments_after_manual_review": not blocked,
            "can_run_autonomous_tuning_now": False,
            "can_write_production_config": False,
            "can_write_learning_memory": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _model_performance_smoke() -> dict:
    return {
        "mode": "model_performance_agent",
        "report": {
            "metrics_snapshot": {
                "metrics": {},
                "threshold_failures": ["missing_evaluation_timestamp", "missing_recognized_metrics"],
            },
        },
    }


def _write_inputs(
    tmp_path: Path,
    *,
    readiness: dict | None = None,
    instance: dict | None = None,
    model_report: dict | None = None,
) -> dict[str, Path]:
    return {
        "readiness": _write_json(tmp_path / "readiness" / "latest.json", readiness or _readiness()),
        "instance": _write_json(tmp_path / "instance" / "latest.json", instance or _instance()),
        "model_report": _write_json(tmp_path / "model_performance" / "smoke.json", model_report or _model_performance_smoke()),
        "data_quality": _write_json(tmp_path / "data_quality" / "latest.json", {"warnings": [], "leakage_flags": []}),
    }


def test_pipeline_control_caution_review_records_missing_metric_evidence(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = PipelineControlCautionReviewPacket(tmp_path / "reports").build(
        pipeline_metric_input_readiness_json=paths["readiness"],
        pipeline_control_instance_json=paths["instance"],
        model_performance_report_json=paths["model_report"],
        data_quality_json=paths["data_quality"],
        save=False,
    )

    assert payload["summary"]["caution_review_status"] == "pipeline_cautions_need_reviewed_inputs"
    assert payload["summary"]["blocked_metric_planes"] == []
    assert payload["summary"]["caution_metric_planes"] == ["risk", "validation", "feature_stability"]
    assert payload["summary"]["missing_evidence_planes"] == ["risk", "validation", "feature_stability"]
    assert payload["summary"]["can_propose_reviewed_experiments_after_manual_caution_acceptance"] is True
    assert payload["summary"]["can_run_autonomous_tuning_now"] is False
    assert payload["summary"]["can_trade"] is False
    assert any(check["code"] == "risk_drawdown_evidence_present" and check["status"] == "warn" for check in payload["review_checks"])
    assert any(item["artifact_id"] == "model_performance_report" and item["triage_status"] == "warning_evidence_only" for item in payload["artifact_triage"])
    assert any(item["artifact_class"] == "code_audit_reports" for item in payload["excluded_artifact_classes"])


def test_pipeline_control_caution_review_blocks_hard_metric_planes(tmp_path):
    paths = _write_inputs(
        tmp_path,
        readiness=_readiness(blocked=["data_quality"], caution=[]),
        instance=_instance(blocked=["data_quality"], caution=[]),
    )

    payload = PipelineControlCautionReviewPacket(tmp_path / "reports").build(
        pipeline_metric_input_readiness_json=paths["readiness"],
        pipeline_control_instance_json=paths["instance"],
        model_performance_report_json=paths["model_report"],
        data_quality_json=paths["data_quality"],
        save=False,
    )

    assert payload["summary"]["caution_review_status"] == "pipeline_caution_review_blocked_by_hard_planes"
    assert payload["summary"]["blocked_metric_planes"] == ["data_quality"]
    assert payload["summary"]["can_propose_reviewed_experiments_after_manual_caution_acceptance"] is False
    assert any(check["code"] == "no_hard_blocked_metric_planes" and check["status"] == "fail" for check in payload["review_checks"])


def test_pipeline_control_caution_review_marks_clear_state_ready(tmp_path):
    paths = _write_inputs(
        tmp_path,
        readiness=_readiness(caution=[]),
        instance=_instance(caution=[]),
    )

    payload = PipelineControlCautionReviewPacket(tmp_path / "reports").build(
        pipeline_metric_input_readiness_json=paths["readiness"],
        pipeline_control_instance_json=paths["instance"],
        model_performance_report_json=paths["model_report"],
        data_quality_json=paths["data_quality"],
        save=False,
    )

    assert payload["summary"]["caution_review_status"] == "pipeline_ready_for_manual_proposal_review"
    assert payload["summary"]["caution_metric_planes"] == []
    assert payload["summary"]["missing_evidence_planes"] == []
    assert payload["summary"]["can_clear_cautions_with_current_artifacts"] is True


def test_pipeline_control_caution_review_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = PipelineControlCautionReviewPacket(tmp_path / "reports").build(
        pipeline_metric_input_readiness_json=paths["readiness"],
        pipeline_control_instance_json=paths["instance"],
        model_performance_report_json=paths["model_report"],
        data_quality_json=paths["data_quality"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Pipeline Control Caution Review Packet" in markdown
    assert "Can trade: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_pipeline_control_caution_review_packet.py"),
            "--pipeline-metric-input-readiness-json",
            str(paths["readiness"]),
            "--pipeline-control-instance-json",
            str(paths["instance"]),
            "--model-performance-report-json",
            str(paths["model_report"]),
            "--data-quality-json",
            str(paths["data_quality"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Caution review status: pipeline_cautions_need_reviewed_inputs" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
