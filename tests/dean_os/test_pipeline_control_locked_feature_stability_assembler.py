from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_control_locked_feature_stability_assembler import PipelineControlLockedFeatureStabilityAssembler


def test_locked_feature_stability_assembler_blocks_importance_only_candidate(tmp_path):
    candidate = _write_json(tmp_path / "feature_candidate.json", _feature_candidate(stability=False))

    payload = PipelineControlLockedFeatureStabilityAssembler(tmp_path / "reports").build(
        feature_stability_candidate_json=candidate,
        save=False,
    )

    summary = payload["summary"]
    assert summary["assembly_status"] == "blocked_missing_measured_feature_stability"
    assert summary["feature_importances_present"] is True
    assert summary["stability_signal_present"] is False
    assert summary["locked_feature_stability_written"] is False
    assert summary["can_supply_feature_stability_to_real_runner"] is False
    assert summary["can_run_real_metric_evidence_now"] is False
    assert summary["can_trade"] is False
    assert "feature_stability_signal_present" in summary["blocked_check_codes"]
    assert "feature_stability_candidate_not_partial" in summary["blocked_check_codes"]
    assert payload["next_runner_inputs"]["feature_stability_report"] is None


def test_locked_feature_stability_assembler_writes_when_signal_and_lineage_present(tmp_path):
    candidate = _write_json(tmp_path / "feature_candidate.json", _feature_candidate(stability=True))

    payload = PipelineControlLockedFeatureStabilityAssembler(tmp_path / "reports").build(
        feature_stability_candidate_json=candidate,
    )

    summary = payload["summary"]
    assert summary["assembly_status"] == "locked_feature_stability_assembled"
    assert summary["feature_importances_present"] is True
    assert summary["stability_signal_present"] is True
    assert summary["lineage_present"] is True
    assert summary["locked_feature_stability_written"] is True
    assert summary["can_supply_feature_stability_to_real_runner"] is True
    assert summary["model_evaluation_json_required_separately"] is True
    assert summary["can_run_real_metric_evidence_now"] is False
    assert summary["can_trade"] is False

    locked_path = Path(payload["next_runner_inputs"]["feature_stability_report"])
    assert locked_path.exists()
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    assert locked["artifact_class"] == "locked_feature_stability_report"
    assert locked["feature_importance_count"] == 2
    assert locked["feature_stability_score"] == 0.83
    assert locked["unstable_feature_count"] == 1
    assert locked["training_lineage"]["ticker"] == "AMD"
    assert locked["assembly_contract"]["partial_artifact_promotion_allowed"] is False


def test_locked_feature_stability_assembler_rejects_synthetic_candidate(tmp_path):
    payload_data = _feature_candidate(stability=True)
    payload_data["fixture_not_evidence"] = True
    candidate = _write_json(tmp_path / "feature_candidate.json", payload_data)

    payload = PipelineControlLockedFeatureStabilityAssembler(tmp_path / "reports").build(
        feature_stability_candidate_json=candidate,
        save=False,
    )

    assert payload["summary"]["assembly_status"] == "blocked_missing_measured_feature_stability"
    assert "feature_stability_candidate_not_synthetic" in payload["summary"]["blocked_check_codes"]
    assert payload["summary"]["can_trade"] is False


def test_locked_feature_stability_assembler_resolves_manifest_and_cli_runs(tmp_path):
    candidate = _write_json(tmp_path / "artifacts" / "feature_candidate.json", _feature_candidate(stability=True))
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "artifact_class": "pipeline_control_metric_artifacts_manifest",
            "artifacts": [
                {
                    "artifact_type": "feature_stability_report",
                    "path": str(candidate),
                    "contract_status": "ready_feature_stability_candidate",
                }
            ],
        },
    )
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_pipeline_control_locked_feature_stability_assembler.py"),
            "--feature-stability-candidate-json",
            str(manifest),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Assembly status: locked_feature_stability_assembled" in result.stdout
    assert "Can run real metric evidence now: False" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()


def _feature_candidate(*, stability: bool) -> dict:
    payload = {
        "artifact_class": "pipeline_control_feature_stability_candidate",
        "evidence_class": "pipeline_training_output",
        "ticker": "AMD",
        "target_name": "target_up_1d",
        "model_type": "random_forest",
        "timeframe": "1d",
        "context_fingerprint": "ctx_semis_2026_01",
        "feature_importance": {"macro_score": 0.6, "news_score": 0.4},
        "feature_importance_status": "measured_from_trained_model",
        "stability_signal_status": "not_measured",
        "contract_status": "partial_feature_stability_candidate",
        "missing_for_locked_feature_stability": ["stability_signal"],
    }
    if stability:
        payload.update(
            {
                "feature_stability_score": 0.83,
                "unstable_feature_count": 1,
                "unstable_features": ["news_score"],
                "stability_signal_status": "measured",
                "contract_status": "ready_feature_stability_candidate",
                "missing_for_locked_feature_stability": [],
            }
        )
    return payload


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
