from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_control.pipeline_control_metric_fixture_validation import PipelineControlMetricFixtureValidation


def test_pipeline_control_metric_fixture_validation_passes_clean_synthetic_flow(tmp_path):
    payload = PipelineControlMetricFixtureValidation(tmp_path / "reports").build(save=False)

    assert payload["summary"]["validation_status"] == "synthetic_fixture_control_flow_passed"
    assert payload["summary"]["fixture_is_evidence"] is False
    assert payload["summary"]["can_use_fixture_as_metric_evidence"] is False
    assert payload["summary"]["current_artifacts_overwritten"] is False
    assert payload["summary"]["readiness_status"] == "metric_inputs_ready"
    assert payload["summary"]["surface_status"] == "clear"
    assert payload["summary"]["instance_status"] == "pipeline_control_instance_review_ready"
    assert payload["summary"]["caution_review_status"] == "pipeline_ready_for_manual_proposal_review"
    assert payload["summary"]["can_write_production_config"] is False
    assert payload["summary"]["can_trade"] is False
    assert all(check["status"] == "pass" for check in payload["review_checks"])


def test_pipeline_control_metric_fixture_validation_saves_markdown_and_cli_runs(tmp_path):
    payload = PipelineControlMetricFixtureValidation(tmp_path / "reports").build()
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Pipeline Control Metric Fixture Validation" in markdown
    assert "Fixture is evidence: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_pipeline_control_metric_fixture_validation.py"),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Validation status: synthetic_fixture_control_flow_passed" in result.stdout
    assert "Fixture is evidence: False" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
