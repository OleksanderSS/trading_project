from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dean_os.pipeline_control.pipeline_control_forward_data_accrual_plan import (
    PipelineControlForwardDataAccrualPlan,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _walk_forward_payload(*, test_rows: int = 0, contract_passed: bool = False) -> dict:
    return {
        "mode": "pipeline_control_walk_forward_validation_run",
        "walk_forward_candidate": {
            "artifact_class": "pipeline_control_walk_forward_validation_candidate",
            "evidence_class": "development_train_validation_only",
            "ticker": "NVDA",
            "timeframe": "15m",
            "target_name": "target_intraday_up_15m",
            "context_fingerprint": "ctx-123",
            "metrics": {"contract_passed": contract_passed},
            "folds": [
                {
                    "validation_window": {
                        "start": "2026-04-01T14:30:00+00:00",
                        "end": "2026-04-08T19:30:00+00:00",
                        "sample_count": 120,
                    }
                },
                {
                    "validation_window": {
                        "start": "2026-04-09T14:30:00+00:00",
                        "end": "2026-04-16T19:30:00+00:00",
                        "sample_count": 120,
                    }
                },
            ],
            "test_contract": {
                "eligible_as_locked_test_evidence": False,
                "test_rows_loaded": test_rows,
                "past_evaluation_rows_loaded": 0,
                "frozen_test_windows_accessed": False,
            },
            "source_lineage": {
                "development_artifacts": {
                    "development_15m": {
                        "partition": "development",
                        "sha256": "abc123",
                    },
                    "development_60m": {
                        "partition": "development",
                        "sha256": "def456",
                    },
                }
            },
        },
    }


def test_forward_data_accrual_plan_registers_only_new_development_boundary(tmp_path):
    source = _write_json(tmp_path / "walk_forward.json", _walk_forward_payload())

    payload = PipelineControlForwardDataAccrualPlan(tmp_path / "reports").build(
        walk_forward_json=source,
        acknowledge_development_refresh_only=True,
        save=False,
    )

    assert payload["summary"]["plan_status"] == "forward_development_accrual_plan_ready"
    assert payload["summary"]["development_candidate_was_blocked"] is True
    assert payload["summary"]["can_accept_existing_artifact_as_new"] is False
    assert payload["summary"]["can_call_next_data_virgin_holdout"] is False
    assert payload["summary"]["can_train"] is False
    plan = payload["accrual_plan"]
    assert plan["lane"] == "development_refresh_only"
    assert plan["baseline"]["last_used_validation_timestamp"] == "2026-04-16T19:30:00+00:00"
    assert plan["baseline"]["seen_development_source_sha256"] == ["abc123", "def456"]
    assert plan["acceptance_boundary"]["minimum_new_base_timeframe_rows"] == 120
    assert plan["acceptance_boundary"]["source_artifact_acquired_after"] == payload["created_at"]
    assert plan["acceptance_boundary"]["may_be_used_as_locked_test_evidence"] is False
    assert plan["acceptance_boundary"]["may_be_called_virgin_holdout"] is False


@pytest.mark.parametrize(
    "test_rows,contract_passed,failed_check",
    [
        (1, False, "test_and_past_evaluation_untouched"),
        (0, True, "blocked_candidate_requires_development_refresh"),
    ],
)
def test_forward_data_accrual_plan_blocks_invalid_boundary(
    tmp_path,
    test_rows,
    contract_passed,
    failed_check,
):
    source = _write_json(
        tmp_path / "walk_forward.json",
        _walk_forward_payload(
            test_rows=test_rows,
            contract_passed=contract_passed,
        ),
    )

    payload = PipelineControlForwardDataAccrualPlan(tmp_path / "reports").build(
        walk_forward_json=source,
        acknowledge_development_refresh_only=True,
        save=False,
    )

    assert payload["summary"]["plan_status"] == "blocked_invalid_walk_forward_development_boundary"
    assert payload["accrual_plan"] is None
    failed = {
        check["check_id"]
        for check in payload["checks"]
        if check["status"] == "fail"
    }
    assert failed_check in failed


def test_forward_data_accrual_plan_requires_acknowledgement(tmp_path):
    source = _write_json(tmp_path / "walk_forward.json", _walk_forward_payload())

    with pytest.raises(ValueError, match="development-refresh-only"):
        PipelineControlForwardDataAccrualPlan(tmp_path / "reports").build(
            walk_forward_json=source,
            acknowledge_development_refresh_only=False,
            save=False,
        )


def test_forward_data_accrual_plan_saves_and_cli_runs(tmp_path):
    source = _write_json(tmp_path / "walk_forward.json", _walk_forward_payload())
    payload = PipelineControlForwardDataAccrualPlan(tmp_path / "reports").build(
        walk_forward_json=source,
        acknowledge_development_refresh_only=True,
    )

    assert payload["saved_paths"]["latest_json"].endswith("latest.json")
    assert "Forward Data Accrual Plan" in (
        tmp_path / "reports" / "latest.md"
    ).read_text(encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(
                repo_root
                / "run_agent_pipeline_control_forward_data_accrual_plan.py"
            ),
            "--walk-forward-json",
            str(source),
            "--acknowledge-development-refresh-only",
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Status: forward_development_accrual_plan_ready" in result.stdout
    assert "Can call next data virgin holdout: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
