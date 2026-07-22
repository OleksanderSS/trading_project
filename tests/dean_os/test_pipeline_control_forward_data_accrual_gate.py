from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dean_os.pipeline_control_forward_data_accrual_gate import (
    PipelineControlForwardDataAccrualGate,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_prices(
    path: Path,
    *,
    rows: int = 130,
    add_target: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(
        "2026-06-01T14:30:00Z",
        periods=rows,
        freq="15min",
    )
    close = 100.0 + np.arange(rows) * 0.01
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "ticker": "NVDA",
            "interval": "15m",
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 1_000 + np.arange(rows),
        }
    )
    if add_target:
        frame["target_intraday_up_15m"] = 1
    frame.to_parquet(path, index=False)
    return path


def _plan_payload(
    *,
    registered_at: str = "2026-01-01T00:00:00+00:00",
    seen_hashes: list[str] | None = None,
) -> dict:
    return {
        "mode": "pipeline_control_forward_data_accrual_plan",
        "summary": {
            "plan_status": "forward_development_accrual_plan_ready",
        },
        "accrual_plan": {
            "artifact_class": "pipeline_control_forward_data_accrual_plan",
            "plan_id": "plan-123",
            "registered_at": registered_at,
            "lane": "development_refresh_only",
            "baseline": {
                "seen_development_source_sha256": seen_hashes or [],
            },
            "acceptance_boundary": {
                "source_artifact_acquired_after": registered_at,
                "observation_timestamp_strictly_after": (
                    "2026-05-06T17:30:00+00:00"
                ),
                "minimum_new_base_timeframe_rows": 120,
                "ticker_must_equal": "NVDA",
                "timeframe_must_equal": "15m",
                "target_contract_must_equal": "target_intraday_up_15m",
            },
        },
    }


def test_forward_data_accrual_gate_accepts_only_new_development_rows(tmp_path):
    source = _write_prices(tmp_path / "new_prices.parquet")
    plan = _write_json(tmp_path / "plan.json", _plan_payload())

    payload = PipelineControlForwardDataAccrualGate(tmp_path / "reports").build(
        accrual_plan_json=plan,
        source_path=source,
        save=False,
    )

    assert payload["summary"]["gate_status"] == "forward_development_artifact_ready"
    assert payload["summary"]["candidate_new_row_count"] == 130
    assert payload["summary"]["eligible_new_row_count"] == 130
    assert payload["summary"]["can_supply_next_development_run"] is True
    assert payload["summary"]["can_use_as_locked_test_evidence"] is False
    artifact = payload["eligible_development_artifact"]
    assert artifact["source_sha256"] == _sha256(source)
    assert artifact["start_exclusive"] == "2026-05-06T17:30:00+00:00"
    assert artifact["may_be_used_as_locked_test_evidence"] is False
    assert payload["next_runner_inputs"]["start_exclusive"] == (
        "2026-05-06T17:30:00+00:00"
    )


def test_forward_data_accrual_gate_rejects_pre_registration_file(tmp_path):
    source = _write_prices(tmp_path / "old_prices.parquet")
    plan = _write_json(
        tmp_path / "plan.json",
        _plan_payload(registered_at="2100-01-01T00:00:00+00:00"),
    )

    payload = PipelineControlForwardDataAccrualGate(tmp_path / "reports").build(
        accrual_plan_json=plan,
        source_path=source,
        save=False,
    )

    assert payload["summary"]["gate_status"] == "blocked_forward_development_artifact"
    assert payload["summary"]["can_supply_next_development_run"] is False
    assert _failed(payload, "source_acquired_after_registration")


def test_forward_data_accrual_gate_rejects_seen_sha_and_target_columns(tmp_path):
    source = _write_prices(
        tmp_path / "contaminated_prices.parquet",
        add_target=True,
    )
    plan = _write_json(
        tmp_path / "plan.json",
        _plan_payload(seen_hashes=[_sha256(source)]),
    )

    payload = PipelineControlForwardDataAccrualGate(tmp_path / "reports").build(
        accrual_plan_json=plan,
        source_path=source,
        save=False,
    )

    assert payload["summary"]["gate_status"] == "blocked_forward_development_artifact"
    assert _failed(payload, "source_sha_is_new")
    assert _failed(payload, "raw_source_has_no_target_columns")
    assert payload["eligible_development_artifact"] is None


def test_forward_data_accrual_gate_blocks_insufficient_rows(tmp_path):
    source = _write_prices(tmp_path / "short_prices.parquet", rows=40)
    plan = _write_json(tmp_path / "plan.json", _plan_payload())

    payload = PipelineControlForwardDataAccrualGate(tmp_path / "reports").build(
        accrual_plan_json=plan,
        source_path=source,
        save=False,
    )

    assert payload["summary"]["candidate_new_row_count"] == 40
    assert payload["summary"]["eligible_new_row_count"] == 0
    assert _failed(payload, "minimum_new_rows_after_watermark")
    assert payload["next_runner_inputs"]["source_path"] is None


def test_forward_data_accrual_gate_saves_and_cli_runs(tmp_path):
    source = _write_prices(tmp_path / "new_prices.parquet")
    plan = _write_json(tmp_path / "plan.json", _plan_payload())
    payload = PipelineControlForwardDataAccrualGate(tmp_path / "reports").build(
        accrual_plan_json=plan,
        source_path=source,
    )

    assert payload["saved_paths"]["latest_json"].endswith("latest.json")
    assert "Forward Data Accrual Gate" in (
        tmp_path / "reports" / "latest.md"
    ).read_text(encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(
                repo_root
                / "run_agent_pipeline_control_forward_data_accrual_gate.py"
            ),
            "--accrual-plan-json",
            str(plan),
            "--source-path",
            str(source),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Status: forward_development_artifact_ready" in result.stdout
    assert "Can use as locked test evidence: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()


def _failed(payload: dict, check_id: str) -> bool:
    return any(
        check["check_id"] == check_id and check["status"] == "fail"
        for check in payload["checks"]
    )
