from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.pipeline_control.pipeline_control_instance_contract import PipelineControlInstanceContract


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _axis(name: str, status: str = "clear") -> dict:
    return {
        "name": name,
        "status": status,
        "score": 0.8 if status == "clear" else 0.4,
        "metrics": {},
        "constraints": {},
        "reasons": [f"{name} fixture reason"],
    }


def _surface(*, status: str = "clear", blocked_axis: str | None = None, unsafe_config: bool = False) -> dict:
    axes = [
        _axis("profitability"),
        _axis("risk"),
        _axis("validation"),
        _axis("feature_stability"),
        _axis("data_quality"),
        _axis("replay_repeatability"),
    ]
    if blocked_axis:
        axes = [_axis(axis["name"], "blocked" if axis["name"] == blocked_axis else axis["status"]) for axis in axes]
        status = "blocked"
    return {
        "run_id": "pipeline_surface_fixture",
        "mode": "pipeline_control_surface",
        "surface": {
            "status": status,
            "feasible": status != "blocked",
            "axis_status_counts": {"clear": sum(1 for axis in axes if axis["status"] == "clear")},
            "axes": axes,
            "allowed_variation": {
                "policy": "reviewed_experiment" if status == "clear" else "no_tuning_experiment",
                "max_trials": 25 if status == "clear" else 0,
                "production_write_allowed": unsafe_config,
            },
        },
        "proposal_gate": {
            "status": "review_required" if status != "blocked" else "blocked",
            "can_propose_tuning": status != "blocked",
            "can_change_production_config": unsafe_config,
            "reason": "fixture",
        },
        "constraints": {},
    }


def _architecture() -> dict:
    return {
        "mode": "current_architecture_map",
        "summary": {
            "can_write_production_config_now": False,
            "can_trade": False,
        },
    }


def _domain_instance() -> dict:
    return {
        "mode": "domain_analyst_instance_contract",
        "summary": {
            "can_scale_to_other_domains_now": False,
            "can_trade": False,
        },
    }


def _write_inputs(tmp_path: Path, *, surface: dict | None = None) -> dict[str, Path]:
    return {
        "surface": _write_json(tmp_path / "surface" / "latest.json", surface or _surface()),
        "architecture": _write_json(tmp_path / "architecture" / "latest.json", _architecture()),
        "domain_instance": _write_json(tmp_path / "domain" / "latest.json", _domain_instance()),
    }


def test_pipeline_control_instance_contract_marks_clear_surface_review_ready(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = PipelineControlInstanceContract(tmp_path / "reports").build(
        pipeline_surface_json=paths["surface"],
        architecture_map_json=paths["architecture"],
        domain_instance_contract_json=paths["domain_instance"],
        save=False,
    )

    assert payload["summary"]["instance_status"] == "pipeline_control_instance_review_ready"
    assert payload["summary"]["can_propose_reviewed_experiments_after_manual_review"] is True
    assert payload["summary"]["can_write_production_config"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["metric_plane_contract"]["required_planes_covered"] is True


def test_pipeline_control_instance_contract_blocks_when_metric_plane_blocked(tmp_path):
    paths = _write_inputs(tmp_path, surface=_surface(blocked_axis="data_quality"))

    payload = PipelineControlInstanceContract(tmp_path / "reports").build(
        pipeline_surface_json=paths["surface"],
        architecture_map_json=paths["architecture"],
        domain_instance_contract_json=paths["domain_instance"],
        save=False,
    )

    assert payload["summary"]["instance_status"] == "blocked_pipeline_control_instance"
    assert payload["summary"]["blocked_metric_planes"] == ["data_quality"]
    assert payload["summary"]["can_propose_reviewed_experiments_after_manual_review"] is False
    assert any(check["code"] == "no_blocked_metric_planes" and check["status"] == "fail" for check in payload["review_checks"])


def test_pipeline_control_instance_contract_blocks_unsafe_config_gate(tmp_path):
    paths = _write_inputs(tmp_path, surface=_surface(unsafe_config=True))

    payload = PipelineControlInstanceContract(tmp_path / "reports").build(
        pipeline_surface_json=paths["surface"],
        architecture_map_json=paths["architecture"],
        domain_instance_contract_json=paths["domain_instance"],
        save=False,
    )

    assert payload["summary"]["instance_status"] == "blocked_pipeline_control_instance"
    assert any(check["code"] == "allowed_variation_no_production_write" and check["status"] == "fail" for check in payload["review_checks"])
    assert any(check["code"] == "proposal_gate_no_config_write" and check["status"] == "fail" for check in payload["review_checks"])


def test_pipeline_control_instance_contract_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = PipelineControlInstanceContract(tmp_path / "reports").build(
        pipeline_surface_json=paths["surface"],
        architecture_map_json=paths["architecture"],
        domain_instance_contract_json=paths["domain_instance"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can trade: False" in markdown
    assert "Metric Plane Contract" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_pipeline_control_instance_contract.py"),
            "--pipeline-surface-json",
            str(paths["surface"]),
            "--architecture-map-json",
            str(paths["architecture"]),
            "--domain-instance-contract-json",
            str(paths["domain_instance"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Instance status: pipeline_control_instance_review_ready" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
