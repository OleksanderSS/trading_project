from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dean_os.prospective_accumulation_runbook import ProspectiveAccumulationRunbookBuilder


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    plan = _write(
        tmp_path / "plan.json",
        {"contract": "dean_replay_outcome_evidence_plan_v1", "task_plans": []},
    )
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    monitor = _write(
        tmp_path / "monitor.json",
        {
            "contract": "dean_replay_checkpoint_monitor_v1",
            "inputs": {"evidence_plan": {"sha256": plan_sha}},
            "summary": {"task_count": 9},
            "tasks": [
                {
                    "pre_due_source_review": "2026-08-03T00:00:00+00:00",
                    "due_outcome_review": "2026-08-10T00:00:00+00:00",
                }
            ],
        },
    )
    return plan, monitor


def test_builds_review_only_checkpoint_bound_runbook(tmp_path: Path) -> None:
    plan, monitor = _inputs(tmp_path)
    payload = ProspectiveAccumulationRunbookBuilder(tmp_path / "out").build(
        plan, monitor, as_of="2026-07-12T00:00:00+00:00", save=False
    )
    assert payload["summary"]["replay_task_count"] == 9
    assert payload["summary"]["lane_count"] == 7
    assert payload["summary"]["nearest_pre_due_review"].startswith("2026-08-03")
    assert payload["safety"]["collector_execution_performed"] is False
    assert payload["safety"]["scheduler_write_performed"] is False
    assert payload["summary"]["early_outcome_evaluation_allowed"] is False
    market = next(lane for lane in payload["collection_lanes"] if lane["lane_id"] == "clean_market_15m_60m_1d")
    assert "--timeframe 15m" in market["command"]
    assert "--timeframe 60m" in market["command"]
    assert "--timeframe 1d" in market["command"]
    assert market["command_executable"] is True
    macro = next(lane for lane in payload["collection_lanes"] if lane["lane_id"] == "macro_context")
    assert macro["command_executable"] is False
    assert "source_path" in macro["missing_parameters"]


def test_rejects_monitor_bound_to_another_plan(tmp_path: Path) -> None:
    plan, monitor = _inputs(tmp_path)
    payload = json.loads(monitor.read_text(encoding="utf-8"))
    payload["inputs"]["evidence_plan"]["sha256"] = "0" * 64
    _write(monitor, payload)
    with pytest.raises(ValueError, match="not bound"):
        ProspectiveAccumulationRunbookBuilder(tmp_path / "out").build(
            plan, monitor, as_of="2026-07-12T00:00:00+00:00", save=False
        )
