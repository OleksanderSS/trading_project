from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dean_os.agents.pipeline_readiness import (
    _EXPECTED_MODES,
    _load_binding,
    _summarize_binding,
    load_pipeline_readiness,
    PipelineReadinessAgent,
)
from dean_os.schemas import MarketContext


def test_expected_modes_defined():
    assert "timeframe_lane_readiness" in _EXPECTED_MODES
    assert "feature_timeframe_audit" in _EXPECTED_MODES
    assert "target_readiness" in _EXPECTED_MODES
    assert "stage4_review" in _EXPECTED_MODES
    assert "prediction_review" in _EXPECTED_MODES
    assert "sector_to_ticker_review" in _EXPECTED_MODES


def test_load_binding_missing_path():
    import pytest
    with pytest.raises(FileNotFoundError):
        _load_binding("test", Path("/nonexistent/path.json"), "test_mode")


def test_summarize_binding_empty():
    result = _summarize_binding("test", {})
    assert result["status"] == "unknown"
    assert result["summary_keys"] == []


def test_load_pipeline_readiness_empty_paths():
    result = load_pipeline_readiness({})
    assert result["is_ready"] is True
    assert result["bound_count"] == 0
    assert result["blockers"] == []
    assert result["blocking_reasons"] == []
    assert result["status"] == "pipeline_readiness_no_checks_bound"


def test_multitimeframe_readiness_is_first_class(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps(
            {
                "mode": "pipeline_timeframe_lane_readiness",
                "summary": {
                    "requested_lane_count": 3,
                    "source_valid_lane_count": 3,
                    "can_condition_world_model": True,
                },
            }
        ),
        encoding="utf-8",
    )
    result = load_pipeline_readiness({"timeframe_lane_readiness": path})
    assert result["is_ready"] is True
    assert result["status"] == "pipeline_readiness_ready"


def test_multitimeframe_readiness_blocks_incomplete_lane_set(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps(
            {
                "mode": "pipeline_timeframe_lane_readiness",
                "summary": {
                    "requested_lane_count": 3,
                    "source_valid_lane_count": 2,
                    "can_condition_world_model": False,
                },
            }
        ),
        encoding="utf-8",
    )
    result = load_pipeline_readiness({"timeframe_lane_readiness": path})
    assert result["is_ready"] is False
    assert result["blocking_reasons"] == ["multitimeframe_context_not_ready"]


def test_pipeline_readiness_no_config():
    agent = PipelineReadinessAgent("pipeline_readiness", {})
    report = asyncio.run(
        agent.run(
            MarketContext(
                as_of="2026-07-09T09:00:00+00:00",
            )
        )
    )
    assert report.agent_name == "pipeline_readiness"
    assert report.verdict == "needs_more_data"


def test_pipeline_readiness_empty_artifact_paths():
    agent = PipelineReadinessAgent("pipeline_readiness", {"artifact_paths": {}})
    report = asyncio.run(
        agent.run(
            MarketContext(
                as_of="2026-07-09T09:00:00+00:00",
            )
        )
    )
    assert report.verdict == "needs_more_data"
    assert "No pipeline artifact paths" in " ".join(report.reasons)


def test_pipeline_readiness_with_bad_path():
    agent = PipelineReadinessAgent(
        "pipeline_readiness",
        {"artifact_paths": {"feature_timeframe_audit": "/nonexistent/latest.json"}},
    )
    report = asyncio.run(
        agent.run(
            MarketContext(
                as_of="2026-07-09T09:00:00+00:00",
            )
        )
    )
    assert report.verdict == "blocked"
