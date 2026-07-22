from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from dean_os.pipeline_timeframe_lane_readiness import (
    PIPELINE_TIMEFRAME_LANE_READINESS_CONTRACT,
    PipelineTimeframeLaneReadinessPlan,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path, *, include_1d: bool = True) -> Path:
    rows = []
    for timeframe in ["15m", "60m"] + (["1d"] if include_1d else []):
        frequency = {
            "15m": "15min",
            "60m": "60min",
            "1d": "1D",
        }[timeframe]
        timestamps = pd.date_range(
            "2026-06-01T09:30:00Z",
            periods=5,
            freq=frequency,
        )
        for i in range(5):
            rows.append(
                {
                    "ticker": "NVDA",
                    "datetime": timestamps[i],
                    "interval": timeframe,
                    "open": 100 + i,
                    "high": 101 + i,
                    "low": 99 + i,
                    "close": 100.5 + i,
                    "volume": 1000 + i,
                    "hash": f"{timeframe}-{i}",
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _stage23_with_batch_artifacts(tmp_path: Path, source: Path) -> Path:
    batch = tmp_path / "batch"
    batch.mkdir()
    features = batch / "features.parquet"
    targets = batch / "targets.parquet"
    metadata = batch / "batch_metadata.json"
    pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "datetime": "2026-06-01T09:30:00Z",
                "interval": "15m",
                "feature": 1.0,
            }
        ]
    ).to_parquet(features, index=False)
    pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "datetime": "2026-06-01T09:30:00Z",
                "interval": "15m",
                "target": 1,
            }
        ]
    ).to_parquet(targets, index=False)
    metadata.write_text("{}", encoding="utf-8")
    artifact = tmp_path / "stage23.json"
    artifact.write_text(
        json.dumps(
            {
                "run_id": "stage23_15m",
                "mode": "pipeline_stage23_regeneration",
                "schema_version": "dean_pipeline_stage23_regeneration_v1",
                "status": "stage23_regeneration_review_ready",
                "source_artifact": {
                    "path": str(source),
                    "sha256": _sha(source),
                    "format": "parquet",
                },
                "scope": {
                    "tickers": ["NVDA"],
                    "timeframe": "15m",
                },
                "batch_artifacts": {
                    "metadata_path": str(metadata),
                    "features_path": str(features),
                    "targets_path": str(targets),
                    "features_sha256": _sha(features),
                    "targets_sha256": _sha(targets),
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact


def _context(tmp_path: Path, source: Path) -> dict:
    stage23 = _stage23_with_batch_artifacts(tmp_path, source)
    return {
        "run_id": "context_1",
        "contract": "dean_world_model_pipeline_context_v1",
        "summary": {"status": "pipeline_context_bundle_ready_with_gaps"},
        "timeframe_lanes": [
            {
                "timeframe": "15m",
                "status": "pipeline_lane_exact_context_available",
                "stage23_ready": True,
                "stage3_cache_status": (
                    "stage3_cache_missing_from_ready_stage23_artifact"
                ),
                "stage3_shard_count": 0,
                "stage4_exact_context_count": 1,
                "artifacts": {
                    "stage23_regeneration": {
                        "available": True,
                        "path": str(stage23),
                    }
                },
                "warnings": ["stage3_cache_metadata_missing_for_15m"],
            },
            {
                "timeframe": "60m",
                "status": "pipeline_lane_missing",
                "stage3_cache_status": "stage23_artifact_missing",
                "artifacts": {
                    "stage23_regeneration": {
                        "available": False,
                        "path": None,
                    }
                },
            },
            {
                "timeframe": "1d",
                "status": "pipeline_lane_missing",
                "stage3_cache_status": "stage23_artifact_missing",
                "artifacts": {
                    "stage23_regeneration": {
                        "available": False,
                        "path": None,
                    }
                },
            },
        ],
    }


def test_pipeline_timeframe_lane_readiness_plans_source_backed_missing_artifacts(tmp_path):
    source = _source(tmp_path / "source.parquet")
    payload = PipelineTimeframeLaneReadinessPlan().build(
        source_path=source,
        tickers=["NVDA"],
        timeframes=["15m", "60m", "1d"],
        pipeline_context_json=_context(tmp_path, source),
        save=False,
    )

    assert payload["contract"] == PIPELINE_TIMEFRAME_LANE_READINESS_CONTRACT
    assert payload["summary"]["status"] == "pipeline_timeframe_lanes_ready_with_gaps"
    assert payload["summary"]["source_available_lane_count"] == 3
    assert payload["summary"]["source_valid_lane_count"] == 3
    assert payload["summary"]["source_invalid_lane_count"] == 0
    assert payload["summary"]["exact_context_lane_count"] == 1
    assert payload["summary"]["artifact_missing_lane_count"] == 2
    assert payload["summary"]["stage3_cache_missing_ready_lane_count"] == 1
    assert payload["summary"]["batch_artifact_lane_count"] == 1
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_trade"] is False

    lanes = {lane["timeframe"]: lane for lane in payload["timeframe_lanes"]}
    assert lanes["15m"]["lane_status"] == "exact_lane_ready_but_stage3_cache_missing"
    assert lanes["15m"]["batch_artifact_status"] == (
        "batch_artifacts_verified_not_reusable_stage3_cache"
    )
    assert lanes["60m"]["lane_status"] == "source_available_but_stage23_artifact_missing"
    assert lanes["1d"]["lane_status"] == "source_available_but_stage23_artifact_missing"
    assert "--timeframe 60m" in lanes["60m"]["suggested_stage23_command"]
    assert "--shard-cache-dir data\\colab\\stage3_shard_cache\\dean_review" in lanes[
        "60m"
    ]["suggested_stage23_command"]


def test_pipeline_timeframe_lane_readiness_marks_missing_source_lane(tmp_path):
    source = _source(tmp_path / "source.parquet", include_1d=False)
    context = _context(tmp_path, source)
    payload = PipelineTimeframeLaneReadinessPlan().build(
        source_path=source,
        tickers=["NVDA"],
        timeframes=["1d"],
        pipeline_context_json=context,
        save=False,
    )

    lane = payload["timeframe_lanes"][0]
    assert lane["timeframe"] == "1d"
    assert lane["lane_status"] == "source_rows_missing"
    assert lane["suggested_stage23_command"] is None
    assert payload["summary"]["source_available_lane_count"] == 0


def test_pipeline_timeframe_lane_readiness_blocks_invalid_cadence_lane(tmp_path):
    source = tmp_path / "source.parquet"
    rows = []
    for i, timestamp in enumerate(
        pd.date_range(
            "2026-06-01T09:30:00Z",
            periods=8,
            freq="60min",
        )
    ):
        rows.append(
            {
                "ticker": "NVDA",
                "datetime": timestamp,
                "interval": "1d",
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100.5 + i,
                "volume": 1000 + i,
            }
        )
    pd.DataFrame(rows).to_parquet(source, index=False)

    payload = PipelineTimeframeLaneReadinessPlan().build(
        source_path=source,
        tickers=["NVDA"],
        timeframes=["1d"],
        pipeline_context_json=_context(tmp_path, source),
        save=False,
    )

    lane = payload["timeframe_lanes"][0]
    assert lane["lane_status"] == "source_rows_invalid"
    assert lane["source"]["validation"]["valid"] is False
    assert "timeframe_cadence" in lane["source"]["validation"]["blocking_reasons"]
    assert lane["suggested_stage23_command"] is None
    assert payload["summary"]["source_invalid_lane_count"] == 1


def test_pipeline_timeframe_lane_readiness_ignores_stage23_from_other_source(tmp_path):
    old_source = _source(tmp_path / "old_source.parquet")
    context = _context(tmp_path, old_source)
    current_source = _source(tmp_path / "current_source.parquet")
    changed = pd.read_parquet(current_source)
    changed.loc[0, "close"] = 777.0
    changed.to_parquet(current_source, index=False)

    payload = PipelineTimeframeLaneReadinessPlan().build(
        source_path=current_source,
        tickers=["NVDA"],
        timeframes=["15m"],
        pipeline_context_json=context,
        save=False,
    )

    lane = payload["timeframe_lanes"][0]
    assert lane["lane_status"] == "source_available_but_stage23_artifact_missing"
    assert lane["context"]["stage23_ready"] is False
    assert lane["stage23_artifact"]["effective_available"] is False
    assert "stage23_source_sha256_mismatch" in lane["stage23_artifact"][
        "source_compatibility"
    ]["blocking_reasons"]
    assert payload["summary"]["exact_context_lane_count"] == 0
    assert payload["summary"]["can_condition_world_model"] is False
