from __future__ import annotations

import hashlib
import json

import pytest

from dean_os.analyst_core.pipeline_context_evidence_loader import (
    PipelineContextEvidenceLoader,
)
from dean_os.analyst_core.sector_analyst import SectorAnalyst


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path):
    stage23 = _write_json(tmp_path / "stage23.json", {"status": "ready"})
    stage4 = _write_json(
        tmp_path / "stage4.json",
        {"summary": {"contract_passed": False}},
    )
    bundle = {
        "contract": "dean_world_model_pipeline_context_v1",
        "mode": "world_model_pipeline_context_discovery",
        "created_at": "2026-07-10T19:00:00+00:00",
        "summary": {"status": "pipeline_context_bundle_ready"},
        "safety": {
            "review_only": True,
            "can_trade": False,
            "learning_memory_write_performed": False,
        },
        "timeframe_lanes": [
            {
                "timeframe": "60m",
                "status": "pipeline_lane_exact_context_available",
                "tickers": ["NVDA", "TSM"],
                "stage3_shard_count": 2,
                "stage3_cache_status": (
                    "stage3_cache_materialized_in_stage23_artifact"
                ),
                "artifacts": {
                    "stage23_regeneration": {
                        "path": str(stage23),
                        "sha256": _sha(stage23),
                    },
                    "stage4_exact_context": [
                        {"path": str(stage4), "sha256": _sha(stage4)}
                    ],
                },
            }
        ],
    }
    return _write_json(tmp_path / "context.json", bundle)


def test_pipeline_context_loader_emits_verified_but_not_eligible_weak_model(tmp_path):
    items = PipelineContextEvidenceLoader().load(
        _bundle(tmp_path),
        domain_id="semiconductor_ai_infrastructure",
        as_of="2026-07-10T20:00:00+00:00",
        tickers=["NVDA", "TSM"],
    )

    assert len(items) == 1
    item = items[0]
    assert item.evidence_type == "market_confirmation"
    assert item.provenance["timeframe"] == "60m"
    assert item.provenance["required_lane_eligible"] is False
    assert item.provenance["stage4_validation_contract_passed"] is False
    assert item.point_in_time["status"] == "point_in_time_compatible"


def test_pipeline_context_loader_rejects_linked_hash_mismatch(tmp_path):
    path = _bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["timeframe_lanes"][0]["artifacts"]["stage23_regeneration"][
        "sha256"
    ] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        PipelineContextEvidenceLoader().load(
            path,
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-10T20:00:00+00:00",
        )


def test_pipeline_context_loader_rejects_future_bundle(tmp_path):
    with pytest.raises(ValueError, match="future evidence"):
        PipelineContextEvidenceLoader().load(
            _bundle(tmp_path),
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-10T18:00:00+00:00",
        )


def test_pipeline_timeframe_lanes_remain_distinct_after_merge(tmp_path):
    path = _bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    lane = payload["timeframe_lanes"][0]
    payload["timeframe_lanes"] = [
        {**lane, "timeframe": timeframe}
        for timeframe in ("15m", "60m", "1d")
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    items = PipelineContextEvidenceLoader().load(
        path,
        domain_id="semiconductor_ai_infrastructure",
        as_of="2026-07-10T20:00:00+00:00",
        tickers=["NVDA"],
    )

    report = SectorAnalyst(
        "semiconductor_ai_infrastructure"
    ).run_from_evidence(
        items,
        as_of="2026-07-10T20:00:00+00:00",
        tickers=["NVDA"],
    )

    assert len(items) == 3
    assert report.evidence_count == 3
