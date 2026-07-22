from __future__ import annotations

import json
from pathlib import Path

from dean_os.schemas import MarketContext
from dean_os.world_model_event_learning import WorldModelEventLearningPacket
from dean_os.world_model_pipeline_context import (
    WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT,
    WorldModelPipelineContextDiscovery,
    metadata_from_pipeline_context_bundle,
)

AS_OF = "2026-07-01T12:00:00+00:00"
DOMAIN_ID = "semiconductor_ai_infrastructure"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _semantic_news() -> dict:
    return {
        "title": (
            "Nvidia AI demand growth confirms semiconductor memory shortage "
            "and data center capex pressure"
        ),
        "summary": (
            "AI demand growth is increasing HBM memory shortage risk, "
            "supporting data center capex but raising supply-chain constraints."
        ),
        "published_at": "2026-07-01T10:00:00+00:00",
        "url": "https://example.test/news/ai-memory-shortage",
        "tickers": ["NVDA"],
        "_dean_semantic_evidence": {
            "producer_contract": "test_news_contract",
            "evidence_type": "sector_demand",
            "matched_terms": ["ai demand", "data center demand"],
            "required_lane_eligible": True,
            "source_tier": "tier_2_strong_context",
            "source_identity": "reuters",
            "candidate_sha256": "abc123",
        },
    }


def _pipeline_artifacts(base: Path) -> None:
    _write_json(
        base / "pipeline_stage23_regeneration_current" / "latest.json",
        {
            "run_id": "stage23_1",
            "mode": "pipeline_stage23_regeneration",
            "schema_version": "dean_pipeline_stage23_regeneration_v1",
            "status": "stage23_regeneration_review_ready",
            "scope": {"tickers": ["NVDA"], "timeframe": "15m"},
            "batch_artifacts": {
                "features_sha256": "features_15m_sha",
                "targets_sha256": "targets_15m_sha",
            },
            "stage3_cache": {
                "schema_version": "dean_pipeline_stage23_stage3_cache_v1",
                "shard_count": 1,
                "shards": [
                    {
                        "ticker": "NVDA",
                        "timeframe": "15m",
                        "cache_key": "cache_1",
                    }
                ],
            },
        },
    )
    _write_json(
        base / "pipeline_stage4_exact_context_review_nvda_15m_current" / "latest.json",
        {
            "run_id": "stage4_1",
            "mode": "pipeline_stage4_exact_context_review",
            "schema_version": "dean_pipeline_stage4_exact_context_review_v1",
            "status": "walk_forward_candidate_blocked_by_validation_contract",
            "scope": {
                "ticker": "NVDA",
                "timeframe": "15m",
                "target_name": "target_intraday_up_15m",
            },
            "parent_lineage": {
                "all_parent_hashes_verified": True,
                "features": {"sha256": "features_15m_sha"},
                "targets": {"sha256": "targets_15m_sha"},
            },
            "timeframe_lineage": {"safe_for_prediction_lineage": True},
            "summary": {
                "contract_passed": False,
                "can_trade": False,
            },
        },
    )
    _write_json(
        base / "pipeline_prediction_review_packet_current" / "latest.json",
        {
            "run_id": "stage5_1",
            "mode": "pipeline_prediction_review_packet",
            "schema_version": "dean_stage5_prediction_review_v1",
            "status": "stage5_prediction_review_partial",
            "context_count": 1,
            "complete_context_count": 1,
            "contexts": [
                {
                    "context_key": "prediction_nvda_15m",
                    "ticker": "NVDA",
                    "timeframe": "15m",
                    "target_name": "target_intraday_up_15m",
                    "lineage_status": "complete",
                    "review_issues": [],
                    "prediction": {
                        "value": 0.61,
                        "confidence": 0.7,
                        "as_of": AS_OF,
                    },
                }
            ],
        },
    )
    _write_json(
        base / "pipeline_metric_input_readiness_gate_current" / "latest.json",
        {
            "run_id": "metric_1",
            "mode": "pipeline_metric_input_readiness_gate",
            "summary": {
                "readiness_status": "metric_inputs_ready_with_cautions",
                "axis_status_counts": {"clear": 3, "caution": 2},
                "blocked_metric_planes": [],
                "can_trade": False,
            },
        },
    )


def test_world_model_pipeline_context_discovers_exact_15m_and_missing_lanes(tmp_path):
    _pipeline_artifacts(tmp_path)

    payload = WorldModelPipelineContextDiscovery(
        base_path=tmp_path,
        output_dir=tmp_path / "out",
    ).build(
        tickers=["NVDA"],
        timeframes=["15m", "60m", "1d"],
        save=False,
    )

    assert payload["contract"] == WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT
    assert payload["summary"]["status"] == "pipeline_context_bundle_ready_with_gaps"
    assert payload["summary"]["available_lane_count"] == 1
    assert payload["summary"]["exact_context_lane_count"] == 1
    assert payload["summary"]["missing_lane_count"] == 2
    assert payload["summary"]["stage3_shard_count"] == 1
    assert payload["summary"]["stage3_cache_materialized_lane_count"] == 1
    assert payload["summary"]["stage3_cache_missing_ready_lane_count"] == 0
    assert payload["summary"]["can_trade"] is False

    lanes = {lane["timeframe"]: lane for lane in payload["timeframe_lanes"]}
    assert lanes["15m"]["status"] == "pipeline_lane_exact_context_available"
    assert lanes["15m"]["exact_context_ready"] is True
    assert lanes["15m"]["stage3_cache_status"] == (
        "stage3_cache_materialized_in_stage23_artifact"
    )
    assert lanes["15m"]["stage3_shard_count"] == 1
    assert lanes["60m"]["status"] == "pipeline_lane_missing"
    assert lanes["1d"]["status"] == "pipeline_lane_missing"
    assert "pipeline_lane_15m_exact_context" in payload["pipeline_context"][
        "context_tags"
    ]
    assert "pipeline_lane_60m_missing" in payload["pipeline_context"][
        "context_tags"
    ]
    assert payload["stage5_prediction_review"]["contexts_included"] is False
    assert "contexts" not in payload["stage5_prediction_review"]


def test_discovered_pipeline_context_conditions_world_model_packet(tmp_path):
    _pipeline_artifacts(tmp_path)
    bundle = WorldModelPipelineContextDiscovery(
        base_path=tmp_path,
        output_dir=tmp_path / "out",
    ).build(
        tickers=["NVDA"],
        timeframes=["15m", "60m", "1d"],
        save=False,
    )
    metadata = metadata_from_pipeline_context_bundle(bundle)
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA"],
        timeframes=["15m", "60m", "1d"],
        news=[_semantic_news()],
        metadata=metadata,
        pipeline_result=bundle["pipeline_context"],
    )

    payload = WorldModelEventLearningPacket().build(
        context,
        domain_id=DOMAIN_ID,
        save=False,
    )

    summary = payload["summary"]
    assert summary["pipeline_indicator_context_status"] == (
        "pipeline_indicator_context_ready"
    )
    assert summary["indicator_metric_count"] >= 8
    assert "pipeline_lane_15m_exact_context" in summary["pipeline_context_tags"]
    assert "pipeline_lane_60m_missing" in summary["pipeline_context_tags"]
    assert "stage3_shard_count" in summary["watch_metrics"]

    replay_context = payload["replay_tasks"][0]["pipeline_context_snapshot"]
    assert replay_context["indicator_metric_count"] >= 8
    assert "pipeline_lane_1d_missing" in replay_context["context_tags"]
    assert replay_context["pipeline_indicator_context_status"] == (
        "pipeline_indicator_context_ready"
    )


def test_world_model_pipeline_context_marks_ready_stage23_missing_cache_metadata(tmp_path):
    _write_json(
        tmp_path / "pipeline_stage23_regeneration_current" / "latest.json",
        {
            "run_id": "stage23_old",
            "mode": "pipeline_stage23_regeneration",
            "schema_version": "dean_pipeline_stage23_regeneration_v1",
            "status": "stage23_regeneration_review_ready",
            "scope": {"tickers": ["NVDA"], "timeframe": "15m"},
        },
    )

    payload = WorldModelPipelineContextDiscovery(
        base_path=tmp_path,
        output_dir=tmp_path / "out",
    ).build(
        tickers=["NVDA"],
        timeframes=["15m", "60m", "1d"],
        save=False,
    )

    lanes = {lane["timeframe"]: lane for lane in payload["timeframe_lanes"]}
    assert lanes["15m"]["status"] == "pipeline_lane_stage23_context_available"
    assert lanes["15m"]["stage3_cache_status"] == (
        "stage3_cache_missing_from_ready_stage23_artifact"
    )
    assert "stage3_cache_metadata_missing_for_15m" in lanes["15m"]["warnings"]
    assert payload["summary"]["stage3_cache_materialized_lane_count"] == 0
    assert payload["summary"]["stage3_cache_missing_ready_lane_count"] == 1
    assert "pipeline_lane_15m_stage3_cache_missing" in payload[
        "pipeline_context"
    ]["context_tags"]


def test_world_model_pipeline_context_ignores_stage4_from_other_stage23_batch(tmp_path):
    _pipeline_artifacts(tmp_path)
    _write_json(
        tmp_path / "pipeline_stage4_exact_context_review_incompatible" / "latest.json",
        {
            "run_id": "stage4_incompatible",
            "mode": "pipeline_stage4_exact_context_review",
            "schema_version": "dean_pipeline_stage4_exact_context_review_v1",
            "scope": {
                "ticker": "NVDA",
                "timeframe": "15m",
                "target_name": "target_intraday_up_15m",
            },
            "parent_lineage": {
                "all_parent_hashes_verified": True,
                "features": {"sha256": "other_features"},
                "targets": {"sha256": "other_targets"},
            },
            "timeframe_lineage": {"safe_for_prediction_lineage": True},
        },
    )

    payload = WorldModelPipelineContextDiscovery(
        base_path=tmp_path,
        output_dir=tmp_path / "out",
    ).build(
        tickers=["NVDA"],
        timeframes=["15m"],
        save=False,
    )

    lane = payload["timeframe_lanes"][0]
    assert lane["stage4_exact_context_count"] == 1
    assert lane["stage4_incompatible_context_count"] == 1
    assert "stage4_parent_hash_mismatch_count=1" in lane["warnings"]
