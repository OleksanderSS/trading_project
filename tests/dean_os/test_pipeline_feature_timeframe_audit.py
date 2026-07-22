from __future__ import annotations

import json

import pandas as pd

from dean_os.pipeline_feature_timeframe_audit import (
    PipelineFeatureTimeframeAudit,
)


def test_feature_timeframe_audit_blocks_mislabeled_intraday_data(
    tmp_path,
):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    timestamps = pd.date_range(
        "2026-06-01T13:30:00",
        periods=12,
        freq="15min",
    )
    rows = []
    for ticker in ("AMD", "NVDA"):
        for timestamp in timestamps:
            rows.append(
                {
                    "ticker": ticker,
                    "datetime": timestamp,
                    "interval": "1d",
                    "feature": 1.0,
                }
            )
    features_path = batch_dir / "features.parquet"
    pd.DataFrame(rows).to_parquet(features_path, index=False)
    stage5_path = batch_dir / "stage_5_results.json"
    stage5_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-07-02T01:22:26",
                "batch_name": "batch",
                "prediction_results": {
                    "amd": {"ticker": "AMD"},
                    "nvda": {"ticker": "NVDA"},
                },
                "models_metadata": {
                    "amd": {"ticker": "AMD"},
                    "nvda": {"ticker": "NVDA"},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = PipelineFeatureTimeframeAudit(
        tmp_path / "reports"
    ).build(
        features_path=features_path,
        stage5_path=stage5_path,
        tickers=["AMD", "NVDA"],
        save=False,
    )

    assert payload["status"] == (
        "pipeline_feature_timeframe_audit_blocked_mismatch"
    )
    assert payload["summary"][
        "timeframe_mismatch_tickers"
    ] == ["AMD", "NVDA"]
    assert payload["summary"]["can_use_for_stage4"] is False
    assert payload["stage5_candidate_binding"][
        "relationship_status"
    ] == "co_located_same_batch_candidate_not_hash_bound"
    assert payload["stage5_candidate_binding"][
        "can_assert_feature_parentage"
    ] is False
    assert payload["safety"]["can_trade"] is False


def test_feature_timeframe_audit_accepts_verified_intraday_cadence(
    tmp_path,
):
    timestamps = pd.date_range(
        "2026-06-01T13:30:00Z",
        periods=12,
        freq="15min",
    )
    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "ticker": ["AMD"] * len(timestamps),
            "datetime": timestamps,
            "interval": ["15m"] * len(timestamps),
        }
    ).to_parquet(features_path, index=False)

    payload = PipelineFeatureTimeframeAudit().build(
        features_path=features_path,
        tickers=["AMD"],
        save=False,
    )

    assert payload["status"] == (
        "pipeline_feature_timeframe_audit_ready"
    )
    assert payload["summary"]["can_use_for_stage4"] is True
    assert payload["summary"]["timezone_aware_ticker_count"] == 1
