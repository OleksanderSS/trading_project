from __future__ import annotations

import json

import pandas as pd

from dean_os.pipeline_target_readiness_audit import (
    PipelineTargetReadinessAudit,
    _sha256,
)


def _write_batch(tmp_path, *, target_values, target_name):
    timestamps = pd.date_range(
        "2026-06-01T13:30:00Z",
        periods=8,
        freq="15min",
    )
    targets = pd.DataFrame(
        {
            "ticker": ["AMD"] * 8,
            "datetime": timestamps,
            "interval": ["15m"] * 8,
            target_name: target_values,
        }
    )
    features = targets[
        ["ticker", "datetime", "interval"]
    ].assign(feature=range(8))
    target_path = tmp_path / "targets.parquet"
    feature_path = tmp_path / "features.parquet"
    metadata_path = tmp_path / "batch_metadata.json"
    targets.to_parquet(target_path, index=False)
    features.to_parquet(feature_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "lineage": {
                    "targets_sha256": _sha256(target_path),
                    "features_sha256": _sha256(feature_path),
                }
            }
        ),
        encoding="utf-8",
    )
    return target_path, feature_path, metadata_path


def test_target_readiness_accepts_bound_semantic_target(tmp_path):
    target_path, feature_path, metadata_path = _write_batch(
        tmp_path,
        target_values=[0, 1] * 4,
        target_name="target_intraday_up_15m",
    )

    payload = PipelineTargetReadinessAudit(
        tmp_path / "reports"
    ).build(
        targets_path=target_path,
        features_path=feature_path,
        batch_metadata_path=metadata_path,
        tickers=["AMD"],
        timeframe="15m",
        save=False,
    )

    assert payload["status"] == "pipeline_target_readiness_ready"
    assert payload["summary"]["ready_target_count"] == 1
    assert payload["summary"]["can_use_for_stage4"] is True
    assert payload["lineage_bindings"]["target_hash_verified"] is True
    assert (
        payload["lineage_bindings"]["feature_artifact"][
            "hash_verified"
        ]
        is True
    )


def test_target_readiness_blocks_daily_indicator_on_intraday(tmp_path):
    target_path, feature_path, metadata_path = _write_batch(
        tmp_path,
        target_values=list(range(8)),
        target_name="target_rsi_14_f1",
    )

    payload = PipelineTargetReadinessAudit(
        tmp_path / "reports"
    ).build(
        targets_path=target_path,
        features_path=feature_path,
        batch_metadata_path=metadata_path,
        tickers=["AMD"],
        timeframe="15m",
        save=False,
    )

    assert payload["status"] == "pipeline_target_readiness_blocked"
    assert payload["target_reports"][0]["blocking_reasons"] == [
        "target_not_applicable_to_timeframe"
    ]


def test_target_readiness_blocks_single_class_target(tmp_path):
    target_path, feature_path, metadata_path = _write_batch(
        tmp_path,
        target_values=[1] * 8,
        target_name="target_intraday_up_15m",
    )

    payload = PipelineTargetReadinessAudit(
        tmp_path / "reports"
    ).build(
        targets_path=target_path,
        features_path=feature_path,
        batch_metadata_path=metadata_path,
        tickers=["AMD"],
        timeframe="15m",
        save=False,
    )

    assert payload["status"] == "pipeline_target_readiness_blocked"
    assert payload["target_reports"][0]["blocking_reasons"] == [
        "classification_has_fewer_than_two_classes"
    ]


def test_target_readiness_allows_ready_subset_and_excludes_degenerate_target(tmp_path):
    target_path, feature_path, metadata_path = _write_batch(
        tmp_path,
        target_values=[0, 1] * 4,
        target_name="target_intraday_up_15m",
    )
    targets = pd.read_parquet(target_path)
    targets["target_hourly_volume_spike_1h"] = 0
    targets.to_parquet(target_path, index=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["lineage"]["targets_sha256"] = _sha256(target_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    payload = PipelineTargetReadinessAudit(
        tmp_path / "reports"
    ).build(
        targets_path=target_path,
        features_path=feature_path,
        batch_metadata_path=metadata_path,
        tickers=["AMD"],
        timeframe="15m",
        save=False,
    )

    assert payload["status"] == "pipeline_target_readiness_ready_with_gaps"
    assert payload["summary"]["can_use_for_stage4"] is True
    assert payload["summary"]["ready_target_names"] == [
        "target_intraday_up_15m"
    ]
    assert payload["summary"]["blocked_target_names"] == [
        "target_hourly_volume_spike_1h"
    ]
    assert payload["blocking_reasons"] == []
    assert payload["target_exclusions"] == [
        "target_not_ready:target_hourly_volume_spike_1h"
    ]
