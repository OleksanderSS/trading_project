from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dean_os.pipeline_stage4_exact_context_review import (
    PipelineStage4ExactContextReview,
    _assemble_exact_context,
    _sha256,
)


def _frames(rows: int = 240):
    timestamps = pd.date_range(
        "2026-01-01T14:30:00Z",
        periods=rows,
        freq="15min",
    )
    target = np.asarray([0, 1] * (rows // 2))
    features = pd.DataFrame(
        {
            "ticker": ["NVDA"] * rows,
            "datetime": timestamps,
            "interval": ["15m"] * rows,
            "signal": target + np.sin(np.arange(rows) / 5.0),
            "noise": np.cos(np.arange(rows) / 7.0),
        }
    )
    targets = features[
        ["ticker", "datetime", "interval"]
    ].assign(target_intraday_up_15m=target)
    return features, targets


def test_exact_context_assembler_requires_one_to_one_identity():
    features, targets = _frames(20)
    targets = pd.concat([targets, targets.iloc[[0]]])

    with pytest.raises(ValueError, match="duplicate exact identities"):
        _assemble_exact_context(
            features,
            targets,
            ticker="NVDA",
            timeframe="15m",
            target_name="target_intraday_up_15m",
        )


@pytest.mark.parametrize(
    "target_audit_status",
    [
        "pipeline_target_readiness_ready",
        "pipeline_target_readiness_ready_with_gaps",
    ],
)
def test_stage4_exact_context_review_binds_parents_and_stays_review_only(
    tmp_path,
    target_audit_status,
):
    features, targets = _frames()
    feature_path = tmp_path / "features.parquet"
    target_path = tmp_path / "targets.parquet"
    metadata_path = tmp_path / "batch_metadata.json"
    feature_audit_path = tmp_path / "feature_audit.json"
    target_audit_path = tmp_path / "target_audit.json"
    features.to_parquet(feature_path, index=False)
    targets.to_parquet(target_path, index=False)
    feature_sha = _sha256(feature_path)
    target_sha = _sha256(target_path)
    metadata_path.write_text(
        json.dumps(
            {
                "lineage": {
                    "features_sha256": feature_sha,
                    "targets_sha256": target_sha,
                }
            }
        ),
        encoding="utf-8",
    )
    feature_audit_path.write_text(
        json.dumps(
            {
                "mode": "pipeline_feature_timeframe_audit",
                "status": "pipeline_feature_timeframe_audit_ready",
                "inputs": {"features_sha256": feature_sha},
            }
        ),
        encoding="utf-8",
    )
    target_audit_path.write_text(
        json.dumps(
            {
                "mode": "pipeline_target_readiness_audit",
                "status": target_audit_status,
                "lineage_bindings": {
                    "target_sha256": target_sha,
                    "feature_artifact": {
                        "sha256": feature_sha,
                    },
                },
                "target_reports": [
                    {
                        "target_name": "target_intraday_up_15m",
                        "status": "target_ready",
                        "applies_to_timeframe": True,
                        "per_ticker": {
                            "NVDA": {
                                "non_null_count": len(targets)
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = PipelineStage4ExactContextReview(
        tmp_path / "reports"
    ).run(
        features_path=feature_path,
        targets_path=target_path,
        batch_metadata_path=metadata_path,
        feature_audit_path=feature_audit_path,
        target_audit_path=target_audit_path,
        ticker="NVDA",
        timeframe="15m",
        target_name="target_intraday_up_15m",
        min_train_rows=120,
        validation_rows=40,
        step_rows=40,
        max_folds=2,
        max_features=2,
        save=False,
    )

    assert payload["parent_lineage"][
        "all_parent_hashes_verified"
    ] is True
    assert payload["scope"]["context_row_count"] == 240
    assert payload["summary"]["fold_count"] == 2
    assert payload["summary"]["can_promote_model"] is False
    assert isinstance(payload["summary"]["failed_contract_checks"], list)
    assert payload["safety"]["model_persisted"] is False
    assert payload["summary"]["can_trade"] is False
