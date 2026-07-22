import asyncio
import hashlib
import json

import pandas as pd
import pytest

from dean_os.pipeline_control_walk_forward_validation_run import (
    PipelineControlWalkForwardValidationRun,
    _load_development_frames,
    _load_forward_development_frames,
    _merge_development_frames,
    _normalize_macro_long_form,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_loader_uses_only_required_context_artifacts(tmp_path):
    artifacts = {}
    for timeframe, periods, frequency in (
        ("15m", 20, "15min"),
        ("60m", 8, "60min"),
        ("1d", 5, "1D"),
    ):
        path = tmp_path / f"development_{timeframe}.parquet"
        pd.DataFrame(
            {
                "datetime": pd.date_range(
                    "2025-01-01",
                    periods=periods,
                    freq=frequency,
                    tz="UTC",
                ),
                "ticker": ["NVDA"] * periods,
                "interval": [timeframe] * periods,
                "close": range(periods),
            }
        ).to_parquet(path, index=False)
        artifacts[f"development_{timeframe}"] = {
            "path": str(path),
            "sha256": _sha256(path),
            "synthetic": False,
        }
    artifacts["past_evaluation_15m"] = {
        "path": str(tmp_path / "must_not_be_opened.parquet"),
        "sha256": "not-used",
        "synthetic": False,
    }

    frames, lineage = _load_development_frames(
        {"artifacts": artifacts},
        ticker="NVDA",
        base_timeframe="15m",
    )

    assert set(frames) == {"15m", "60m", "1d"}
    assert all(
        set(frame["partition_id"]) == {"development"}
        for frame in frames.values()
    )
    assert set(lineage) == {
        "development_15m",
        "development_60m",
        "development_1d",
    }


def test_development_loader_rejects_hash_mismatch(tmp_path):
    path = tmp_path / "development_1d.parquet"
    pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2025-01-01",
                periods=5,
                freq="1D",
                tz="UTC",
            ),
            "ticker": ["NVDA"] * 5,
            "interval": ["1d"] * 5,
            "close": range(5),
        }
    ).to_parquet(path, index=False)
    recovery = {
        "artifacts": {
            "development_1d": {
                "path": str(path),
                "sha256": "wrong",
                "synthetic": False,
            }
        }
    }

    with pytest.raises(ValueError, match="hash mismatch"):
        _load_development_frames(
            recovery,
            ticker="NVDA",
            base_timeframe="1d",
        )


def test_walk_forward_run_requires_development_only_acknowledgement(tmp_path):
    runner = PipelineControlWalkForwardValidationRun(tmp_path / "reports")

    with pytest.raises(ValueError, match="development-only acknowledgement"):
        asyncio.run(
            runner.run(
                historical_recovery_json=tmp_path / "missing.json",
                ticker="NVDA",
                timeframe="15m",
                target_name="target_intraday_up_15m",
                acknowledge_development_only=False,
            )
        )


def test_macro_input_is_canonicalized_before_active_stage3():
    frame = pd.DataFrame(
        {
            "datetime": ["2025-01-02", "2025-01-01", "bad"],
            "series": ["DGS10", "DGS10", "DGS10"],
            "value": [4.4, 4.3, 5.0],
        }
    )

    normalized = _normalize_macro_long_form(frame)

    assert list(normalized.columns) == ["datetime", "series_id", "value"]
    assert str(normalized["datetime"].dtype) == "datetime64[ns, UTC]"
    assert normalized["series_id"].tolist() == ["DGS10", "DGS10"]
    assert normalized["value"].tolist() == [4.3, 4.4]


def test_forward_gate_loader_adds_only_post_watermark_development_rows(tmp_path):
    dates = pd.date_range(
        "2026-06-01T14:30:00Z",
        periods=130,
        freq="15min",
    )
    source = tmp_path / "forward.parquet"
    pd.DataFrame(
        {
            "datetime": dates,
            "ticker": "NVDA",
            "interval": "15m",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000,
        }
    ).to_parquet(source, index=False)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "mode": "pipeline_control_forward_data_accrual_gate",
                "summary": {
                    "gate_status": "forward_development_artifact_ready",
                    "can_supply_next_development_run": True,
                },
                "eligible_development_artifact": {
                    "artifact_class": "pipeline_control_forward_development_artifact",
                    "evidence_class": "validated_forward_development_source",
                    "context_key": "NVDA/15m/target_intraday_up_15m",
                    "lane": "development_refresh_only",
                    "may_be_used_as_locked_test_evidence": False,
                    "may_be_called_virgin_holdout": False,
                    "source_path": str(source),
                    "source_sha256": _sha256(source),
                    "start_exclusive": "2026-05-06T17:30:00+00:00",
                    "eligible_new_row_count": 130,
                },
            }
        ),
        encoding="utf-8",
    )

    frames, lineage = _load_forward_development_frames(
        gate,
        ticker="NVDA",
        base_timeframe="15m",
        target_name="target_intraday_up_15m",
    )

    assert set(frames) == {"15m", "60m", "1d"}
    assert len(frames["15m"]) == 130
    assert 30 <= len(frames["60m"]) <= 34
    assert len(frames["1d"]) == 2
    assert all(
        set(frame["partition_id"]) == {"forward_development"}
        for frame in frames.values()
    )
    assert lineage["base_row_count"] == 130
    assert lineage["test_rows_loaded"] == 0
    assert lineage["past_evaluation_rows_loaded"] == 0


def test_forward_gate_loader_rejects_blocked_gate(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "mode": "pipeline_control_forward_data_accrual_gate",
                "summary": {
                    "gate_status": "blocked_forward_development_artifact",
                    "can_supply_next_development_run": False,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not ready"):
        _load_forward_development_frames(
            gate,
            ticker="NVDA",
            base_timeframe="15m",
            target_name="target_intraday_up_15m",
        )


def test_merge_development_frames_preserves_partition_boundary():
    historical = {
        "15m": pd.DataFrame(
            {
                "datetime": pd.date_range(
                    "2026-05-01",
                    periods=2,
                    freq="15min",
                    tz="UTC",
                ),
                "ticker": "NVDA",
                "interval": "15m",
                "partition_id": "development",
                "close": [100.0, 101.0],
            }
        )
    }
    forward = {
        "15m": pd.DataFrame(
            {
                "datetime": pd.date_range(
                    "2026-06-01",
                    periods=2,
                    freq="15min",
                    tz="UTC",
                ),
                "ticker": "NVDA",
                "interval": "15m",
                "partition_id": "forward_development",
                "close": [110.0, 111.0],
            }
        )
    }

    merged = _merge_development_frames(historical, forward)["15m"]

    assert len(merged) == 4
    assert set(merged["partition_id"]) == {
        "development",
        "forward_development",
    }
