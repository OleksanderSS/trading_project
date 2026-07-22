from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dean_os.pipeline_control_bounded_evidence_run import (
    PipelineControlBoundedEvidenceRun,
    _prepare_macro_input,
    _resolve_enriched_datetime,
)


def test_bounded_evidence_run_builds_matching_locked_pair_from_enriched_table(tmp_path):
    source = _write_enriched_source(tmp_path / "enriched.csv", row_count=280)

    payload = asyncio.run(
        PipelineControlBoundedEvidenceRun(tmp_path / "reports").run(
            source_path=source,
            ticker="AMD",
            timeframe="15m",
            target_name="target_intraday_up_15m",
            max_rows=280,
            max_features=8,
            min_rows=180,
            input_is_enriched=True,
            run_real_metric_review=False,
        )
    )

    summary = payload["summary"]
    assert summary["bounded_evidence_status"] == "bounded_locked_metric_pair_ready"
    assert summary["train_sample_count"] > summary["validation_sample_count"]
    assert summary["validation_sample_count"] > 0
    assert summary["test_sample_count"] > 0
    assert summary["locked_model_evaluation_ready"] is True
    assert summary["locked_feature_stability_ready"] is True
    assert summary["real_metric_review_invoked"] is False
    assert summary["can_trade"] is False

    locked_model = json.loads(
        Path(payload["artifacts"]["locked_model_evaluation"]).read_text(encoding="utf-8")
    )
    assert locked_model["metrics"]["train_score"] is not None
    assert locked_model["metrics"]["validation_score"] is not None
    assert locked_model["metrics"]["test_score"] is not None
    assert locked_model["joined_lineage"]["ticker"] == "AMD"
    assert Path(payload["artifacts"]["model_path"]).exists()


def test_bounded_evidence_run_blocks_short_source_before_training(tmp_path):
    source = _write_enriched_source(tmp_path / "short.csv", row_count=40)

    payload = asyncio.run(
        PipelineControlBoundedEvidenceRun(tmp_path / "reports").run(
            source_path=source,
            ticker="AMD",
            timeframe="15m",
            target_name="target_intraday_up_15m",
            input_is_enriched=True,
            min_rows=180,
        )
    )

    assert payload["summary"]["bounded_evidence_status"] == "blocked_source_quality"
    assert payload["summary"]["locked_model_evaluation_ready"] is False
    assert payload["summary"]["can_trade"] is False
    failed = {
        check["code"]
        for check in payload["source_quality_checks"]
        if check["status"] == "fail"
    }
    assert "minimum_rows" in failed


def test_bounded_evidence_run_accepts_stage3_style_suffixed_datetime(tmp_path):
    source = _write_enriched_source(tmp_path / "enriched.csv", row_count=240)
    frame = pd.read_csv(source)
    frame = frame.rename(columns={"datetime": "datetime_15m"})
    frame.to_csv(source, index=False)

    payload = asyncio.run(
        PipelineControlBoundedEvidenceRun(tmp_path / "reports").run(
            source_path=source,
            ticker="AMD",
            timeframe="15m",
            target_name="target_intraday_up_15m",
            max_features=6,
            input_is_enriched=True,
            run_real_metric_review=False,
        )
    )

    assert payload["summary"]["locked_model_evaluation_ready"] is True
    assert payload["summary"]["locked_feature_stability_ready"] is True


def test_enriched_datetime_prefers_exact_service_column_over_suffixed_feature():
    exact = pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC")
    misleading = pd.date_range("2030-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "datetime": exact,
            "date_15m": misleading,
        }
    )

    resolved = _resolve_enriched_datetime(frame)

    assert resolved.tolist() == exact.tolist()


def test_bounded_macro_input_normalizes_series_and_excludes_future_rows(tmp_path):
    macro_path = tmp_path / "macro.csv"
    pd.DataFrame(
        {
            "datetime": ["2025-01-01", "2025-02-01"],
            "series": ["DGS10", "DGS10"],
            "value": [4.1, 9.9],
        }
    ).to_csv(macro_path, index=False)
    captured_at = pd.Timestamp("2025-01-05T00:00:00Z").timestamp()
    os.utime(macro_path, (captured_at, captured_at))
    index = pd.date_range("2025-01-10", periods=20, freq="D", tz="UTC")
    bounded = pd.DataFrame({"close": np.arange(20) + 100.0}, index=index)

    macro, provenance, checks = _prepare_macro_input(
        macro_path,
        bounded_frame=bounded,
    )

    assert macro is not None
    assert macro["series_id"].tolist() == ["DGS10"]
    assert macro["value"].tolist() == [4.1]
    assert provenance["future_rows_excluded"] == 1
    assert provenance["availability_basis"] == "artifact_mtime_conservative"
    assert not [check for check in checks if check["status"] == "fail"]


def _write_enriched_source(path: Path, *, row_count: int) -> Path:
    timestamps = pd.date_range("2025-01-01", periods=row_count, freq="15min", tz="UTC")
    next_returns = np.where(np.arange(row_count) % 4 < 2, 0.004, -0.003)
    close = np.empty(row_count)
    close[0] = 100.0
    for index in range(1, row_count):
        close[index] = close[index - 1] * (1.0 + next_returns[index - 1])
    target = (next_returns > 0.001).astype(int)
    frame = pd.DataFrame(
        {
            "datetime": timestamps,
            "ticker": "AMD",
            "interval": "15m",
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1_000_000 + np.arange(row_count) * 100,
            "momentum_15m": np.sin(np.arange(row_count) / 5.0),
            "volatility_15m": np.cos(np.arange(row_count) / 7.0),
            "trend_15m": np.arange(row_count) % 4,
            "volume_ratio_15m": 1.0 + (np.arange(row_count) % 9) / 10.0,
            "target_intraday_up_15m": target,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
