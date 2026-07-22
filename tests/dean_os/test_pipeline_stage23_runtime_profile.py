from __future__ import annotations

import asyncio

import pandas as pd

import dean_os.pipeline_stage23_runtime_profile as runtime_profile
from dean_os.pipeline_stage23_runtime_profile import (
    PIPELINE_STAGE23_RUNTIME_PROFILE_CONTRACT,
    PipelineStage23RuntimeProfile,
)


def _market_frame(
    *,
    interval: str = "15m",
    frequency: str = "15min",
    rows: int = 10,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-06-01T13:30:00Z",
        periods=rows,
        freq=frequency,
    )
    return pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "datetime": timestamp,
                "interval": interval,
                "open": 100.0 + index,
                "high": 100.5 + index,
                "low": 99.5 + index,
                "close": 100.2 + index,
                "volume": 1000.0 + index,
            }
            for index, timestamp in enumerate(timestamps)
        ]
    )


def test_stage23_runtime_profile_stage2_only_is_review_only(tmp_path, monkeypatch):
    source = tmp_path / "legacy_stage1.parquet"
    pd.to_pickle({"market_data": _market_frame()}, source)

    def fake_stage2(frame, *, timeframe):
        return frame.copy(), {"status": "accepted"}

    monkeypatch.setattr(runtime_profile, "_run_bounded_stage2", fake_stage2)

    payload = asyncio.run(
        PipelineStage23RuntimeProfile().build(
            source_path=source,
            tickers=["NVDA"],
            timeframes=["15m"],
            max_rows_per_ticker=8,
            include_stage2=True,
            save=False,
        )
    )

    lane = payload["timeframe_lanes"][0]
    assert payload["contract"] == PIPELINE_STAGE23_RUNTIME_PROFILE_CONTRACT
    assert payload["summary"]["status"] == "pipeline_stage23_runtime_profile_ready"
    assert payload["summary"]["stage3_included"] is False
    assert payload["summary"]["stage2_included"] is True
    assert payload["summary"]["can_create_stage23_artifacts"] is False
    assert payload["safety"]["stage23_batch_write_performed"] is False
    assert payload["safety"]["stage3_cache_write_performed"] is False
    assert payload["safety"]["can_trade"] is False
    assert lane["lane_status"] == "stage2_profile_ready"
    assert lane["row_counts"]["selected"] == 8
    assert "--shard-cache-dir data\\colab\\stage3_shard_cache\\dean_review" in lane[
        "suggested_stage23_command"
    ]


def test_stage23_runtime_profile_blocks_bad_timeframe_cadence(tmp_path, monkeypatch):
    source = tmp_path / "legacy_stage1.parquet"
    pd.to_pickle(
        {
            "market_data": _market_frame(
                interval="1d",
                frequency="15min",
                rows=8,
            )
        },
        source,
    )

    def fail_if_called(frame, *, timeframe):  # pragma: no cover - assertion helper
        raise AssertionError("Stage2 must not run after source-check blockers")

    monkeypatch.setattr(runtime_profile, "_run_bounded_stage2", fail_if_called)

    payload = asyncio.run(
        PipelineStage23RuntimeProfile().build(
            source_path=source,
            tickers=["NVDA"],
            timeframes=["1d"],
            max_rows_per_ticker=8,
            include_stage2=True,
            save=False,
        )
    )

    lane = payload["timeframe_lanes"][0]
    assert payload["summary"]["status"] == "pipeline_stage23_runtime_profile_blocked"
    assert lane["lane_status"] == "source_checks_blocked"
    assert "timeframe_cadence" in lane["blocking_reasons"]
    assert lane["suggested_stage23_command"] is None
