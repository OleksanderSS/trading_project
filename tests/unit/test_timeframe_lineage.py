from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.stages.feature_engineering.orchestrator import (
    FeatureEngineeringStage,
)
from src.pipeline.timeframe_lineage import (
    partition_market_frame_by_timeframe,
    timeframe_lineage_report,
)


def _intraday_frame(*, declared: str | None = None) -> pd.DataFrame:
    rows = []
    for ticker in ("AMD", "NVDA"):
        for timestamp in pd.date_range(
            "2026-06-01T13:30:00Z",
            periods=12,
            freq="15min",
        ):
            row = {
                "ticker": ticker,
                "datetime": timestamp,
                "close": 100.0,
            }
            if declared is not None:
                row["interval"] = declared
            rows.append(row)
    return pd.DataFrame(rows)


def test_missing_timeframe_is_inferred_from_observed_cadence():
    grouped = partition_market_frame_by_timeframe(
        _intraday_frame()
    )

    assert list(grouped) == ["15m"]
    assert set(grouped["15m"]["interval"]) == {"15m"}
    assert grouped["15m"].attrs["timeframe_source"] == (
        "inferred_from_observed_cadence"
    )


def test_declared_daily_timeframe_rejects_intraday_cadence():
    with pytest.raises(ValueError, match="observed 15m cadence"):
        partition_market_frame_by_timeframe(
            _intraday_frame(declared="1d")
        )


def test_feature_stage_does_not_default_flat_intraday_frame_to_daily():
    stage = object.__new__(FeatureEngineeringStage)
    cleaned, grouped = stage._validate_and_prepare_market_data(
        cleaned_data={"prices": _intraday_frame()}
    )

    assert cleaned["prices"] is not None
    assert list(grouped) == ["15m"]


def test_feature_stage_rejects_mapping_key_that_conflicts_with_cadence():
    stage = object.__new__(FeatureEngineeringStage)

    with pytest.raises(ValueError, match="conflicts with observed 15m"):
        stage._validate_and_prepare_market_data(
            cleaned_data={"prices": {"1d": _intraday_frame()}}
        )


def test_prediction_timeframe_report_fails_closed_on_mismatch():
    report = timeframe_lineage_report(
        _intraday_frame(declared="1d")
    )

    assert report["status"] == "timeframe_cadence_mismatch"
    assert report["declared_timeframe"] == "1d"
    assert report["observed_timeframe"] == "15m"
    assert report["resolved_timeframe"] is None
    assert report["safe_for_prediction_lineage"] is False
