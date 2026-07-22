from __future__ import annotations

import pandas as pd

from dean_os.relative_return_direction_policy import (
    calibrate_relative_return_direction_contract,
    classify_relative_total_return,
)


def _prices(path, *, include_post_cutoff: bool) -> None:
    sessions = pd.bdate_range("2024-01-02", "2026-02-27", tz="UTC")
    rows = []
    for index, session in enumerate(sessions):
        benchmark = 100.0 * (1.0005 ** index)
        cycle = ((index % 31) - 15) / 1000.0
        for offset, ticker in enumerate(("A", "B", "C")):
            close = benchmark * (1.0 + cycle * (offset + 1) / 3.0)
            if include_post_cutoff and session >= pd.Timestamp("2026-01-01", tz="UTC"):
                close *= 10.0
            rows.append({"datetime": session, "ticker": ticker, "close": close})
        rows.append({"datetime": session, "ticker": "BM", "close": benchmark})
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_calibration_uses_only_windows_completed_before_cutoff(tmp_path):
    clean = tmp_path / "clean.parquet"
    contaminated = tmp_path / "contaminated.parquet"
    _prices(clean, include_post_cutoff=False)
    _prices(contaminated, include_post_cutoff=True)
    kwargs = {
        "members": ["A", "B", "C"],
        "benchmark": "BM",
        "calibration_cutoff_at": "2026-01-01T00:00:00+00:00",
        "horizon_days": 20,
        "expected_direction": "negative",
        "minimum_sample_count": 60,
    }
    first = calibrate_relative_return_direction_contract(
        price_paths=[clean], **kwargs
    )
    second = calibrate_relative_return_direction_contract(
        price_paths=[contaminated], **kwargs
    )

    assert first["status"] == "calibrated_pre_outcome_direction_contract"
    assert first["neutral_band_absolute_return"] == second["neutral_band_absolute_return"]
    assert first["calibration"]["last_checkpoint_session"] < "2026-01-01"


def test_same_day_daily_close_is_not_available_before_market_close(tmp_path):
    prices = tmp_path / "prices.parquet"
    _prices(prices, include_post_cutoff=False)
    contract = calibrate_relative_return_direction_contract(
        price_paths=[prices],
        members=["A", "B", "C"],
        benchmark="BM",
        calibration_cutoff_at="2025-12-31T15:00:00+00:00",
        horizon_days=20,
        expected_direction="negative",
    )

    assert contract["calibration"]["last_checkpoint_session"] < "2025-12-31"


def test_direction_classification_is_symmetric_and_rewards_correct_decline(tmp_path):
    prices = tmp_path / "prices.parquet"
    _prices(prices, include_post_cutoff=False)
    negative = calibrate_relative_return_direction_contract(
        price_paths=[prices],
        members=["A", "B", "C"],
        benchmark="BM",
        calibration_cutoff_at="2026-01-01T00:00:00+00:00",
        horizon_days=20,
        expected_direction="negative",
    )
    band = negative["neutral_band_absolute_return"]

    assert classify_relative_total_return(-2 * band, negative)["classification"] == "support"
    assert classify_relative_total_return(0.0, negative)["classification"] == "neutral"
    assert classify_relative_total_return(2 * band, negative)["classification"] == "contradict"

    positive = dict(negative, expected_direction="positive")
    assert classify_relative_total_return(2 * band, positive)["classification"] == "support"
    assert classify_relative_total_return(-2 * band, positive)["classification"] == "contradict"
