from __future__ import annotations

import pandas as pd
import pytest

from dean_os.pipeline_control_historical_price_recovery import (
    PipelineControlHistoricalPriceRecovery,
)


def _intraday_rows(start: str, *, days: int) -> pd.DataFrame:
    rows = []
    for ticker, offset in (("AAA", 0.0), ("BBB", 50.0)):
        for day_index, day in enumerate(
            pd.date_range(start, periods=days, freq="D", tz="UTC")
        ):
            for bar in range(4):
                price = 100.0 + offset + day_index + bar * 0.1
                rows.append(
                    {
                        "datetime": day + pd.Timedelta(hours=14, minutes=30 + 15 * bar),
                        "ticker": ticker,
                        "interval": "15m",
                        "open": price,
                        "high": price + 0.2,
                        "low": price - 0.2,
                        "close": price + 0.1,
                        "volume": 1000.0 + offset + bar,
                    }
                )
    return pd.DataFrame(rows)


def _daily_rows(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ticker, day), group in intraday.assign(
        day=pd.to_datetime(intraday["datetime"], utc=True).dt.normalize()
    ).groupby(["ticker", "day"]):
        rows.append(
            {
                "datetime": day,
                "ticker": ticker,
                "interval": "1d",
                "open": group.iloc[0]["open"],
                "high": group["high"].max(),
                "low": group["low"].min(),
                "close": group.iloc[-1]["close"],
                "volume": group["volume"].sum(),
            }
        )
    return pd.DataFrame(rows)


def test_historical_price_recovery_keeps_development_and_evaluation_separate(
    tmp_path,
):
    historical = _intraday_rows("2025-01-01", days=5)
    current = _intraday_rows("2025-02-01", days=3)
    daily = _daily_rows(historical)
    historical_path = tmp_path / "historical_15m.parquet"
    current_path = tmp_path / "current_15m.parquet"
    daily_path = tmp_path / "historical_1d.parquet"
    historical.to_parquet(historical_path, index=False)
    current.to_parquet(current_path, index=False)
    daily.to_parquet(daily_path, index=False)

    payload = PipelineControlHistoricalPriceRecovery(tmp_path / "reports").build(
        historical_15m_path=historical_path,
        current_15m_path=current_path,
        historical_1d_path=daily_path,
        required_development_rows=5,
        minimum_past_evaluation_rows=3,
        min_daily_source_bars=4,
    )

    assert payload["summary"]["development_timeframes_ready"] == [
        "15m",
        "60m",
        "1d",
    ]
    assert payload["summary"]["past_evaluation_timeframes_ready"] == [
        "15m",
        "60m",
        "1d",
    ]
    assert payload["summary"]["ready_for_bounded_offline_intraday_evaluation"] is True
    assert payload["summary"]["can_train_automatically"] is False
    assert payload["context_and_target_contract"]["15m"][
        "one_hour_target_shift_bars"
    ] == 4
    assert payload["context_and_target_contract"]["60m"][
        "one_hour_target_shift_bars"
    ] == 1
    assert payload["context_and_target_contract"][
        "targets_may_cross_partition_boundary"
    ] is False
    assert payload["source_quality"]["daily_overlap_consistency"]["consistent"] is True
    assert set(payload["artifacts"]) == {
        "development_15m",
        "development_60m",
        "development_1d",
        "past_evaluation_15m",
        "past_evaluation_60m",
        "past_evaluation_1d_context_tail",
    }


def test_historical_price_recovery_rejects_pickle_disguised_as_parquet(tmp_path):
    fake_path = tmp_path / "fake.parquet"
    fake_path.write_bytes(b"\x80\x04not-real-parquet")

    with pytest.raises(ValueError, match="not a real Parquet"):
        PipelineControlHistoricalPriceRecovery(tmp_path / "reports").build(
            historical_15m_path=fake_path,
            current_15m_path=fake_path,
            historical_1d_path=fake_path,
            save=False,
        )
