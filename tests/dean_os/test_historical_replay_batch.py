from __future__ import annotations

import asyncio

import pandas as pd

from dean_os.historical_replay_batch import HistoricalReplayBatchRunner


def test_historical_replay_batch_summarizes_clean_runs(tmp_path):
    price_path = tmp_path / "prices.csv"
    dates = pd.date_range("2026-01-01", periods=18, freq="D", tz="UTC")
    rows = []
    for index, dt in enumerate(dates):
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 100.0 + index * 2.0, "interval": "1d"})
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": 100.0 + index * 0.5, "interval": "1d"})
    pd.DataFrame(rows).to_csv(price_path, index=False)

    payload = asyncio.run(
        HistoricalReplayBatchRunner(output_dir=tmp_path / "reports").run(
            price_data_path=price_path,
            tickers=["AAPL", "SPY"],
            as_of_dates=["2026-01-08T00:00:00+00:00", "2026-01-10T00:00:00+00:00"],
            lookback_days=7,
            horizon_days=[3, 5],
        )
    )

    assert payload["summary"]["total_runs"] == 4
    assert payload["summary"]["evaluated_runs"] == 4
    assert payload["summary"]["quality_blocked_runs"] == 0
    assert payload["learning_gate"]["status"] == "insufficient_sample"
    assert "AAPL" in payload["summary"]["by_ticker"]


def test_historical_replay_batch_blocks_learning_on_quality_warnings(tmp_path):
    price_path = tmp_path / "prices.csv"
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    spy = [100.0, 90.0, 70.0, 50.0, 30.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    rows = []
    for index, dt in enumerate(dates):
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": spy[index], "interval": "1d"})
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 50.0 + index, "interval": "1d"})
    pd.DataFrame(rows).to_csv(price_path, index=False)

    payload = asyncio.run(
        HistoricalReplayBatchRunner(output_dir=tmp_path / "reports").run(
            price_data_path=price_path,
            tickers=["AAPL", "SPY"],
            as_of_dates=["2026-01-06T00:00:00+00:00"],
            lookback_days=7,
            horizon_days=3,
        )
    )

    assert payload["summary"]["quality_blocked_runs"] == 1
    assert payload["learning_gate"]["status"] == "blocked"
    assert payload["learning_gate"]["can_write_learning_memory"] is False
    assert payload["summary"]["quality_warnings"]
