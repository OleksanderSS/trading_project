from __future__ import annotations

import asyncio

import pandas as pd

from dean_os.replay_price_normalizer import ReplayPriceNormalizer


def test_replay_price_normalizer_collapses_daily_like_rows(tmp_path):
    raw_path = tmp_path / "raw_prices.csv"
    artifact_path = tmp_path / "normalized_prices.csv"
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "datetime": "2026-01-01T10:00:00Z",
                "open": 10.0,
                "high": 12.0,
                "low": 9.0,
                "close": 11.0,
                "volume": 100,
                "interval": "1d",
            },
            {
                "ticker": "AAPL",
                "datetime": "2026-01-01T15:00:00Z",
                "open": 11.0,
                "high": 13.0,
                "low": 10.0,
                "close": 12.0,
                "volume": 150,
                "interval": "1d",
            },
            {
                "ticker": "AAPL",
                "datetime": "2026-01-02T16:00:00Z",
                "open": 12.0,
                "high": 14.0,
                "low": 11.5,
                "close": 13.0,
                "volume": 200,
                "interval": "1d",
            },
            {
                "ticker": "SPY",
                "datetime": "2026-01-01T16:00:00Z",
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 500,
                "interval": "1d",
            },
            {
                "ticker": "SPY",
                "datetime": "2026-01-02T16:00:00Z",
                "open": 101.0,
                "high": 103.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 550,
                "interval": "1d",
            },
        ]
    ).to_csv(raw_path, index=False)

    payload = asyncio.run(
        ReplayPriceNormalizer(output_dir=tmp_path / "reports", artifact_dir=tmp_path / "artifacts").run(
            price_data_path=raw_path,
            tickers=["AAPL", "SPY"],
            output_path=artifact_path,
        )
    )

    normalized = pd.read_csv(payload["artifact"]["path"])
    aapl_day = normalized[(normalized["ticker"] == "AAPL") & (normalized["date"] == "2026-01-01")].iloc[0]
    assert float(aapl_day["open"]) == 10.0
    assert float(aapl_day["high"]) == 13.0
    assert float(aapl_day["low"]) == 9.0
    assert float(aapl_day["close"]) == 12.0
    assert float(aapl_day["volume"]) == 250.0
    assert int(aapl_day["source_row_count"]) == 2
    assert payload["quality"]["raw"]["max_rows_per_ticker_day"] == 2
    assert payload["quality"]["normalized"]["max_rows_per_ticker_day"] == 1
    assert payload["learning_gate"]["status"] == "clear"


def test_replay_price_normalizer_removes_leaky_columns(tmp_path):
    raw_path = tmp_path / "raw_prices.csv"
    pd.DataFrame(
        [
            {
                "ticker": "AMD",
                "datetime": "2026-01-01T16:00:00Z",
                "close": 100.0,
                "TARGET_60D": 0.2,
                "future_close": 120.0,
            },
            {
                "ticker": "AMD",
                "datetime": "2026-01-02T16:00:00Z",
                "close": 101.0,
                "TARGET_60D": 0.18,
                "future_close": 119.0,
            },
        ]
    ).to_csv(raw_path, index=False)

    payload = asyncio.run(
        ReplayPriceNormalizer(output_dir=tmp_path / "reports", artifact_dir=tmp_path / "artifacts").run(
            price_data_path=raw_path,
            tickers=["AMD"],
            output_path=tmp_path / "normalized.csv",
        )
    )

    removed = payload["data_guard"]["prices"]["removed_columns"]
    assert "TARGET_60D" in removed
    assert "future_close" in removed


def test_replay_price_normalizer_blocks_learning_when_quality_warnings_remain(tmp_path):
    raw_path = tmp_path / "raw_prices.csv"
    pd.DataFrame(
        [
            {"ticker": "SPY", "datetime": "2026-01-01T16:00:00Z", "close": 100.0, "interval": "1d"},
            {"ticker": "SPY", "datetime": "2026-01-02T16:00:00Z", "close": 20.0, "interval": "1d"},
        ]
    ).to_csv(raw_path, index=False)

    payload = asyncio.run(
        ReplayPriceNormalizer(output_dir=tmp_path / "reports", artifact_dir=tmp_path / "artifacts").run(
            price_data_path=raw_path,
            tickers=["SPY"],
            output_path=tmp_path / "normalized.csv",
        )
    )

    assert payload["learning_gate"]["status"] == "blocked"
    assert payload["learning_gate"]["can_write_learning_memory"] is False
    assert any("Benchmark SPY lookback return is extreme" in warning for warning in payload["quality"]["warnings"])


def test_replay_price_normalizer_blocks_learning_when_compare_window_warns(tmp_path):
    raw_path = tmp_path / "raw_prices.csv"
    dates = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    spy_closes = [100.0, 80.0, 60.0, 40.0, 20.0, 30.0, 80.0, 120.0]
    aapl_closes = [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0]
    rows = []
    for dt, spy_close, aapl_close in zip(dates, spy_closes, aapl_closes, strict=True):
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": spy_close, "interval": "1d"})
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": aapl_close, "interval": "1d"})
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    payload = asyncio.run(
        ReplayPriceNormalizer(output_dir=tmp_path / "reports", artifact_dir=tmp_path / "artifacts").run(
            price_data_path=raw_path,
            tickers=["AAPL", "SPY"],
            output_path=tmp_path / "normalized.csv",
            compare_replay=True,
            as_of="2026-01-05T00:00:00+00:00",
            lookback_days=10,
            horizon_days=3,
        )
    )

    assert payload["quality"]["normalized"]["warnings"] == []
    assert payload["learning_gate"]["status"] == "blocked"
    assert payload["quality"]["comparison_window_warnings"]
    assert any("Benchmark SPY lookback return is extreme" in warning for warning in payload["learning_gate"]["warnings"])
