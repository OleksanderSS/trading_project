from __future__ import annotations

import pandas as pd

from dean_os.replay_price_artifact_repair import ReplayPriceArtifactRepairPlan


def test_replay_price_artifact_repair_prefers_midnight_daily_anchor(tmp_path):
    raw_path = tmp_path / "mixed_prices.csv"
    artifact_path = tmp_path / "repaired_prices.csv"
    pd.DataFrame(
        [
            {"ticker": "SPY", "datetime": "2026-01-01T00:00:00Z", "close": 100.0, "interval": "1d"},
            {"ticker": "SPY", "datetime": "2026-01-01T15:30:00Z", "close": 20.0, "interval": "1d"},
            {"ticker": "SPY", "datetime": "2026-01-02T00:00:00Z", "close": 102.0, "interval": "1d"},
            {"ticker": "SPY", "datetime": "2026-01-02T15:30:00Z", "close": 21.0, "interval": "1d"},
            {"ticker": "AAPL", "datetime": "2026-01-01T00:00:00Z", "close": 50.0, "interval": "1d"},
            {"ticker": "AAPL", "datetime": "2026-01-02T00:00:00Z", "close": 51.0, "interval": "1d"},
        ]
    ).to_csv(raw_path, index=False)

    payload = ReplayPriceArtifactRepairPlan(output_dir=tmp_path / "reports", artifact_dir=tmp_path).build(
        price_data_path=raw_path,
        tickers=["SPY", "AAPL"],
        output_path=artifact_path,
        write_artifact=True,
    )

    repaired = pd.read_csv(payload["artifact"]["path"])
    spy = repaired[repaired["ticker"] == "SPY"].sort_values("datetime")
    assert spy["close"].tolist() == [100.0, 102.0]
    assert payload["quarantine"]["by_reason"]["same_day_anchor_deviation"] == 2
    assert payload["quality"]["candidate_repaired"]["warnings"] == []
    assert payload["learning_gate"]["status"] == "candidate_ready_for_replay_review"


def test_replay_price_artifact_repair_quarantines_unanchored_outlier_block(tmp_path):
    raw_path = tmp_path / "mixed_prices.csv"
    artifact_path = tmp_path / "repaired_prices.csv"
    pd.DataFrame(
        [
            {"ticker": "SPY", "datetime": "2026-01-01T00:00:00Z", "close": 100.0, "interval": "1d"},
            {"ticker": "SPY", "datetime": "2026-01-02T15:30:00Z", "close": 20.0, "interval": "1d"},
            {"ticker": "SPY", "datetime": "2026-01-03T15:30:00Z", "close": 21.0, "interval": "1d"},
            {"ticker": "SPY", "datetime": "2026-01-04T00:00:00Z", "close": 103.0, "interval": "1d"},
            {"ticker": "AAPL", "datetime": "2026-01-01T16:00:00Z", "close": 50.0, "interval": "1d"},
            {"ticker": "AAPL", "datetime": "2026-01-02T16:00:00Z", "close": 51.0, "interval": "1d"},
        ]
    ).to_csv(raw_path, index=False)

    payload = ReplayPriceArtifactRepairPlan(output_dir=tmp_path / "reports", artifact_dir=tmp_path).build(
        price_data_path=raw_path,
        tickers=["SPY", "AAPL"],
        output_path=artifact_path,
        write_artifact=True,
    )

    repaired = pd.read_csv(payload["artifact"]["path"])
    spy = repaired[repaired["ticker"] == "SPY"].sort_values("datetime")
    assert spy["close"].tolist() == [100.0, 103.0]
    assert payload["quarantine"]["by_reason"]["unanchored_price_level_outlier"] == 2
    assert payload["summary"]["quarantined_ticker_date_count"] == 2
    assert payload["quality"]["candidate_repaired"]["warnings"] == []


def test_replay_price_artifact_repair_dry_run_does_not_write_artifact(tmp_path):
    raw_path = tmp_path / "prices.csv"
    pd.DataFrame(
        [
            {"ticker": "SPY", "datetime": "2026-01-01T00:00:00Z", "close": 100.0, "interval": "1d"},
            {"ticker": "SPY", "datetime": "2026-01-02T00:00:00Z", "close": 101.0, "interval": "1d"},
        ]
    ).to_csv(raw_path, index=False)

    payload = ReplayPriceArtifactRepairPlan(output_dir=tmp_path / "reports", artifact_dir=tmp_path).build(
        price_data_path=raw_path,
        tickers=["SPY"],
        output_path=tmp_path / "would_not_write.csv",
        write_artifact=False,
    )

    assert payload["artifact"]["path"] is None
    assert payload["safety"]["candidate_artifact_written"] is False
    assert payload["learning_gate"]["status"] == "dry_run_review_required"
