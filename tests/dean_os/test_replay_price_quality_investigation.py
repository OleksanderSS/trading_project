from __future__ import annotations

import json

import pandas as pd

from dean_os.replay_price_quality_investigation import ReplayPriceQualityInvestigationPlan


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_replay_price_quality_investigation_detects_extreme_benchmark_window(tmp_path):
    price_path = tmp_path / "prices.csv"
    report_path = tmp_path / "replay_batch.json"
    dates = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    rows = []
    spy_prices = [100.0, 98.0, 95.0, 22.0, 23.0, 24.0, 25.0, 26.0]
    for index, dt in enumerate(dates):
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": spy_prices[index], "interval": "1d_normalized"})
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 50.0 + index, "interval": "1d_normalized"})
    pd.DataFrame(rows).to_csv(price_path, index=False)
    _write_json(
        report_path,
        {
            "mode": "historical_replay_batch",
            "inputs": {"price_data_path": str(price_path), "lookback_days": 7},
            "summary": {
                "quality_warnings": {
                    "Benchmark SPY lookback return is extreme (-0.740); review splits, interval mixing, or price normalization.": 1
                }
            },
            "runs": [{"as_of": "2026-01-08T00:00:00+00:00", "quality_warnings": []}],
        },
    )

    payload = ReplayPriceQualityInvestigationPlan(tmp_path / "reports").build(
        report_paths=[report_path],
        price_data_paths=[price_path],
        save=False,
    )

    assert payload["summary"]["investigation_status"] == "blocked_price_quality"
    assert payload["summary"]["extreme_benchmark_warning_count"] == 1
    assert payload["window_diagnostics"][0]["lookback_return"] < -0.7
    assert any(item["hypothesis"] == "benchmark_window_price_anomaly" for item in payload["hypotheses"])
    assert payload["safety"]["data_mutation_performed"] is False


def test_replay_price_quality_investigation_detects_interval_mixing(tmp_path):
    price_path = tmp_path / "prices.csv"
    report_path = tmp_path / "normalizer.json"
    rows = [
        {"ticker": "SPY", "datetime": "2026-01-01T09:00:00+00:00", "close": 100.0, "interval": "1d"},
        {"ticker": "SPY", "datetime": "2026-01-01T15:00:00+00:00", "close": 101.0, "interval": "1d"},
        {"ticker": "SPY", "datetime": "2026-01-02T15:00:00+00:00", "close": 102.0, "interval": "1d"},
        {"ticker": "AAPL", "datetime": "2026-01-01T09:00:00+00:00", "close": 50.0, "interval": "1d"},
    ]
    pd.DataFrame(rows).to_csv(price_path, index=False)
    _write_json(
        report_path,
        {
            "mode": "replay_price_normalization",
            "inputs": {"price_data_path": str(price_path)},
            "quality": {
                "raw": {
                    "warnings": [
                        "Rows are labelled 1d but multiple rows per ticker/day exist; normalize daily bars before relying on replay scores."
                    ]
                }
            },
        },
    )

    payload = ReplayPriceQualityInvestigationPlan(tmp_path / "reports").build(
        report_paths=[report_path],
        price_data_paths=[price_path],
        save=False,
    )

    assert payload["summary"]["warning_record_count"] == 1
    assert payload["artifact_diagnostics"][0]["multi_row_ticker_day_count"] >= 1
    assert any(item["hypothesis"] == "interval_mixing_or_daily_label_issue" for item in payload["hypotheses"])


def test_replay_price_quality_investigation_can_skip_default_reports(tmp_path):
    price_path = tmp_path / "prices.csv"
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    rows = [{"ticker": "SPY", "datetime": dt.isoformat(), "close": 100.0 + index, "interval": "1d_repaired"} for index, dt in enumerate(dates)]
    pd.DataFrame(rows).to_csv(price_path, index=False)

    payload = ReplayPriceQualityInvestigationPlan(tmp_path / "reports").build(
        report_paths=[],
        price_data_paths=[price_path],
        save=False,
    )

    assert payload["summary"]["reports_loaded"] == 0
    assert payload["summary"]["warning_record_count"] == 0
    assert payload["summary"]["investigation_status"] == "clear"
