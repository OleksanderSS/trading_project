from __future__ import annotations

from pathlib import Path

import pandas as pd

from dean_os.pipeline_control.pipeline_control_saved_data_coverage import PipelineControlSavedDataCoverage


def test_saved_data_coverage_builds_eligible_asset_contexts(tmp_path):
    assets_path = tmp_path / "assets.yaml"
    assets_path.write_text(
        """
assets:
  active_preset: test
  presets:
    test:
      tickers: [AAA, BBB]
  sectors:
    semiconductors:
      assets: [AAA]
""".strip(),
        encoding="utf-8",
    )
    price_path = tmp_path / "prices_15m_test.csv"
    timestamps = pd.date_range("2025-01-01", periods=220, freq="15min", tz="UTC")
    rows = []
    for ticker, offset in (("AAA", 0.0), ("BBB", 20.0)):
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "datetime": timestamp,
                    "ticker": ticker,
                    "interval": "15m",
                    "close": 100.0 + offset + index * 0.05,
                }
            )
    pd.DataFrame(rows).to_csv(price_path, index=False)

    payload = PipelineControlSavedDataCoverage(tmp_path / "reports").build(
        assets_yaml=assets_path,
        price_paths=[price_path],
        macro_paths=[],
        min_rows=180,
        save=False,
    )

    assert payload["summary"]["eligible_context_count"] == 2
    assert payload["summary"]["configured_assets_with_price_data"] == 2
    assert {item["ticker"] for item in payload["eligible_contexts"]} == {"AAA", "BBB"}
    assert all(item["timeframe"] == "15m" for item in payload["eligible_contexts"])
    assert payload["summary"]["can_trade"] is False


def test_saved_data_coverage_reports_empty_macro_snapshot(tmp_path):
    assets_path = tmp_path / "assets.yaml"
    assets_path.write_text(
        "assets:\n  active_preset: test\n  presets:\n    test:\n      tickers: []\n",
        encoding="utf-8",
    )
    macro_path = tmp_path / "macro_data_current.csv"
    pd.DataFrame(columns=["datetime", "series", "value"]).to_csv(macro_path, index=False)

    payload = PipelineControlSavedDataCoverage(tmp_path / "reports").build(
        assets_yaml=assets_path,
        price_paths=[],
        macro_paths=[macro_path],
        save=False,
    )

    assert payload["macro_sources"][0]["status"] == "blocked_empty"
    assert payload["summary"]["recommended_macro_source"] is None


def test_saved_data_coverage_handles_mixed_cadence_with_interval_column(tmp_path):
    assets_path = tmp_path / "assets.yaml"
    assets_path.write_text(
        "assets:\n  active_preset: test\n  presets:\n    test:\n      tickers: [AAA]\n",
        encoding="utf-8",
    )
    # Filename does not contain _15m_ or _1d_, should rely entirely on `interval` column
    price_path = tmp_path / "latest.csv"
    timestamps_15m = pd.date_range("2025-01-01", periods=220, freq="15min", tz="UTC")
    timestamps_1d = pd.date_range("2025-01-01", periods=220, freq="1D", tz="UTC")
    rows = []
    for index, timestamp in enumerate(timestamps_15m):
        rows.append({
            "datetime": timestamp,
            "ticker": "AAA",
            "interval": "15m",
            "close": 100.0 + index * 0.05,
        })
    for index, timestamp in enumerate(timestamps_1d):
        rows.append({
            "datetime": timestamp,
            "ticker": "AAA",
            "interval": "1d",
            "close": 100.0 + index * 0.1,
        })
    pd.DataFrame(rows).to_csv(price_path, index=False)

    payload = PipelineControlSavedDataCoverage(tmp_path / "reports").build(
        assets_yaml=assets_path,
        price_paths=[price_path],
        macro_paths=[],
        min_rows=180,
        save=False,
    )

    # We expect 2 eligible contexts for AAA (one for 15m, one for 1d)
    assert payload["summary"]["eligible_context_count"] == 2
    timeframes = sorted([item["timeframe"] for item in payload["eligible_contexts"]])
    assert timeframes == ["15m", "1d"]
