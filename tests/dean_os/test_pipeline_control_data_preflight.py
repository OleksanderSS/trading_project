from __future__ import annotations

import pandas as pd

from dean_os.pipeline_control.pipeline_control_data_preflight import (
    PipelineControlDataPreflight,
)


def test_data_preflight_runs_coverage_then_non_destructive_repair(tmp_path):
    assets_path = tmp_path / "assets.yaml"
    assets_path.write_text(
        """
assets:
  active_preset: test
  presets:
    test:
      tickers: [AAA, BBB]
  sectors: {}
""".strip(),
        encoding="utf-8",
    )
    price_path = tmp_path / "prices_15m_test.csv"
    rows = []
    for ticker, offset in (("AAA", 0.0), ("BBB", 50.0)):
        for day in pd.date_range("2025-01-06", periods=3, freq="D", tz="UTC"):
            for bar in range(24):
                timestamp = day + pd.Timedelta(hours=14, minutes=30 + 15 * bar)
                price = 100.0 + offset + bar * 0.1
                rows.append(
                    {
                        "datetime": timestamp,
                        "ticker": ticker,
                        "interval": "15m",
                        "open": price,
                        "high": price + 0.2,
                        "low": price - 0.2,
                        "close": price + 0.1,
                        "volume": 1000.0 + bar,
                    }
                )
    pd.DataFrame(rows).to_csv(price_path, index=False)

    payload = PipelineControlDataPreflight(
        tmp_path / "reports" / "pipeline_control_data_preflight_current"
    ).build(
        assets_yaml=assets_path,
        price_paths=[price_path],
        macro_paths=[],
        required_model_rows=10,
    )

    assert [step["step_id"] for step in payload["steps"]] == [
        "saved_data_coverage",
        "saved_price_repair",
    ]
    assert payload["summary"]["eligible_context_count"] == 2
    assert payload["summary"]["can_start_bounded_15m_review"] is True
    assert payload["summary"]["can_train"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["repaired_artifacts"]["prices_15m_clean"]["synthetic"] is False
