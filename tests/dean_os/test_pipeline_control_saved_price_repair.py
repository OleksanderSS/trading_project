from __future__ import annotations

import json
import hashlib

import pandas as pd

from dean_os.pipeline_control.pipeline_control_saved_price_repair import PipelineControlSavedPriceRepair
from dean_os.pipeline_control.pipeline_control_saved_data_coverage import (
    CONTRACT as COVERAGE_CONTRACT,
)


def _coverage(contexts):
    return {
        "contract": COVERAGE_CONTRACT,
        "mode": "pipeline_control_saved_data_coverage",
        "summary": {"coverage_status": "saved_data_coverage_ready"},
        "eligible_contexts": contexts,
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_saved_price_repair_builds_real_resampled_candidates(tmp_path):
    source_path = tmp_path / "prices_15m_source.csv"
    coverage_path = tmp_path / "coverage.json"
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
    pd.DataFrame(rows).to_csv(source_path, index=False)
    coverage_path.write_text(
        json.dumps(
            _coverage(
                [
                    {
                        "source_path": str(source_path),
                        "source_sha256": _sha(source_path),
                        "ticker": ticker,
                        "timeframe": "15m",
                        "effective_start": "2025-01-06T00:00:00+00:00",
                    }
                    for ticker in ("AAA", "BBB")
                ]
            )
        ),
        encoding="utf-8",
    )

    payload = PipelineControlSavedPriceRepair(tmp_path / "reports").build(
        coverage_json=coverage_path,
        required_model_rows=10,
        min_daily_source_bars=24,
    )

    assert payload["summary"]["cross_ticker_identity_groups"] == 0
    assert payload["summary"]["clean_15m_row_count"] == 144
    assert payload["summary"]["resampled_60m_row_count"] == 36
    assert payload["summary"]["resampled_1d_row_count"] == 6
    assert payload["coverage_by_timeframe"]["60m"]["all_tickers_meet_required_rows"] is True
    assert payload["coverage_by_timeframe"]["1d"]["all_tickers_meet_required_rows"] is False
    assert all(
        artifact["synthetic"] is False
        for artifact in payload["artifacts"].values()
    )
    assert payload["summary"]["can_trade"] is False


def test_saved_price_repair_rejects_nat_effective_start(tmp_path):
    source_path = tmp_path / "prices_15m_source.csv"
    coverage_path = tmp_path / "coverage.json"
    rows = []
    # Create valid 15m data for AAA
    for day in pd.date_range("2025-01-06", periods=3, freq="D", tz="UTC"):
        for bar in range(24):
            timestamp = day + pd.Timedelta(hours=14, minutes=30 + 15 * bar)
            rows.append({
                "datetime": timestamp,
                "ticker": "AAA",
                "interval": "15m",
                "open": 100.0,
                "high": 100.2,
                "low": 99.8,
                "close": 100.1,
                "volume": 1000.0,
            })
    pd.DataFrame(rows).to_csv(source_path, index=False)
    
    # Coverage manifest with effective_start as null
    coverage_path.write_text(
        json.dumps(_coverage([{
                "source_path": str(source_path),
                "source_sha256": _sha(source_path),
                "ticker": "AAA",
                "timeframe": "15m",
                "effective_start": None,
            }])),
        encoding="utf-8",
    )

    import pytest
    with pytest.raises(ValueError, match="effective_start is required"):
        PipelineControlSavedPriceRepair(tmp_path / "reports").build(
            coverage_json=coverage_path,
            required_model_rows=10,
            min_daily_source_bars=24,
        )


def test_saved_price_repair_detects_cross_ticker_contamination(tmp_path):
    source_path = tmp_path / "prices_15m_source.csv"
    coverage_path = tmp_path / "coverage.json"
    rows = []
    # Both AAA and BBB will have the exact same OHLCV rows for the same datetime
    for ticker in ("AAA", "BBB"):
        for bar in range(24):
            timestamp = pd.Timestamp("2025-01-06T14:30:00Z") + pd.Timedelta(minutes=15 * bar)
            rows.append({
                "datetime": timestamp,
                "ticker": ticker,
                "interval": "15m",
                "open": 100.0,
                "high": 100.2,
                "low": 99.8,
                "close": 100.1,
                "volume": 1000.0,
            })
    pd.DataFrame(rows).to_csv(source_path, index=False)
    
    coverage_path.write_text(
        json.dumps(_coverage(
            [
                {
                    "source_path": str(source_path),
                    "source_sha256": _sha(source_path),
                    "ticker": ticker,
                    "timeframe": "15m",
                    "effective_start": "2025-01-06T00:00:00+00:00",
                }
                for ticker in ("AAA", "BBB")
            ])),
        encoding="utf-8",
    )

    import pytest
    with pytest.raises(ValueError, match="cross-ticker identical OHLCV groups"):
        PipelineControlSavedPriceRepair(tmp_path / "reports").build(
            coverage_json=coverage_path,
            required_model_rows=10,
            min_daily_source_bars=24,
        )
