from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dean_os.pipeline_control.pipeline_control_bounded_evidence_batch import (
    PipelineControlBoundedEvidenceBatch,
)


def test_bounded_evidence_batch_runs_predeclared_non_frozen_contexts(tmp_path):
    source = _write_enriched_source(tmp_path / "prices_15m_fixture.csv")
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(
        json.dumps(
            {
                "summary": {"recommended_macro_source": None},
                "eligible_contexts": [
                    _coverage_context(source, "AAA"),
                    _coverage_context(source, "BBB"),
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = asyncio.run(
        PipelineControlBoundedEvidenceBatch(tmp_path / "reports").run(
            coverage_json=coverage_path,
            tickers=["AAA", "BBB"],
            rows_per_context=220,
            max_features=6,
            input_is_enriched=True,
            run_real_metric_review=False,
            frozen_contexts=["BBB/15m"],
        )
    )

    assert payload["manifest"]["locked_before_fit"] is True
    assert payload["summary"]["context_count"] == 1
    assert payload["summary"]["completed_context_count"] == 1
    assert payload["results"][0]["context_key"] == "AAA/15m"
    assert payload["results"][0]["locked_model_evaluation_ready"] is True
    assert payload["summary"]["can_trade"] is False


def _coverage_context(source: Path, ticker: str) -> dict:
    return {
        "source_path": str(source),
        "ticker": ticker,
        "timeframe": "15m",
        "target_name": "target_intraday_up_15m",
        "effective_start": "2025-01-01T00:00:00+00:00",
        "rows_after_effective_start": 240,
        "evidence_eligible": True,
    }


def _write_enriched_source(path: Path) -> Path:
    timestamps = pd.date_range("2025-01-01", periods=240, freq="15min", tz="UTC")
    rows = []
    for ticker, offset in (("AAA", 0.0), ("BBB", 10.0)):
        next_returns = np.where(np.arange(240) % 4 < 2, 0.004, -0.003)
        close = np.empty(240)
        close[0] = 100.0 + offset
        for index in range(1, 240):
            close[index] = close[index - 1] * (1.0 + next_returns[index - 1])
        target = (next_returns > 0.001).astype(int)
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "datetime": timestamp,
                    "ticker": ticker,
                    "interval": "15m",
                    "open": close[index] * 0.999,
                    "high": close[index] * 1.002,
                    "low": close[index] * 0.998,
                    "close": close[index],
                    "volume": 1_000_000 + index * 100,
                    "momentum_15m": np.sin(index / 5.0),
                    "volatility_15m": np.cos(index / 7.0),
                    "trend_15m": index % 4,
                    "target_intraday_up_15m": int(target[index]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
