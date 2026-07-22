from __future__ import annotations

import asyncio
import json

import pandas as pd

from dean_os.historical_research_replay_batch import HistoricalResearchReplayBatchRunner


def test_historical_research_replay_batch_summarizes_multiple_slices(tmp_path):
    price_path = tmp_path / "prices.csv"
    news_path = tmp_path / "news.csv"
    dates = pd.date_range("2026-01-01", periods=16, freq="D", tz="UTC")
    rows = []
    for index, dt in enumerate(dates):
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 100.0 + index * 2.0, "interval": "1d"})
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": 200.0 + index * 0.2, "interval": "1d"})
    pd.DataFrame(rows).to_csv(price_path, index=False)
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "published_date": "2026-01-03T00:00:00+00:00",
                "title": "AAPL AI demand expands",
                "text": "AI compute cycle and pricing power support AAPL.",
            },
            {
                "ticker": "AAPL",
                "published_date": "2026-01-08T00:00:00+00:00",
                "title": "AAPL future contract win",
                "text": "Long-term contract and recurring revenue improve the thesis.",
            },
        ]
    ).to_csv(news_path, index=False)

    payload = asyncio.run(
        HistoricalResearchReplayBatchRunner(output_dir=tmp_path / "reports").run(
            price_data_path=price_path,
            news_data_paths=[news_path],
            tickers=["AAPL", "SPY"],
            as_of_dates=["2026-01-06T00:00:00+00:00", "2026-01-10T00:00:00+00:00"],
            lookback_days=10,
            horizon_days=[3],
        )
    )

    assert payload["mode"] == "historical_research_replay_batch"
    assert payload["summary"]["total_runs"] == 2
    assert payload["summary"]["evaluated_runs"] == 2
    assert payload["learning_gate"]["can_write_learning_memory"] is False
    assert payload["runs"][0]["evidence_document_count"] == 1
    assert payload["runs"][1]["evidence_document_count"] == 2
    assert (tmp_path / "reports" / "latest.json").exists()


def test_historical_research_replay_batch_normalizes_horizons(tmp_path):
    price_path = tmp_path / "prices.csv"
    dates = pd.date_range("2026-01-01", periods=14, freq="D", tz="UTC")
    rows = []
    for index, dt in enumerate(dates):
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 100.0 + index, "interval": "1d"})
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": 100.0 + index * 0.1, "interval": "1d"})
    pd.DataFrame(rows).to_csv(price_path, index=False)

    payload = asyncio.run(
        HistoricalResearchReplayBatchRunner(output_dir=tmp_path / "reports").run(
            price_data_path=price_path,
            tickers=["AAPL", "SPY"],
            as_of_dates=["2026-01-06T00:00:00+00:00"],
            lookback_days=5,
            horizon_days=[3, 3, 5],
        )
    )

    assert payload["inputs"]["horizon_days"] == [3, 5]
    assert payload["summary"]["total_runs"] == 2


def test_historical_research_replay_batch_passes_focused_overlay(tmp_path):
    price_path = tmp_path / "prices.csv"
    dates = pd.date_range("2026-01-01", periods=14, freq="D", tz="UTC")
    rows = []
    for index, dt in enumerate(dates):
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 100.0 + index * 2.0, "interval": "1d"})
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": 200.0 + index * 0.2, "interval": "1d"})
    pd.DataFrame(rows).to_csv(price_path, index=False)
    overlay_path = tmp_path / "focused_overlay.json"
    overlay_path.write_text(
        json.dumps(
            {
                "run_overlays": [
                    {
                        "as_of": "2026-01-06T00:00:00+00:00",
                        "horizon_days": 3,
                        "price_ticker": "AAPL",
                        "overlay_status": "focused_overlay_ready",
                        "issues": [],
                        "focused_exam": {
                            "research_thesis": "Focused AAPL thesis.",
                            "research_stance": "constructive",
                            "research_expected_direction": "bullish",
                            "research_confidence": 0.82,
                            "research_data_quality": "partial",
                            "ticker_specificity": "single_ticker",
                            "price_expected_direction": "bullish",
                            "research_price_agreement": "aligned",
                            "exam_verdict": "aligned_hit",
                            "learning_gate": {"status": "review_required", "can_write_learning_memory": False},
                        },
                        "focused_note": {"note_id": "note-aapl"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = asyncio.run(
        HistoricalResearchReplayBatchRunner(output_dir=tmp_path / "reports").run(
            price_data_path=price_path,
            tickers=["AAPL", "SPY"],
            as_of_dates=["2026-01-06T00:00:00+00:00"],
            lookback_days=5,
            horizon_days=[3],
            focused_overlay_path=overlay_path,
            apply_focused_overlay=True,
        )
    )

    assert payload["inputs"]["focused_overlay_path"] == str(overlay_path)
    assert payload["inputs"]["apply_focused_overlay"] is True
    assert payload["runs"][0]["focused_overlay_applied"] is True
    assert payload["runs"][0]["focused_overlay_status"] == "focused_overlay_ready"
