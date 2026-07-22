from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pandas as pd

from dean_os.historical_research_replay import HistoricalResearchReplayRunner, _research_direction, _research_stance
from dean_os.schemas import ResearchNote


def test_historical_research_replay_combines_research_and_price_outcome(tmp_path):
    price_path = tmp_path / "prices.csv"
    news_path = tmp_path / "news.csv"
    macro_path = tmp_path / "macro.csv"
    dates = pd.date_range("2026-01-01", periods=14, freq="D", tz="UTC")
    rows = []
    for index, dt in enumerate(dates):
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 100.0 + index * 2.0, "interval": "1d"})
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": 200.0 + index * 0.2, "interval": "1d"})
    pd.DataFrame(rows).to_csv(price_path, index=False)
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "published_at": "2026-01-03T00:00:00+00:00",
                "title": "AAPL AI accelerator demand expands",
                "text": "AI data center accelerator demand, pricing power, and supply chain reshoring support AAPL.",
            },
            {
                "ticker": "AAPL",
                "published_at": "2026-01-06T00:00:00+00:00",
                "title": "AAPL contract backlog improves",
                "text": "Long-term contract wins and recurring revenue improve the capital cycle.",
            },
        ]
    ).to_csv(news_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-01-05T00:00:00+00:00",
                "series": "rates",
                "value": "policy easing expectations improve liquidity",
            }
        ]
    ).to_csv(macro_path, index=False)

    payload = asyncio.run(
        HistoricalResearchReplayRunner(output_dir=tmp_path / "reports").run(
            price_data_path=price_path,
            news_data_paths=[news_path],
            macro_data_paths=[macro_path],
            tickers=["AAPL", "SPY"],
            as_of="2026-01-07T00:00:00+00:00",
            lookback_days=5,
            horizon_days=3,
        )
    )

    assert payload["mode"] == "historical_research_replay"
    assert payload["safety"]["learning_records_created"] == 0
    assert payload["safety"]["operation_proposals_created"] == 0
    assert payload["evidence_pack"]["coverage"]["document_count"] >= 3
    assert payload["agent_lab"]["learning_record_count"] == 0
    assert payload["agent_lab"]["action_proposal_count"] == 0
    assert payload["price_replay"]["evaluation"]["status"] == "evaluated"
    assert payload["research_exam"]["learning_gate"]["can_write_learning_memory"] is False
    assert (tmp_path / "reports" / "latest.json").exists()
    assert (tmp_path / "reports" / "latest.md").exists()


def test_historical_research_replay_filters_future_evidence(tmp_path):
    price_path = tmp_path / "prices.csv"
    news_path = tmp_path / "news.csv"
    dates = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    rows = []
    for index, dt in enumerate(dates):
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 100.0 + index, "interval": "1d"})
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": 100.0 + index * 0.1, "interval": "1d"})
    pd.DataFrame(rows).to_csv(price_path, index=False)
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "published_at": "2026-01-04T00:00:00+00:00",
                "title": "Visible AAPL AI setup",
                "text": "AI compute cycle and pricing power were visible before the cutoff.",
            },
            {
                "ticker": "AAPL",
                "published_at": "2026-01-10T00:00:00+00:00",
                "title": "Future AAPL result",
                "text": "This future result must not be visible to the replay agent.",
            },
        ]
    ).to_csv(news_path, index=False)

    payload = asyncio.run(
        HistoricalResearchReplayRunner(output_dir=tmp_path / "reports").run(
            price_data_path=price_path,
            news_data_paths=[news_path],
            tickers=["AAPL", "SPY"],
            as_of="2026-01-06T00:00:00+00:00",
            lookback_days=5,
            horizon_days=3,
        )
    )

    date_range = payload["evidence_pack"]["coverage"]["date_range"]
    assert datetime.fromisoformat(date_range["end"]) <= datetime.fromisoformat("2026-01-06T00:00:00+00:00")
    assert payload["evidence_pack"]["coverage"]["document_count"] == 1


def test_research_stance_uses_structured_patterns_before_mixed_thesis_text():
    note = ResearchNote(
        agent_name="evidence_synthesis",
        topic="diagnostic",
        thesis="Cited research is mixed; dominant patterns are ai_compute_cycle.",
        patterns=["research_corpus_ingestion", "ai_compute_cycle"],
        tickers=["AMD", "NVDA"],
        confidence=0.9,
        data_quality="strong",
    )

    stance = _research_stance(note)

    assert stance == "constructive"
    assert _research_direction(stance) == "bullish"


def test_historical_research_replay_can_apply_focused_overlay(tmp_path):
    price_path = tmp_path / "prices.csv"
    news_path = tmp_path / "news.csv"
    dates = pd.date_range("2026-01-01", periods=14, freq="D", tz="UTC")
    rows = []
    for index, dt in enumerate(dates):
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 100.0 + index * 2.0, "interval": "1d"})
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": 200.0 + index * 0.2, "interval": "1d"})
    pd.DataFrame(rows).to_csv(price_path, index=False)
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "published_at": "2026-01-03T00:00:00+00:00",
                "title": "AAPL AI accelerator demand expands",
                "text": "AI compute cycle and pricing power support AAPL.",
            }
        ]
    ).to_csv(news_path, index=False)
    overlay_path = tmp_path / "focused_overlay.json"
    overlay_path.write_text(
        json.dumps(
            {
                "run_overlays": [
                    {
                        "as_of": "2026-01-07T00:00:00+00:00",
                        "horizon_days": 3,
                        "price_ticker": "AAPL",
                        "overlay_status": "focused_overlay_ready",
                        "issues": [],
                        "focused_exam": {
                            "selected_note_agent": "ticker_focused_research_note_builder",
                            "selected_note_id": "note-aapl",
                            "research_thesis": "Focused AAPL thesis.",
                            "research_stance": "constructive",
                            "research_expected_direction": "bullish",
                            "research_confidence": 0.82,
                            "research_data_quality": "partial",
                            "ticker_specificity": "single_ticker",
                            "price_expected_direction": "bullish",
                            "research_price_agreement": "aligned",
                            "exam_verdict": "aligned_hit",
                            "learning_gate": {
                                "status": "review_required",
                                "can_write_learning_memory": False,
                                "reason": "review required",
                            },
                        },
                        "focused_note": {"note_id": "note-aapl", "direct_document_count": 3},
                        "comparison": {"specificity_improved": True},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = asyncio.run(
        HistoricalResearchReplayRunner(output_dir=tmp_path / "reports").run(
            price_data_path=price_path,
            news_data_paths=[news_path],
            tickers=["AAPL", "SPY"],
            as_of="2026-01-07T00:00:00+00:00",
            lookback_days=5,
            horizon_days=3,
            focused_overlay_path=overlay_path,
            apply_focused_overlay=True,
        )
    )

    assert payload["research_exam"]["focused_overlay_applied"] is True
    assert payload["research_exam"]["focused_overlay_status"] == "focused_overlay_ready"
    assert payload["research_exam"]["ticker_specificity"] == "single_ticker"
    assert payload["research_exam_original"]["ticker_specificity"] == "basket_or_sector"
    assert payload["focused_research_exam_overlay"]["focused_note"]["note_id"] == "note-aapl"
    assert payload["safety"]["focused_overlay_applied"] is True


def test_historical_research_replay_preserves_original_when_overlay_not_applied(tmp_path):
    price_path = tmp_path / "prices.csv"
    dates = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    rows = []
    for index, dt in enumerate(dates):
        rows.append({"ticker": "AAPL", "datetime": dt.isoformat(), "close": 100.0 + index, "interval": "1d"})
        rows.append({"ticker": "SPY", "datetime": dt.isoformat(), "close": 100.0 + index * 0.1, "interval": "1d"})
    pd.DataFrame(rows).to_csv(price_path, index=False)
    overlay_path = tmp_path / "focused_overlay.json"
    overlay_path.write_text(json.dumps({"run_overlays": []}), encoding="utf-8")

    payload = asyncio.run(
        HistoricalResearchReplayRunner(output_dir=tmp_path / "reports").run(
            price_data_path=price_path,
            tickers=["AAPL", "SPY"],
            as_of="2026-01-06T00:00:00+00:00",
            lookback_days=5,
            horizon_days=3,
            focused_overlay_path=overlay_path,
            apply_focused_overlay=False,
        )
    )

    assert payload["research_exam"]["focused_overlay_available"] is True
    assert payload["research_exam"]["focused_overlay_applied"] is False
    assert payload["research_exam"]["focused_overlay_status"] == "blocked_missing_focused_overlay"
    assert payload["research_exam_original"]["exam_verdict"] == payload["research_exam"]["exam_verdict"]
