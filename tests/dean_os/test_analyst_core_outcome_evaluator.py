"""Tests for OutcomeEvaluator."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dean_os.analyst_core.outcome_evaluator import (
    OutcomeEvaluator,
    EvaluationResult,
    TickerOutcome,
    HorizonOutcome,
    render_evaluation_markdown,
)
from dean_os.analyst_core.sector_analyst import SectorAnalyst, SectorReport
from dean_os.analyst_core.artifact_evidence_loader import load_evidence_from_artifacts


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_price_data(tmp: Path) -> Path:
    """Create a minimal price parquet for testing."""
    dates = pd.date_range("2026-05-01", "2026-08-01", freq="B")
    tickers = ["NVDA", "AMD", "TSM"]

    rows = []
    for ticker in tickers:
        base_price = {"NVDA": 100.0, "AMD": 50.0, "TSM": 150.0}[ticker]
        for i, date in enumerate(dates):
            price = base_price * (1 + 0.001 * i)  # Slight upward trend
            rows.append({
                "datetime": date,
                "ticker": ticker,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "volume": 1000000,
                "interval": "1d",
                "source_bar_count": 1,
                "hash": "test",
            })

    df = pd.DataFrame(rows)
    path = tmp / "prices_1d.parquet"
    df.to_parquet(path)
    return path


def _make_report_with_tickers(tmp: Path) -> SectorReport:
    """Create a SectorReport with known ticker recommendations."""
    evidence = [
        {
            "evidence_id": "ev_001",
            "source_type": "news",
            "source": "reuters",
            "as_of": "2026-06-15T00:00:00Z",
            "domain_id": "semiconductor_ai_infrastructure",
            "tickers": ["NVDA"],
            "sectors": ["semiconductor_ai_infrastructure"],
            "evidence_type": "sector_demand",
            "summary": "NVIDIA reports record AI GPU orders",
            "stance_hint": "positive",
            "strength": 0.9,
            "freshness_score": 0.8,
            "directness": "ticker",
            "reliability_score": 0.85,
        },
        {
            "evidence_id": "ev_002",
            "source_type": "news",
            "source": "bloomberg",
            "as_of": "2026-06-15T00:00:00Z",
            "domain_id": "semiconductor_ai_infrastructure",
            "tickers": ["AMD"],
            "sectors": ["semiconductor_ai_infrastructure"],
            "evidence_type": "sector_demand",
            "summary": "AMD faces margin pressure from competition",
            "stance_hint": "negative",
            "strength": 0.7,
            "freshness_score": 0.7,
            "directness": "ticker",
            "reliability_score": 0.7,
        },
    ]

    analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
    return analyst.run_from_evidence(
        evidence=evidence,
        as_of="2026-06-15T00:00:00Z",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOutcomeEvaluatorConstruction:
    def test_creates_with_valid_path(self, tmp_path):
        price_path = _make_price_data(tmp_path)
        evaluator = OutcomeEvaluator(price_path)
        assert evaluator.prices is not None

    def test_prices_loaded_lazily(self, tmp_path):
        price_path = _make_price_data(tmp_path)
        evaluator = OutcomeEvaluator(price_path)
        assert evaluator._prices is None
        _ = evaluator.prices
        assert evaluator._prices is not None


class TestTickerOutcome:
    def test_ticker_outcome_fields(self):
        outcome = TickerOutcome(
            ticker="NVDA",
            horizon_days=5,
            entry_price=100.0,
            exit_price=105.0,
            actual_return=0.05,
            predicted_direction="bullish",
            direction_correct=True,
            entry_date="2026-06-15",
            exit_date="2026-06-20",
        )
        assert outcome.ticker == "NVDA"
        assert outcome.direction_correct is True


class TestEvaluateHorizon:
    def test_evaluate_at_5d_horizon(self, tmp_path):
        price_path = _make_price_data(tmp_path)
        evaluator = OutcomeEvaluator(price_path)
        report = _make_report_with_tickers(tmp_path)

        recs = evaluator._extract_recommendations(report)
        outcome = evaluator._evaluate_horizon(
            recs,
            pd.Timestamp("2026-06-15"),
            horizon_days=5,
        )

        assert outcome.horizon_days == 5
        assert outcome.tickers_evaluated > 0
        assert outcome.direction_accuracy is not None

    def test_evaluate_at_20d_horizon(self, tmp_path):
        price_path = _make_price_data(tmp_path)
        evaluator = OutcomeEvaluator(price_path)
        report = _make_report_with_tickers(tmp_path)

        recs = evaluator._extract_recommendations(report)
        outcome = evaluator._evaluate_horizon(
            recs,
            pd.Timestamp("2026-06-15"),
            horizon_days=20,
        )

        assert outcome.horizon_days == 20
        assert outcome.tickers_evaluated > 0


class TestEvaluateReport:
    def test_full_evaluation(self, tmp_path):
        price_path = _make_price_data(tmp_path)
        evaluator = OutcomeEvaluator(price_path)
        report = _make_report_with_tickers(tmp_path)

        result = evaluator.evaluate(
            report,
            as_of="2026-06-15T00:00:00Z",
            horizons=[1, 5, 20],
        )

        assert isinstance(result, EvaluationResult)
        assert result.report_domain_id == "semiconductor_ai_infrastructure"
        assert len(result.horizons) == 3
        assert result.tickers_in_basket > 0

    def test_evaluation_summary(self, tmp_path):
        price_path = _make_price_data(tmp_path)
        evaluator = OutcomeEvaluator(price_path)
        report = _make_report_with_tickers(tmp_path)

        result = evaluator.evaluate(
            report,
            as_of="2026-06-15T00:00:00Z",
            horizons=[5],
        )

        assert "overall_direction_accuracy" in result.summary
        assert "horizon_5d" in result.summary


class TestRenderMarkdown:
    def test_render_evaluation_markdown(self, tmp_path):
        price_path = _make_price_data(tmp_path)
        evaluator = OutcomeEvaluator(price_path)
        report = _make_report_with_tickers(tmp_path)

        result = evaluator.evaluate(
            report,
            as_of="2026-06-15T00:00:00Z",
            horizons=[5, 20],
        )

        md = render_evaluation_markdown(result)
        assert "# Outcome Evaluation" in md
        assert "semiconductor_ai_infrastructure" in md
        assert "| 5d |" in md
        assert "| 20d |" in md
