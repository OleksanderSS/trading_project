from __future__ import annotations

import asyncio

import pytest

from dean_os.agents.paper_portfolio import PaperPortfolioAgent
from dean_os.paper_portfolio import PaperPortfolioSimulator
from dean_os.schemas import MarketContext


def test_missing_market_file_returns_explicit_data_gap(tmp_path):
    agent = PaperPortfolioAgent(
        "paper_portfolio",
        {
            "market_data_path": str(tmp_path / "missing.parquet"),
            "latest_processed_prices": None,
        },
    )

    report = asyncio.run(agent.run(MarketContext()))

    assert report.verdict == "needs_more_data"
    assert report.metrics_snapshot["simulation_skipped"] is True
    assert report.input_hash


def test_unexpected_simulator_failure_is_not_hidden(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("programming defect")

    monkeypatch.setattr(PaperPortfolioSimulator, "simulate", fail)
    agent = PaperPortfolioAgent("paper_portfolio", {})

    with pytest.raises(RuntimeError, match="programming defect"):
        asyncio.run(agent.run(MarketContext()))
