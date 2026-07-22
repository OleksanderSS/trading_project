from pathlib import Path

import asyncio
from pathlib import Path

import pandas as pd

from dean_os.agents.historical_analogies import HistoricalAnalogiesAgent
from dean_os.data_loader import load_local_tabular_data
from dean_os.schemas import MarketContext, PipelineReport


def test_load_local_tabular_data_reads_csv_and_parquet(tmp_path: Path) -> None:
    csv_path = tmp_path / "prices.csv"
    df_csv = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "close": [170.0, 330.0]})
    df_csv.to_csv(csv_path, index=False)

    parquet_path = tmp_path / "macro.parquet"
    df_parquet = pd.DataFrame({"series_id": ["GDP", "CPI"], "value": [1000, 4.2]})
    df_parquet.to_parquet(parquet_path)

    frames = load_local_tabular_data(directory=tmp_path)

    assert "prices" in frames
    assert "macro" in frames
    assert frames["prices"].reset_index(drop=True).equals(df_csv)
    assert frames["macro"].reset_index(drop=True).equals(df_parquet)


def test_historical_analogies_agent_can_read_agent_reports_from_context_metadata(tmp_path: Path) -> None:
    ctx = MarketContext(as_of="2026-07-10T15:00:00Z")
    report = PipelineReport(
        agent_name="macro_analyst",
        agent_version="0.1.0",
        branch="pipeline",
        verdict="bullish",
        confidence=0.8,
        data_quality_score=0.9,
        signal_strength=0.5,
        reasons=["Macro stance bullish"],
        risks=[],
        blind_spots=[],
        evidence=[],
        input_hash="test",
        metrics_snapshot={},
    )
    ctx.metadata["agent_reports"] = [report.model_dump(mode="json")]

    agent = HistoricalAnalogiesAgent(name="historical_analogies", config={})
    actual = asyncio.run(agent.run(ctx))

    assert actual.agent_name == "historical_analogies"
    assert actual.verdict in {"bullish", "neutral", "bearish", "needs_more_data"}
    assert actual.thesis.startswith("Current World State") or actual.verdict == "needs_more_data"
