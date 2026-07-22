from __future__ import annotations

import asyncio

from dean_os.agent_observability import AgentRunTraceStore
from dean_os.base import AnalyticalAgent
from dean_os.branches import AnalyticalBranch
from dean_os.schemas import AnalyticalReport, MarketContext


class _ObservedAgent(AnalyticalAgent):
    name = "observed_agent"
    version = "2.0.0"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            branch="analytical",
            verdict="needs_more_data",
            confidence=0.5,
            data_quality_score=0.7,
            evidence=[self.evidence("news", "wire", "event-1", {"known": True})],
        )


def test_analytical_branch_writes_real_execution_trace(tmp_path) -> None:
    store = AgentRunTraceStore(tmp_path / "traces.jsonl")
    agent = _ObservedAgent(
        config={"prompt_version": "observed_v2", "model_version": "luna"}
    )
    reports = asyncio.run(
        AnalyticalBranch([agent], trace_store=store).run_parallel(
            MarketContext(
                tickers=["AMD"],
                timeframes=["15m", "60m", "1d"],
                metadata={"task_id": "task-1"},
            )
        )
    )

    traces = store.list_traces()
    assert len(reports) == 1
    assert len(traces) == 1
    assert traces[0].agent_name == "observed_agent"
    assert traces[0].prompt_version == "observed_v2"
    assert traces[0].model_version == "luna"
    assert traces[0].schema_valid is True
    assert traces[0].task_success is None
    assert traces[0].retrieved_documents[0].document_id == "wire:event-1"
    assert [step.to_state for step in traces[0].state_transitions] == [
        "running",
        "completed",
    ]
