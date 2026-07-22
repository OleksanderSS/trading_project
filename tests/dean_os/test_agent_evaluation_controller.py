from __future__ import annotations

import asyncio

from dean_os.agent_observability import AgentRunTrace, AgentRunTraceStore
from dean_os.agents.agent_evaluation_controller import AgentEvaluationControllerAgent
from dean_os.schemas import MarketContext


def _saved_trace(path, *, unsafe: int = 0, grounded: float | None = None) -> None:
    trace = AgentRunTrace.start(
        agent_name="domain_analyst",
        input_packet={"case": unsafe, "grounded": grounded},
        prompt_version="domain_v1",
        model_version="luna",
    )
    trace.unsafe_action_attempts = unsafe
    trace.finish(
        final_output={"done": True},
        final_verdict="needs_more_data",
        schema_valid=True,
        source_grounding_score=grounded,
    )
    AgentRunTraceStore(path).append(trace)


def test_controller_does_not_fail_unknown_review_metrics(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    _saved_trace(path)
    agent = AgentEvaluationControllerAgent(
        name="agent_evaluation_controller",
        config={"trace_store_path": str(path), "min_reviewed_runs": 1},
    )

    report = asyncio.run(agent.run(MarketContext()))

    assert report.verdict == "clear"
    unavailable = report.metrics_snapshot["unavailable_metrics"]
    assert "task_success_rate" in unavailable
    assert "source_grounding" in unavailable


def test_controller_blocks_directly_observed_unsafe_attempt(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    _saved_trace(path, unsafe=1, grounded=1.0)
    agent = AgentEvaluationControllerAgent(
        name="agent_evaluation_controller",
        config={"trace_store_path": str(path), "min_reviewed_runs": 1},
    )

    report = asyncio.run(agent.run(MarketContext()))

    assert report.verdict == "blocked"
    assert "unsafe_action_attempts=1" in report.metrics_snapshot["violations"]


def test_controller_warns_on_reviewed_grounding_below_threshold(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    _saved_trace(path, grounded=0.4)
    agent = AgentEvaluationControllerAgent(
        name="agent_evaluation_controller",
        config={
            "trace_store_path": str(path),
            "min_reviewed_runs": 1,
            "min_source_grounding": 0.9,
        },
    )

    report = asyncio.run(agent.run(MarketContext()))

    assert report.verdict == "caution"
    assert report.metrics_snapshot["warnings"] == [
        "source_grounding=0.400 below 0.900"
    ]
