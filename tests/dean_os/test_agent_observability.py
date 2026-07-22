from __future__ import annotations

import pytest

from dean_os.agent_observability import (
    AgentRunTrace,
    AgentRunTraceStore,
    ToolCallTrace,
    build_agent_evaluation_scorecard,
)


def test_trace_hashes_payloads_and_round_trips(tmp_path) -> None:
    trace = AgentRunTrace.start(
        agent_name="semiconductor_analyst",
        agent_version="1.2.0",
        branch="analytical",
        prompt_version="semiconductor_v3",
        model_version="luna",
        input_packet={"secret_payload": [1, 2, 3]},
    )
    trace.tool_calls.append(
        ToolCallTrace(
            tool_name="evidence_search",
            input_hash="input-sha",
            output_hash="output-sha",
            success=True,
            judged_correct=True,
        )
    )
    trace.steps_to_completion = 4
    trace.cost = 2.5
    trace.schema_valid = True
    trace.source_grounding_score = 0.9
    trace.finish(
        final_output={"verdict": "needs_more_data"},
        final_verdict="needs_more_data",
        task_success=True,
        schema_valid=True,
        source_grounding_score=0.9,
    )

    store = AgentRunTraceStore(tmp_path / "traces.jsonl")
    store.append(trace)
    loaded = store.list_traces()

    assert loaded == [trace]
    persisted = (tmp_path / "traces.jsonl").read_text(encoding="utf-8")
    assert "secret_payload" not in persisted
    assert trace.input_packet_hash in persisted


def test_scorecard_never_converts_missing_review_into_zero() -> None:
    trace = AgentRunTrace.start(agent_name="analyst", input_packet={"x": 1})
    trace.tool_calls.append(
        ToolCallTrace(tool_name="search", success=True)
    )
    trace.finish(final_output={}, final_verdict="review")

    scorecard = build_agent_evaluation_scorecard([trace])

    assert scorecard.metrics["task_success_rate"]["status"] == "unavailable"
    assert scorecard.metrics["tool_call_accuracy"]["status"] == "unavailable"
    assert scorecard.metrics["source_grounding"]["status"] == "unavailable"
    assert scorecard.metrics["loop_rate"]["value"] == 0.0


def test_scorecard_computes_only_observed_metrics() -> None:
    first = AgentRunTrace.start(agent_name="analyst", input_packet={"x": 1})
    first.cost = 3.0
    first.steps_to_completion = 3
    first.human_intervention = True
    first.error_labels = ["unsupported_inference", "overconfidence"]
    first.tool_calls = [
        ToolCallTrace(tool_name="search", success=True, judged_correct=True)
    ]
    first.finish(
        final_output={"ok": True},
        final_verdict="pass",
        task_success=True,
        schema_valid=True,
        source_grounding_score=0.8,
    )
    second = AgentRunTrace.start(agent_name="analyst", input_packet={"x": 2})
    second.cost = 1.0
    second.steps_to_completion = 5
    second.loop_detected = True
    second.unsafe_action_attempts = 2
    second.error_labels = ["unsupported_inference", "loop_detected"]
    second.tool_calls = [
        ToolCallTrace(tool_name="search", success=True, judged_correct=False)
    ]
    second.finish(
        final_output={"ok": False},
        final_verdict="fail",
        task_success=False,
        schema_valid=False,
        source_grounding_score=0.4,
    )

    scorecard = build_agent_evaluation_scorecard([first, second])

    assert scorecard.metrics["task_success_rate"]["value"] == 0.5
    assert scorecard.metrics["tool_call_accuracy"]["value"] == 0.5
    assert scorecard.metrics["steps_to_completion"]["value"] == 4.0
    assert scorecard.metrics["cost_per_success"]["value"] == 4.0
    assert scorecard.metrics["human_intervention_rate"]["value"] == 0.5
    assert scorecard.metrics["schema_validity"]["value"] == 0.5
    assert scorecard.metrics["source_grounding"]["value"] == pytest.approx(0.6)
    assert scorecard.metrics["loop_rate"]["value"] == 0.5
    assert scorecard.metrics["unsafe_action_attempts"]["value"] == 2
    assert scorecard.error_taxonomy_counts["unsupported_inference"] == 2
