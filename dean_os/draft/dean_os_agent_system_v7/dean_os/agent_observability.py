from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from dean_os.utils import json_ready, sha256_json


class RetrievedDocumentTrace(BaseModel):
    document_id: str
    content_hash: str = ""
    source_type: str = ""
    as_of: str = ""


class ToolCallTrace(BaseModel):
    tool_name: str
    input_hash: str = ""
    output_hash: str = ""
    success: bool
    judged_correct: bool | None = None
    error_type: str = ""


class StateTransitionTrace(BaseModel):
    from_state: str
    to_state: str
    timestamp: str
    reason: str = ""


class AgentRunTrace(BaseModel):
    """One privacy-conscious, replayable execution trace for an agent run.

    Payloads are represented by stable hashes. Small identifiers and error
    labels remain readable so controllers can audit behavior without copying
    whole evidence packets or tool outputs into an operational log.
    """

    schema_version: str = "dean_agent_run_trace_v1"
    run_id: str = Field(default_factory=lambda: f"agent_{uuid4().hex}")
    task_id: str = ""
    agent_name: str
    agent_version: str = "unknown"
    branch: str = "unknown"
    prompt_version: str = "unknown"
    model_version: str = "unknown"
    input_packet_hash: str
    retrieved_documents: list[RetrievedDocumentTrace] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    state_transitions: list[StateTransitionTrace] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    human_corrections: list[str] = Field(default_factory=list)
    final_output_hash: str = ""
    final_verdict: str = ""
    started_at: str
    finished_at: str = ""
    latency_ms: float | None = None
    steps_to_completion: int | None = None
    cost: float | None = None
    task_success: bool | None = None
    schema_valid: bool | None = None
    source_grounding_score: float | None = None
    error_recovery_success: bool | None = None
    loop_detected: bool = False
    unsafe_action_attempts: int = 0
    human_intervention: bool = False
    status: Literal["running", "completed", "failed", "blocked"] = "running"
    error_labels: list[str] = Field(default_factory=list)

    @classmethod
    def start(
        cls,
        *,
        agent_name: str,
        input_packet: Any,
        agent_version: str = "unknown",
        branch: str = "unknown",
        prompt_version: str = "unknown",
        model_version: str = "unknown",
        task_id: str = "",
    ) -> "AgentRunTrace":
        return cls(
            task_id=task_id,
            agent_name=agent_name,
            agent_version=agent_version,
            branch=branch,
            prompt_version=prompt_version,
            model_version=model_version,
            input_packet_hash=sha256_json(input_packet),
            started_at=_utc_now(),
        )

    def finish(
        self,
        *,
        final_output: Any,
        final_verdict: str,
        status: Literal["completed", "failed", "blocked"] = "completed",
        task_success: bool | None = None,
        schema_valid: bool | None = None,
        source_grounding_score: float | None = None,
    ) -> None:
        finished = datetime.now(UTC)
        self.finished_at = finished.isoformat()
        self.final_output_hash = sha256_json(final_output)
        self.final_verdict = final_verdict
        self.status = status
        self.task_success = task_success
        self.schema_valid = schema_valid
        self.source_grounding_score = source_grounding_score
        started = datetime.fromisoformat(self.started_at)
        self.latency_ms = max(0.0, (finished - started).total_seconds() * 1000)


class AgentRunTraceStore:
    def __init__(self, path: str | Path = "logs/dean_os/agent_run_traces.jsonl"):
        self.path = Path(path)

    def append(self, trace: AgentRunTrace) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    json_ready(trace.model_dump(mode="json")),
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n"
            )
        return trace.run_id

    def list_traces(self, *, agent_name: str | None = None) -> list[AgentRunTrace]:
        if not self.path.exists():
            return []
        traces = [
            AgentRunTrace.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if agent_name is None:
            return traces
        return [trace for trace in traces if trace.agent_name == agent_name]


class AgentEvaluationScorecard(BaseModel):
    schema_version: str = "dean_agent_evaluation_scorecard_v1"
    run_count: int
    metrics: dict[str, dict[str, Any]]
    error_taxonomy_counts: dict[str, int]
    trace_completeness: dict[str, float]


def build_agent_evaluation_scorecard(
    traces: list[AgentRunTrace],
) -> AgentEvaluationScorecard:
    """Compute only evidenced metrics; never turn missing telemetry into zero."""

    metrics = {
        "task_success_rate": _rate(traces, "task_success"),
        "tool_call_accuracy": _tool_accuracy(traces),
        "steps_to_completion": _average(traces, "steps_to_completion"),
        "cost_per_success": _cost_per_success(traces),
        "latency_ms": _average(traces, "latency_ms"),
        "human_intervention_rate": _boolean_rate(traces, "human_intervention"),
        "schema_validity": _rate(traces, "schema_valid"),
        "source_grounding": _average(traces, "source_grounding_score"),
        "error_recovery_success": _rate(traces, "error_recovery_success"),
        "loop_rate": _boolean_rate(traces, "loop_detected"),
        "unsafe_action_attempts": {
            "status": "available",
            "value": sum(trace.unsafe_action_attempts for trace in traces),
            "sample_size": len(traces),
        },
    }
    error_counts = Counter(label for trace in traces for label in trace.error_labels)
    completeness_fields = (
        "prompt_version",
        "model_version",
        "input_packet_hash",
        "final_output_hash",
    )
    denominator = len(traces)
    completeness = {
        field: (
            sum(
                bool(getattr(trace, field))
                and getattr(trace, field) != "unknown"
                for trace in traces
            )
            / denominator
            if denominator
            else 0.0
        )
        for field in completeness_fields
    }
    return AgentEvaluationScorecard(
        run_count=denominator,
        metrics=metrics,
        error_taxonomy_counts=dict(sorted(error_counts.items())),
        trace_completeness=completeness,
    )


def _rate(traces: list[AgentRunTrace], field: str) -> dict[str, Any]:
    observed = [getattr(trace, field) for trace in traces if getattr(trace, field) is not None]
    if not observed:
        return _unavailable(f"No reviewed values for {field}")
    return {
        "status": "available",
        "value": sum(bool(value) for value in observed) / len(observed),
        "sample_size": len(observed),
    }


def _boolean_rate(traces: list[AgentRunTrace], field: str) -> dict[str, Any]:
    if not traces:
        return _unavailable("No run traces")
    return {
        "status": "available",
        "value": sum(bool(getattr(trace, field)) for trace in traces) / len(traces),
        "sample_size": len(traces),
    }


def _average(traces: list[AgentRunTrace], field: str) -> dict[str, Any]:
    observed = [getattr(trace, field) for trace in traces if getattr(trace, field) is not None]
    if not observed:
        return _unavailable(f"No observed values for {field}")
    return {
        "status": "available",
        "value": sum(observed) / len(observed),
        "sample_size": len(observed),
    }


def _tool_accuracy(traces: list[AgentRunTrace]) -> dict[str, Any]:
    reviewed = [
        call.judged_correct
        for trace in traces
        for call in trace.tool_calls
        if call.judged_correct is not None
    ]
    if not reviewed:
        return _unavailable("Tool success is logged, but correctness has not been reviewed")
    return {
        "status": "available",
        "value": sum(bool(value) for value in reviewed) / len(reviewed),
        "sample_size": len(reviewed),
    }


def _cost_per_success(traces: list[AgentRunTrace]) -> dict[str, Any]:
    observed = [trace for trace in traces if trace.cost is not None and trace.task_success is not None]
    successes = sum(bool(trace.task_success) for trace in observed)
    if not observed or not successes:
        return _unavailable("Cost and at least one reviewed successful task are required")
    return {
        "status": "available",
        "value": sum(float(trace.cost or 0.0) for trace in observed) / successes,
        "sample_size": len(observed),
        "successful_tasks": successes,
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "value": None, "sample_size": 0, "reason": reason}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "AgentEvaluationScorecard",
    "AgentRunTrace",
    "AgentRunTraceStore",
    "RetrievedDocumentTrace",
    "StateTransitionTrace",
    "ToolCallTrace",
    "build_agent_evaluation_scorecard",
]
