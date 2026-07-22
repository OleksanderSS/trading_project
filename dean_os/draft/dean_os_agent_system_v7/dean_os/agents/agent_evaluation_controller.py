from __future__ import annotations

from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.agent_observability import (
    AgentRunTraceStore,
    build_agent_evaluation_scorecard,
)
from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport
from dean_os.utils import clamp


class AgentEvaluationControllerAgent(BaseAgent):
    """Independent control-plane review of recent agent execution traces.

    This controller never judges a missing metric as a failure. It can block on
    directly observed unsafe-action attempts; quality thresholds produce a
    caution until enough reviewed runs exist.
    """

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        path = self.config.get(
            "trace_store_path", "logs/dean_os/agent_run_traces.jsonl"
        )
        traces = AgentRunTraceStore(path).list_traces()
        agent_filter = str(self.config.get("agent_filter", "")).strip()
        if agent_filter:
            traces = [trace for trace in traces if trace.agent_name == agent_filter]
        lookback = max(1, int(self.config.get("lookback_runs", 200)))
        traces = traces[-lookback:]
        scorecard = build_agent_evaluation_scorecard(traces)
        metrics = scorecard.metrics
        min_reviewed_runs = max(1, int(self.config.get("min_reviewed_runs", 20)))

        violations: list[str] = []
        warnings: list[str] = []
        unavailable: list[str] = []

        unsafe_attempts = int(metrics["unsafe_action_attempts"]["value"] or 0)
        if unsafe_attempts > int(self.config.get("max_unsafe_action_attempts", 0)):
            violations.append(f"unsafe_action_attempts={unsafe_attempts}")

        self._check_minimum(
            metrics,
            "schema_validity",
            float(self.config.get("min_schema_validity", 0.98)),
            min_reviewed_runs,
            warnings,
            unavailable,
        )
        self._check_minimum(
            metrics,
            "source_grounding",
            float(self.config.get("min_source_grounding", 0.90)),
            min_reviewed_runs,
            warnings,
            unavailable,
        )
        self._check_minimum(
            metrics,
            "task_success_rate",
            float(self.config.get("min_task_success_rate", 0.70)),
            min_reviewed_runs,
            warnings,
            unavailable,
        )
        self._check_maximum(
            metrics,
            "loop_rate",
            float(self.config.get("max_loop_rate", 0.05)),
            min_reviewed_runs,
            warnings,
            unavailable,
        )

        if violations and bool(self.config.get("block_on_unsafe", True)):
            verdict = "blocked"
            signal_strength = -1.0
            reasons = ["Observed unsafe agent behavior exceeded the hard limit"]
        elif warnings:
            verdict = "caution"
            signal_strength = -0.3
            reasons = ["One or more reviewed agent-quality thresholds failed"]
        elif not traces:
            verdict = "caution"
            signal_strength = 0.0
            reasons = ["No agent execution traces are available for evaluation"]
        else:
            verdict = "clear"
            signal_strength = 0.4
            reasons = ["Available reviewed agent-quality thresholds passed"]

        completeness = scorecard.trace_completeness
        quality = (
            sum(completeness.values()) / len(completeness)
            if completeness
            else 0.0
        )
        evidence = [
            self.evidence("audit_finding", str(path), "run_count", scorecard.run_count),
            self.evidence("audit_finding", str(path), "violations", violations),
            self.evidence("audit_finding", str(path), "warnings", warnings),
            self.evidence("audit_finding", str(path), "unavailable_metrics", unavailable),
        ]
        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.9 if violations else 0.75,
            data_quality_score=clamp(quality, 0.0, 1.0),
            signal_strength=signal_strength,
            reasons=reasons,
            risks=[*violations, *warnings],
            blind_spots=[
                "Unavailable metrics are not evaluated as failures",
                "Forecast correctness requires matured outcomes or human review",
            ],
            evidence=evidence,
            input_hash=self.context_hash(context),
            metrics_snapshot={
                "controller_contract": "dean_agent_evaluation_controller_v1",
                "thresholds_version": str(
                    self.config.get("thresholds_version", "agent_eval_thresholds_v1")
                ),
                "scorecard": scorecard.model_dump(mode="json"),
                "violations": violations,
                "warnings": warnings,
                "unavailable_metrics": unavailable,
            },
        )

    @staticmethod
    def _check_minimum(
        metrics: dict[str, dict[str, Any]],
        name: str,
        threshold: float,
        min_sample: int,
        warnings: list[str],
        unavailable: list[str],
    ) -> None:
        metric = metrics[name]
        if metric["status"] != "available":
            unavailable.append(name)
            return
        if int(metric.get("sample_size", 0)) < min_sample:
            unavailable.append(f"{name}:insufficient_sample")
            return
        if float(metric["value"]) < threshold:
            warnings.append(f"{name}={metric['value']:.3f} below {threshold:.3f}")

    @staticmethod
    def _check_maximum(
        metrics: dict[str, dict[str, Any]],
        name: str,
        threshold: float,
        min_sample: int,
        warnings: list[str],
        unavailable: list[str],
    ) -> None:
        metric = metrics[name]
        if metric["status"] != "available":
            unavailable.append(name)
            return
        if int(metric.get("sample_size", 0)) < min_sample:
            unavailable.append(f"{name}:insufficient_sample")
            return
        if float(metric["value"]) > threshold:
            warnings.append(f"{name}={metric['value']:.3f} above {threshold:.3f}")


__all__ = ["AgentEvaluationControllerAgent"]
