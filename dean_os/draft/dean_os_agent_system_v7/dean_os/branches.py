from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import ClassVar

from dean_os.draft.dean_os_agent_system_v7.dean_os.agent_observability import (
    AgentRunTrace,
    AgentRunTraceStore,
    RetrievedDocumentTrace,
    StateTransitionTrace,
)
from dean_os.base import BaseAgent
from dean_os.schemas import AnalyticalReport, MarketContext, PipelineReport
from dean_os.utils import sha256_json

logger = logging.getLogger(__name__)


class PipelineBranch:
    hard_veto_agents: ClassVar[set[str]] = {"pipeline_audit", "data_quality", "risk"}

    def __init__(
        self,
        agents: Iterable[BaseAgent],
        soft_mode: bool = False,
        trace_store: AgentRunTraceStore | None = None,
    ):
        self.agents = list(agents)
        self.soft_mode = soft_mode
        self.trace_store = trace_store

    async def run(self, context: MarketContext) -> list[PipelineReport]:
        reports: list[PipelineReport] = []
        for agent in self.agents:
            trace = _start_trace(agent, context) if self.trace_store else None
            try:
                async with asyncio.timeout(agent.capabilities.timeout_seconds):
                    report = await agent.run(context)
                if not isinstance(report, PipelineReport):
                    raise TypeError(f"{agent.name} returned {type(report).__name__}, expected PipelineReport")
                reports.append(report)
                _finish_trace(trace, report)
                if trace is not None:
                    self.trace_store.append(trace)
                if (
                    not self.soft_mode
                    and agent.capabilities.can_veto
                    and report.verdict == "blocked"
                ):
                    break
            except TimeoutError as exc:
                self._handle_error(agent, exc, reports)
                _fail_trace(trace, exc, "timeout")
                if trace is not None:
                    self.trace_store.append(trace)
            except Exception as exc:
                self._handle_error(agent, exc, reports)
                label = "schema_violation" if isinstance(exc, TypeError) else "agent_execution_error"
                _fail_trace(trace, exc, label)
                if trace is not None:
                    self.trace_store.append(trace)
        return reports

    def _handle_error(self, agent: BaseAgent, exc: Exception, reports: list[PipelineReport]) -> None:
        behavior = agent.capabilities.error_behavior
        if behavior == "block":
            reports.append(
                PipelineReport(
                    agent_name=agent.name,
                    agent_version=agent.version,
                    verdict="blocked",
                    confidence=1.0,
                    data_quality_score=0.0,
                    reasons=[f"{agent.name} failed"],
                    risks=[repr(exc)],
                    blind_spots=["Agent failed before producing normal diagnostics"],
                    evidence=[agent.evidence("metric", agent.name, "exception", repr(exc))],
                    metrics_snapshot={"exception": repr(exc)},
                )
            )
        elif behavior == "warn":
            logger.warning("%s failed: %s", agent.name, exc)
        else:
            logger.info("%s skipped after failure: %s", agent.name, exc)


class AnalyticalBranch:
    def __init__(
        self,
        agents: Iterable[BaseAgent],
        trace_store: AgentRunTraceStore | None = None,
    ):
        self.agents = list(agents)
        self.trace_store = trace_store

    async def run_parallel(self, context: MarketContext) -> list[AnalyticalReport]:
        tasks = [self._run_agent(agent, context) for agent in self.agents]
        if not tasks:
            return []
        reports = await asyncio.gather(*tasks)
        return [report for report in reports if report is not None]

    async def _run_agent(self, agent: BaseAgent, context: MarketContext) -> AnalyticalReport | None:
        trace = _start_trace(agent, context) if self.trace_store else None
        try:
            async with asyncio.timeout(agent.capabilities.timeout_seconds):
                report = await agent.run(context)
            if not isinstance(report, AnalyticalReport):
                raise TypeError(f"{agent.name} returned {type(report).__name__}, expected AnalyticalReport")
            _finish_trace(trace, report)
            if trace is not None:
                self.trace_store.append(trace)
            return report
        except Exception as exc:
            if agent.capabilities.error_behavior == "warn":
                logger.warning("%s failed: %s", agent.name, exc)
            label = "schema_violation" if isinstance(exc, TypeError) else "agent_execution_error"
            _fail_trace(trace, exc, label)
            if trace is not None:
                self.trace_store.append(trace)
            return None


def _start_trace(agent: BaseAgent, context: MarketContext) -> AgentRunTrace:
    compact_input = {
        "context_hash": agent.context_hash(context),
        "phase": context.phase,
        "as_of": context.as_of,
        "tickers": context.tickers,
        "timeframes": context.timeframes,
    }
    trace = AgentRunTrace.start(
        agent_name=agent.name,
        agent_version=str(agent.version),
        branch=str(agent.branch),
        prompt_version=str(agent.config.get("prompt_version", "unknown")),
        model_version=str(agent.config.get("model_version", "unknown")),
        input_packet=compact_input,
        task_id=str(context.metadata.get("task_id", "")),
    )
    trace.state_transitions.append(
        StateTransitionTrace(
            from_state="scheduled",
            to_state="running",
            timestamp=datetime.now(UTC).isoformat(),
            reason=f"phase:{context.phase}",
        )
    )
    return trace


def _finish_trace(trace: AgentRunTrace | None, report: PipelineReport | AnalyticalReport) -> None:
    if trace is None:
        return
    trace.retrieved_documents = [
        RetrievedDocumentTrace(
            document_id=f"{item.source}:{item.key}",
            content_hash=sha256_json(item.value),
            source_type=item.source_type,
            as_of=item.timestamp or "",
        )
        for item in report.evidence
    ]
    status = "blocked" if report.verdict == "blocked" else "completed"
    trace.finish(
        final_output=report,
        final_verdict=report.verdict,
        status=status,
        schema_valid=True,
    )
    trace.state_transitions.append(
        StateTransitionTrace(
            from_state="running",
            to_state=status,
            timestamp=trace.finished_at,
            reason=f"verdict:{report.verdict}",
        )
    )


def _fail_trace(trace: AgentRunTrace | None, exc: Exception, label: str) -> None:
    if trace is None:
        return
    trace.validation_errors.append(repr(exc))
    trace.error_labels.append(label)
    trace.finish(
        final_output={"exception": repr(exc)},
        final_verdict="failed",
        status="failed",
        task_success=False,
        schema_valid=False if label == "schema_violation" else None,
    )
    trace.state_transitions.append(
        StateTransitionTrace(
            from_state="running",
            to_state="failed",
            timestamp=trace.finished_at,
            reason=label,
        )
    )
