from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from dean_os.base import BaseAgent
from dean_os.schemas import AnalyticalReport, MarketContext, PipelineReport

logger = logging.getLogger(__name__)


class PipelineBranch:
    hard_veto_agents = {"pipeline_audit", "data_quality", "risk"}

    def __init__(self, agents: Iterable[BaseAgent]):
        self.agents = list(agents)

    async def run(self, context: MarketContext) -> list[PipelineReport]:
        reports: list[PipelineReport] = []
        for agent in self.agents:
            try:
                async with asyncio.timeout(agent.capabilities.timeout_seconds):
                    report = await agent.run(context)
                if not isinstance(report, PipelineReport):
                    raise TypeError(f"{agent.name} returned {type(report).__name__}, expected PipelineReport")
                reports.append(report)
                if agent.name in self.hard_veto_agents and report.verdict == "blocked":
                    break
            except TimeoutError as exc:
                self._handle_error(agent, exc, reports)
            except Exception as exc:
                self._handle_error(agent, exc, reports)
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
    def __init__(self, agents: Iterable[BaseAgent]):
        self.agents = list(agents)

    async def run_parallel(self, context: MarketContext) -> list[AnalyticalReport]:
        tasks = [self._run_agent(agent, context) for agent in self.agents]
        if not tasks:
            return []
        reports = await asyncio.gather(*tasks)
        return [report for report in reports if report is not None]

    async def _run_agent(self, agent: BaseAgent, context: MarketContext) -> AnalyticalReport | None:
        try:
            async with asyncio.timeout(agent.capabilities.timeout_seconds):
                report = await agent.run(context)
            if not isinstance(report, AnalyticalReport):
                raise TypeError(f"{agent.name} returned {type(report).__name__}, expected AnalyticalReport")
            return report
        except Exception as exc:
            if agent.capabilities.error_behavior == "warn":
                logger.warning("%s failed: %s", agent.name, exc)
            return None
