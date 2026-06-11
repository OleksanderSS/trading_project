from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from dean_os.branches import AnalyticalBranch, PipelineBranch
from dean_os.consensus import ConsensusEngine
from dean_os.decision_logger import DecisionLogger
from dean_os.registry import AgentRegistry
from dean_os.schemas import ConsensusDecision, MarketContext

PipelineRunner = Callable[[MarketContext], Awaitable[dict[str, Any]] | dict[str, Any]]


class DEANOrchestrator:
    def __init__(
        self,
        registry: AgentRegistry,
        pipeline_runner: PipelineRunner | None = None,
        consensus: ConsensusEngine | None = None,
        decision_logger: DecisionLogger | None = None,
    ):
        self.registry = registry
        self.pipeline_runner = pipeline_runner
        self.consensus = consensus or ConsensusEngine()
        self.decision_logger = decision_logger

    async def run(self, context: MarketContext) -> ConsensusDecision:
        pipeline_reports = await PipelineBranch(self.registry.load_branch("pipeline", context)).run(context)
        blocked = [report for report in pipeline_reports if report.verdict == "blocked"]
        if blocked:
            decision = self.consensus.combine(pipeline_reports, {}, [])
            self._log_if_enabled(decision, pipeline_reports, [], context)
            return decision

        pipeline_result = await self._run_pipeline(context)
        context.pipeline_result.update(pipeline_result)
        analytical_reports = await AnalyticalBranch(self.registry.load_branch("analytical", context)).run_parallel(context)
        decision = self.consensus.combine(pipeline_reports, pipeline_result, analytical_reports)
        self._log_if_enabled(decision, pipeline_reports, analytical_reports, context)
        return decision

    async def _run_pipeline(self, context: MarketContext) -> dict[str, Any]:
        if self.pipeline_runner is None:
            return {"tickers": context.tickers, "timeframe": context.timeframe or (context.timeframes[0] if context.timeframes else None)}
        result = self.pipeline_runner(context)
        if hasattr(result, "__await__"):
            return await result
        return result

    def _log_if_enabled(self, decision, pipeline_reports, analytical_reports, context: MarketContext) -> None:
        if self.decision_logger is None:
            return
        self.decision_logger.log(
            decision=decision,
            pipeline_reports=pipeline_reports,
            analytical_reports=analytical_reports,
            input_snapshot=context.model_dump(mode="json", exclude={"dataframes", "returns"}),
            config={"registry": str(self.registry.config_path)},
        )
