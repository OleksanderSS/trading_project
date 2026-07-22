from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.agent_observability import AgentRunTraceStore
from dean_os.draft.dean_os_agent_system_v7.dean_os.anxiety_kill_switch import AnxietyConfig, AnxietyKillSwitch
from dean_os.draft.dean_os_agent_system_v7.dean_os.branches import AnalyticalBranch, PipelineBranch
from dean_os.draft.dean_os_agent_system_v7.dean_os.consensus import ConsensusEngine
from dean_os.draft.dean_os_agent_system_v7.dean_os.decision_logger import DecisionLogger
from dean_os.draft.dean_os_agent_system_v7.src.models.prototypes.registry import AgentRegistry
from dean_os.schemas import ConsensusDecision, MarketContext, PipelineReport
from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state import WorldStateBuilder

PipelineRunner = Callable[[MarketContext], Awaitable[dict[str, Any]] | dict[str, Any]]
PipelineContextBridge = Callable[[dict[str, Any], MarketContext], None]


class DEANOrchestrator:
    def __init__(
        self,
        registry: AgentRegistry,
        pipeline_runner: PipelineRunner | None = None,
        consensus: ConsensusEngine | None = None,
        decision_logger: DecisionLogger | None = None,
        soft_mode: bool = False,
        anxiety_config: AnxietyConfig | None = None,
        trace_store: AgentRunTraceStore | None = None,
        pipeline_context_bridge: PipelineContextBridge | None = None,
    ):
        self.registry = registry
        self.pipeline_runner = pipeline_runner
        self.consensus = consensus or ConsensusEngine()
        self.decision_logger = decision_logger
        self.soft_mode = soft_mode
        self.kill_switch = AnxietyKillSwitch(config=anxiety_config)
        self.trace_store = trace_store
        self.pipeline_context_bridge = pipeline_context_bridge

    async def run(self, context: MarketContext) -> ConsensusDecision:
        context.phase = "pre_pipeline"
        preflight_reports = await self._run_pipeline_review(context)
        if self._blocked(preflight_reports):
            decision = self.consensus.combine(preflight_reports, {}, [])
            self._log_if_enabled(decision, preflight_reports, [], context)
            return decision

        # Make preflight diagnostics and the pipeline execution policy available
        # to the injected runner before any expensive stage is started.
        if preflight_reports:
            context._agent_reports = preflight_reports
            context.metadata["preflight_agent_reports"] = [
                report.model_dump(mode="json") for report in preflight_reports
            ]

        pipeline_result = await self._run_pipeline(context)
        context.pipeline_result.update(pipeline_result)
        if self.pipeline_context_bridge is not None:
            self.pipeline_context_bridge(pipeline_result, context)

        # Allow analytical agents to consume earlier pipeline diagnostics and reports.
        if preflight_reports:
            context.metadata["agent_reports"] = [
                report.model_dump(mode="json") for report in preflight_reports
            ]

        context.phase = "post_pipeline"
        analytical_reports = await AnalyticalBranch(
            self.registry.load_branch("analytical", context),
            trace_store=self.trace_store,
        ).run_parallel(context)

        context.phase = "pre_trade"
        post_pipeline_reports = await self._run_pipeline_review(context)
        pipeline_reports = self._merge_pipeline_reports(
            post_pipeline_reports,
            preflight_reports,
        )
        # Expose reports so post-hoc agents (coherence_scan) can read them
        if not isinstance(context.metadata, dict):
            context.metadata = {}
        context.metadata["agent_reports"] = [
            r.model_dump(mode="json") for r in pipeline_reports + analytical_reports
        ]

        decision = self.consensus.combine(
            pipeline_reports,
            pipeline_result,
            analytical_reports,
        )

        # Stage 6: Anxiety Kill-Switch — автоматичний перехід в review_only
        ks_result = self.kill_switch.evaluate(context, decision)
        if ks_result.triggered:
            decision = self.kill_switch.apply(decision, ks_result)
            context.metadata["anxiety_kill_switch_triggered"] = True
            context.metadata["kill_switch_reasons"] = ks_result.reasons
            context.metadata["kill_switch_metrics"] = ks_result.metrics

        # Compatibility decision summary. The canonical immutable historical
        # object is built later by `WorldStateSnapshotBuilder` in the minimal
        # composition root, after ContextGrid, IndicatorStateGrid, and the
        # ScenarioOutcomeGraph are aligned.
        legacy_world_state = WorldStateBuilder().build(
            reports=pipeline_reports + analytical_reports,
            decision=decision,
            as_of=context.as_of,
            total_agents=len(pipeline_reports) + len(analytical_reports)
        )
        legacy_payload = legacy_world_state.model_dump(mode="json")
        context.metadata["legacy_world_state"] = legacy_payload
        context.metadata["legacy_world_state_summary"] = legacy_world_state.summary()
        # Backward-compatible aliases; new code should consume the canonical
        # immutable snapshot returned by `DEANMinimalSystem`.
        context.metadata["world_state"] = legacy_payload
        context.metadata["world_state_summary"] = legacy_world_state.summary()
        decision.world_state = legacy_payload

        self._stats_log(pipeline_reports + analytical_reports, decision)
        self._log_if_enabled(
            decision,
            pipeline_reports,
            analytical_reports,
            context,
        )
        return decision

    async def _run_pipeline_review(
        self,
        context: MarketContext,
    ) -> list[PipelineReport]:
        agents = self.registry.load_branch("pipeline", context)
        reports = await PipelineBranch(
            agents,
            soft_mode=self.soft_mode,
            trace_store=self.trace_store,
        ).run(context)
        synthetic_reports = self.registry.get_synthetic_reports()
        existing_names = {report.agent_name for report in reports}
        reports.extend(
            report
            for name, report in synthetic_reports.items()
            if name not in existing_names
        )
        return reports

    def _blocked(self, reports: list[PipelineReport]) -> bool:
        if self.soft_mode:
            return False
        resolver = getattr(
            self.registry,
            "hard_veto_agent_names",
            None,
        )
        hard_veto_agents = (
            set(resolver())
            if callable(resolver)
            else set(PipelineBranch.hard_veto_agents)
        )
        return any(
            report.agent_name in hard_veto_agents
            and report.verdict == "blocked"
            for report in reports
        )

    def _merge_pipeline_reports(
        self,
        post_pipeline_reports: list[PipelineReport],
        preflight_reports: list[PipelineReport],
    ) -> list[PipelineReport]:
        """Prefer post-pipeline safety evidence for each agent."""
        merged = list(post_pipeline_reports)
        seen = {report.agent_name for report in merged}
        merged.extend(
            report
            for report in preflight_reports
            if report.agent_name not in seen
        )
        return merged

    async def _run_pipeline(self, context: MarketContext) -> dict[str, Any]:
        if self.pipeline_runner is None:
            return {"tickers": context.tickers, "timeframe": context.timeframe or (context.timeframes[0] if context.timeframes else None)}
        result = self.pipeline_runner(context)
        if hasattr(result, "__await__"):
            result = await result
        if result is None:
            return {"status": "pipeline_returned_none"}
        if not isinstance(result, dict):
            return {"status": "pipeline_returned_non_mapping", "raw_result": result}
        return result

    def _stats_log(self, all_reports: list, decision: ConsensusDecision) -> None:
        try:
            from dean_os.draft.dean_os_agent_system_v7.dean_os.agent_stats import AgentStatsStore
            store = AgentStatsStore()
            for r in all_reports:
                store.log_run(
                    agent_name=r.agent_name,
                    agent_version=getattr(r, "agent_version", "0.0.0"),
                    branch=getattr(r, "branch", "?"),
                    verdict=r.verdict,
                    confidence=r.confidence,
                    data_quality_score=r.data_quality_score,
                    duration_ms=0.0,
                    ticker=getattr(r, "ticker", ""),
                )
            store.log_orchestrator_run(
                agent_count=len(all_reports),
                decision=decision.decision,
                confidence=decision.confidence,
            )
        except Exception:
            pass

    def _log_if_enabled(self, decision: ConsensusDecision, pipeline_reports, analytical_reports, context: MarketContext) -> None:
        if self.decision_logger is None:
            return
        self.decision_logger.log(
            decision=decision,
            pipeline_reports=pipeline_reports,
            analytical_reports=analytical_reports,
            input_snapshot=context.model_dump(mode="json", exclude={"dataframes", "returns"}),
            config={"registry": str(self.registry.config_path)},
        )
