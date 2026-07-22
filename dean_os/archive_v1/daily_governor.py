import logging
from typing import Any

from dean_os.evals.output_quality_scorer import OutputQualityScorer
from dean_os.evals.source_grounding_eval import SourceGroundingEval
from dean_os.evals.time_leakage_guard import TimeLeakageGuard
from dean_os.observability.daily_run_audit_log import (
    CollectorRunStats,
    DailyRunAuditLog,
)
from dean_os.observability.safety_counters import SafetyCounters
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.system_run_manifest import SystemRunManifest
from dean_os.schemas import MarketContext
from dean_os.world_model.world_model_event_learning import WorldModelEventLearningPacket
from dean_os.world_model.world_model_replay_review_gate import WorldModelReplayReviewGate

logger = logging.getLogger(__name__)


class DailyPipelineGovernor:
    """
    Orchestrates the daily process from data collection to event packets
    and pushes to the review gate. Logs everything into the SystemRunManifest.

    Phase 6 additions:
    - TimeLeakageGuard: checks news items for future-date violations before analysis
    - SourceGroundingEval: validates hypothesis outputs don't contain unsafe claims
    - OutputQualityScorer: computes AGENT_OUTPUT_QUALITY_METRICS for the run
    - SafetyCounters: tracks any forbidden output generation attempts
    - DailyRunAuditLog: produces a full Codex-compliant audit log of the run
    """

    def __init__(self, domain_id: str, as_of: str):
        self.domain_id = domain_id
        self.as_of = as_of
        self.manifest = SystemRunManifest.initialize(as_of=as_of, domain_id=domain_id)
        self.safety_counters = SafetyCounters()
        self.audit_log = DailyRunAuditLog.start(run_id=self.manifest.run_id)

    def _run_collectors(self) -> bool:
        """
        Simulates the data collection process and checks health.
        In a full system, this would call actual scraper/collector modules.
        """
        self.manifest.log("Starting collector validation...")
        is_healthy = True
        details = "Mock collectors verified healthy."

        self.manifest.set_collector_status(is_healthy=is_healthy, details=details)

        # Audit log — collector stats
        self.audit_log.add_collector(CollectorRunStats(
            collector_id="mock_collector",
            source_count_attempted=1,
            source_count_succeeded=1 if is_healthy else 0,
            source_count_failed=0 if is_healthy else 1,
        ))
        return is_healthy

    def execute_daily_run(self, pipeline_context: dict) -> SystemRunManifest:
        """
        Executes the daily run through the required state machine:
        COLLECTION -> VALIDATION -> EVENT_LEARNING -> EVAL -> REVIEW_GATE
        """
        try:
            # ── COLLECTION & VALIDATION ────────────────────────────────────────
            self.manifest.log("=== STATE: COLLECTION & VALIDATION ===")
            if not self._run_collectors():
                self.manifest.log("Run aborted due to unhealthy collectors.")
                self.manifest.mark_completed()
                return self.manifest

            context_id = pipeline_context.get("as_of", self.as_of)
            self.manifest.register_pipeline_context(
                f"pipeline_context_{context_id}", pipeline_context
            )

            # ── TIME LEAKAGE CHECK (Phase 6) ────────────────────────────────────
            self.manifest.log("=== STATE: TIME LEAKAGE CHECK ===")
            leakage_guard = TimeLeakageGuard(as_of=self.as_of)
            leakage_count = leakage_guard.check_news_list(
                pipeline_context.get("news", [])
            )
            leakage_summary = leakage_guard.summary()
            if leakage_count > 0:
                self.manifest.log(
                    f"WARNING: {leakage_count} time-leakage violation(s) detected: "
                    f"{leakage_summary['violations']}"
                )
            else:
                self.manifest.log("Time leakage check: CLEAN")

            # ── EVENT_LEARNING ──────────────────────────────────────────────────
            self.manifest.log("=== STATE: EVENT_LEARNING ===")
            context = MarketContext(
                news=pipeline_context.get("news", []),
                metadata=pipeline_context,
                as_of=self.as_of,
            )

            event_packet = WorldModelEventLearningPacket()
            event_result = event_packet.build(
                context=context,
                domain_id=self.domain_id,
                as_of=self.as_of,
                save=False,
            )
            bundle_id = event_result.get("run_id", "unknown_bundle_id")
            self.manifest.set_event_packet(bundle_id)

            # Populate audit log with event extraction stats
            self.audit_log.populate_from_event_result(event_result)

            # ── SOURCE GROUNDING EVAL (Phase 6) ─────────────────────────────────
            self.manifest.log("=== STATE: SOURCE GROUNDING EVAL ===")
            grounding_eval = SourceGroundingEval()
            grounding_eval.check_event_packet(event_result)
            grounding_summary = grounding_eval.summary()
            if grounding_summary["unsafe_outputs"] > 0:
                self.safety_counters.buy_sell_hold_generated += grounding_summary["unsafe_outputs"]
                self.manifest.log(
                    f"SAFETY VIOLATION: {grounding_summary['unsafe_outputs']} unsafe output(s) found!"
                )
            else:
                self.manifest.log("Source grounding eval: SAFE")

            # ── OUTPUT QUALITY METRICS (Phase 6) ────────────────────────────────
            self.manifest.log("=== STATE: OUTPUT QUALITY SCORING ===")
            scorer = OutputQualityScorer()
            quality_report = scorer.score(event_result, leakage_violations=leakage_count)
            self.manifest.gate_summary["quality_metrics"] = quality_report["metrics"]
            self.manifest.gate_summary["quality_targets_passed"] = quality_report["all_targets_passed"]
            self.audit_log.quality_metrics = quality_report["metrics"]
            self.manifest.log(
                f"Quality scoring complete. All targets passed: {quality_report['all_targets_passed']}"
            )

            # ── REVIEW_GATE ──────────────────────────────────────────────────────
            self.manifest.log("=== STATE: REVIEW_GATE ===")
            review_gate = WorldModelReplayReviewGate()
            gate_result = review_gate.build(
                packet_json=event_result,
                approve=False,
                reviewer="daily_governor_auto",
                save=False,
            )
            self.manifest.set_review_gate(
                gate_result.get("run_id", "unknown_gate_id"),
                gate_result.get("summary", {}),
            )
            self.audit_log.populate_from_gate_result(gate_result)

            # ── SAFETY COUNTER SNAPSHOT ──────────────────────────────────────────
            if not self.safety_counters.is_clean():
                self.manifest.log(
                    f"SAFETY COUNTERS NON-ZERO: {self.safety_counters.violations()}"
                )
            self.manifest.gate_summary["safety_counters"] = self.safety_counters.as_dict()
            self.manifest.gate_summary["safety_clean"] = self.safety_counters.is_clean()

        except Exception as e:
            self.manifest.status = "failed"
            self.manifest.log(f"Exception during run: {str(e)}")
            logger.exception("Daily run failed")

        finally:
            self.manifest.mark_completed()
            self.audit_log.finish()

        return self.manifest

    def get_audit_log(self) -> DailyRunAuditLog:
        return self.audit_log
