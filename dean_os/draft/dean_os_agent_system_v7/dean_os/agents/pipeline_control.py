from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_surface import PipelineControlSurface
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_metrics import PipelineMetricSnapshot
from dean_os.schemas import EvidenceItem, MarketContext, PipelineReport
from dean_os.utils import sha256_json


class PipelineControlAgent(BaseAgent):
    """Build the bounded pipeline-control plane before and after execution.

    ``pre_pipeline`` decides whether the expensive pipeline may run and which
    stages are allowed. ``pre_trade`` evaluates the metrics produced by the
    current run, controls whether tuning proposals are allowed, and sets the
    boundary for the next run. Neither phase can write production config,
    promote a model, write learning memory, or trade.
    """

    version = "1.1.0"
    branch = "pipeline"

    def __init__(self, name: str | None = None, config: dict[str, Any] | None = None):
        super().__init__(name=name, config=config)
        self.capabilities.can_modify_pipeline = True

    async def run(self, context: MarketContext) -> PipelineReport:
        phase = str(context.phase or "unknown")
        surfaces = context.metadata.setdefault("pipeline_control_surfaces", {})
        existing = surfaces.get(phase) if isinstance(surfaces, dict) else None

        if isinstance(existing, dict) and existing and self.config.get("reuse_context_surface", True):
            surface = existing
        else:
            operating_profile = str(
                context.metadata.get("pipeline_operating_profile") or "full_ml_pipeline"
            )
            if operating_profile in {"stage03_data_only", "agent_only_no_pipeline"}:
                surface = _upstream_data_control_surface(
                    context,
                    phase=phase,
                    operating_profile=operating_profile,
                )
            else:
                output_dir = Path(
                    self.config.get(
                        "output_dir",
                        "reports/dean_os/pipeline_control_surface_runtime",
                    )
                )
                direct = _snapshot_control_payloads(context.metadata.get("pipeline_metric_snapshot"))
                surface = PipelineControlSurface(output_dir=output_dir).run(
                    model_performance_path=self.config.get("model_performance_path"),
                    replay_batch_path=self.config.get("replay_batch_path"),
                    feature_report_path=self.config.get("feature_report_path"),
                    data_quality_path=self.config.get("data_quality_path"),
                    constraints_path=self.config.get("constraints_path"),
                    constraints=self.config.get("constraints"),
                    save=bool(self.config.get("save_surface", False)),
                    model_performance_payload=direct.get("model_performance"),
                    replay_batch_payload=direct.get("replay_batch"),
                    feature_report_payload=direct.get("feature_report"),
                    data_quality_payload=direct.get("data_quality"),
                )
            surfaces[phase] = surface

        status = str(surface.get("surface", {}).get("status") or "blocked")
        proposal_gate = surface.get("proposal_gate", {})
        allowed_variation = surface.get("surface", {}).get("allowed_variation", {})
        configured_stages = _integer_list(self.config.get("allowed_stages"))
        if context.metadata.get("pipeline_operating_profile") == "stage03_data_only":
            configured_stages = [stage for stage in configured_stages if stage <= 3]
        execution_policy = self._execution_policy(
            context,
            phase=phase,
            status=status,
            proposal_gate=proposal_gate,
            allowed_variation=allowed_variation,
            configured_stages=configured_stages,
            surface=surface,
        )

        context.metadata["pipeline_control_surface"] = surface
        context.metadata["pipeline_execution_policy"] = execution_policy

        verdict = "blocked" if status == "blocked" else "caution" if status == "caution" else "clear"
        reasons = list(surface.get("recommendations", []))
        if not reasons:
            reasons = [f"Pipeline control surface status is {status} during {phase}."]
        risks = [
            str(axis_reason)
            for axis in surface.get("surface", {}).get("axes", [])
            if axis.get("status") in {"blocked", "caution"}
            for axis_reason in axis.get("reasons", [])[:2]
        ]
        evidence = [
            EvidenceItem(
                source_type="metric",
                source=self.name,
                key=f"pipeline_control_surface:{phase}",
                value=surface,
                timestamp=context.as_of,
            ),
            EvidenceItem(
                source_type="config",
                source=self.name,
                key="pipeline_execution_policy",
                value=execution_policy,
                timestamp=context.as_of,
            ),
        ]
        snapshot = context.metadata.get("pipeline_metric_snapshot")
        if isinstance(snapshot, dict):
            evidence.append(
                EvidenceItem(
                    source_type="metric",
                    source=self.name,
                    key="pipeline_metric_snapshot",
                    value=snapshot,
                    timestamp=context.as_of,
                )
            )

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=1.0,
            data_quality_score=_surface_quality(surface),
            signal_strength=0.0,
            reasons=reasons,
            risks=risks,
            blind_spots=[
                "The surface can only evaluate supplied metric artifacts or the normalized current-run snapshot.",
                "A clear surface permits a bounded experiment proposal; it does not prove model quality or authorize promotion.",
            ],
            evidence=evidence,
            input_hash=self.context_hash(context),
            config_hash=sha256_json(self.config),
            metrics_snapshot={
                "phase": phase,
                "pipeline_control_surface": surface,
                "pipeline_execution_policy": execution_policy,
                "pipeline_metric_snapshot_status": (
                    snapshot.get("status") if isinstance(snapshot, dict) else None
                ),
                "can_trade": False,
            },
        )

    def _execution_policy(
        self,
        context: MarketContext,
        *,
        phase: str,
        status: str,
        proposal_gate: dict[str, Any],
        allowed_variation: dict[str, Any],
        configured_stages: list[int],
        surface: dict[str, Any],
    ) -> dict[str, Any]:
        run_allowed = status != "blocked"
        base_boundary = {
            "schema_version": "dean_pipeline_execution_policy_v1",
            "production_config_write_allowed": False,
            "model_promotion_allowed": False,
            "learning_memory_write_allowed": False,
            "paper_or_live_trade_allowed": False,
            "requires_human_review": True,
        }
        if phase == "pre_pipeline":
            return {
                **base_boundary,
                "assessment_phase": phase,
                "status": "allowed" if run_allowed else "blocked",
                "pipeline_run_allowed": run_allowed,
                "next_pipeline_run_allowed": run_allowed,
                "allowed_stages": configured_stages,
                "tuning_proposal_allowed": False,
                "allowed_variation": allowed_variation,
                "source_surface_run_id": surface.get("run_id"),
            }

        previous = context.metadata.get("pipeline_execution_policy")
        previous = dict(previous) if isinstance(previous, dict) else {}
        post_assessment = {
            "schema_version": "dean_pipeline_post_run_assessment_v1",
            "assessment_phase": phase,
            "status": status,
            "current_run_evidence_accepted": bool(
                context.metadata.get("pipeline_metric_snapshot")
            ),
            "next_pipeline_run_allowed": run_allowed,
            "tuning_proposal_allowed": bool(
                proposal_gate.get("can_propose_tuning", False)
            ),
            "allowed_variation": allowed_variation,
            "source_surface_run_id": surface.get("run_id"),
        }
        context.metadata["pipeline_post_run_assessment"] = post_assessment
        return {
            **base_boundary,
            **previous,
            "assessment_phase": phase,
            # Do not retroactively rewrite whether the already-finished run was
            # allowed. The post-run result controls the next run and tuning.
            "pipeline_run_allowed": bool(previous.get("pipeline_run_allowed", True)),
            "next_pipeline_run_allowed": run_allowed,
            "tuning_proposal_allowed": bool(
                proposal_gate.get("can_propose_tuning", False)
            ),
            "allowed_stages": previous.get("allowed_stages", configured_stages),
            "allowed_variation": allowed_variation,
            "post_run_assessment": post_assessment,
            "source_surface_run_id": surface.get("run_id"),
        }


def _snapshot_control_payloads(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        return {}
    try:
        snapshot = PipelineMetricSnapshot.model_validate(value)
    except Exception:
        return {}
    # An empty/skipped snapshot is still useful as explicit evidence, but the
    # control surface should treat absent metric families as caution, not clear.
    return snapshot.to_control_surface_payloads()


def _integer_list(value: Any) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[int] = []
    for item in values:
        try:
            stage = int(item)
        except (TypeError, ValueError):
            continue
        if stage not in result:
            result.append(stage)
    return result


def _surface_quality(surface: dict[str, Any]) -> float:
    axes = surface.get("surface", {}).get("axes", [])
    if not axes:
        return 0.0
    scores = [float(axis.get("score", 0.0) or 0.0) for axis in axes]
    return max(0.0, min(sum(scores) / len(scores), 1.0))


def _upstream_data_control_surface(
    context: MarketContext,
    *,
    phase: str,
    operating_profile: str,
) -> dict[str, Any]:
    packet = context.metadata.get("pipeline_stage03_packet")
    packet = dict(packet) if isinstance(packet, dict) else {}
    packet_status = str(packet.get("status") or "missing")
    stages_present = [int(item) for item in packet.get("stages_present", []) if str(item).isdigit()]
    news_count = len(packet.get("news_items", []) or [])
    warnings = [str(item) for item in packet.get("warnings", [])]

    if packet_status == "failed":
        status = "blocked"
        reasons = ["Pipeline stages 0-3 packet reports a failed upstream data run."]
    elif operating_profile == "agent_only_no_pipeline":
        status = "caution"
        reasons = ["No pipeline packet supplied; agent system is running from manual or research evidence only."]
    elif packet_status == "partial":
        status = "caution"
        reasons = ["Pipeline stages 0-3 packet is partial; downstream analysis remains review-only."]
    else:
        status = "clear"
        reasons = ["Existing stages 0-3 outputs are available for bounded analytical use."]
    if warnings:
        status = "caution" if status == "clear" else status
        reasons.extend(warnings[:3])

    axes = [
        {
            "name": "upstream_stage_availability",
            "status": status,
            "score": 1.0 if status == "clear" else 0.55 if status == "caution" else 0.0,
            "metrics": {"stages_present": stages_present, "packet_status": packet_status},
            "reasons": reasons,
        },
        {
            "name": "news_and_evidence_intake",
            "status": "clear" if news_count > 0 else "caution",
            "score": 0.8 if news_count > 0 else 0.45,
            "metrics": {"news_item_count": news_count},
            "reasons": [
                f"Normalized pipeline news items available: {news_count}."
                if news_count > 0
                else "No normalized news rows were found in the stage 0-3 packet."
            ],
        },
        {
            "name": "model_metrics_applicability",
            "status": "clear",
            "score": 1.0,
            "metrics": {"model_stages_active": False},
            "reasons": [
                "PnL, train/test, model stability, and replay thresholds are not applicable before stage 4+."
            ],
        },
    ]
    overall = "blocked" if any(item["status"] == "blocked" for item in axes) else "caution" if any(item["status"] == "caution" for item in axes) else "clear"
    return {
        "run_id": f"pipeline_upstream_control_{phase}_{sha256_json({'as_of': context.as_of, 'packet': packet})[:16]}",
        "created_at": context.as_of,
        "mode": "pipeline_upstream_data_control_surface",
        "operating_profile": operating_profile,
        "surface": {
            "status": overall,
            "feasible": overall != "blocked",
            "axes": axes,
            "allowed_variation": {
                "policy": "stage03_data_quality_only",
                "production_write_allowed": False,
                "allowed_stage_boundary": [0, 1, 2, 3],
            },
        },
        "proposal_gate": {
            "status": "disabled_until_model_stages",
            "can_propose_tuning": False,
        },
        "recommendations": reasons,
    }


__all__ = ["PipelineControlAgent"]
