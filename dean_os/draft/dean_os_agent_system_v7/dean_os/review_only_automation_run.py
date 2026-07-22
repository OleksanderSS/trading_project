from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.packets.build_focus_review_packet import BuildFocusReviewPacket
from dean_os.draft.dean_os_agent_system_v7.dean_os.current_architecture_map import CurrentArchitectureMap
from dean_os.draft.dean_os_agent_system_v7.dean_os.current_system_alignment_review import CurrentSystemAlignmentReview
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_evidence_inventory import (
    DEFAULT_CANDIDATE_PATHS,
    PipelineControlEvidenceInventory,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_locked_evaluation_assembler import (
    DEFAULT_EVALUATION_CANDIDATE_JSON,
    DEFAULT_TRAINING_CANDIDATE_JSON,
    PipelineControlLockedEvaluationAssembler,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_locked_feature_stability_assembler import (
    DEFAULT_FEATURE_STABILITY_CANDIDATE_JSON,
    PipelineControlLockedFeatureStabilityAssembler,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_metric_artifact_materializer import PipelineControlMetricArtifactMaterializer
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_real_metric_evidence_run import (
    DEFAULT_DATA_QUALITY_JSON,
    DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON,
    DEFAULT_REPLAY_BATCH_JSON,
    PipelineControlRealMetricEvidenceRun,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class DeanOSReviewOnlyAutomationRun:
    """Run the safe DEAN-OS review chain without starting the trading pipeline."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/review_only_automation_run_current"):
        self.output_dir = Path(output_dir)
        self.base_reports_dir = self.output_dir.parent

    def build(
        self,
        *,
        candidate_paths: list[str | Path] | None = None,
        training_candidate_json: str | Path | None = None,
        evaluation_candidate_json: str | Path | None = None,
        feature_stability_candidate_json: str | Path | None = None,
        replay_batch_json: str | Path | None = DEFAULT_REPLAY_BATCH_JSON,
        data_quality_json: str | Path | None = DEFAULT_DATA_QUALITY_JSON,
        constraints_path: str | Path | None = None,
        domain_instance_contract_json: str | Path | None = DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON,
        run_real_metric_when_ready: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        bounded_candidates = _discover_bounded_candidate_inputs(self.base_reports_dir)
        training_candidate_json = (
            training_candidate_json
            or bounded_candidates.get("training_candidate_json")
            or DEFAULT_TRAINING_CANDIDATE_JSON
        )
        evaluation_candidate_json = (
            evaluation_candidate_json
            or bounded_candidates.get("evaluation_candidate_json")
            or DEFAULT_EVALUATION_CANDIDATE_JSON
        )
        feature_stability_candidate_json = (
            feature_stability_candidate_json
            or bounded_candidates.get("feature_stability_candidate_json")
            or DEFAULT_FEATURE_STABILITY_CANDIDATE_JSON
        )
        if candidate_paths is None:
            candidate_paths = list(
                dict.fromkeys(
                    [
                        *(str(path) for path in DEFAULT_CANDIDATE_PATHS),
                        *(str(path) for path in bounded_candidates.values()),
                    ]
                )
            )
        replay_batch_json = replay_batch_json or DEFAULT_REPLAY_BATCH_JSON
        data_quality_json = data_quality_json or DEFAULT_DATA_QUALITY_JSON
        domain_instance_contract_json = (
            domain_instance_contract_json or DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON
        )
        run_id = _run_id("review_only_automation_run")
        architecture = CurrentArchitectureMap(self.base_reports_dir / "current_architecture_map_current").build()
        alignment = CurrentSystemAlignmentReview(
            self.base_reports_dir / "current_system_alignment_review_two_branch_current"
        ).build(
            architecture_map_json=architecture["saved_paths"]["latest_json"],
        )
        focus = BuildFocusReviewPacket(self.base_reports_dir / "build_focus_review_packet_current").build(
            alignment_review_json=alignment["saved_paths"]["latest_json"],
        )
        inventory = PipelineControlEvidenceInventory(
            self.base_reports_dir / "pipeline_control_evidence_inventory_current"
        ).build(
            candidate_paths=candidate_paths,
        )
        locked_evaluation = PipelineControlLockedEvaluationAssembler(
            self.base_reports_dir / "pipeline_control_locked_evaluation_assembler_current"
        ).build(
            training_candidate_json=training_candidate_json,
            evaluation_candidate_json=evaluation_candidate_json,
        )
        locked_feature_stability = PipelineControlLockedFeatureStabilityAssembler(
            self.base_reports_dir / "pipeline_control_locked_feature_stability_assembler_current"
        ).build(
            feature_stability_candidate_json=feature_stability_candidate_json,
        )
        materializer = PipelineControlMetricArtifactMaterializer(
            self.base_reports_dir / "pipeline_control_metric_artifact_materializer_current"
        ).build(
            candidate_paths=candidate_paths,
        )

        model_evaluation_json, feature_stability_report = _resolved_metric_inputs(
            locked_evaluation,
            locked_feature_stability,
            materializer,
        )
        real_metric = _real_metric_step(
            base_reports_dir=self.base_reports_dir,
            model_evaluation_json=model_evaluation_json,
            feature_stability_report=feature_stability_report,
            replay_batch_json=replay_batch_json,
            data_quality_json=data_quality_json,
            constraints_path=constraints_path,
            architecture_map_json=architecture["saved_paths"]["latest_json"],
            domain_instance_contract_json=domain_instance_contract_json,
            run_real_metric_when_ready=run_real_metric_when_ready,
        )
        steps = _steps(
            architecture=architecture,
            alignment=alignment,
            focus=focus,
            inventory=inventory,
            locked_evaluation=locked_evaluation,
            locked_feature_stability=locked_feature_stability,
            materializer=materializer,
            real_metric=real_metric,
        )
        summary = _summary(steps, model_evaluation_json, feature_stability_report, real_metric)
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "dean_os_review_only_automation_run",
            "inputs": {
                "candidate_paths": [str(path) for path in candidate_paths] if candidate_paths else None,
                "training_candidate_json": str(training_candidate_json) if training_candidate_json else None,
                "evaluation_candidate_json": str(evaluation_candidate_json) if evaluation_candidate_json else None,
                "feature_stability_candidate_json": str(feature_stability_candidate_json)
                if feature_stability_candidate_json
                else None,
                "replay_batch_json": str(replay_batch_json) if replay_batch_json else None,
                "data_quality_json": str(data_quality_json) if data_quality_json else None,
                "constraints_path": str(constraints_path) if constraints_path else None,
                "domain_instance_contract_json": str(domain_instance_contract_json)
                if domain_instance_contract_json
                else None,
                "bounded_candidate_discovery": bounded_candidates or None,
                "run_real_metric_when_ready": run_real_metric_when_ready,
            },
            "summary": summary,
            "steps": steps,
            "next_runner_inputs": {
                "model_evaluation_json": model_evaluation_json,
                "feature_stability_report": feature_stability_report,
                "can_invoke_real_metric_evidence_run": bool(model_evaluation_json and feature_stability_report),
            },
            "report_paths": _report_paths(
                architecture,
                alignment,
                focus,
                inventory,
                locked_evaluation,
                locked_feature_stability,
                materializer,
                real_metric,
            ),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(summary),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_review_only_automation_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_review_only_automation_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Review-Only Automation Run",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Automation status: `{summary.get('automation_status')}`",
        f"- Ready locked model evaluation: {summary.get('ready_locked_model_evaluation')}",
        f"- Ready locked feature stability: {summary.get('ready_locked_feature_stability')}",
        f"- Real metric evidence invoked: {summary.get('real_metric_evidence_invoked')}",
        f"- Can use as metric evidence: {summary.get('can_use_as_metric_evidence')}",
        f"- Can clear current real cautions: {summary.get('can_clear_current_real_cautions')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Steps",
        "",
    ]
    for step in payload.get("steps", []):
        lines.append(f"- `{step.get('step_id')}`: {step.get('status')} ({step.get('primary_status')})")
    lines.extend(["", "## Next Runner Inputs", ""])
    next_inputs = payload.get("next_runner_inputs", {})
    lines.append(f"- Model evaluation JSON: `{next_inputs.get('model_evaluation_json')}`")
    lines.append(f"- Feature stability report: `{next_inputs.get('feature_stability_report')}`")
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _real_metric_step(
    *,
    base_reports_dir: Path,
    model_evaluation_json: str | None,
    feature_stability_report: str | None,
    replay_batch_json: str | Path | None,
    data_quality_json: str | Path | None,
    constraints_path: str | Path | None,
    architecture_map_json: str,
    domain_instance_contract_json: str | Path | None,
    run_real_metric_when_ready: bool,
) -> dict[str, Any]:
    if not run_real_metric_when_ready:
        return {"invoked": False, "skip_reason": "disabled_by_operator"}
    if not model_evaluation_json or not feature_stability_report:
        return {"invoked": False, "skip_reason": "missing_locked_metric_inputs"}
    payload = PipelineControlRealMetricEvidenceRun(
        base_reports_dir / "pipeline_control_real_metric_evidence_run_current"
    ).build(
        model_evaluation_json=model_evaluation_json,
        feature_stability_report=feature_stability_report,
        replay_batch_json=replay_batch_json,
        data_quality_json=data_quality_json,
        constraints_path=constraints_path,
        architecture_map_json=architecture_map_json,
        domain_instance_contract_json=domain_instance_contract_json,
    )
    return {"invoked": True, "payload": payload}


def _resolved_metric_inputs(
    locked_evaluation: dict[str, Any],
    locked_feature_stability: dict[str, Any],
    materializer: dict[str, Any],
) -> tuple[str | None, str | None]:
    materializer_inputs = materializer.get("next_runner_inputs", {})
    model = materializer_inputs.get("model_evaluation_json")
    feature = materializer_inputs.get("feature_stability_report")
    if not model:
        model = locked_evaluation.get("next_runner_inputs", {}).get("model_evaluation_json")
    if not feature:
        feature = locked_feature_stability.get("next_runner_inputs", {}).get("feature_stability_report")
    return model, feature


def _discover_bounded_candidate_inputs(base_reports_dir: Path) -> dict[str, str]:
    report_path = base_reports_dir / "pipeline_control_bounded_evidence_run_current" / "latest.json"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
    training = artifacts.get("training_candidates", {}) if isinstance(artifacts, dict) else {}
    evaluation = artifacts.get("evaluation_candidate", {}) if isinstance(artifacts, dict) else {}
    candidates = {
        "training_candidate_json": training.get("model_evaluation_json"),
        "evaluation_candidate_json": evaluation.get("evaluation_metric_candidate"),
        "feature_stability_candidate_json": training.get("feature_stability_report"),
    }
    return {
        key: str(path)
        for key, path in candidates.items()
        if path and Path(path).exists()
    }


def _steps(**payloads: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _step("current_architecture_map", payloads["architecture"], ("architecture_status",)),
        _step("current_system_alignment_review", payloads["alignment"], ("alignment_status",)),
        _step("build_focus_review_packet", payloads["focus"], ("focus_status",)),
        _step("pipeline_control_evidence_inventory", payloads["inventory"], ("inventory_status",)),
        _step("pipeline_control_locked_evaluation_assembler", payloads["locked_evaluation"], ("assembly_status",)),
        _step(
            "pipeline_control_locked_feature_stability_assembler",
            payloads["locked_feature_stability"],
            ("assembly_status",),
        ),
        _step("pipeline_control_metric_artifact_materializer", payloads["materializer"], ("materialization_status",)),
        _real_metric_status_step(payloads["real_metric"]),
    ]


def _step(step_id: str, payload: dict[str, Any], status_keys: tuple[str, ...]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    primary_status = next((summary.get(key) for key in status_keys if summary.get(key)), None)
    return {
        "step_id": step_id,
        "status": "completed",
        "primary_status": primary_status,
        "can_trade": summary.get("can_trade", False),
        "report_json": payload.get("saved_paths", {}).get("latest_json"),
        "report_markdown": payload.get("saved_paths", {}).get("latest_markdown"),
    }


def _real_metric_status_step(real_metric: dict[str, Any]) -> dict[str, Any]:
    if not real_metric.get("invoked"):
        return {
            "step_id": "pipeline_control_real_metric_evidence_run",
            "status": "skipped",
            "primary_status": real_metric.get("skip_reason"),
            "can_trade": False,
            "report_json": None,
            "report_markdown": None,
        }
    return _step(
        "pipeline_control_real_metric_evidence_run",
        real_metric["payload"],
        ("real_metric_evidence_status",),
    )


def _summary(
    steps: list[dict[str, Any]],
    model_evaluation_json: str | None,
    feature_stability_report: str | None,
    real_metric: dict[str, Any],
) -> dict[str, Any]:
    real_payload = real_metric.get("payload") if real_metric.get("invoked") else {}
    real_summary = real_payload.get("summary", {}) if isinstance(real_payload, dict) else {}
    ready_model = bool(model_evaluation_json)
    ready_feature = bool(feature_stability_report)
    invoked = bool(real_metric.get("invoked"))
    can_use = bool(real_summary.get("can_use_as_metric_evidence", False))
    can_clear = bool(real_summary.get("can_clear_current_real_cautions", False))
    status = "review_automation_completed"
    if invoked and can_clear:
        status = "review_automation_completed_real_metric_chain_clear"
    elif invoked:
        status = "review_automation_completed_real_metric_chain_review_needed"
    elif ready_model and ready_feature:
        status = "review_automation_completed_metric_pair_ready_real_run_skipped"
    elif ready_model or ready_feature:
        status = "review_automation_completed_waiting_for_metric_pair"
    else:
        status = "review_automation_completed_missing_locked_metric_inputs"
    return {
        "automation_status": status,
        "completed_step_count": sum(1 for step in steps if step.get("status") == "completed"),
        "skipped_step_count": sum(1 for step in steps if step.get("status") == "skipped"),
        "ready_locked_model_evaluation": ready_model,
        "ready_locked_feature_stability": ready_feature,
        "real_metric_evidence_invoked": invoked,
        "can_use_as_metric_evidence": can_use,
        "can_clear_current_real_cautions": can_clear,
        "can_write_learning_memory": False,
        "can_write_production_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _report_paths(*payloads: dict[str, Any]) -> dict[str, Any]:
    paths = {}
    for payload in payloads:
        if not payload:
            continue
        if payload.get("invoked") and payload.get("payload"):
            payload = payload["payload"]
        mode = payload.get("mode")
        if mode:
            paths[mode] = payload.get("saved_paths", {})
    return paths


def _operator_next_steps(summary: dict[str, Any]) -> list[str]:
    if summary["real_metric_evidence_invoked"]:
        return [
            "Review the real metric evidence run output before changing any pipeline or learning setting.",
            "Keep production config, recommendations, and trading disabled unless a separate human-approved action gate is added.",
        ]
    if summary["ready_locked_model_evaluation"] and summary["ready_locked_feature_stability"]:
        return [
            "Rerun without --no-real-metric-run to execute the review-only real metric evidence chain.",
            "Review its caution result before changing any pipeline or learning setting.",
        ]
    if summary["ready_locked_model_evaluation"] or summary["ready_locked_feature_stability"]:
        return [
            "Complete the missing locked metric counterpart, then rerun this automation.",
            "Do not treat one locked artifact as sufficient to clear current real cautions.",
        ]
    return [
        "Run or ingest real Stage 4/Stage 7 artifacts so the locked assemblers can build a model-evaluation and feature-stability pair.",
        "Use this automation as the regular review refresh after new artifacts are produced.",
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training, Stage 7 evaluation, replay, backtest, or tuning run is started.",
        "No synthetic metric artifact is generated.",
        "No learning memory, model promotion, or production config write is performed.",
        "No recommendation, allocation, paper order, broker call, or live trade is generated.",
    ]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', '').replace('-', '')}"
