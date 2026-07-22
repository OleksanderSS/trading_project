from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.pipeline_control.pipeline_control_caution_review_packet import PipelineControlCautionReviewPacket
from dean_os.pipeline_control.pipeline_control_instance_contract import PipelineControlInstanceContract
from dean_os.pipeline_control.pipeline_control_surface import PipelineControlSurface
from dean_os.pipeline_metric_input_readiness_gate import PipelineMetricInputReadinessGate
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class PipelineControlMetricFixtureValidation:
    """Synthetic control-flow validation for pipeline metric gates.

    The fixture inputs are deliberately not evidence. They only prove that the
    pipeline-control contracts can move from caution to clear when real metric
    artifacts with the required fields are later supplied.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_control_metric_fixture_validation"):
        self.output_dir = Path(output_dir)

    def build(self, *, save: bool = True) -> dict[str, Any]:
        run_id = _run_id("pipeline_control_metric_fixture_validation")
        work_dir = self.output_dir / run_id
        fixture_paths = _write_fixture_inputs(work_dir / "fixture_inputs")

        readiness = PipelineMetricInputReadinessGate(output_dir=work_dir / "readiness").build(
            model_performance_path=fixture_paths["model_performance"],
            replay_batch_path=fixture_paths["replay_batch"],
            feature_report_path=fixture_paths["feature_report"],
            data_quality_path=fixture_paths["data_quality"],
        )
        surface = PipelineControlSurface(output_dir=work_dir / "surface").run(
            model_performance_path=fixture_paths["model_performance"],
            replay_batch_path=fixture_paths["replay_batch"],
            feature_report_path=fixture_paths["feature_report"],
            data_quality_path=fixture_paths["data_quality"],
        )
        instance = PipelineControlInstanceContract(output_dir=work_dir / "instance").build(
            pipeline_surface_json=surface["saved_paths"]["latest_json"],
            architecture_map_json=None,
            domain_instance_contract_json=None,
        )
        caution_review = PipelineControlCautionReviewPacket(output_dir=work_dir / "caution_review").build(
            pipeline_metric_input_readiness_json=readiness["saved_paths"]["latest_json"],
            pipeline_control_instance_json=instance["saved_paths"]["latest_json"],
            model_performance_report_json=None,
            feature_report_json=fixture_paths["feature_report"],
            data_quality_json=fixture_paths["data_quality"],
        )

        chain = _chain_results(readiness, surface, instance, caution_review)
        checks = _review_checks(chain)
        status = _validation_status(checks)
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_metric_fixture_validation",
            "summary": _summary(status, chain),
            "fixture_inputs": {key: str(value) for key, value in fixture_paths.items()},
            "chain_results": chain,
            "review_checks": checks,
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(status),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_control_metric_fixture_validation_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_control_metric_fixture_validation_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Pipeline Control Metric Fixture Validation",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Validation status: `{summary.get('validation_status')}`",
        f"- Fixture is evidence: {summary.get('fixture_is_evidence')}",
        f"- Current artifacts overwritten: {summary.get('current_artifacts_overwritten')}",
        f"- Readiness status: `{summary.get('readiness_status')}`",
        f"- Surface status: `{summary.get('surface_status')}`",
        f"- Instance status: `{summary.get('instance_status')}`",
        f"- Caution review status: `{summary.get('caution_review_status')}`",
        f"- Can write production config: {summary.get('can_write_production_config')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Chain Results",
        "",
    ]
    for item in payload.get("chain_results", []):
        lines.append(f"- `{item.get('step_id')}`: {item.get('status')} path=`{item.get('latest_json')}`")

    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")

    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))

    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _write_fixture_inputs(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_performance": output_dir / "synthetic_model_performance.json",
        "replay_batch": output_dir / "synthetic_replay_batch.json",
        "feature_report": output_dir / "synthetic_feature_report.json",
        "data_quality": output_dir / "synthetic_data_quality.json",
    }
    _atomic_json(
        paths["model_performance"],
        {
            "mode": "synthetic_pipeline_metric_fixture",
            "fixture_not_evidence": True,
            "metrics": {
                "total_return": 0.12,
                "pnl": 1200.0,
                "sharpe": 1.1,
                "max_drawdown": 0.08,
                "train_score": 0.72,
                "validation_score": 0.68,
                "sample_count": 250,
            },
        },
    )
    _atomic_json(
        paths["replay_batch"],
        {
            "mode": "synthetic_pipeline_metric_fixture",
            "fixture_not_evidence": True,
            "summary": {"clear_hit_rate": 0.72, "clear_evaluated_runs": 8, "quality_blocked_runs": 0},
        },
    )
    _atomic_json(
        paths["feature_report"],
        {
            "mode": "synthetic_pipeline_metric_fixture",
            "fixture_not_evidence": True,
            "feature_importance": {"momentum": 0.2, "volume": 0.19, "sentiment": 0.18, "macro": 0.17},
            "feature_stability_score": 0.82,
            "unstable_features": [],
        },
    )
    _atomic_json(
        paths["data_quality"],
        {
            "mode": "synthetic_pipeline_metric_fixture",
            "fixture_not_evidence": True,
            "warnings": [],
            "leakage_flags": [],
        },
    )
    return paths


def _chain_results(
    readiness: dict[str, Any],
    surface: dict[str, Any],
    instance: dict[str, Any],
    caution_review: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "step_id": "pipeline_metric_input_readiness",
            "status": readiness.get("summary", {}).get("readiness_status"),
            "latest_json": readiness.get("saved_paths", {}).get("latest_json"),
            "blocked_metric_planes": readiness.get("summary", {}).get("blocked_metric_planes", []),
            "caution_metric_planes": readiness.get("summary", {}).get("caution_metric_planes", []),
        },
        {
            "step_id": "pipeline_control_surface",
            "status": surface.get("surface", {}).get("status"),
            "latest_json": surface.get("saved_paths", {}).get("latest_json"),
            "blocked_metric_planes": surface.get("surface", {}).get("allowed_variation", {}).get("blocked_axes", []),
            "caution_metric_planes": surface.get("surface", {}).get("allowed_variation", {}).get("caution_axes", []),
        },
        {
            "step_id": "pipeline_control_instance_contract",
            "status": instance.get("summary", {}).get("instance_status"),
            "latest_json": instance.get("saved_paths", {}).get("latest_json"),
            "blocked_metric_planes": instance.get("summary", {}).get("blocked_metric_planes", []),
            "caution_metric_planes": instance.get("summary", {}).get("caution_metric_planes", []),
        },
        {
            "step_id": "pipeline_control_caution_review",
            "status": caution_review.get("summary", {}).get("caution_review_status"),
            "latest_json": caution_review.get("saved_paths", {}).get("latest_json"),
            "blocked_metric_planes": caution_review.get("summary", {}).get("blocked_metric_planes", []),
            "caution_metric_planes": caution_review.get("summary", {}).get("caution_metric_planes", []),
        },
    ]


def _summary(status: str, chain: list[dict[str, Any]]) -> dict[str, Any]:
    by_step = {item["step_id"]: item for item in chain}
    return {
        "validation_status": status,
        "fixture_is_evidence": False,
        "can_use_fixture_as_metric_evidence": False,
        "current_artifacts_overwritten": False,
        "readiness_status": by_step["pipeline_metric_input_readiness"]["status"],
        "surface_status": by_step["pipeline_control_surface"]["status"],
        "instance_status": by_step["pipeline_control_instance_contract"]["status"],
        "caution_review_status": by_step["pipeline_control_caution_review"]["status"],
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _review_checks(chain: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_step = {item["step_id"]: item for item in chain}
    return [
        _check("pass", "fixture_marked_not_evidence", "Synthetic inputs are marked as validation fixtures only."),
        _check(
            "pass" if by_step["pipeline_metric_input_readiness"]["status"] == "metric_inputs_ready" else "fail",
            "readiness_clears_with_complete_metrics",
            str(by_step["pipeline_metric_input_readiness"]["status"]),
        ),
        _check(
            "pass" if by_step["pipeline_control_surface"]["status"] == "clear" else "fail",
            "surface_clears_with_complete_metrics",
            str(by_step["pipeline_control_surface"]["status"]),
        ),
        _check(
            "pass"
            if by_step["pipeline_control_instance_contract"]["status"] == "pipeline_control_instance_review_ready"
            else "fail",
            "instance_review_ready_with_complete_metrics",
            str(by_step["pipeline_control_instance_contract"]["status"]),
        ),
        _check(
            "pass"
            if by_step["pipeline_control_caution_review"]["status"] == "pipeline_ready_for_manual_proposal_review"
            else "fail",
            "caution_review_clear_with_complete_metrics",
            str(by_step["pipeline_control_caution_review"]["status"]),
        ),
        _check("pass", "current_pipeline_artifacts_not_overwritten", "All fixture outputs are written under this validation output directory."),
        _check("pass", "downstream_actions_disabled", "No autonomous tuning, config write, learning write, recommendation, or trade is enabled."),
    ]


def _validation_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "synthetic_fixture_control_flow_failed"
    return "synthetic_fixture_control_flow_passed"


def _operator_next_steps(status: str) -> list[str]:
    if status == "synthetic_fixture_control_flow_failed":
        return [
            "Do not rely on the current pipeline-control caution interpretation until the failed fixture check is fixed.",
            "Inspect the failed chain step and keep current real artifacts unchanged.",
        ]
    return [
        "Treat this as a contract sanity check only; it is not model evidence.",
        "Use real historical or locked evaluation artifacts with the same fields before clearing current risk, validation, and feature-stability cautions.",
        "Keep current production config, learning, recommendation, paper trading, and live trading gates closed.",
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No replay is rerun on market data.",
        "No model training or hyperparameter search is executed.",
        "No current pipeline-control artifact is overwritten.",
        "No production config is written.",
        "No learning memory or analyst-weight update is written.",
        "No recommendation, allocation, paper order, broker call, or live trade is generated.",
    ]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
    ReviewArtifactWriter.atomic_write_text(path, rendered)


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
