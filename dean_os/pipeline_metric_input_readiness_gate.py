from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.pipeline_control.pipeline_control_surface import (
    _as_list,
    _audit_warning_items,
    _data_quality_axis,
    _extract_feature_importances,
    _feature_concentration,
    _feature_stability_axis,
    _first_number,
    _leakage_items,
    _load_constraints,
    _load_optional_payload,
    _model_metrics,
    _profitability_axis,
    _replay_axis,
    _replay_summary,
    _risk_axis,
    _unstable_feature_count,
    _validation_axis,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_MODEL_PERFORMANCE_PATH = "performance_data.json"
DEFAULT_REPLAY_BATCH_PATH = "reports/dean_os/historical_replay_batch/latest.json"
DEFAULT_FEATURE_REPORT_PATH = None
DEFAULT_DATA_QUALITY_PATH = "diagnostic_reports/feature_lineage_report.json"


class PipelineMetricInputReadinessGate:
    """Review-only inventory for PipelineControlSurface inputs.

    This gate does not train, tune, run replay, write config, or trade. It only
    checks whether the saved metric artifacts are coherent enough to feed the
    pipeline-control surface and explains known blockers before rerunning it.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_metric_input_readiness_gate"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        model_performance_path: str | Path | None = DEFAULT_MODEL_PERFORMANCE_PATH,
        replay_batch_path: str | Path | None = DEFAULT_REPLAY_BATCH_PATH,
        feature_report_path: str | Path | None = DEFAULT_FEATURE_REPORT_PATH,
        data_quality_path: str | Path | None = DEFAULT_DATA_QUALITY_PATH,
        constraints_path: str | Path | None = None,
        constraints: dict[str, Any] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        merged_constraints = _load_constraints(constraints_path, constraints)
        evidence = {
            "model_performance": _load_optional_payload(model_performance_path),
            "replay_batch": _load_optional_payload(replay_batch_path),
            "feature_report": _load_optional_payload(feature_report_path),
            "data_quality": _load_optional_payload(data_quality_path),
        }
        axes = [
            _profitability_axis(evidence, merged_constraints),
            _risk_axis(evidence, merged_constraints),
            _validation_axis(evidence, merged_constraints),
            _feature_stability_axis(evidence, merged_constraints),
            _data_quality_axis(evidence, merged_constraints),
            _replay_axis(evidence, merged_constraints),
        ]
        requested_paths = {
            "model_performance": model_performance_path,
            "replay_batch": replay_batch_path,
            "feature_report": feature_report_path,
            "data_quality": data_quality_path,
        }
        inventory = _input_inventory(evidence, requested_paths)
        status = _readiness_status(axes, inventory)
        payload = {
            "run_id": _run_id("pipeline_metric_input_readiness_gate"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_metric_input_readiness_gate",
            "inputs": {
                "model_performance_path": str(model_performance_path) if model_performance_path else None,
                "replay_batch_path": str(replay_batch_path) if replay_batch_path else None,
                "feature_report_path": str(feature_report_path) if feature_report_path else None,
                "data_quality_path": str(data_quality_path) if data_quality_path else None,
                "constraints_path": str(constraints_path) if constraints_path else None,
            },
            "summary": _summary(status, axes, inventory),
            "input_inventory": inventory,
            "metric_plane_readiness": axes,
            "constraints": merged_constraints,
            "commands": _commands(
                model_performance_path=model_performance_path,
                replay_batch_path=replay_batch_path,
                feature_report_path=feature_report_path,
                data_quality_path=data_quality_path,
                constraints_path=constraints_path,
            ),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(status, axes, inventory),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_metric_input_readiness_gate_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_metric_input_readiness_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Pipeline Metric Input Readiness Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Readiness status: `{summary.get('readiness_status')}`",
        f"- Available inputs: {summary.get('available_input_count')}",
        f"- Missing inputs: {summary.get('missing_input_count')}",
        f"- Blocked planes: {', '.join(summary.get('blocked_metric_planes', [])) or 'none'}",
        f"- Caution planes: {', '.join(summary.get('caution_metric_planes', [])) or 'none'}",
        f"- Can refresh PipelineControlSurface now: {summary.get('can_refresh_pipeline_control_surface_now')}",
        f"- Can propose reviewed tuning after surface/manual review: {summary.get('can_propose_reviewed_tuning_after_surface_and_manual_review')}",
        f"- Can run autonomous tuning now: {summary.get('can_run_autonomous_tuning_now')}",
        f"- Can write production config: {summary.get('can_write_production_config')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Input Inventory",
        "",
    ]
    for item in payload.get("input_inventory", []):
        lines.append(f"- `{item.get('input_id')}`: {item.get('status')} path=`{item.get('path')}`")
        for note in item.get("notes", [])[:3]:
            lines.append(f"  - {note}")

    lines.extend(["", "## Metric Plane Readiness", ""])
    for axis in payload.get("metric_plane_readiness", []):
        lines.append(f"- `{axis.get('name')}`: {axis.get('status')} score={axis.get('score')}")
        for reason in axis.get("reasons", [])[:3]:
            lines.append(f"  - {reason}")

    commands = payload.get("commands", {})
    lines.extend(["", "## Commands", ""])
    for command_id, command in commands.items():
        lines.append(f"- `{command_id}`: `{command}`")

    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _summary(status: str, axes: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = [axis["name"] for axis in axes if axis.get("status") == "blocked"]
    caution = [axis["name"] for axis in axes if axis.get("status") == "caution"]
    supplied_errors = [item["input_id"] for item in inventory if item.get("path") and item.get("status") == "unreadable"]
    available = [item for item in inventory if item.get("available")]
    missing = [item for item in inventory if item.get("status") == "missing"]
    return {
        "readiness_status": status,
        "available_input_count": len(available),
        "missing_input_count": len(missing),
        "unreadable_input_ids": supplied_errors,
        "axis_status_counts": dict(Counter(axis.get("status") for axis in axes)),
        "blocked_metric_planes": blocked,
        "caution_metric_planes": caution,
        "can_refresh_pipeline_control_surface_now": bool(available) and not supplied_errors,
        "can_propose_reviewed_tuning_after_surface_and_manual_review": not blocked and status != "blocked_metric_inputs",
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _input_inventory(evidence: dict[str, dict[str, Any]], requested_paths: dict[str, str | Path | None]) -> list[dict[str, Any]]:
    return [
        _model_performance_inventory(evidence, requested_paths.get("model_performance")),
        _replay_batch_inventory(evidence, requested_paths.get("replay_batch")),
        _feature_report_inventory(evidence, requested_paths.get("feature_report")),
        _data_quality_inventory(evidence, requested_paths.get("data_quality")),
    ]


def _base_inventory(
    input_id: str,
    evidence: dict[str, Any],
    path: str | Path | None,
    required_for_planes: list[str],
) -> dict[str, Any]:
    if not path:
        return {
            "input_id": input_id,
            "path": None,
            "available": False,
            "status": "missing",
            "required_for_planes": required_for_planes,
            "recognized_metrics": {},
            "notes": ["No path was supplied for this input."],
        }
    if not evidence.get("available"):
        return {
            "input_id": input_id,
            "path": str(path),
            "available": False,
            "status": "unreadable",
            "required_for_planes": required_for_planes,
            "recognized_metrics": {},
            "notes": [str(evidence.get("error") or "Input path could not be loaded.")],
        }
    return {
        "input_id": input_id,
        "path": str(path),
        "available": True,
        "status": "available",
        "required_for_planes": required_for_planes,
        "recognized_metrics": {},
        "notes": [],
    }


def _model_performance_inventory(evidence: dict[str, dict[str, Any]], path: str | Path | None) -> dict[str, Any]:
    item = _base_inventory("model_performance", evidence["model_performance"], path, ["profitability", "risk", "validation"])
    if not item["available"]:
        return item
    metrics = _model_metrics(evidence)
    recognized = {
        "profitability_metric_present": any(
            _first_number(metrics, keys) is not None
            for keys in [["total_return", "return", "realized_return", "pnl_pct"], ["pnl", "profit", "net_profit"], ["sharpe", "sharpe_ratio"]]
        ),
        "max_drawdown_present": _first_number(metrics, ["max_drawdown", "maximum_drawdown", "mdd", "drawdown"]) is not None,
        "validation_score_present": _first_number(metrics, ["validation_score", "val_score", "test_score", "out_of_sample_score"]) is not None,
        "train_score_present": _first_number(metrics, ["train_score", "training_score", "in_sample_score"]) is not None,
        "sample_count_present": _first_number(metrics, ["sample_count", "n_samples", "test_samples", "validation_samples", "observations"]) is not None,
    }
    notes = []
    if not recognized["profitability_metric_present"]:
        notes.append("No direct profitability metric recognized; replay may still provide a proxy.")
    if not recognized["max_drawdown_present"]:
        notes.append("Max drawdown is missing.")
    if not (recognized["validation_score_present"] and recognized["train_score_present"] and recognized["sample_count_present"]):
        notes.append("Train/validation/sample-count metrics are incomplete.")
    item["recognized_metrics"] = recognized
    item["notes"] = notes or ["Model performance metrics are recognized."]
    return item


def _replay_batch_inventory(evidence: dict[str, dict[str, Any]], path: str | Path | None) -> dict[str, Any]:
    item = _base_inventory("replay_batch", evidence["replay_batch"], path, ["profitability", "replay_repeatability"])
    if not item["available"]:
        return item
    summary = _replay_summary(evidence)
    recognized = {
        "clear_hit_rate": _first_number(summary, ["clear_hit_rate"]),
        "clear_evaluated_runs": _first_number(summary, ["clear_evaluated_runs"]),
        "quality_blocked_runs": _first_number(summary, ["quality_blocked_runs"]) or 0,
        "clear_average_realized_return": _first_number(summary, ["clear_average_realized_return", "average_realized_return"]),
    }
    notes = []
    if not summary:
        notes.append("Replay summary is missing.")
    if recognized["quality_blocked_runs"]:
        notes.append(f"Replay has quality-blocked runs: {recognized['quality_blocked_runs']:.0f}.")
    if recognized["clear_hit_rate"] is None:
        notes.append("Clear replay hit rate is missing.")
    item["recognized_metrics"] = recognized
    item["notes"] = notes or ["Replay summary metrics are recognized."]
    return item


def _feature_report_inventory(evidence: dict[str, dict[str, Any]], path: str | Path | None) -> dict[str, Any]:
    item = _base_inventory("feature_report", evidence["feature_report"], path, ["feature_stability"])
    if not item["available"]:
        return item
    payload = evidence["feature_report"].get("payload", {})
    importances = _extract_feature_importances(payload)
    stability_score = _first_number(payload, ["feature_stability_score", "stability_score"])
    recognized = {
        "feature_count": len(importances),
        "feature_concentration": _feature_concentration(importances),
        "feature_stability_score": stability_score,
        "unstable_feature_count": _unstable_feature_count(payload),
    }
    notes = []
    if not importances and stability_score is None:
        notes.append("No recognized feature importance or stability score found.")
    item["recognized_metrics"] = recognized
    item["notes"] = notes or ["Feature stability metrics are recognized."]
    return item


def _data_quality_inventory(evidence: dict[str, dict[str, Any]], path: str | Path | None) -> dict[str, Any]:
    item = _base_inventory("data_quality", evidence["data_quality"], path, ["data_quality"])
    if not item["available"]:
        return item
    payload = evidence["data_quality"].get("payload", {})
    warnings = _as_list(payload.get("warnings") or payload.get("data_quality_warnings"))
    leakage_flags = _as_list(payload.get("leakage_flags") or payload.get("leakage_warnings") or payload.get("leakage"))
    warnings.extend(_audit_warning_items(payload))
    leakage_flags.extend(_leakage_items(payload))
    recognized = {
        "warning_count": len(warnings),
        "leakage_flag_count": len(leakage_flags),
        "sample_warnings": warnings[:5],
        "sample_leakage_flags": leakage_flags[:5],
    }
    notes = []
    if leakage_flags:
        notes.append(f"Leakage flags present: {len(leakage_flags)}.")
    if warnings:
        notes.append(f"Data-quality warnings present: {len(warnings)}.")
    item["recognized_metrics"] = recognized
    item["notes"] = notes or ["No data-quality warnings or leakage flags recognized."]
    return item


def _readiness_status(axes: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> str:
    supplied_errors = [item for item in inventory if item.get("path") and item.get("status") == "unreadable"]
    if supplied_errors or any(axis.get("status") == "blocked" for axis in axes):
        return "blocked_metric_inputs"
    if any(axis.get("status") == "caution" for axis in axes) or any(item.get("status") == "missing" for item in inventory):
        return "metric_inputs_ready_with_cautions"
    return "metric_inputs_ready"


def _commands(
    *,
    model_performance_path: str | Path | None,
    replay_batch_path: str | Path | None,
    feature_report_path: str | Path | None,
    data_quality_path: str | Path | None,
    constraints_path: str | Path | None,
) -> dict[str, str | None]:
    parts = ["python run_agent_pipeline_control_surface.py"]
    if model_performance_path:
        parts.extend(["--model-performance", _quote_arg(model_performance_path)])
    if replay_batch_path:
        parts.extend(["--replay-batch", _quote_arg(replay_batch_path)])
    if feature_report_path:
        parts.extend(["--feature-report", _quote_arg(feature_report_path)])
    if data_quality_path:
        parts.extend(["--data-quality", _quote_arg(data_quality_path)])
    if constraints_path:
        parts.extend(["--constraints", _quote_arg(constraints_path)])
    parts.extend(["--output-dir", "reports/dean_os/pipeline_control_surface"])
    return {
        "pipeline_control_surface": " ".join(parts),
        "pipeline_control_instance_contract": (
            "python run_agent_pipeline_control_instance_contract.py "
            "--pipeline-surface-json reports/dean_os/pipeline_control_surface/latest.json "
            "--architecture-map-json reports/dean_os/current_architecture_map_current/latest.json "
            "--domain-instance-contract-json reports/dean_os/domain_analyst_instance_contract_current/latest.json "
            "--output-dir reports/dean_os/pipeline_control_instance_contract_current"
        ),
    }


def _quote_arg(value: str | Path) -> str:
    text = str(value)
    if " " in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training, hyperparameter search, or replay rerun is executed.",
        "No PipelineControlSurface artifact is written by this gate.",
        "No production config is written.",
        "No learning memory or analyst-weight update is written.",
        "No recommendation, allocation, paper order, broker call, or live trade is generated.",
    ]


def _operator_next_steps(status: str, axes: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> list[str]:
    blocked = [axis["name"] for axis in axes if axis.get("status") == "blocked"]
    caution = [axis["name"] for axis in axes if axis.get("status") == "caution"]
    missing = [item["input_id"] for item in inventory if item.get("status") == "missing"]
    unreadable = [item["input_id"] for item in inventory if item.get("status") == "unreadable"]
    steps = []
    if unreadable:
        steps.append("Fix unreadable supplied input paths first: " + ", ".join(unreadable) + ".")
    if blocked:
        steps.append("Do not pass these inputs to tuning proposals; blocked metric planes: " + ", ".join(blocked) + ".")
    if missing:
        steps.append("Decide whether missing inputs should remain cautions or be supplied before surface refresh: " + ", ".join(missing) + ".")
    if caution:
        steps.append("Review caution planes before widening any experiment bounds: " + ", ".join(caution) + ".")
    if status == "metric_inputs_ready":
        steps.append("Refresh PipelineControlSurface, then build PipelineControlInstanceContract; both remain review-only.")
    else:
        steps.append("Use the command preview only after accepting the blockers/cautions; this gate itself performs no surface refresh.")
    return steps


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
