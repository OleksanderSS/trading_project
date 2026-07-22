from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import clamp, json_ready

DEFAULT_CONSTRAINTS: dict[str, Any] = {
    "min_total_return": 0.0,
    "min_pnl": 0.0,
    "min_sharpe": 0.0,
    "max_drawdown": 0.25,
    "min_validation_score": 0.55,
    "max_train_test_gap": 0.15,
    "min_sample_count": 50,
    "max_feature_concentration": 0.35,
    "max_feature_weight_abs": 0.6,
    "max_unstable_features": 0,
    "min_feature_stability_score": 0.7,
    "max_data_quality_warnings": 0,
    "max_leakage_flags": 0,
    "min_clear_replay_hit_rate": 0.55,
    "max_quality_blocked_replay_runs": 0,
    "min_clear_replay_runs": 5,
}


class PipelineControlSurface:
    """Builds a safe variation surface for pipeline tuning proposals.

    This is intentionally not a tuner. It defines whether tuning experiments are
    allowed to be proposed and which guardrails must be active.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_control_surface"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        model_performance_path: str | Path | None = None,
        replay_batch_path: str | Path | None = None,
        feature_report_path: str | Path | None = None,
        data_quality_path: str | Path | None = None,
        constraints_path: str | Path | None = None,
        constraints: dict[str, Any] | None = None,
        save: bool = True,
        *,
        model_performance_payload: dict[str, Any] | None = None,
        replay_batch_payload: dict[str, Any] | None = None,
        feature_report_payload: dict[str, Any] | None = None,
        data_quality_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the control surface from saved artifacts or in-memory evidence.

        Direct payloads take precedence over paths. This is the canonical path
        for the post-pipeline assessment because it prevents the current run's
        metrics from being lost behind stale or absent files.
        """

        merged_constraints = _load_constraints(constraints_path, constraints)
        evidence = {
            "model_performance": _resolve_evidence(model_performance_path, model_performance_payload),
            "replay_batch": _resolve_evidence(replay_batch_path, replay_batch_payload),
            "feature_report": _resolve_evidence(feature_report_path, feature_report_payload),
            "data_quality": _resolve_evidence(data_quality_path, data_quality_payload),
        }
        axes = [
            _profitability_axis(evidence, merged_constraints),
            _risk_axis(evidence, merged_constraints),
            _validation_axis(evidence, merged_constraints),
            _feature_stability_axis(evidence, merged_constraints),
            _data_quality_axis(evidence, merged_constraints),
            _replay_axis(evidence, merged_constraints),
        ]
        status = _surface_status(axes)
        payload = {
            "run_id": "pipeline_control_surface_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_surface",
            "inputs": {
                "model_performance_path": str(model_performance_path) if model_performance_path else None,
                "replay_batch_path": str(replay_batch_path) if replay_batch_path else None,
                "feature_report_path": str(feature_report_path) if feature_report_path else None,
                "data_quality_path": str(data_quality_path) if data_quality_path else None,
                "constraints_path": str(constraints_path) if constraints_path else None,
                "in_memory_payloads": {
                    "model_performance": model_performance_payload is not None,
                    "replay_batch": replay_batch_payload is not None,
                    "feature_report": feature_report_payload is not None,
                    "data_quality": data_quality_payload is not None,
                },
            },
            "constraints": merged_constraints,
            "surface": {
                "status": status,
                "feasible": status != "blocked",
                "axis_status_counts": dict(Counter(axis["status"] for axis in axes)),
                "axes": axes,
                "allowed_variation": _allowed_variation(status, axes),
            },
            "proposal_gate": _proposal_gate(status, axes),
            "recommendations": _recommendations(status, axes),
        }
        if save:
            self.save_report(payload)
        return payload

    def save_report(self, payload: dict[str, Any]) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = payload["run_id"]
        json_path = self.output_dir / f"{run_id}.json"
        md_path = self.output_dir / f"{run_id}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        paths = {"json": json_path, "markdown": md_path, "latest_json": latest_json, "latest_markdown": latest_md}
        payload["saved_paths"] = {key: str(value) for key, value in paths.items()}
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False)
        rendered_md = render_pipeline_control_surface_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return paths


def render_pipeline_control_surface_markdown(payload: dict[str, Any]) -> str:
    surface = payload.get("surface", {})
    gate = payload.get("proposal_gate", {})
    lines = [
        "# DEAN-OS Pipeline Control Surface",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Surface status: `{surface.get('status')}`",
        f"- Feasible: {surface.get('feasible')}",
        f"- Proposal gate: `{gate.get('status')}`",
        f"- Can propose tuning: {gate.get('can_propose_tuning')}",
        "",
        "## Axes",
        "",
    ]
    for axis in surface.get("axes", []):
        lines.append(f"- `{axis.get('name')}`: {axis.get('status')} score={axis.get('score')}")
        for reason in axis.get("reasons", [])[:3]:
            lines.append(f"  - {reason}")
    lines.extend(["", "## Allowed Variation", ""])
    variation = surface.get("allowed_variation", {})
    lines.append(f"- Policy: `{variation.get('policy')}`")
    lines.append(f"- Production writes allowed: {variation.get('production_write_allowed')}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _load_constraints(path: str | Path | None, overrides: dict[str, Any] | None) -> dict[str, Any]:
    constraints = dict(DEFAULT_CONSTRAINTS)
    if path:
        constraints.update(_load_payload(Path(path)))
    if overrides:
        constraints.update(overrides)
    return constraints


def _resolve_evidence(
    path: str | Path | None,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if payload is not None:
        if not isinstance(payload, dict):
            return {
                "available": False,
                "payload": {},
                "path": None,
                "source": "in_memory",
                "error": "In-memory evidence payload must be a mapping.",
            }
        return {
            "available": bool(payload),
            "payload": payload,
            "path": None,
            "source": "in_memory",
        }
    resolved = _load_optional_payload(path)
    resolved["source"] = "path" if path else "missing"
    return resolved


def _load_optional_payload(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"available": False, "payload": {}, "path": None}
    resolved = Path(path)
    if not resolved.exists():
        return {"available": False, "payload": {}, "path": str(resolved), "error": f"Missing file: {resolved}"}
    try:
        return {"available": True, "payload": _load_payload(resolved), "path": str(resolved)}
    except Exception as exc:
        return {"available": False, "payload": {}, "path": str(resolved), "error": f"{type(exc).__name__}: {exc}"}


def _load_payload(path: Path) -> dict[str, Any]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    try:
        return DeanPaths.load_json(path)
    except Exception:
        # Fallback to CSV if JSON fails
        try:
            frame = DeanPaths.load_data_file(path)
            if frame.empty:
                return {}
            return frame.iloc[-1].to_dict()
        except Exception as exc:
            raise ValueError(f"Failed to load payload from {path}: {exc}")


def _profitability_axis(evidence: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    metrics = _model_metrics(evidence)
    replay_summary = _replay_summary(evidence)
    total_return = _first_number(metrics, ["total_return", "return", "realized_return", "pnl_pct"])
    pnl = _first_number(metrics, ["pnl", "profit", "net_profit"])
    sharpe = _first_number(metrics, ["sharpe", "sharpe_ratio"])
    if total_return is None:
        total_return = _first_number(replay_summary, ["clear_average_realized_return", "average_realized_return"])
    reasons: list[str] = []
    status = "clear"
    if total_return is None and pnl is None and sharpe is None:
        status = "caution"
        reasons.append("No profitability metrics were supplied; tuning cannot optimize an invisible objective.")
    if total_return is not None and total_return < float(constraints["min_total_return"]):
        status = "blocked"
        reasons.append(f"Total return {total_return:.4f} is below floor {float(constraints['min_total_return']):.4f}.")
    if pnl is not None and pnl < float(constraints["min_pnl"]):
        status = "blocked"
        reasons.append(f"PnL {pnl:.4f} is below floor {float(constraints['min_pnl']):.4f}.")
    if sharpe is not None and sharpe < float(constraints["min_sharpe"]):
        status = "blocked"
        reasons.append(f"Sharpe {sharpe:.4f} is below floor {float(constraints['min_sharpe']):.4f}.")
    if not reasons:
        reasons.append("Profitability floor is satisfied or supported by replay proxy evidence.")
    score_inputs = [value for value in [total_return, pnl, sharpe] if value is not None]
    score = clamp(0.55 + (sum(score_inputs) / len(score_inputs) if score_inputs else 0.0), 0.0, 1.0)
    return _axis("profitability", status, score, {"total_return": total_return, "pnl": pnl, "sharpe": sharpe}, constraints, reasons)


def _risk_axis(evidence: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    metrics = _model_metrics(evidence)
    drawdown = _first_number(metrics, ["max_drawdown", "maximum_drawdown", "mdd", "drawdown"])
    status = "clear"
    reasons: list[str] = []
    if drawdown is None:
        status = "caution"
        reasons.append("No max drawdown metric supplied; downside boundary is not proven.")
    else:
        drawdown = abs(drawdown)
        if drawdown > float(constraints["max_drawdown"]):
            status = "blocked"
            reasons.append(f"Max drawdown {drawdown:.4f} exceeds cap {float(constraints['max_drawdown']):.4f}.")
        else:
            reasons.append("Drawdown is inside the configured cap.")
    score = 0.5 if drawdown is None else clamp(1.0 - drawdown / max(float(constraints["max_drawdown"]), 1e-9), 0.0, 1.0)
    return _axis("risk", status, score, {"max_drawdown": drawdown}, constraints, reasons)


def _validation_axis(evidence: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    metrics = _model_metrics(evidence)
    train_score = _first_number(metrics, ["train_score", "training_score", "in_sample_score"])
    validation_score = _first_number(metrics, ["validation_score", "val_score", "test_score", "out_of_sample_score"])
    sample_count = _first_number(metrics, ["sample_count", "n_samples", "test_samples", "validation_samples", "observations"])
    gap = abs(train_score - validation_score) if train_score is not None and validation_score is not None else None
    status = "clear"
    reasons: list[str] = []
    if validation_score is None:
        status = "caution"
        reasons.append("No validation/test score supplied.")
    elif validation_score < float(constraints["min_validation_score"]):
        status = "blocked"
        reasons.append(f"Validation score {validation_score:.4f} is below floor {float(constraints['min_validation_score']):.4f}.")
    if gap is None:
        status = _worse_status(status, "caution")
        reasons.append("Train/test gap cannot be computed.")
    elif gap > float(constraints["max_train_test_gap"]):
        status = "blocked"
        reasons.append(f"Train/test gap {gap:.4f} exceeds cap {float(constraints['max_train_test_gap']):.4f}.")
    if sample_count is None:
        status = _worse_status(status, "caution")
        reasons.append("Validation sample count is missing.")
    elif sample_count < float(constraints["min_sample_count"]):
        status = "blocked"
        reasons.append(f"Sample count {sample_count:.0f} is below floor {float(constraints['min_sample_count']):.0f}.")
    if not reasons:
        reasons.append("Validation score, train/test gap, and sample count are inside bounds.")
    score_parts = [
        validation_score if validation_score is not None else 0.45,
        1.0 - gap / max(float(constraints["max_train_test_gap"]), 1e-9) if gap is not None else 0.45,
    ]
    score = clamp(sum(score_parts) / len(score_parts), 0.0, 1.0)
    return _axis(
        "validation",
        status,
        score,
        {"train_score": train_score, "validation_score": validation_score, "train_test_gap": gap, "sample_count": sample_count},
        constraints,
        reasons,
    )


def _feature_stability_axis(evidence: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    feature_payload = evidence["feature_report"]["payload"]
    if not evidence["feature_report"]["available"]:
        return _axis(
            "feature_stability",
            "caution",
            0.45,
            {},
            constraints,
            ["No feature stability report supplied; feature-weight sanity is not proven."],
        )
    importances = _extract_feature_importances(feature_payload)
    concentration = _feature_concentration(importances)
    max_abs = max((abs(value) for value in importances.values()), default=None)
    stability_score = _first_number(feature_payload, ["feature_stability_score", "stability_score"])
    unstable_count = _unstable_feature_count(feature_payload)
    status = "clear"
    reasons: list[str] = []
    if concentration is not None and concentration > float(constraints["max_feature_concentration"]):
        status = "blocked"
        reasons.append(f"Feature concentration {concentration:.4f} exceeds cap {float(constraints['max_feature_concentration']):.4f}.")
    if max_abs is not None and max_abs > float(constraints["max_feature_weight_abs"]):
        status = "blocked"
        reasons.append(f"Max feature weight {max_abs:.4f} exceeds cap {float(constraints['max_feature_weight_abs']):.4f}.")
    if unstable_count > int(constraints["max_unstable_features"]):
        status = "blocked"
        reasons.append(f"Unstable feature count {unstable_count} exceeds cap {int(constraints['max_unstable_features'])}.")
    if stability_score is not None and stability_score < float(constraints["min_feature_stability_score"]):
        status = "blocked"
        reasons.append(f"Feature stability score {stability_score:.4f} is below floor {float(constraints['min_feature_stability_score']):.4f}.")
    if not importances and stability_score is None and unstable_count == 0:
        status = "caution"
        reasons.append("Feature report supplied, but no recognized feature importance or stability metrics were found.")
    if not reasons:
        reasons.append("Feature weights and stability metrics are inside bounds.")
    score = stability_score if stability_score is not None else 1.0 - (concentration or 0.0)
    return _axis(
        "feature_stability",
        status,
        clamp(score, 0.0, 1.0),
        {
            "feature_count": len(importances),
            "feature_concentration": concentration,
            "max_feature_weight_abs": max_abs,
            "feature_stability_score": stability_score,
            "unstable_feature_count": unstable_count,
        },
        constraints,
        reasons,
    )


def _data_quality_axis(evidence: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    payload = evidence["data_quality"]["payload"]
    warnings = _as_list(payload.get("warnings") or payload.get("data_quality_warnings"))
    leakage_flags = _as_list(payload.get("leakage_flags") or payload.get("leakage_warnings") or payload.get("leakage"))
    warnings.extend(_audit_warning_items(payload))
    leakage_flags.extend(_leakage_items(payload))
    status = "clear"
    reasons: list[str] = []
    if not evidence["data_quality"]["available"]:
        status = "caution"
        reasons.append("No data-quality report supplied; leakage/data warnings are unknown.")
    if len(leakage_flags) > int(constraints["max_leakage_flags"]):
        status = "blocked"
        reasons.append(f"Leakage flags present: {len(leakage_flags)}.")
    if len(warnings) > int(constraints["max_data_quality_warnings"]):
        status = _worse_status(status, "caution")
        reasons.append(f"Data-quality warnings present: {len(warnings)}.")
    if not reasons:
        reasons.append("No data-quality or leakage warnings supplied.")
    score = clamp(1.0 - 0.25 * len(warnings) - 0.6 * len(leakage_flags), 0.0, 1.0)
    return _axis("data_quality", status, score, {"warnings": warnings, "leakage_flags": leakage_flags}, constraints, reasons)


def _replay_axis(evidence: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    summary = _replay_summary(evidence)
    if not summary:
        return _axis(
            "replay_repeatability",
            "caution",
            0.45,
            {},
            constraints,
            ["No historical replay batch supplied; repeatability is unknown."],
        )
    clear_hit_rate = _first_number(summary, ["clear_hit_rate"])
    clear_runs = _first_number(summary, ["clear_evaluated_runs"])
    blocked_runs = _first_number(summary, ["quality_blocked_runs"]) or 0
    status = "clear"
    reasons: list[str] = []
    if blocked_runs > float(constraints["max_quality_blocked_replay_runs"]):
        status = "blocked"
        reasons.append(f"Replay quality-blocked runs {blocked_runs:.0f} exceed cap {float(constraints['max_quality_blocked_replay_runs']):.0f}.")
    if clear_runs is None or clear_runs < float(constraints["min_clear_replay_runs"]):
        status = _worse_status(status, "caution")
        reasons.append("Clean replay sample is too small for calibration.")
    if clear_hit_rate is None:
        status = _worse_status(status, "caution")
        reasons.append("Clear replay hit rate is missing.")
    elif clear_hit_rate < float(constraints["min_clear_replay_hit_rate"]):
        status = "blocked"
        reasons.append(f"Clear replay hit rate {clear_hit_rate:.4f} is below floor {float(constraints['min_clear_replay_hit_rate']):.4f}.")
    if not reasons:
        reasons.append("Replay repeatability is inside configured bounds.")
    score = clear_hit_rate if clear_hit_rate is not None else 0.45
    return _axis(
        "replay_repeatability",
        status,
        clamp(score, 0.0, 1.0),
        {"clear_hit_rate": clear_hit_rate, "clear_evaluated_runs": clear_runs, "quality_blocked_runs": blocked_runs},
        constraints,
        reasons,
    )


def _axis(
    name: str,
    status: str,
    score: float,
    metrics: dict[str, Any],
    constraints: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "score": round(clamp(score, 0.0, 1.0), 6),
        "metrics": json_ready(metrics),
        "constraints": _constraints_for_axis(name, constraints),
        "reasons": reasons,
    }


def _constraints_for_axis(name: str, constraints: dict[str, Any]) -> dict[str, Any]:
    keys_by_axis = {
        "profitability": ["min_total_return", "min_pnl", "min_sharpe"],
        "risk": ["max_drawdown"],
        "validation": ["min_validation_score", "max_train_test_gap", "min_sample_count"],
        "feature_stability": [
            "max_feature_concentration",
            "max_feature_weight_abs",
            "max_unstable_features",
            "min_feature_stability_score",
        ],
        "data_quality": ["max_data_quality_warnings", "max_leakage_flags"],
        "replay_repeatability": ["min_clear_replay_hit_rate", "max_quality_blocked_replay_runs", "min_clear_replay_runs"],
    }
    return {key: constraints[key] for key in keys_by_axis.get(name, [])}


def _surface_status(axes: list[dict[str, Any]]) -> str:
    statuses = {axis["status"] for axis in axes}
    if "blocked" in statuses:
        return "blocked"
    if "caution" in statuses:
        return "caution"
    return "clear"


def _allowed_variation(status: str, axes: list[dict[str, Any]]) -> dict[str, Any]:
    common = {
        "production_write_allowed": False,
        "required_guards": [
            "proposal_only",
            "human_review_required",
            "locked_holdout",
            "walk_forward_validation",
            "leakage_scan",
            "feature_stability_check",
            "regime_slice_report",
        ],
        "blocked_axes": [axis["name"] for axis in axes if axis["status"] == "blocked"],
        "caution_axes": [axis["name"] for axis in axes if axis["status"] == "caution"],
    }
    if status == "blocked":
        return {
            **common,
            "policy": "no_tuning_experiment",
            "max_trials": 0,
            "parameter_delta_pct": 0.0,
            "reason": "At least one control-surface axis is blocked.",
        }
    if status == "caution":
        return {
            **common,
            "policy": "limited_reviewed_experiment",
            "max_trials": 10,
            "parameter_delta_pct": 0.1,
            "max_feature_additions": 1,
            "max_model_candidates": 2,
        }
    return {
        **common,
        "policy": "reviewed_experiment",
        "max_trials": 25,
        "parameter_delta_pct": 0.2,
        "max_feature_additions": 3,
        "max_model_candidates": 3,
    }


def _proposal_gate(status: str, axes: list[dict[str, Any]]) -> dict[str, Any]:
    if status == "blocked":
        return {
            "status": "blocked",
            "can_propose_tuning": False,
            "can_change_production_config": False,
            "reason": "Blocked control-surface axes must be fixed before tuning proposals.",
        }
    return {
        "status": "review_required",
        "can_propose_tuning": True,
        "can_change_production_config": False,
        "reason": "Tuning can only be proposed as a reviewed, bounded experiment.",
    }


def _recommendations(status: str, axes: list[dict[str, Any]]) -> list[str]:
    recommendations = [
        "Treat this as the allowed search surface for proposal-only tuning, not as an automatic tuner.",
        "Production config writes remain forbidden until review, experiment evidence, and promotion gates pass.",
    ]
    blocked = [axis for axis in axes if axis["status"] == "blocked"]
    caution = [axis for axis in axes if axis["status"] == "caution"]
    if blocked:
        recommendations.append("Fix blocked axes before letting TuningAgent propose experiments.")
    if caution:
        recommendations.append("Collect missing caution-axis evidence before widening tuning bounds.")
    if status != "blocked":
        recommendations.append("Next step: feed this surface into TuningAgent so proposals include explicit bounds.")
    return recommendations


def _model_metrics(evidence: dict[str, Any]) -> dict[str, Any]:
    payload = evidence["model_performance"]["payload"]
    if not isinstance(payload, dict):
        return {}
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return {**_flatten(payload), **_flatten(metrics)}
    return _flatten(payload)


def _replay_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    payload = evidence["replay_batch"]["payload"]
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else {}


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = _normalize_key(str(key))
        full_key = f"{prefix}.{normalized_key}" if prefix else normalized_key
        if isinstance(value, dict):
            flattened.update(_flatten(value, full_key))
        else:
            flattened[normalized_key] = value
            flattened[full_key] = value
    return flattened


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _first_number(payload: dict[str, Any], keys: list[str]) -> float | None:
    lowered = {_normalize_key(str(key)): value for key, value in payload.items()}
    for key in keys:
        value = lowered.get(_normalize_key(key))
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_feature_importances(payload: dict[str, Any]) -> dict[str, float]:
    candidates = [
        payload.get("feature_importance"),
        payload.get("feature_importances"),
        payload.get("feature_weights"),
        payload.get("weights"),
    ]
    for candidate in candidates:
        parsed = _parse_feature_importance(candidate)
        if parsed:
            return parsed
    return {}


def _parse_feature_importance(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        parsed = {}
        for key, item in value.items():
            try:
                parsed[str(key)] = float(item)
            except (TypeError, ValueError):
                continue
        return parsed
    if isinstance(value, list):
        parsed = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            name = item.get("feature") or item.get("name") or item.get("column")
            raw_weight = item.get("importance", item.get("weight", item.get("value")))
            if not name:
                continue
            try:
                parsed[str(name)] = float(raw_weight)
            except (TypeError, ValueError):
                continue
        return parsed
    return {}


def _feature_concentration(importances: dict[str, float]) -> float | None:
    weights = [abs(value) for value in importances.values()]
    total = sum(weights)
    if not total:
        return None
    return max(weights) / total


def _unstable_feature_count(payload: dict[str, Any]) -> int:
    for key in ["unstable_feature_count", "unstable_features_count"]:
        value = payload.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    unstable = payload.get("unstable_features")
    if isinstance(unstable, list):
        return len(unstable)
    return 0


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [key for key, item in value.items() if item]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _audit_warning_items(payload: dict[str, Any]) -> list[str]:
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    warnings: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", ""))
        rule_id = str(item.get("rule_id", "unknown_rule"))
        file_path = str(item.get("file", "unknown_file"))
        if severity in {"P0", "P1", "P2"}:
            warnings.append(f"{severity}:{rule_id}:{file_path}")
    return warnings


def _leakage_items(payload: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    items = payload.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            text = json.dumps(item, ensure_ascii=False).lower()
            if _looks_leaky(text):
                flags.append(str(item.get("rule_id") or item.get("problem") or item))
    for column in _walk_columns(payload):
        if _looks_leaky(str(column).lower()):
            flags.append(f"leaky_column:{column}")
    return _unique(flags)


def _walk_columns(value: Any) -> list[str]:
    columns: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "columns" and isinstance(item, list):
                columns.extend(str(column) for column in item)
            else:
                columns.extend(_walk_columns(item))
    elif isinstance(value, list):
        for item in value:
            columns.extend(_walk_columns(item))
    return columns


def _looks_leaky(text: str) -> bool:
    tokens = ("target", "future", "_after_", "after_", "leak", "label", "prediction", "predicted")
    return any(token in text for token in tokens)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _worse_status(left: str, right: str) -> str:
    order = {"clear": 0, "caution": 1, "blocked": 2}
    return left if order[left] >= order[right] else right
