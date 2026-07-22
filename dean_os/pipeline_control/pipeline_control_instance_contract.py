from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_PIPELINE_CONTROL_SURFACE_JSON = "reports/dean_os/pipeline_control_surface/latest.json"
DEFAULT_ARCHITECTURE_MAP_JSON = "reports/dean_os/current_architecture_map_current/latest.json"
DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON = "reports/dean_os/domain_analyst_instance_contract_current/latest.json"
REQUIRED_METRIC_PLANES = {
    "profitability",
    "risk",
    "validation",
    "feature_stability",
    "data_quality",
    "replay_repeatability",
}


class PipelineControlInstanceContract:
    """Review-only passport for the pipeline-control branch.

    It does not tune, train, write production config, or trade. It only checks
    whether the current saved metric surface is coherent enough for reviewed
    experiment proposals.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_control_instance_contract"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        pipeline_surface_json: str | Path = DEFAULT_PIPELINE_CONTROL_SURFACE_JSON,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        domain_instance_contract_json: str | Path | None = DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        surface = _load_json(pipeline_surface_json)
        architecture = _load_optional_json(architecture_map_json)
        domain_instance = _load_optional_json(domain_instance_contract_json)
        checks = _review_checks(surface=surface, architecture=architecture, domain_instance=domain_instance)
        status = _instance_status(surface, checks)
        payload = {
            "run_id": _run_id("pipeline_control_instance_contract"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_instance_contract",
            "inputs": {
                "pipeline_surface_json": str(pipeline_surface_json),
                "architecture_map_json": str(architecture_map_json) if architecture_map_json else None,
                "domain_instance_contract_json": str(domain_instance_contract_json) if domain_instance_contract_json else None,
            },
            "summary": _summary(status, surface),
            "metric_plane_contract": _metric_plane_contract(surface),
            "fixed_contract_sequence": _fixed_contract_sequence(),
            "review_checks": checks,
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(status, surface, checks),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_control_instance_contract_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_control_instance_contract_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    contract = payload.get("metric_plane_contract", {})
    lines = [
        "# DEAN-OS Pipeline Control Instance Contract",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Instance status: `{summary.get('instance_status')}`",
        f"- Surface status: `{summary.get('surface_status')}`",
        f"- Proposal gate: `{summary.get('proposal_gate_status')}`",
        f"- Metric planes: {summary.get('metric_plane_count')}",
        f"- Blocked planes: {', '.join(summary.get('blocked_metric_planes', [])) or 'none'}",
        f"- Caution planes: {', '.join(summary.get('caution_metric_planes', [])) or 'none'}",
        f"- Can propose reviewed experiments after manual review: {summary.get('can_propose_reviewed_experiments_after_manual_review')}",
        f"- Can run autonomous tuning now: {summary.get('can_run_autonomous_tuning_now')}",
        f"- Can write production config: {summary.get('can_write_production_config')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Metric Plane Contract",
        "",
        f"- Required planes covered: {contract.get('required_planes_covered')}",
        f"- Policy: `{contract.get('allowed_variation', {}).get('policy')}`",
        f"- Max trials: {contract.get('allowed_variation', {}).get('max_trials')}",
        f"- Production writes allowed: {contract.get('allowed_variation', {}).get('production_write_allowed')}",
        "",
        "## Fixed Contract Sequence",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("fixed_contract_sequence", []))
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _summary(status: str, surface_payload: dict[str, Any]) -> dict[str, Any]:
    surface = surface_payload.get("surface", {})
    gate = surface_payload.get("proposal_gate", {})
    axes = surface.get("axes", [])
    blocked = [axis.get("name") for axis in axes if axis.get("status") == "blocked"]
    caution = [axis.get("name") for axis in axes if axis.get("status") == "caution"]
    return {
        "instance_status": status,
        "surface_run_id": surface_payload.get("run_id"),
        "surface_status": surface.get("status"),
        "surface_feasible": surface.get("feasible"),
        "proposal_gate_status": gate.get("status"),
        "metric_plane_count": len(axes),
        "blocked_metric_planes": blocked,
        "caution_metric_planes": caution,
        "can_propose_reviewed_experiments_after_manual_review": status in {
            "pipeline_control_instance_review_ready",
            "pipeline_control_instance_review_ready_with_cautions",
        },
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _metric_plane_contract(surface_payload: dict[str, Any]) -> dict[str, Any]:
    surface = surface_payload.get("surface", {})
    axes = surface.get("axes", [])
    axis_names = {str(axis.get("name")) for axis in axes}
    return {
        "required_planes": sorted(REQUIRED_METRIC_PLANES),
        "available_planes": sorted(axis_names),
        "required_planes_covered": REQUIRED_METRIC_PLANES.issubset(axis_names),
        "axis_status_counts": surface.get("axis_status_counts", {}),
        "allowed_variation": surface.get("allowed_variation", {}),
        "constraints": surface_payload.get("constraints", {}),
        "portable_rule": (
            "For another pipeline experiment family, change saved metric artifacts and constraints; keep "
            "proposal-only gates, production-write bans, leakage checks, replay checks, and review sequence unchanged."
        ),
    }


def _review_checks(
    *,
    surface: dict[str, Any],
    architecture: dict[str, Any] | None,
    domain_instance: dict[str, Any] | None,
) -> list[dict[str, str]]:
    surface_summary = surface.get("surface", {})
    gate = surface.get("proposal_gate", {})
    variation = surface_summary.get("allowed_variation", {})
    axes = surface_summary.get("axes", [])
    axis_names = {str(axis.get("name")) for axis in axes}
    blocked_axes = [str(axis.get("name")) for axis in axes if axis.get("status") == "blocked"]
    checks = [
        _check("pass" if surface.get("mode") == "pipeline_control_surface" else "fail", "pipeline_surface_artifact_type", str(surface.get("mode"))),
        _check("pass" if REQUIRED_METRIC_PLANES.issubset(axis_names) else "fail", "required_metric_planes_present", ", ".join(sorted(axis_names))),
        _check("pass" if not blocked_axes else "fail", "no_blocked_metric_planes", ", ".join(blocked_axes) if blocked_axes else "No blocked metric planes."),
        _check("pass" if variation.get("production_write_allowed") is False else "fail", "allowed_variation_no_production_write", f"production_write_allowed={variation.get('production_write_allowed')!r}."),
        _check("pass" if gate.get("can_change_production_config") is False else "fail", "proposal_gate_no_config_write", f"can_change_production_config={gate.get('can_change_production_config')!r}."),
        _check("pass" if gate.get("status") in {"review_required", "blocked"} else "fail", "proposal_gate_review_or_block", str(gate.get("status"))),
    ]
    if gate.get("can_propose_tuning") is True:
        checks.append(_check("pass", "tuning_is_proposal_only", "Tuning may be proposed only as a reviewed bounded experiment."))
    else:
        checks.append(_check("warn", "tuning_proposals_blocked", str(gate.get("reason"))))
    if architecture:
        arch_summary = architecture.get("summary", {})
        checks.append(_check("pass" if arch_summary.get("can_write_production_config_now") is False else "fail", "architecture_no_production_config_write", "Architecture keeps production config writes disabled."))
        checks.append(_check("pass" if arch_summary.get("can_trade") is False else "fail", "architecture_no_trading", "Architecture keeps trading disabled."))
    if domain_instance:
        domain_summary = domain_instance.get("summary", {})
        checks.append(_check("pass" if domain_summary.get("can_scale_to_other_domains_now") is False else "fail", "domain_branch_not_scaled_by_pipeline", "Pipeline control does not scale domain analysts."))
        checks.append(_check("pass" if domain_summary.get("can_trade") is False else "fail", "domain_branch_no_trading", "Domain branch also keeps trading disabled."))
    return checks


def _instance_status(surface: dict[str, Any], checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "blocked_pipeline_control_instance"
    if surface.get("surface", {}).get("status") == "caution" or any(check["status"] == "warn" for check in checks):
        return "pipeline_control_instance_review_ready_with_cautions"
    return "pipeline_control_instance_review_ready"


def _fixed_contract_sequence() -> list[str]:
    return [
        "saved model/replay/feature/data-quality artifacts -> PipelineControlSurface",
        "PipelineControlSurface -> PipelineControlInstanceContract",
        "blocked planes stop tuning proposals",
        "clear/caution planes allow only bounded reviewed experiment proposals",
        "TuningAgent may propose, but never write production config",
        "model promotion, learning promotion, paper trading, and live trading remain separate gates",
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training or hyperparameter search is executed.",
        "No production config is written.",
        "No learning memory or analyst-weight update is written.",
        "No recommendation, allocation, paper order, broker call, or live trade is generated.",
        "No domain analyst is cloned or scaled by this pipeline contract.",
    ]


def _operator_next_steps(status: str, surface: dict[str, Any], checks: list[dict[str, str]]) -> list[str]:
    failed = [check["code"] for check in checks if check["status"] == "fail"]
    if status == "blocked_pipeline_control_instance":
        blocked_axes = [
            str(axis.get("name"))
            for axis in surface.get("surface", {}).get("axes", [])
            if axis.get("status") == "blocked"
        ]
        steps = ["Do not pass this surface to tuning proposals yet."]
        if blocked_axes:
            steps.append("Fix blocked metric planes first: " + ", ".join(blocked_axes) + ".")
        if failed:
            steps.append("Failed contract checks: " + ", ".join(failed) + ".")
        return steps
    steps = ["Manually review the control surface before allowing proposal-only tuning."]
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    if warnings:
        steps.append("Review caution checks before widening bounds: " + ", ".join(warnings) + ".")
    steps.append("After manual acceptance, connect this contract to the orchestrator; do not write config or trade.")
    return steps


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return _load_json(resolved)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
