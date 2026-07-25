from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_PIPELINE_METRIC_INPUT_READINESS_JSON = "reports/dean_os/pipeline_metric_input_readiness_gate_current/latest.json"
DEFAULT_PIPELINE_CONTROL_INSTANCE_JSON = "reports/dean_os/pipeline_control_instance_contract_current/latest.json"
DEFAULT_MODEL_PERFORMANCE_REPORT_JSON = "reports/dean_os/model_performance/smoke.json"
DEFAULT_FEATURE_REPORT_JSON = None
DEFAULT_DATA_QUALITY_JSON = "diagnostic_reports/feature_lineage_report_current_cache.json"


class PipelineControlCautionReviewPacket:
    """Review-only packet for pipeline-control caution planes.

    This packet does not clear cautions by inventing data. It records which
    evidence is missing, which saved artifacts are useful, and what remains
    forbidden before any tuning or orchestrator integration.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_control_caution_review_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        pipeline_metric_input_readiness_json: str | Path = DEFAULT_PIPELINE_METRIC_INPUT_READINESS_JSON,
        pipeline_control_instance_json: str | Path = DEFAULT_PIPELINE_CONTROL_INSTANCE_JSON,
        model_performance_report_json: str | Path | None = DEFAULT_MODEL_PERFORMANCE_REPORT_JSON,
        feature_report_json: str | Path | None = DEFAULT_FEATURE_REPORT_JSON,
        data_quality_json: str | Path | None = DEFAULT_DATA_QUALITY_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        readiness = _load_json(pipeline_metric_input_readiness_json)
        instance = _load_json(pipeline_control_instance_json)
        optional_artifacts = {
            "model_performance_report": _load_optional_json(model_performance_report_json),
            "feature_report": _load_optional_json(feature_report_json),
            "data_quality": _load_optional_json(data_quality_json),
        }
        plane_reviews = _plane_reviews(readiness, instance, optional_artifacts)
        checks = _review_checks(readiness, instance, plane_reviews)
        status = _caution_review_status(readiness, instance, checks)
        payload = {
            "run_id": _run_id("pipeline_control_caution_review_packet"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_caution_review_packet",
            "inputs": {
                "pipeline_metric_input_readiness_json": str(pipeline_metric_input_readiness_json),
                "pipeline_control_instance_json": str(pipeline_control_instance_json),
                "model_performance_report_json": str(model_performance_report_json) if model_performance_report_json else None,
                "feature_report_json": str(feature_report_json) if feature_report_json else None,
                "data_quality_json": str(data_quality_json) if data_quality_json else None,
            },
            "summary": _summary(status, readiness, instance, plane_reviews),
            "artifact_triage": _artifact_triage(readiness, instance, optional_artifacts),
            "caution_plane_reviews": plane_reviews,
            "review_checks": checks,
            "excluded_artifact_classes": _excluded_artifact_classes(),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(status, plane_reviews),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_control_caution_review_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_control_caution_review_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Pipeline Control Caution Review Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Caution review status: `{summary.get('caution_review_status')}`",
        f"- Blocked planes: {', '.join(summary.get('blocked_metric_planes', [])) or 'none'}",
        f"- Caution planes: {', '.join(summary.get('caution_metric_planes', [])) or 'none'}",
        f"- Missing evidence planes: {', '.join(summary.get('missing_evidence_planes', [])) or 'none'}",
        f"- Can propose reviewed experiments after manual caution acceptance: {summary.get('can_propose_reviewed_experiments_after_manual_caution_acceptance')}",
        f"- Can run autonomous tuning now: {summary.get('can_run_autonomous_tuning_now')}",
        f"- Can write production config: {summary.get('can_write_production_config')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Caution Plane Reviews",
        "",
    ]
    for review in payload.get("caution_plane_reviews", []):
        lines.append(f"- `{review.get('plane_id')}`: {review.get('evidence_status')}")
        for item in review.get("missing_evidence", [])[:4]:
            lines.append(f"  - missing: {item}")
        for action in review.get("required_next_evidence", [])[:4]:
            lines.append(f"  - next: {action}")

    lines.extend(["", "## Artifact Triage", ""])
    for item in payload.get("artifact_triage", []):
        lines.append(f"- `{item.get('artifact_id')}`: {item.get('triage_status')} - {item.get('role')}")

    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")

    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))

    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _summary(status: str, readiness: dict[str, Any], instance: dict[str, Any], plane_reviews: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = _unique(
        _summary_list(readiness, "blocked_metric_planes")
        + _summary_list(instance, "blocked_metric_planes")
    )
    caution = _unique(
        _summary_list(readiness, "caution_metric_planes")
        + _summary_list(instance, "caution_metric_planes")
    )
    missing_evidence = [review["plane_id"] for review in plane_reviews if review.get("evidence_status") != "sufficient_to_clear"]
    safe = _safe_pipeline_flags(readiness.get("summary", {})) and _safe_pipeline_flags(instance.get("summary", {}))
    return {
        "caution_review_status": status,
        "readiness_status": readiness.get("summary", {}).get("readiness_status"),
        "instance_status": instance.get("summary", {}).get("instance_status"),
        "blocked_metric_planes": blocked,
        "caution_metric_planes": caution,
        "missing_evidence_planes": missing_evidence,
        "caution_plane_count": len(caution),
        "can_propose_reviewed_experiments_after_manual_caution_acceptance": safe and not blocked,
        "can_clear_cautions_with_current_artifacts": not missing_evidence and not blocked,
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _artifact_triage(
    readiness: dict[str, Any],
    instance: dict[str, Any],
    optional_artifacts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": "pipeline_metric_input_readiness",
            "path": readiness.get("saved_paths", {}).get("latest_json") or readiness.get("inputs", {}).get("pipeline_metric_input_readiness_json"),
            "triage_status": "authoritative_for_current_plane_states",
            "role": "Identifies clear, caution, and blocked metric planes; does not itself clear missing empirical evidence.",
            "acceptable_to_clear_caution_planes": False,
        },
        {
            "artifact_id": "pipeline_control_instance_contract",
            "path": instance.get("saved_paths", {}).get("latest_json") or instance.get("inputs", {}).get("pipeline_control_instance_json"),
            "triage_status": "authoritative_for_proposal_only_guardrails",
            "role": "Proves review-only boundaries and production-write bans for the current surface.",
            "acceptable_to_clear_caution_planes": False,
        },
        _model_performance_report_triage(optional_artifacts["model_performance_report"]),
        _feature_report_triage(optional_artifacts["feature_report"]),
        _data_quality_triage(optional_artifacts["data_quality"]),
    ]


def _model_performance_report_triage(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return {
            "artifact_id": "model_performance_report",
            "path": artifact.get("path"),
            "triage_status": "optional_missing",
            "role": "Can corroborate whether model metrics are absent, stale, or below threshold.",
            "acceptable_to_clear_caution_planes": False,
        }
    payload = artifact.get("payload", {})
    snapshot = payload.get("report", {}).get("metrics_snapshot", {})
    metrics = snapshot.get("metrics") if isinstance(snapshot, dict) else None
    failures = snapshot.get("threshold_failures") if isinstance(snapshot, dict) else []
    if not metrics and isinstance(payload.get("metrics"), dict):
        metrics = payload.get("metrics")
        failures = payload.get("threshold_failures", [])
    has_metrics = bool(metrics)
    return {
        "artifact_id": "model_performance_report",
        "path": artifact.get("path"),
        "triage_status": "metric_source" if has_metrics else "warning_evidence_only",
        "role": "Useful as a warning artifact; only clears risk/validation if it contains recognized drawdown and holdout metrics.",
        "recognized_metric_count": len(metrics or {}),
        "threshold_failures": failures or [],
        "acceptable_to_clear_caution_planes": has_metrics,
    }


def _feature_report_triage(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return {
            "artifact_id": "feature_report",
            "path": artifact.get("path"),
            "triage_status": "missing",
            "role": "Required to clear feature_stability cautions.",
            "acceptable_to_clear_caution_planes": False,
        }
    payload = artifact.get("payload", {})
    has_stability = any(
        key in payload
        for key in ["feature_importance", "feature_importances", "feature_stability_score", "stability_score", "unstable_features"]
    )
    return {
        "artifact_id": "feature_report",
        "path": artifact.get("path"),
        "triage_status": "metric_source" if has_stability else "unrecognized_feature_report",
        "role": "Clears feature_stability only if it contains feature importances, concentration, or stability metrics.",
        "acceptable_to_clear_caution_planes": has_stability,
    }


def _data_quality_triage(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return {
            "artifact_id": "data_quality",
            "path": artifact.get("path"),
            "triage_status": "optional_missing",
            "role": "Useful for data_quality/leakage, not for drawdown, validation, or feature stability.",
            "acceptable_to_clear_caution_planes": False,
        }
    return {
        "artifact_id": "data_quality",
        "path": artifact.get("path"),
        "triage_status": "supporting_clear_data_quality",
        "role": "Can support data_quality status; cannot substitute for risk, validation, or feature stability evidence.",
        "acceptable_to_clear_caution_planes": False,
    }


def _plane_reviews(
    readiness: dict[str, Any],
    instance: dict[str, Any],
    optional_artifacts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    caution_planes = _unique(
        _summary_list(readiness, "caution_metric_planes")
        + _summary_list(instance, "caution_metric_planes")
        + _summary_list(readiness, "blocked_metric_planes")
        + _summary_list(instance, "blocked_metric_planes")
    )
    if not caution_planes:
        return []
    axes = {str(axis.get("name")): axis for axis in readiness.get("metric_plane_readiness", [])}
    inventory = {str(item.get("input_id")): item for item in readiness.get("input_inventory", [])}
    reviews = []
    for plane_id in caution_planes:
        reviews.append(_single_plane_review(plane_id, axes.get(plane_id, {}), inventory, optional_artifacts))
    return reviews


def _single_plane_review(
    plane_id: str,
    axis: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    optional_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if plane_id == "risk":
        return _risk_review(axis, inventory, optional_artifacts)
    if plane_id == "validation":
        return _validation_review(axis, inventory, optional_artifacts)
    if plane_id == "feature_stability":
        return _feature_stability_review(axis, inventory, optional_artifacts)
    return {
        "plane_id": plane_id,
        "current_plane_status": axis.get("status"),
        "current_reasons": axis.get("reasons", []),
        "evidence_status": "needs_manual_review",
        "missing_evidence": ["No specialized caution-review rule is defined for this plane."],
        "required_next_evidence": ["Review the upstream plane reasons and add a purpose-built evidence artifact before clearing it."],
        "accepted_current_use": "May block or caution proposal review; cannot be auto-cleared.",
    }


def _risk_review(
    axis: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    optional_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = axis.get("metrics", {})
    max_drawdown = metrics.get("max_drawdown")
    model_inventory = inventory.get("model_performance", {})
    perf_triage = _model_performance_report_triage(optional_artifacts["model_performance_report"])
    missing = []
    if max_drawdown is None:
        missing.append("max_drawdown")
    if not model_inventory.get("recognized_metrics", {}).get("max_drawdown_present"):
        missing.append("recognized drawdown metric in model performance input")
    return {
        "plane_id": "risk",
        "current_plane_status": axis.get("status"),
        "current_reasons": axis.get("reasons", []),
        "observed_metrics": metrics,
        "supporting_artifacts": ["pipeline_metric_input_readiness", "model_performance_report"],
        "model_performance_report_status": perf_triage["triage_status"],
        "evidence_status": "sufficient_to_clear" if not missing else "missing_required_metric",
        "missing_evidence": missing,
        "required_next_evidence": [
            "Provide an evaluation JSON with max_drawdown calculated on the same locked evaluation window.",
            "Keep drawdown separate from code-audit risk findings; code audits cannot clear downside evidence.",
        ],
        "accepted_current_use": "Can remain a caution for a manually reviewed tiny proposal; cannot justify autonomous tuning.",
    }


def _validation_review(
    axis: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    optional_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = axis.get("metrics", {})
    required = ["train_score", "validation_score", "sample_count"]
    missing = [key for key in required if metrics.get(key) is None]
    if metrics.get("train_test_gap") is None:
        missing.append("train_test_gap")
    model_inventory = inventory.get("model_performance", {})
    recognized = model_inventory.get("recognized_metrics", {})
    if not (recognized.get("validation_score_present") and recognized.get("train_score_present") and recognized.get("sample_count_present")):
        missing.append("recognized train/validation/sample-count metrics in model performance input")
    perf_triage = _model_performance_report_triage(optional_artifacts["model_performance_report"])
    return {
        "plane_id": "validation",
        "current_plane_status": axis.get("status"),
        "current_reasons": axis.get("reasons", []),
        "observed_metrics": metrics,
        "supporting_artifacts": ["pipeline_metric_input_readiness", "model_performance_report"],
        "model_performance_report_status": perf_triage["triage_status"],
        "evidence_status": "sufficient_to_clear" if not missing else "missing_holdout_metrics",
        "missing_evidence": _unique(missing),
        "required_next_evidence": [
            "Provide locked holdout or walk-forward metrics with train_score, validation_score/test_score, train_test_gap, and sample_count.",
            "Do not reuse replay hit-rate as a substitute for holdout validation quality.",
        ],
        "accepted_current_use": "Can remain a caution for review-only experiments; cannot support model promotion.",
    }


def _feature_stability_review(
    axis: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    optional_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = axis.get("metrics", {})
    feature_inventory = inventory.get("feature_report", {})
    feature_triage = _feature_report_triage(optional_artifacts["feature_report"])
    missing = []
    if not metrics:
        missing.append("feature stability metrics on the current feature set")
    if feature_inventory.get("status") == "missing" or feature_triage["triage_status"] == "missing":
        missing.append("feature_report artifact")
    return {
        "plane_id": "feature_stability",
        "current_plane_status": axis.get("status"),
        "current_reasons": axis.get("reasons", []),
        "observed_metrics": metrics,
        "supporting_artifacts": ["pipeline_metric_input_readiness", "feature_report"],
        "feature_report_status": feature_triage["triage_status"],
        "evidence_status": "sufficient_to_clear" if not missing else "missing_feature_stability_report",
        "missing_evidence": _unique(missing),
        "required_next_evidence": [
            "Provide a feature stability report with importances, concentration, unstable feature count, or stability score.",
            "Keep leakage/data-quality lineage separate; clean lineage does not prove feature stability.",
        ],
        "accepted_current_use": "Can remain a caution for one bounded proposal; cannot widen experiment bounds.",
    }


def _review_checks(readiness: dict[str, Any], instance: dict[str, Any], plane_reviews: list[dict[str, Any]]) -> list[dict[str, str]]:
    blocked = _unique(
        _summary_list(readiness, "blocked_metric_planes")
        + _summary_list(instance, "blocked_metric_planes")
    )
    checks = [
        _check(
            "pass" if readiness.get("mode") == "pipeline_metric_input_readiness_gate" else "fail",
            "metric_input_readiness_artifact_type",
            str(readiness.get("mode")),
        ),
        _check(
            "pass" if instance.get("mode") == "pipeline_control_instance_contract" else "fail",
            "pipeline_control_instance_artifact_type",
            str(instance.get("mode")),
        ),
        _check("pass" if not blocked else "fail", "no_hard_blocked_metric_planes", ", ".join(blocked) if blocked else "No hard blocked planes."),
        _safe_check(readiness.get("summary", {}), "metric_input_gate"),
        _safe_check(instance.get("summary", {}), "pipeline_instance"),
        _plane_check(plane_reviews, "risk", "risk_drawdown_evidence_present"),
        _plane_check(plane_reviews, "validation", "validation_holdout_evidence_present"),
        _plane_check(plane_reviews, "feature_stability", "feature_stability_report_present"),
        _check("pass", "data_quality_not_reused_for_other_planes", "Clean data-quality lineage is supporting evidence only."),
        _check("pass", "code_audits_not_used_as_metric_evidence", "Code-audit reports cannot clear drawdown, validation, or feature-stability planes."),
    ]
    return checks


def _safe_check(summary: dict[str, Any], prefix: str) -> dict[str, str]:
    safe = _safe_pipeline_flags(summary)
    return _check(
        "pass" if safe else "fail",
        f"{prefix}_keeps_downstream_actions_disabled",
        (
            "autonomous_tuning=False, config=False, learning=False, recommendation=False, trade=False"
            if safe
            else "One or more downstream action flags are enabled."
        ),
    )


def _plane_check(plane_reviews: list[dict[str, Any]], plane_id: str, code: str) -> dict[str, str]:
    review = next((item for item in plane_reviews if item.get("plane_id") == plane_id), None)
    if not review:
        return _check("pass", code, "Plane is not currently cautioned or blocked.")
    if review.get("evidence_status") == "sufficient_to_clear":
        return _check("pass", code, "Current artifacts contain the required evidence.")
    return _check("warn", code, ", ".join(review.get("missing_evidence", [])) or "Evidence is incomplete.")


def _caution_review_status(readiness: dict[str, Any], instance: dict[str, Any], checks: list[dict[str, str]]) -> str:
    blocked = _unique(
        _summary_list(readiness, "blocked_metric_planes")
        + _summary_list(instance, "blocked_metric_planes")
    )
    if blocked or any(check["status"] == "fail" for check in checks):
        return "pipeline_caution_review_blocked_by_hard_planes"
    caution = _unique(
        _summary_list(readiness, "caution_metric_planes")
        + _summary_list(instance, "caution_metric_planes")
    )
    if caution or any(check["status"] == "warn" for check in checks):
        return "pipeline_cautions_need_reviewed_inputs"
    return "pipeline_ready_for_manual_proposal_review"


def _operator_next_steps(status: str, plane_reviews: list[dict[str, Any]]) -> list[str]:
    if status == "pipeline_caution_review_blocked_by_hard_planes":
        return [
            "Stop proposal handoff until hard blocked planes are repaired.",
            "Rebuild PipelineMetricInputReadinessGate and PipelineControlInstanceContract after the repair.",
        ]
    missing_planes = [review["plane_id"] for review in plane_reviews if review.get("evidence_status") != "sufficient_to_clear"]
    if missing_planes:
        return [
            "Either manually accept these cautions for one tiny bounded review-only proposal, or supply the missing evidence first: "
            + ", ".join(missing_planes)
            + ".",
            "Preferred evidence: model evaluation JSON with max_drawdown, train/validation/test metrics, sample_count, and a feature stability report.",
            "Do not use code-audit reports or clean data lineage as substitutes for drawdown, validation, or feature-stability metrics.",
            "After evidence is supplied, rerun readiness -> surface -> instance -> caution review; still no autonomous tuning or config writes.",
        ]
    return [
        "Caution planes have enough current evidence to enter manual proposal review.",
        "Keep the next step proposal-only; production config, learning promotion, recommendations, and trading remain separate gates.",
    ]


def _excluded_artifact_classes() -> list[dict[str, Any]]:
    return [
        {
            "artifact_class": "code_audit_reports",
            "accepted_for_planes": [],
            "reason": "They can reveal code risks, but they do not measure realized drawdown, holdout validation, or feature stability.",
        },
        {
            "artifact_class": "clean_data_lineage_only",
            "accepted_for_planes": ["data_quality"],
            "reason": "Clean lineage can clear leakage/data-quality cautions, not risk, validation, or feature-stability cautions.",
        },
        {
            "artifact_class": "replay_hit_rate_only",
            "accepted_for_planes": ["profitability", "replay_repeatability"],
            "reason": "Replay can support outcome proxies and repeatability, but it does not replace locked validation metrics.",
        },
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training, hyperparameter search, or replay rerun is executed.",
        "No PipelineControlSurface artifact is refreshed by this packet.",
        "No production config is written.",
        "No learning memory or analyst-weight update is written.",
        "No recommendation, allocation, paper order, broker call, or live trade is generated.",
    ]


def _summary_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get("summary", {}).get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _safe_pipeline_flags(summary: dict[str, Any]) -> bool:
    return (
        summary.get("can_run_autonomous_tuning_now") is False
        and summary.get("can_write_production_config") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )


def _unique(items: list[str]) -> list[str]:
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _load_json(path: str | Path) -> dict[str, Any]:
    # Every downstream reader of this dict already uses .get() defensively
    # (see _review_checks/_caution_review_status), so a missing/corrupt
    # required artifact degrades to review_checks failing ->
    # "pipeline_caution_review_blocked_by_hard_planes", matching how every
    # sibling stage in this fixed chain treats a missing input, instead of
    # an uncaught FileNotFoundError/JSONDecodeError.
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"load_error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"load_error": f"Expected JSON object: {path}"}
    return payload


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"available": False, "path": None, "error": "missing_artifact"}
    resolved = Path(path)
    if not resolved.exists():
        return {"available": False, "path": str(path), "error": "missing_artifact"}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"available": False, "path": str(path), "error": f"invalid_json: {exc}"}
    if not isinstance(payload, dict):
        return {"available": False, "path": str(path), "error": "expected_json_object"}
    return {"available": True, "path": str(path), "payload": payload}


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
