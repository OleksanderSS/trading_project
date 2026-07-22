from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.pipeline_control.pipeline_control_evidence_inventory import (
    verify_locked_feature_stability,
    verify_locked_model_evaluation,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_REAL_METRIC_EVIDENCE_JSON = (
    "reports/dean_os/pipeline_control_real_metric_evidence_run_current/"
    "latest.json"
)
DEFAULT_MODEL_EVALUATION_JSON = (
    "reports/dean_os/pipeline_control_metric_artifact_materializer_current/"
    "model_evaluation/latest.json"
)
DEFAULT_FEATURE_STABILITY_JSON = (
    "reports/dean_os/pipeline_control_metric_artifact_materializer_current/"
    "feature_stability/latest.json"
)

ACCEPTED_CHAIN_STATUSES = {
    "real_metric_evidence_blocked_by_metric_planes",
    "real_metric_evidence_ready_with_cautions",
    "real_metric_evidence_chain_ready",
}
LINEAGE_FIELDS = (
    "ticker",
    "model",
    "target_name",
    "timeframe",
    "context_fingerprint",
)


class PipelineModelCasePacket:
    """Builds a review-only case from one locked pipeline evaluation chain.

    An evaluation block is not a realized forecast miss. This packet preserves
    that distinction and creates no learning, tuning, configuration, model,
    recommendation, or execution state.
    """

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_model_case_packet_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        real_metric_evidence_json: str | Path = (
            DEFAULT_REAL_METRIC_EVIDENCE_JSON
        ),
        model_evaluation_json: str | Path = DEFAULT_MODEL_EVALUATION_JSON,
        feature_stability_json: str | Path = DEFAULT_FEATURE_STABILITY_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        chain = _load_json(real_metric_evidence_json)
        model = _load_json(model_evaluation_json)
        feature = _load_json(feature_stability_json)
        readiness_step = _chain_step(
            chain,
            "pipeline_metric_input_readiness",
        )
        readiness_path = readiness_step.get("latest_json")
        readiness = _load_json(readiness_path) if readiness_path else {}

        hashes = {
            "real_metric_evidence_sha256": _file_sha256(
                real_metric_evidence_json
            ),
            "model_evaluation_sha256": _file_sha256(model_evaluation_json),
            "feature_stability_sha256": _file_sha256(
                feature_stability_json
            ),
            "metric_readiness_sha256": _file_sha256(readiness_path),
        }
        checks = _review_checks(
            chain=chain,
            model=model,
            feature=feature,
            readiness=readiness,
            paths={
                "model": model_evaluation_json,
                "feature": feature_stability_json,
                "readiness": readiness_path,
            },
            hashes=hashes,
            readiness_step=readiness_step,
        )
        chain_summary = _mapping(chain.get("summary"))
        blocked_planes = _string_list(
            chain_summary.get("blocked_metric_planes")
        )
        caution_planes = _string_list(
            chain_summary.get("caution_metric_planes")
        )
        plane_outcomes = _metric_plane_outcomes(readiness)
        lineage = _model_lineage(model)
        evaluation_window = _mapping(lineage.get("evaluation_window"))
        status = _case_status(chain_summary, checks)
        root_causes = _root_cause_categories(plane_outcomes)
        case_fingerprint = _case_fingerprint(
            lineage=lineage,
            evaluation_window=evaluation_window,
            source_hashes=hashes,
            blocked_planes=blocked_planes,
            caution_planes=caution_planes,
            plane_outcomes=plane_outcomes,
        )
        case_id = f"pipeline_model_case:{case_fingerprint[:24]}"
        result_label = _result_label(
            status,
            blocked_planes,
            caution_planes,
        )
        payload = {
            "run_id": _run_id("pipeline_model_case_packet"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_model_case_packet",
            "artifact_class": "pipeline_model_evaluation_review_case",
            "inputs": {
                "real_metric_evidence_json": str(
                    real_metric_evidence_json
                ),
                "model_evaluation_json": str(model_evaluation_json),
                "feature_stability_json": str(feature_stability_json),
                "metric_readiness_json": str(readiness_path)
                if readiness_path
                else None,
                **hashes,
            },
            "summary": _summary(
                status=status,
                case_id=case_id,
                result_label=result_label,
                blocked_planes=blocked_planes,
                caution_planes=caution_planes,
                root_causes=root_causes,
                checks=checks,
            ),
            "case": {
                "case_id": case_id,
                "dedupe_fingerprint": case_fingerprint,
                "case_type": "locked_model_evaluation_case",
                "case_scope": "ticker_model_evaluation_only",
                "domain_profile_association": None,
                "sector_scope": None,
                "eligible_as_domain_evidence": False,
                "case_status": status,
                "case_classification": _case_classification(status),
                "result_label": result_label,
                "outcome_semantics": (
                    "evaluation_contract_result_not_realized_forecast_outcome"
                ),
                "forecast_outcome_label": None,
                "production_incident": False,
                "lineage": {
                    field: lineage.get(field) for field in LINEAGE_FIELDS
                },
                "evaluation_window": evaluation_window,
                "evaluated_at": model.get("evaluated_at")
                or _evaluation_window_end(evaluation_window),
                "source_run_ids": {
                    "real_metric_evidence": chain.get("run_id"),
                    "model_evaluation": model.get("run_id"),
                    "feature_stability": feature.get("run_id"),
                    "metric_readiness": readiness.get("run_id"),
                },
                "source_hashes": hashes,
                "blocked_metric_planes": blocked_planes,
                "caution_metric_planes": caution_planes,
                "metric_plane_outcomes": plane_outcomes,
                "root_cause_categories": root_causes,
                "review_disposition": _review_disposition(status),
            },
            "new_data_requirement": _new_data_requirement(
                status,
                evaluation_window,
            ),
            "evaluation_test_candidates": _evaluation_test_candidates(
                plane_outcomes
            ),
            "learning_bridge": _learning_bridge(status),
            "review_checks": checks,
            "template_alignment": _template_alignment(),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(
                status,
                blocked_planes,
            ),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_model_case_packet_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_model_case_packet_markdown(
    payload: dict[str, Any],
) -> str:
    summary = _mapping(payload.get("summary"))
    case = _mapping(payload.get("case"))
    lineage = _mapping(case.get("lineage"))
    lines = [
        "# DEAN-OS Pipeline Model Evaluation Case",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Case ID: `{summary.get('case_id')}`",
        f"- Case status: `{summary.get('case_status')}`",
        f"- Classification: `{summary.get('case_classification')}`",
        f"- Scope: `{summary.get('case_scope')}`",
        f"- Eligible as domain evidence: {summary.get('eligible_as_domain_evidence')}",
        f"- Result label: `{summary.get('result_label')}`",
        f"- Model: `{lineage.get('ticker')}/{lineage.get('model')}`",
        f"- Target/timeframe: `{lineage.get('target_name')}/{lineage.get('timeframe')}`",
        f"- Evaluated at: `{case.get('evaluated_at')}`",
        f"- Blocked planes: {', '.join(summary.get('blocked_metric_planes', [])) or 'none'}",
        f"- Root causes: {', '.join(summary.get('root_cause_categories', [])) or 'none'}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can launch model variant now: {summary.get('can_launch_model_variant_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Metric Plane Outcomes",
        "",
    ]
    for item in case.get("metric_plane_outcomes", []):
        lines.append(
            f"- `{item.get('plane_id')}`: {item.get('status')} — "
            + "; ".join(item.get("reasons", []))
        )
        for comparison in item.get("constraint_comparisons", []):
            lines.append(
                "  - `{metric}`={actual} {operator} {threshold}: "
                "{result}".format(
                    metric=comparison.get("metric"),
                    actual=comparison.get("actual"),
                    operator=comparison.get("operator"),
                    threshold=comparison.get("threshold"),
                    result=comparison.get("result"),
                )
            )
    lines.extend(["", "## New Data Requirement", ""])
    requirement = _mapping(payload.get("new_data_requirement"))
    lines.append(f"- Status: `{requirement.get('status')}`")
    lines.extend(
        f"- {item}" for item in requirement.get("requirements", [])
    )
    lines.extend(["", "## Proposal-only Evaluation Tests", ""])
    for item in payload.get("evaluation_test_candidates", []):
        lines.append(
            f"- `{item.get('test_id')}`: {item.get('assertion')}"
        )
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(
            f"- {str(check.get('status')).upper()}: "
            f"`{check.get('code')}` - {check.get('message')}"
        )
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(
        f"- {item}" for item in payload.get("explicit_non_actions", [])
    )
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(
        f"- {item}" for item in payload.get("operator_next_steps", [])
    )
    return "\n".join(lines).strip() + "\n"


def inspect_pipeline_model_case(
    case_path: str | Path | None,
    *,
    expected_model_evaluation_path: str | Path | None = None,
    expected_evidence_chain_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read a case for agent/review use without mutating memory."""

    if not case_path:
        return _case_inspection("not_configured", None)
    path = Path(case_path)
    if not path.exists():
        return _case_inspection("missing", path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return _case_inspection(
            "unreadable",
            path,
            error_type=type(exc).__name__,
        )
    if not isinstance(payload, dict):
        return _case_inspection("invalid_shape", path)

    inputs = _mapping(payload.get("inputs"))
    summary = _mapping(payload.get("summary"))
    expected_model_sha = _file_sha256(expected_model_evaluation_path)
    expected_chain_sha = _file_sha256(expected_evidence_chain_path)
    model_path_matches = _same_path(
        inputs.get("model_evaluation_json"),
        expected_model_evaluation_path,
    )
    chain_path_matches = _same_path(
        inputs.get("real_metric_evidence_json"),
        expected_evidence_chain_path,
    )
    model_sha_matches = bool(
        expected_model_sha
        and inputs.get("model_evaluation_sha256") == expected_model_sha
    )
    chain_sha_matches = bool(
        expected_chain_sha
        and inputs.get("real_metric_evidence_sha256")
        == expected_chain_sha
    )
    status = str(summary.get("case_status") or "not_reported")
    usable = (
        payload.get("mode") == "pipeline_model_case_packet"
        and status
        in {
            "evaluation_block_case_ready",
            "evaluation_caution_case_ready",
            "evaluation_clear_case_ready",
        }
        and model_path_matches
        and chain_path_matches
        and model_sha_matches
        and chain_sha_matches
    )
    return {
        "status": status if usable else "case_binding_invalid",
        "reported_status": status,
        "path": str(path),
        "case_id": summary.get("case_id"),
        "case_classification": summary.get("case_classification"),
        "case_scope": summary.get("case_scope"),
        "eligible_as_domain_evidence": summary.get(
            "eligible_as_domain_evidence",
            False,
        ),
        "result_label": summary.get("result_label"),
        "blocked_metric_planes": summary.get(
            "blocked_metric_planes",
            [],
        ),
        "root_cause_categories": summary.get(
            "root_cause_categories",
            [],
        ),
        "model_evaluation_path_matches": model_path_matches,
        "real_metric_evidence_path_matches": chain_path_matches,
        "model_evaluation_sha256_matches": model_sha_matches,
        "real_metric_evidence_sha256_matches": chain_sha_matches,
        "usable_for_review": usable,
        "can_write_learning_memory": False,
        "can_launch_model_variant_now": False,
        "can_promote_model": False,
        "can_trade": False,
    }


def _review_checks(
    *,
    chain: dict[str, Any],
    model: dict[str, Any],
    feature: dict[str, Any],
    readiness: dict[str, Any],
    paths: dict[str, str | Path | None],
    hashes: dict[str, str | None],
    readiness_step: dict[str, Any],
) -> list[dict[str, str]]:
    chain_summary = _mapping(chain.get("summary"))
    model_provenance = verify_locked_model_evaluation(model)
    feature_provenance = verify_locked_feature_stability(feature)
    checks = [
        _check(
            "pass"
            if chain.get("mode")
            == "pipeline_control_real_metric_evidence_run"
            else "fail",
            "real_metric_evidence_artifact_type",
            str(chain.get("mode")),
        ),
        _check(
            "pass"
            if chain_summary.get("real_metric_evidence_status")
            in ACCEPTED_CHAIN_STATUSES
            else "fail",
            "real_metric_evidence_status_accepted",
            str(chain_summary.get("real_metric_evidence_status")),
        ),
        _check(
            "pass" if model_provenance["valid"] else "fail",
            "locked_model_evaluation_provenance",
            model_provenance["proof"]
            if model_provenance["valid"]
            else ", ".join(model_provenance["failures"]),
        ),
        _check(
            "pass" if feature_provenance["valid"] else "fail",
            "locked_feature_stability_provenance",
            feature_provenance["proof"]
            if feature_provenance["valid"]
            else ", ".join(feature_provenance["failures"]),
        ),
        _check(
            "pass"
            if readiness.get("mode")
            == "pipeline_metric_input_readiness_gate"
            else "fail",
            "metric_readiness_artifact_type",
            str(readiness.get("mode")),
        ),
        _check(
            "pass"
            if readiness_step.get("latest_json_sha256")
            and readiness_step.get("latest_json_sha256")
            == hashes["metric_readiness_sha256"]
            else "fail",
            "metric_readiness_sha256_matches_chain",
            (
                "Readiness snapshot matches the evidence-chain hash."
                if readiness_step.get("latest_json_sha256")
                == hashes["metric_readiness_sha256"]
                else "Readiness snapshot hash is missing or mismatched."
            ),
        ),
    ]
    chain_inputs = _mapping(chain.get("inputs"))
    readiness_inputs = _mapping(readiness.get("inputs"))
    checks.extend(
        [
            _path_and_hash_check(
                artifact_id="model_evaluation",
                supplied_path=paths["model"],
                referenced_path=chain_inputs.get(
                    "model_evaluation_json"
                ),
                supplied_hash=hashes["model_evaluation_sha256"],
                referenced_hash=chain_inputs.get(
                    "model_evaluation_sha256"
                ),
            ),
            _path_and_hash_check(
                artifact_id="feature_stability",
                supplied_path=paths["feature"],
                referenced_path=chain_inputs.get(
                    "feature_stability_report"
                ),
                supplied_hash=hashes["feature_stability_sha256"],
                referenced_hash=chain_inputs.get(
                    "feature_stability_sha256"
                ),
            ),
            _check(
                "pass"
                if _same_path(
                    readiness_inputs.get("model_performance_path"),
                    paths["model"],
                )
                else "fail",
                "readiness_model_path_matches_case",
                str(readiness_inputs.get("model_performance_path")),
            ),
            _check(
                "pass"
                if _same_path(
                    readiness_inputs.get("feature_report_path"),
                    paths["feature"],
                )
                else "fail",
                "readiness_feature_path_matches_case",
                str(readiness_inputs.get("feature_report_path")),
            ),
        ]
    )
    model_lineage = _model_lineage(model)
    feature_lineage = _feature_lineage(feature)
    for field in LINEAGE_FIELDS:
        matches = (
            not _missing(model_lineage.get(field))
            and not _missing(feature_lineage.get(field))
            and _normalize(model_lineage.get(field))
            == _normalize(feature_lineage.get(field))
        )
        checks.append(
            _check(
                "pass" if matches else "fail",
                f"metric_pair_{field}_matches",
                (
                    f"{field} lineage matches."
                    if matches
                    else (
                        f"{field} mismatch: "
                        f"model={model_lineage.get(field)!r}, "
                        f"feature={feature_lineage.get(field)!r}."
                    )
                ),
            )
        )
    chain_blocked = set(
        _string_list(chain_summary.get("blocked_metric_planes"))
    )
    readiness_blocked = set(
        _string_list(
            _mapping(readiness.get("summary")).get(
                "blocked_metric_planes"
            )
        )
    )
    checks.append(
        _check(
            "pass" if chain_blocked == readiness_blocked else "fail",
            "blocked_metric_planes_match_readiness",
            (
                "Blocked planes match the hashed readiness snapshot."
                if chain_blocked == readiness_blocked
                else (
                    f"chain={sorted(chain_blocked)}, "
                    f"readiness={sorted(readiness_blocked)}"
                )
            ),
        )
    )
    return checks


def _path_and_hash_check(
    *,
    artifact_id: str,
    supplied_path: str | Path | None,
    referenced_path: Any,
    supplied_hash: str | None,
    referenced_hash: Any,
) -> dict[str, str]:
    path_matches = _same_path(supplied_path, referenced_path)
    hash_matches = bool(
        supplied_hash
        and referenced_hash
        and supplied_hash == referenced_hash
    )
    return _check(
        "pass" if path_matches and hash_matches else "fail",
        f"{artifact_id}_binding_matches_chain",
        (
            "Path and SHA-256 match the real-metric evidence chain."
            if path_matches and hash_matches
            else (
                f"path_matches={path_matches}, "
                f"sha256_matches={hash_matches}"
            )
        ),
    )


def _metric_plane_outcomes(
    readiness: dict[str, Any],
) -> list[dict[str, Any]]:
    outcomes = []
    axes = readiness.get("metric_plane_readiness")
    for axis in axes if isinstance(axes, list) else []:
        if not isinstance(axis, dict):
            continue
        metrics = _mapping(axis.get("metrics"))
        constraints = _mapping(axis.get("constraints"))
        outcomes.append(
            {
                "plane_id": str(axis.get("name") or "unknown"),
                "status": str(axis.get("status") or "unknown"),
                "score": axis.get("score"),
                "metrics": metrics,
                "constraints": constraints,
                "constraint_comparisons": _constraint_comparisons(
                    metrics,
                    constraints,
                ),
                "reasons": _string_list(axis.get("reasons")),
            }
        )
    return outcomes


def _constraint_comparisons(
    metrics: dict[str, Any],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    comparisons = []
    mappings = {
        "min_total_return": ("total_return", ">="),
        "min_pnl": ("pnl", ">="),
        "min_sharpe": ("sharpe", ">="),
        "max_drawdown": ("max_drawdown", "<="),
        "min_validation_score": ("validation_score", ">="),
        "max_train_test_gap": ("train_test_gap", "<="),
        "min_sample_count": ("sample_count", ">="),
        "max_feature_concentration": (
            "feature_concentration",
            "<=",
        ),
        "max_feature_weight_abs": ("max_feature_weight_abs", "<="),
        "max_unstable_features": ("unstable_feature_count", "<="),
        "min_feature_stability_score": (
            "feature_stability_score",
            ">=",
        ),
        "min_clear_replay_hit_rate": ("clear_hit_rate", ">="),
        "max_quality_blocked_replay_runs": (
            "quality_blocked_runs",
            "<=",
        ),
        "min_clear_replay_runs": ("clear_evaluated_runs", ">="),
    }
    for constraint_name, threshold in constraints.items():
        mapping = mappings.get(str(constraint_name))
        if not mapping:
            continue
        metric_name, operator = mapping
        actual = metrics.get(metric_name)
        if not _number(actual) or not _number(threshold):
            continue
        actual_number = float(actual)
        threshold_number = float(threshold)
        passed = (
            actual_number >= threshold_number
            if operator == ">="
            else actual_number <= threshold_number
        )
        comparisons.append(
            {
                "metric": metric_name,
                "constraint": str(constraint_name),
                "actual": actual,
                "operator": operator,
                "threshold": threshold,
                "result": "pass" if passed else "fail",
                "distance_to_boundary": round(
                    (
                        actual_number - threshold_number
                        if operator == ">="
                        else threshold_number - actual_number
                    ),
                    12,
                ),
            }
        )
    return comparisons


def _root_cause_categories(
    plane_outcomes: list[dict[str, Any]],
) -> list[str]:
    causes = []
    for plane in plane_outcomes:
        if plane.get("status") != "blocked":
            continue
        plane_id = plane.get("plane_id")
        failed_metrics = {
            item.get("metric")
            for item in plane.get("constraint_comparisons", [])
            if item.get("result") == "fail"
        }
        if plane_id == "validation":
            if "train_test_gap" in failed_metrics:
                causes.append("generalization_gap")
            if "validation_score" in failed_metrics:
                causes.append("validation_floor_failure")
            if "sample_count" in failed_metrics:
                causes.append("insufficient_evaluation_sample")
        elif plane_id == "feature_stability":
            if "feature_stability_score" in failed_metrics:
                causes.append("feature_instability")
            if {
                "feature_concentration",
                "max_feature_weight_abs",
            }.intersection(failed_metrics):
                causes.append("feature_concentration")
            if "unstable_feature_count" in failed_metrics:
                causes.append("unstable_feature_set")
        else:
            causes.append(f"{plane_id}_metric_plane_block")
    return _unique(causes)


def _case_status(
    chain_summary: dict[str, Any],
    checks: list[dict[str, str]],
) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "pipeline_model_case_rejected"
    status = chain_summary.get("real_metric_evidence_status")
    if status == "real_metric_evidence_blocked_by_metric_planes":
        return "evaluation_block_case_ready"
    if status == "real_metric_evidence_ready_with_cautions":
        return "evaluation_caution_case_ready"
    if status == "real_metric_evidence_chain_ready":
        return "evaluation_clear_case_ready"
    return "pipeline_model_case_rejected"


def _case_classification(status: str) -> str:
    return {
        "evaluation_block_case_ready": "negative_evaluation_block_case",
        "evaluation_caution_case_ready": "caution_evaluation_case",
        "evaluation_clear_case_ready": "clear_evaluation_case",
    }.get(status, "rejected_unbound_case")


def _review_disposition(status: str) -> str:
    return {
        "evaluation_block_case_ready": (
            "retain_as_negative_review_case_wait_for_new_forward_data"
        ),
        "evaluation_caution_case_ready": (
            "retain_for_manual_caution_review"
        ),
        "evaluation_clear_case_ready": (
            "retain_as_clear_evaluation_case_no_automatic_promotion"
        ),
    }.get(status, "repair_case_evidence_binding")


def _result_label(
    status: str,
    blocked_planes: list[str],
    caution_planes: list[str],
) -> str:
    if status == "pipeline_model_case_rejected":
        return "case_evidence_rejected"
    if blocked_planes:
        return "failed_" + "_and_".join(
            _normalize_label(item) for item in blocked_planes
        )
    if caution_planes:
        return "caution_" + "_and_".join(
            _normalize_label(item) for item in caution_planes
        )
    return "evaluation_chain_clear"


def _summary(
    *,
    status: str,
    case_id: str,
    result_label: str,
    blocked_planes: list[str],
    caution_planes: list[str],
    root_causes: list[str],
    checks: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "case_status": status,
        "case_id": case_id,
        "case_classification": _case_classification(status),
        "case_scope": "ticker_model_evaluation_only",
        "domain_profile_association": None,
        "eligible_as_domain_evidence": False,
        "result_label": result_label,
        "blocked_metric_planes": blocked_planes,
        "caution_metric_planes": caution_planes,
        "root_cause_categories": root_causes,
        "failed_review_checks": [
            check["code"]
            for check in checks
            if check["status"] == "fail"
        ],
        "review_disposition": _review_disposition(status),
        "manual_review_required": True,
        "can_use_as_forecast_outcome": False,
        "can_write_case_review_artifact": True,
        "can_create_learning_candidate_now": False,
        "can_write_learning_memory": False,
        "can_change_agent_weights": False,
        "can_launch_model_variant_now": False,
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_promote_model": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _new_data_requirement(
    status: str,
    evaluation_window: dict[str, Any],
) -> dict[str, Any]:
    evaluation_end = _evaluation_window_end(evaluation_window)
    if status != "evaluation_block_case_ready":
        return {
            "status": "not_triggered_by_hard_metric_block",
            "data_after": evaluation_end,
            "requirements": [],
        }
    return {
        "status": "wait_for_new_forward_development_data",
        "data_after": evaluation_end,
        "requirements": [
            "Use accepted post-registration forward data strictly after the current evaluation window.",
            "Keep the new development window distinct from the locked final holdout.",
            "Pass the normal data-quality, lineage, and causal feature-join checks before any new model iteration.",
            "Do not retune or relabel the current candidate on the same folds to erase this negative case.",
        ],
        "same_fold_retry_allowed": False,
        "threshold_weakening_allowed": False,
        "new_variant_launch_authorized": False,
    }


def _evaluation_test_candidates(
    plane_outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for plane in plane_outcomes:
        if plane.get("status") != "blocked":
            continue
        for comparison in plane.get("constraint_comparisons", []):
            if comparison.get("result") != "fail":
                continue
            metric = str(comparison.get("metric"))
            candidates.append(
                {
                    "test_id": (
                        f"regression_{plane.get('plane_id')}_{metric}"
                    ),
                    "plane_id": plane.get("plane_id"),
                    "assertion": (
                        f"{metric} must be "
                        f"{comparison.get('operator')} "
                        f"{comparison.get('threshold')} on a future "
                        "lineage-matched locked evaluation window."
                    ),
                    "current_value": comparison.get("actual"),
                    "proposal_only": True,
                    "auto_run_authorized": False,
                    "threshold_change_authorized": False,
                }
            )
    return candidates


def _learning_bridge(status: str) -> dict[str, Any]:
    return {
        "status": (
            "negative_case_available_for_manual_postmortem"
            if status == "evaluation_block_case_ready"
            else "case_available_for_manual_review"
        ),
        "learning_candidate_created": False,
        "recommendation_memory_record_created": False,
        "outcome_label_written": False,
        "agent_weight_change_proposed": False,
        "eligible_for_manual_postmortem": status
        == "evaluation_block_case_ready",
        "future_promotion_requires": [
            "human causal review",
            "explicit feedback classification",
            "balanced retention of clear, blocked, and inconclusive cases",
            "separate approved learning-write ceremony",
        ],
    }


def _template_alignment() -> dict[str, Any]:
    return {
        "harvested_contracts": [
            {
                "template": "OUTCOME_REVIEW_TEMPLATE",
                "use": (
                    "Preserve expected constraints, actual metric results, "
                    "evidence assessment, and what failed."
                ),
            },
            {
                "template": "FEEDBACK_TO_LEARNING_PIPELINE_TEMPLATE",
                "use": (
                    "Classify evaluation gaps but keep any learning "
                    "candidate and production mutation disabled."
                ),
            },
            {
                "template": "PATTERN_MEMORY_UPDATE_POLICY",
                "use": (
                    "Keep the case reviewable and deduplicated without "
                    "automatic memory promotion."
                ),
            },
        ],
        "non_adopted_behavior": [
            "No direct report-to-training conversion.",
            "No automatic pattern-memory update.",
            "No hit/miss forecast label inferred from evaluation metrics.",
        ],
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No collector, replay, backtest, training, tuning, or model variant is run.",
        "No threshold, fold, target, feature set, or production config is changed.",
        "No learning store, recommendation memory, case database, or agent weight is written.",
        "No model is promoted and no recommendation, allocation, order, broker call, or trade is created.",
    ]


def _operator_next_steps(
    status: str,
    blocked_planes: list[str],
) -> list[str]:
    if status == "pipeline_model_case_rejected":
        return [
            "Repair the case evidence binding or provenance before using this artifact in review.",
            "Do not infer a model lesson from an unbound case.",
        ]
    if status == "evaluation_block_case_ready":
        return [
            "Retain this as a negative evaluation-block case: "
            + ", ".join(blocked_planes)
            + ".",
            "Use it in human review and future regression checks, but do not write learning memory yet.",
            "Wait for accepted post-registration forward development data before considering another model iteration.",
        ]
    if status == "evaluation_caution_case_ready":
        return [
            "Review caution planes manually; no automatic tuning or promotion follows from this case."
        ]
    return [
        "Keep the clear case as review evidence; model promotion and trading remain separate decisions."
    ]


def _case_fingerprint(
    *,
    lineage: dict[str, Any],
    evaluation_window: dict[str, Any],
    source_hashes: dict[str, str | None],
    blocked_planes: list[str],
    caution_planes: list[str],
    plane_outcomes: list[dict[str, Any]],
) -> str:
    canonical = {
        "contract": "pipeline_model_evaluation_review_case_v1",
        "lineage": {
            field: lineage.get(field) for field in LINEAGE_FIELDS
        },
        "evaluation_window": evaluation_window,
        # Wrapper reports contain generated timestamps and run IDs. Dedupe is
        # tied to the primary locked evidence plus semantic plane outcomes so
        # rebuilding the same review chain does not create a new case.
        "primary_evidence_hashes": {
            "model_evaluation_sha256": source_hashes.get(
                "model_evaluation_sha256"
            ),
            "feature_stability_sha256": source_hashes.get(
                "feature_stability_sha256"
            ),
        },
        "blocked_metric_planes": blocked_planes,
        "caution_metric_planes": caution_planes,
        "metric_plane_outcomes": plane_outcomes,
    }
    rendered = json.dumps(
        json_ready(canonical),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _case_inspection(
    status: str,
    path: Path | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "path": str(path) if path else None,
        "usable_for_review": False,
        "can_write_learning_memory": False,
        "can_launch_model_variant_now": False,
        "can_promote_model": False,
        "can_trade": False,
        **extra,
    }


def _model_lineage(payload: dict[str, Any]) -> dict[str, Any]:
    return _mapping(payload.get("joined_lineage"))


def _feature_lineage(payload: dict[str, Any]) -> dict[str, Any]:
    return _mapping(payload.get("training_lineage"))


def _evaluation_window_end(value: dict[str, Any]) -> str | None:
    evaluation = _mapping(value.get("evaluation"))
    end = evaluation.get("end") or value.get("end")
    return str(end) if end is not None else None


def _chain_step(
    chain: dict[str, Any],
    step_id: str,
) -> dict[str, Any]:
    for item in chain.get("chain_results", []):
        if isinstance(item, dict) and item.get("step_id") == step_id:
            return item
    return {}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _same_path(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except OSError:
        return False


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and not value.strip()
    )


def _normalize(value: Any) -> str:
    return "".join(
        character for character in str(value).lower() if character.isalnum()
    )


def _normalize_label(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_"
        for character in value.lower()
    ).strip("_")


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _run_id(prefix: str) -> str:
    return (
        f"{prefix}_"
        + utc_now_iso().replace(":", "").replace("+", "Z")
    )
