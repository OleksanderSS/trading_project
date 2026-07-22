from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_CANDIDATE_PATHS = [
    "performance_data.json",
    "data/colab/accumulated/main_database/stage_5_results.json",
    "data/colab/accumulated/main_database/light_models_results.json",
    "data/colab/accumulated/main_database/colab_results.json",
    "data/colab/accumulated/main_database/selected_features_AMD_target_up_1d_random_forest.json",
    "data/colab/accumulated/main_database/pipeline_control_metric_artifacts_manifest.json",
    "data/results/pipeline_control_evaluation_metric_artifacts_manifest.json",
    "data/results/pipeline_control_stage4_training/pipeline_control_metric_artifacts_manifest.json",
    "reports/dean_os/pipeline_control_locked_evaluation_assembler_current/locked_model_evaluation/latest.json",
    "reports/dean_os/pipeline_control_locked_feature_stability_assembler_current/locked_feature_stability/latest.json",
    "reports/dean_os/historical_replay_batch_repaired_expanded/latest.json",
    "reports/dean_os/pipeline_control_walk_forward_validation_current/latest.json",
    "reports/dean_os/pipeline_control_forward_data_accrual_plan_current/latest.json",
    "reports/dean_os/pipeline_control_forward_data_accrual_gate_current/latest.json",
    "diagnostic_reports/feature_lineage_report_current_cache.json",
]

MODEL_REQUIRED_METRICS = {
    "max_drawdown": ("max_drawdown", "maximum_drawdown", "mdd", "drawdown"),
    "train_score": ("train_score", "training_score", "in_sample_score"),
    "validation_score": ("validation_score", "val_score", "test_score", "out_of_sample_score"),
    "sample_count": ("sample_count", "n_samples", "test_samples", "validation_samples", "observations"),
}

MODEL_METADATA_HINTS = (
    "accuracy",
    "model_accuracy",
    "model_name",
    "model_type",
    "best_model",
    "target_column",
)
LOCKED_MODEL_ARTIFACT_CLASS = "locked_model_evaluation"
LOCKED_FEATURE_ARTIFACT_CLASS = "locked_feature_stability_report"
MODEL_LINEAGE_FIELDS = (
    "ticker",
    "model",
    "target_name",
    "timeframe",
    "context_fingerprint",
    "evaluation_window",
)
FEATURE_LINEAGE_FIELDS = (
    "ticker",
    "model",
    "target_name",
    "timeframe",
    "context_fingerprint",
)

SUPPORTING_ONLY_ARTIFACT_TYPES = {
    "backtest_or_portfolio_performance",
    "data_quality_or_lineage",
    "replay_batch",
    "walk_forward_train_validation",
    "forward_development_accrual_plan",
    "forward_development_accrual_gate",
}


class PipelineControlEvidenceInventory:
    """Inventory real local pipeline outputs before treating them as metric evidence.

    This packet prevents useful pipeline artifacts from being silently ignored,
    while also preventing partial metadata, smoke outputs, or selected-feature
    manifests from being promoted into locked model evidence.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_control_evidence_inventory_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        candidate_paths: list[str | Path] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        paths = expand_pipeline_control_candidate_paths(candidate_paths or DEFAULT_CANDIDATE_PATHS)
        records = [_inventory_path(path) for path in paths]
        status = _inventory_status(records)
        payload = {
            "run_id": _run_id("pipeline_control_evidence_inventory"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_evidence_inventory",
            "inputs": {"candidate_paths": [str(path) for path in paths]},
            "summary": _summary(status, records),
            "candidate_artifacts": records,
            "real_metric_evidence_gap": _real_metric_evidence_gap(records),
            "next_runner_inputs": _next_runner_inputs(records),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(records),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_control_evidence_inventory_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def expand_pipeline_control_candidate_paths(candidate_paths: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for candidate in candidate_paths:
        path = Path(candidate)
        expanded = _manifest_candidate_paths(path)
        if not expanded:
            expanded = [path]
        for item in expanded:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            paths.append(item)
    return paths


def render_pipeline_control_evidence_inventory_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Pipeline Control Evidence Inventory",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Inventory status: `{summary.get('inventory_status')}`",
        f"- Existing candidates: {summary.get('existing_candidate_count')}",
        f"- Ready model evaluation candidates: {summary.get('ready_model_evaluation_candidate_count')}",
        f"- Ready feature stability candidates: {summary.get('ready_feature_stability_candidate_count')}",
        f"- Supporting artifacts: {summary.get('supporting_artifact_count')}",
        f"- Development walk-forward candidates: {summary.get('walk_forward_validation_candidate_count')}",
        f"- Blocked development walk-forward candidates: {summary.get('blocked_walk_forward_candidate_count')}",
        f"- Forward development accrual plans: {summary.get('forward_development_accrual_plan_count')}",
        f"- Forward development accrual gates: {summary.get('forward_development_accrual_gate_count')}",
        f"- Blocked forward development artifacts: {summary.get('blocked_forward_development_artifact_count')}",
        f"- Can run real metric evidence now: {summary.get('can_run_real_metric_evidence_now')}",
        f"- Can clear current real cautions: {summary.get('can_clear_current_real_cautions')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Candidate Artifacts",
        "",
    ]
    for item in payload.get("candidate_artifacts", []):
        lines.append(f"- `{item.get('artifact_id')}`: {item.get('classification')} path=`{item.get('path')}`")
        for note in item.get("notes", [])[:4]:
            lines.append(f"  - {note}")
    lines.extend(["", "## Real Metric Evidence Gap", ""])
    gap = payload.get("real_metric_evidence_gap", {})
    for item in gap.get("missing_for_model_evaluation", []):
        lines.append(f"- Missing model metric: `{item}`")
    for item in gap.get("missing_for_feature_stability", []):
        lines.append(f"- Missing feature stability field: `{item}`")
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _inventory_path(path: Path) -> dict[str, Any]:
    artifact = _load_json(path)
    if not artifact["available"]:
        return {
            "artifact_id": _artifact_id(path),
            "path": str(path),
            "exists": False,
            "classification": "missing",
            "usable_as_model_evaluation": False,
            "usable_as_feature_stability": False,
            "usable_as_supporting_artifact": False,
            "recognized_fields": {},
            "notes": [artifact["message"]],
        }
    payload = artifact["payload"]
    flattened = _flatten(payload)
    metrics_payload = payload.get("metrics")
    metrics_payload = (
        metrics_payload if isinstance(metrics_payload, dict) else {}
    )
    model_metrics = _model_metric_presence(_flatten(metrics_payload))
    feature_fields = _feature_field_presence(payload)
    model_provenance = verify_locked_model_evaluation(payload)
    feature_provenance = verify_locked_feature_stability(payload)
    supporting = _supporting_artifact_type(path, payload, flattened)
    selected_features = _selected_feature_count(payload)
    walk_forward_contract_status = (
        _find_key(payload, "contract_status")
        if supporting == "walk_forward_train_validation"
        else None
    )
    walk_forward_contract_passed = (
        _find_key(payload, "contract_passed")
        if supporting == "walk_forward_train_validation"
        else None
    )
    forward_accrual_gate_status = (
        _find_key(payload, "gate_status")
        if supporting == "forward_development_accrual_gate"
        else None
    )
    classification = _classification(
        path,
        payload,
        model_metrics,
        feature_fields,
        model_provenance,
        feature_provenance,
        supporting,
        selected_features,
    )
    return {
        "artifact_id": _artifact_id(path),
        "path": str(path),
        "exists": True,
        "classification": classification,
        "usable_as_model_evaluation": classification == "ready_locked_model_evaluation_candidate",
        "usable_as_feature_stability": classification == "ready_feature_stability_candidate",
        "usable_as_supporting_artifact": supporting is not None,
        "supporting_artifact_type": supporting,
        "development_only_candidate": supporting == "walk_forward_train_validation",
        "eligible_as_locked_test_evidence": False
        if supporting
        in {
            "walk_forward_train_validation",
            "forward_development_accrual_plan",
            "forward_development_accrual_gate",
        }
        else None,
        "recognized_fields": {
            "model_metrics": model_metrics,
            "feature_stability_fields": feature_fields,
            "model_provenance": model_provenance,
            "feature_provenance": feature_provenance,
            "selected_feature_count": selected_features,
            "walk_forward_contract_status": walk_forward_contract_status,
            "walk_forward_contract_passed": walk_forward_contract_passed,
            "forward_accrual_gate_status": forward_accrual_gate_status,
        },
        "source_sha256": _file_sha256(path),
        "notes": _notes(classification, model_metrics, feature_fields, supporting, selected_features),
    }


def _classification(
    path: Path,
    payload: dict[str, Any],
    model_metrics: dict[str, bool],
    feature_fields: dict[str, bool],
    model_provenance: dict[str, Any],
    feature_provenance: dict[str, Any],
    supporting: str | None,
    selected_features: int,
) -> str:
    if _is_synthetic_or_fixture(payload):
        return "synthetic_or_fixture_not_metric_evidence"
    if supporting in SUPPORTING_ONLY_ARTIFACT_TYPES:
        return f"supporting_{supporting}"
    if all(model_metrics.values()) and model_provenance["valid"]:
        return "ready_locked_model_evaluation_candidate"
    if (
        feature_fields["has_importances"]
        and feature_fields["has_stability_signal"]
        and feature_provenance["valid"]
    ):
        return "ready_feature_stability_candidate"
    if all(model_metrics.values()):
        return "complete_model_shape_without_locked_provenance"
    if feature_fields["has_importances"] and feature_fields["has_stability_signal"]:
        return "complete_feature_shape_without_locked_provenance"
    if selected_features:
        return "selected_feature_manifest_only"
    if any(model_metrics.values()) or _has_model_metadata_hint(payload):
        return "partial_model_metadata_not_locked_evaluation"
    if supporting:
        return f"supporting_{supporting}"
    return "unclassified_not_metric_evidence"


def verify_locked_model_evaluation(
    payload: dict[str, Any],
) -> dict[str, Any]:
    artifact_class = str(payload.get("artifact_class") or "")
    lineage = payload.get("joined_lineage")
    lineage = lineage if isinstance(lineage, dict) else {}
    missing_lineage = [
        field
        for field in MODEL_LINEAGE_FIELDS
        if _lineage_value_missing(lineage.get(field))
    ]
    join_contract = payload.get("join_contract")
    join_contract = join_contract if isinstance(join_contract, dict) else {}
    materialization_contract = payload.get("materialization_contract")
    materialization_contract = (
        materialization_contract
        if isinstance(materialization_contract, dict)
        else {}
    )
    proof = (
        "same_window_lineage_proven"
        if join_contract.get("join_status") == "same_window_lineage_proven"
        else (
            "verified_locked_source"
            if materialization_contract.get(
                "source_locked_artifact_verified"
            ) is True
            else None
        )
    )
    failures = []
    if artifact_class != LOCKED_MODEL_ARTIFACT_CLASS:
        failures.append("artifact_class_not_locked_model_evaluation")
    if missing_lineage:
        failures.append("incomplete_joined_lineage")
    if proof is None:
        failures.append("missing_locked_lineage_proof")
    if _is_synthetic_or_fixture(payload):
        failures.append("synthetic_or_fixture")
    return {
        "valid": not failures,
        "artifact_class": artifact_class or None,
        "proof": proof,
        "missing_lineage": missing_lineage,
        "failures": failures,
    }


def verify_locked_feature_stability(
    payload: dict[str, Any],
) -> dict[str, Any]:
    artifact_class = str(payload.get("artifact_class") or "")
    lineage = payload.get("training_lineage")
    lineage = lineage if isinstance(lineage, dict) else {}
    missing_lineage = [
        field
        for field in FEATURE_LINEAGE_FIELDS
        if _lineage_value_missing(lineage.get(field))
    ]
    assembly_contract = payload.get("assembly_contract")
    assembly_contract = (
        assembly_contract if isinstance(assembly_contract, dict) else {}
    )
    materialization_contract = payload.get("materialization_contract")
    materialization_contract = (
        materialization_contract
        if isinstance(materialization_contract, dict)
        else {}
    )
    proof = (
        "measured_stability_assembled"
        if assembly_contract.get("measured_stability_signal_required") is True
        else (
            "verified_locked_source"
            if materialization_contract.get(
                "source_locked_artifact_verified"
            ) is True
            else None
        )
    )
    failures = []
    if artifact_class != LOCKED_FEATURE_ARTIFACT_CLASS:
        failures.append("artifact_class_not_locked_feature_stability")
    if missing_lineage:
        failures.append("incomplete_training_lineage")
    if proof is None:
        failures.append("missing_locked_stability_proof")
    if _is_synthetic_or_fixture(payload):
        failures.append("synthetic_or_fixture")
    return {
        "valid": not failures,
        "artifact_class": artifact_class or None,
        "proof": proof,
        "missing_lineage": missing_lineage,
        "failures": failures,
    }


def _summary(status: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    existing = [item for item in records if item.get("exists")]
    ready_model = [item for item in records if item.get("usable_as_model_evaluation")]
    ready_feature = [item for item in records if item.get("usable_as_feature_stability")]
    supporting = [item for item in records if item.get("usable_as_supporting_artifact")]
    walk_forward = [
        item
        for item in records
        if item.get("supporting_artifact_type") == "walk_forward_train_validation"
    ]
    forward_accrual = [
        item
        for item in records
        if item.get("supporting_artifact_type")
        == "forward_development_accrual_plan"
    ]
    forward_accrual_gates = [
        item
        for item in records
        if item.get("supporting_artifact_type")
        == "forward_development_accrual_gate"
    ]
    return {
        "inventory_status": status,
        "candidate_count": len(records),
        "existing_candidate_count": len(existing),
        "ready_model_evaluation_candidate_count": len(ready_model),
        "ready_feature_stability_candidate_count": len(ready_feature),
        "supporting_artifact_count": len(supporting),
        "walk_forward_validation_candidate_count": len(walk_forward),
        "blocked_walk_forward_candidate_count": sum(
            1
            for item in walk_forward
            if item.get("recognized_fields", {}).get("walk_forward_contract_passed") is False
        ),
        "forward_development_accrual_plan_count": len(forward_accrual),
        "forward_development_accrual_gate_count": len(
            forward_accrual_gates
        ),
        "blocked_forward_development_artifact_count": sum(
            1
            for item in forward_accrual_gates
            if item.get("recognized_fields", {}).get(
                "forward_accrual_gate_status"
            )
            == "blocked_forward_development_artifact"
        ),
        "selected_feature_manifest_count": sum(1 for item in records if item.get("classification") == "selected_feature_manifest_only"),
        "partial_model_metadata_count": sum(1 for item in records if item.get("classification") == "partial_model_metadata_not_locked_evaluation"),
        "complete_model_shape_without_locked_provenance_count": sum(
            1
            for item in records
            if item.get("classification")
            == "complete_model_shape_without_locked_provenance"
        ),
        "complete_feature_shape_without_locked_provenance_count": sum(
            1
            for item in records
            if item.get("classification")
            == "complete_feature_shape_without_locked_provenance"
        ),
        "can_run_real_metric_evidence_now": bool(ready_model and ready_feature),
        "can_clear_current_real_cautions": False,
        "real_metric_evidence_run_required": bool(
            ready_model and ready_feature
        ),
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _inventory_status(records: list[dict[str, Any]]) -> str:
    ready_model = any(item.get("usable_as_model_evaluation") for item in records)
    ready_feature = any(item.get("usable_as_feature_stability") for item in records)
    if ready_model and ready_feature:
        return "real_metric_evidence_inputs_available"
    if any(item.get("exists") for item in records):
        return "real_pipeline_outputs_found_but_metric_evidence_incomplete"
    return "no_pipeline_metric_candidates_found"


def _real_metric_evidence_gap(records: list[dict[str, Any]]) -> dict[str, Any]:
    model_presence = dict.fromkeys(MODEL_REQUIRED_METRICS, False)
    feature_presence = {"feature_importance": False, "stability_signal": False}
    for item in records:
        if _is_model_evaluation_candidate(item):
            metrics = item.get("recognized_fields", {}).get("model_metrics", {})
            for key in model_presence:
                model_presence[key] = model_presence[key] or bool(metrics.get(key))
        fields = item.get("recognized_fields", {}).get("feature_stability_fields", {})
        feature_presence["feature_importance"] = feature_presence["feature_importance"] or bool(fields.get("has_importances"))
        feature_presence["stability_signal"] = feature_presence["stability_signal"] or bool(fields.get("has_stability_signal"))
    return {
        "missing_for_model_evaluation": [key for key, present in model_presence.items() if not present],
        "missing_for_feature_stability": [key for key, present in feature_presence.items() if not present],
        "missing_locked_model_provenance": not any(
            item.get("usable_as_model_evaluation") for item in records
        ),
        "missing_locked_feature_provenance": not any(
            item.get("usable_as_feature_stability") for item in records
        ),
        "accepted_model_evaluation_shape": list(MODEL_REQUIRED_METRICS),
        "accepted_feature_stability_shape": ["feature_importance or feature_importances or feature_weights", "feature_stability_score or unstable_feature_count or unstable_features"],
    }


def _next_runner_inputs(records: list[dict[str, Any]]) -> dict[str, Any]:
    model = next((item for item in records if item.get("usable_as_model_evaluation")), None)
    feature = next((item for item in records if item.get("usable_as_feature_stability")), None)
    return {
        "model_evaluation_json": model.get("path") if model else None,
        "feature_stability_report": feature.get("path") if feature else None,
        "can_invoke_pipeline_control_real_metric_evidence_run": bool(model and feature),
    }


def _operator_next_steps(records: list[dict[str, Any]]) -> list[str]:
    gap = _real_metric_evidence_gap(records)
    steps = []
    if any(
        item.get("supporting_artifact_type") == "walk_forward_train_validation"
        for item in records
    ):
        steps.append(
            "Keep the walk-forward artifact as development-only train/validation evidence; "
            "do not promote it to locked test evidence or iterate model variants on the same folds."
        )
    if any(
        item.get("supporting_artifact_type")
        == "forward_development_accrual_plan"
        for item in records
    ):
        steps.append(
            "Use the forward accrual plan only to prove that a future immutable "
            "artifact is genuinely new development data; it is not a holdout or metric evidence."
        )
    if any(
        item.get("supporting_artifact_type")
        == "forward_development_accrual_gate"
        for item in records
    ):
        steps.append(
            "Do not pass a blocked forward source into Stage 3 or walk-forward; "
            "wait for a post-registration immutable artifact that clears the accrual gate."
        )
    if gap["missing_for_model_evaluation"]:
        steps.append("Create or supply a locked model evaluation JSON with max_drawdown, train_score, validation_score/test_score, and sample_count.")
    if gap["missing_for_feature_stability"]:
        steps.append("Create or supply a feature stability report with importances plus stability_score or unstable_features.")
    steps.append("Do not treat selected feature manifests, smoke reports, or empty performance histories as real metric evidence.")
    steps.append("After both artifacts exist, run PipelineControlRealMetricEvidenceRun.")
    return steps


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training, evaluation, replay, or backtest is executed.",
        "No synthetic metric artifact is generated.",
        "No production config is written.",
        "No autonomous tuning, recommendation, allocation, paper order, broker call, or live trade is generated.",
    ]


def _is_model_evaluation_candidate(item: dict[str, Any]) -> bool:
    classification = item.get("classification")
    supporting = item.get("supporting_artifact_type")
    if supporting in SUPPORTING_ONLY_ARTIFACT_TYPES:
        return False
    return classification in {
        "ready_locked_model_evaluation_candidate",
        "partial_model_metadata_not_locked_evaluation",
        "complete_model_shape_without_locked_provenance",
    }


def _model_metric_presence(flattened: dict[str, Any]) -> dict[str, bool]:
    return {canonical: _first_present(flattened, aliases) for canonical, aliases in MODEL_REQUIRED_METRICS.items()}


def _feature_field_presence(payload: dict[str, Any]) -> dict[str, bool]:
    return {
        "has_importances": bool(_extract_feature_importances(payload)),
        "has_stability_signal": _find_key(payload, "feature_stability_score") is not None
        or _find_key(payload, "stability_score") is not None
        or _find_key(payload, "unstable_feature_count") is not None
        or isinstance(_find_key(payload, "unstable_features"), list),
    }


def _selected_feature_count(payload: Any) -> int:
    value = _find_key(payload, "selected_features")
    if isinstance(value, list):
        return len(value)
    return 0


def _supporting_artifact_type(path: Path, payload: dict[str, Any], flattened: dict[str, Any]) -> str | None:
    text = str(path).replace("\\", "/").lower()
    artifact_class = str(_find_key(payload, "artifact_class") or "").lower()
    evidence_class = str(_find_key(payload, "evidence_class") or "").lower()
    mode = str(payload.get("mode") or "").lower()
    if (
        "forward_data_accrual_plan" in artifact_class
        or mode == "pipeline_control_forward_data_accrual_plan"
    ):
        return "forward_development_accrual_plan"
    if (
        "forward_development_artifact" in artifact_class
        or mode == "pipeline_control_forward_data_accrual_gate"
    ):
        return "forward_development_accrual_gate"
    if (
        "walk_forward_validation_candidate" in artifact_class
        or mode == "pipeline_control_walk_forward_validation_run"
        or evidence_class == "development_train_validation_only"
    ):
        return "walk_forward_train_validation"
    if "evaluation_metric_candidate" in artifact_class or "pipeline_stage_7_evaluation" in evidence_class:
        return "backtest_or_portfolio_performance"
    if "replay" in text or _first_present(flattened, ("clear_hit_rate", "clear_evaluated_runs", "quality_blocked_runs")):
        return "replay_batch"
    if "feature_lineage" in text or _find_key(payload, "leakage_flags") is not None or _find_key(payload, "warnings") is not None:
        return "data_quality_or_lineage"
    if "data/results" in text or _find_key(payload, "backtest_stats") is not None or _find_key(payload, "portfolio_history") is not None:
        return "backtest_or_portfolio_performance"
    if "stage_5_results" in text:
        return "pipeline_prediction_metadata"
    return None


def _notes(classification: str, model_metrics: dict[str, bool], feature_fields: dict[str, bool], supporting: str | None, selected_features: int) -> list[str]:
    notes = []
    if classification == "ready_locked_model_evaluation_candidate":
        notes.append(
            "Required metrics and verified locked evaluation provenance are present."
        )
    elif classification == "complete_model_shape_without_locked_provenance":
        notes.append(
            "Required metric names are present, but artifact class, same-window "
            "join proof, or joined lineage is incomplete."
        )
    elif classification == "complete_feature_shape_without_locked_provenance":
        notes.append(
            "Feature importances and a stability signal are present, but the "
            "locked measured-stability assembly contract is not proven."
        )
    elif classification == "partial_model_metadata_not_locked_evaluation":
        missing = [key for key, present in model_metrics.items() if not present]
        notes.append("Partial model metrics found, but locked evaluation shape is incomplete: " + ", ".join(missing) + ".")
    if classification == "selected_feature_manifest_only":
        notes.append(f"Selected feature manifest has {selected_features} features, but no importances or stability signal.")
    if feature_fields["has_importances"] and not feature_fields["has_stability_signal"]:
        notes.append("Feature importances are present, but stability signal is missing.")
    if supporting == "walk_forward_train_validation":
        notes.append(
            "Development-only purged walk-forward train/validation evidence; "
            "it is never eligible as locked test evidence."
        )
    if supporting == "forward_development_accrual_plan":
        notes.append(
            "Prospective development-data boundary only; it loads no observations "
            "and is neither model evidence nor a virgin holdout."
        )
    if supporting == "forward_development_accrual_gate":
        notes.append(
            "Forward source intake status only; even a passing artifact remains "
            "development-refresh data and never locked test evidence."
        )
    if supporting:
        notes.append(f"Useful supporting artifact for {supporting}, but not sufficient to clear missing metric planes by itself.")
    if not notes:
        notes.append("No accepted model-evaluation or feature-stability evidence recognized.")
    return notes


def _extract_feature_importances(payload: Any) -> dict[str, float]:
    for key in ("feature_importance", "feature_importances", "feature_weights", "importances"):
        value = _find_key(payload, key)
        if isinstance(value, dict):
            result = {}
            for feature, weight in value.items():
                number = _number(weight)
                if number is not None:
                    result[str(feature)] = number
            return result
        if isinstance(value, list):
            result = {}
            for item in value:
                if isinstance(item, dict):
                    name = item.get("feature") or item.get("name")
                    weight = _number(item.get("importance") or item.get("weight") or item.get("value"))
                    if name and weight is not None:
                        result[str(name)] = weight
            return result
    return {}


def _manifest_candidate_paths(path: Path) -> list[Path]:
    if not path.exists():
        return []
    loaded = _load_json(path)
    if not loaded.get("available"):
        return []
    payload = loaded.get("payload", {})
    artifact_class = str(_find_key(payload, "artifact_class") or "").lower()
    if "metric_artifacts_manifest" not in artifact_class and path.name != "pipeline_control_metric_artifacts_manifest.json":
        return []
    entries = _find_key(payload, "artifacts")
    if isinstance(entries, dict):
        entries = list(entries.values())
    if not isinstance(entries, list):
        return []
    candidates: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in ("path", "model_evaluation_json", "feature_stability_report", "latest_json"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(_resolve_manifest_path(path, value))
    return candidates


def _resolve_manifest_path(manifest_path: Path, candidate_path: str) -> Path:
    path = Path(candidate_path)
    if path.is_absolute() or path.exists():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "payload": {}, "message": f"Missing file: {path}"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "payload": {}, "message": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"available": False, "payload": {}, "message": f"Expected JSON object: {path}"}
    return {"available": True, "payload": payload, "message": f"Loaded {path}"}


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        result = {}
        for key, value in payload.items():
            key_text = _normalize_key(str(key))
            full_key = f"{prefix}.{key_text}" if prefix else key_text
            result[key_text] = value
            result[full_key] = value
            result.update(_flatten(value, prefix=full_key))
        return result
    if isinstance(payload, list):
        return {key: value for item in payload for key, value in _flatten(item, prefix=prefix).items()}
    return {}


def _find_key(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        for current_key, value in payload.items():
            if _normalize_key(str(current_key)) == _normalize_key(key):
                return value
            found = _find_key(value, key)
            if found is not None:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


def _first_present(flattened: dict[str, Any], aliases: tuple[str, ...]) -> bool:
    normalized_aliases = tuple(_normalize_key(alias) for alias in aliases)
    for key, value in flattened.items():
        normalized_key = _normalize_key(key)
        if normalized_key in normalized_aliases or any(normalized_key.endswith(f".{alias}") for alias in normalized_aliases):
            if _number(value) is not None:
                return True
    return False


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _is_synthetic_or_fixture(payload: Any) -> bool:
    if _find_key(payload, "fixture_not_evidence") is True or _find_key(payload, "synthetic") is True:
        return True
    for key in ("mode", "artifact_type", "source_type", "evidence_class"):
        value = _find_key(payload, key)
        if value is not None:
            text = str(value).lower()
            if "synthetic" in text or "fixture" in text:
                return True
    return False


def _has_model_metadata_hint(payload: Any) -> bool:
    return any(_find_key(payload, key) is not None for key in MODEL_METADATA_HINTS)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _lineage_value_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        evaluation = value.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, dict) else value
        return not evaluation.get("start") or not evaluation.get("end")
    return False


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _artifact_id(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
