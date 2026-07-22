from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_evidence_inventory import (
    DEFAULT_CANDIDATE_PATHS,
    PipelineControlEvidenceInventory,
    expand_pipeline_control_candidate_paths,
    verify_locked_feature_stability,
    verify_locked_model_evaluation,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_MATERIALIZATION_CANDIDATE_PATHS = [
    *DEFAULT_CANDIDATE_PATHS,
    "data/results/summary_20260620_191713.json",
    "data/results/summary_20260620_184204.json",
]

MODEL_REQUIRED_METRICS = {
    "max_drawdown": ("max_drawdown", "maximum_drawdown", "mdd", "drawdown"),
    "train_score": ("train_score", "training_score", "in_sample_score"),
    "validation_score": ("validation_score", "val_score", "test_score", "out_of_sample_score"),
    "sample_count": ("sample_count", "n_samples", "test_samples", "validation_samples", "observations"),
}
MODEL_OPTIONAL_METRICS = {
    "total_return": ("total_return", "return", "realized_return", "pnl_pct"),
    "pnl": ("pnl", "profit", "net_profit"),
    "sharpe": ("sharpe", "sharpe_ratio"),
}
METRIC_PAIR_LINEAGE_FIELDS = ("ticker", "model", "target_name", "timeframe", "context_fingerprint")


class PipelineControlMetricArtifactMaterializer:
    """Materialize real metric artifacts only when saved inputs satisfy the contract.

    The materializer is deliberately stricter than a converter. It refuses to
    write model-evaluation or feature-stability evidence from partial pipeline
    metadata, selected-feature manifests, replay summaries, backtest summaries,
    synthetic fixtures, or code-audit reports.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_control_metric_artifact_materializer_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        candidate_paths: list[str | Path] | None = None,
        write_artifacts: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        paths = expand_pipeline_control_candidate_paths(candidate_paths or DEFAULT_MATERIALIZATION_CANDIDATE_PATHS)
        run_id = _run_id("pipeline_control_metric_artifact_materializer")
        inventory = PipelineControlEvidenceInventory(output_dir=self.output_dir / "inventory").build(
            candidate_paths=paths,
            save=False,
        )
        records = inventory.get("candidate_artifacts", [])
        ready_model_records = [item for item in records if item.get("usable_as_model_evaluation")]
        ready_feature_records = [item for item in records if item.get("usable_as_feature_stability")]
        model_record, feature_record = _select_compatible_metric_pair(
            ready_model_records,
            ready_feature_records,
        )

        model_artifact = _build_model_evaluation_artifact(model_record, run_id=run_id) if model_record else None
        feature_artifact = _build_feature_stability_artifact(feature_record, run_id=run_id) if feature_record else None
        status = _materialization_status(model_artifact, feature_artifact)
        materialized_paths: dict[str, Any] = {}
        if status == "materialized_real_metric_artifacts_ready" and write_artifacts:
            materialized_paths = _write_materialized_artifacts(
                output_dir=self.output_dir,
                run_id=run_id,
                model_artifact=model_artifact,
                feature_artifact=feature_artifact,
            )

        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_metric_artifact_materializer",
            "inputs": {
                "candidate_paths": [str(path) for path in paths],
                "write_artifacts": write_artifacts,
            },
            "summary": _summary(
                status,
                records,
                materialized_paths,
                compatible_metric_pair_found=bool(model_record and feature_record),
            ),
            "inventory_summary": inventory.get("summary", {}),
            "candidate_classification_counts": dict(Counter(item.get("classification") for item in records)),
            "candidate_artifacts": records,
            "materialized_artifacts": {
                "model_evaluation_json": _artifact_preview(model_artifact, materialized_paths.get("model_evaluation_json")),
                "feature_stability_report": _artifact_preview(feature_artifact, materialized_paths.get("feature_stability_report")),
            },
            "materialization_gap": _materialization_gap(
                inventory,
                model_candidate_found=bool(ready_model_records),
                feature_candidate_found=bool(ready_feature_records),
                compatible_metric_pair_found=bool(model_record and feature_record),
            ),
            "next_runner_inputs": _next_runner_inputs(materialized_paths),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(status, inventory),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_control_metric_artifact_materializer_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_control_metric_artifact_materializer_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Pipeline Control Metric Artifact Materializer",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Materialization status: `{summary.get('materialization_status')}`",
        f"- Existing candidates: {summary.get('existing_candidate_count')}",
        f"- Ready model candidate found: {summary.get('ready_model_candidate_found')}",
        f"- Ready feature candidate found: {summary.get('ready_feature_candidate_found')}",
        f"- Compatible metric pair found: {summary.get('compatible_metric_pair_found')}",
        f"- Materialized model evaluation: {summary.get('materialized_model_evaluation_json')}",
        f"- Materialized feature stability: {summary.get('materialized_feature_stability_report')}",
        f"- Can run real metric evidence now: {summary.get('can_run_real_metric_evidence_now')}",
        f"- Can clear cautions without real runner: {summary.get('can_clear_cautions_without_real_runner')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Materialization Gap",
        "",
    ]
    gap = payload.get("materialization_gap", {})
    for reason in gap.get("blocking_reasons", []):
        lines.append(f"- {reason}")
    for item in gap.get("missing_for_model_evaluation", []):
        lines.append(f"- Missing model metric: `{item}`")
    for item in gap.get("missing_for_feature_stability", []):
        lines.append(f"- Missing feature stability field: `{item}`")

    lines.extend(["", "## Next Runner Inputs", ""])
    next_inputs = payload.get("next_runner_inputs", {})
    lines.append(f"- Model evaluation JSON: `{next_inputs.get('model_evaluation_json')}`")
    lines.append(f"- Feature stability report: `{next_inputs.get('feature_stability_report')}`")
    lines.append(f"- Can invoke real metric evidence run: {next_inputs.get('can_invoke_pipeline_control_real_metric_evidence_run')}")

    lines.extend(["", "## Candidate Classification Counts", ""])
    for key, value in payload.get("candidate_classification_counts", {}).items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _summary(
    status: str,
    records: list[dict[str, Any]],
    materialized_paths: dict[str, Any],
    *,
    compatible_metric_pair_found: bool,
) -> dict[str, Any]:
    existing = [item for item in records if item.get("exists")]
    ready_model = [item for item in records if item.get("usable_as_model_evaluation")]
    ready_feature = [item for item in records if item.get("usable_as_feature_stability")]
    model_path = materialized_paths.get("model_evaluation_json", {}).get("latest_json")
    feature_path = materialized_paths.get("feature_stability_report", {}).get("latest_json")
    return {
        "materialization_status": status,
        "candidate_count": len(records),
        "existing_candidate_count": len(existing),
        "ready_model_candidate_found": bool(ready_model),
        "ready_feature_candidate_found": bool(ready_feature),
        "compatible_metric_pair_found": compatible_metric_pair_found,
        "materialized_model_evaluation_json": bool(model_path),
        "materialized_feature_stability_report": bool(feature_path),
        "can_run_real_metric_evidence_now": bool(model_path and feature_path),
        "can_clear_cautions_without_real_runner": False,
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _materialization_status(model_artifact: dict[str, Any] | None, feature_artifact: dict[str, Any] | None) -> str:
    if model_artifact and feature_artifact:
        return "materialized_real_metric_artifacts_ready"
    return "blocked_missing_locked_metric_artifacts"


def _materialization_gap(
    inventory: dict[str, Any],
    *,
    model_candidate_found: bool,
    feature_candidate_found: bool,
    compatible_metric_pair_found: bool,
) -> dict[str, Any]:
    inventory_gap = inventory.get("real_metric_evidence_gap", {})
    reasons = []
    if not model_candidate_found:
        reasons.append("No single non-synthetic locked model-evaluation candidate contains max_drawdown, train_score, validation_score/test_score, and sample_count.")
    if not feature_candidate_found:
        reasons.append("No single non-synthetic feature-stability candidate contains importances plus a stability score or unstable-feature signal.")
    if model_candidate_found and feature_candidate_found and not compatible_metric_pair_found:
        reasons.append(
            "Ready candidates exist, but they do not declare matching ticker, model, target_name, timeframe, and context_fingerprint lineage."
        )
    if reasons:
        reasons.append("Do not combine replay/backtest drawdown, selected-feature manifests, and partial model metadata into a fake locked artifact.")
    return {
        "blocking_reasons": reasons,
        "missing_for_model_evaluation": inventory_gap.get("missing_for_model_evaluation", []),
        "missing_for_feature_stability": inventory_gap.get("missing_for_feature_stability", []),
        "accepted_model_evaluation_shape": list(MODEL_REQUIRED_METRICS),
        "accepted_feature_stability_shape": [
            "feature_importance or feature_importances or feature_weights",
            "feature_stability_score or unstable_feature_count or unstable_features",
        ],
        "required_matching_lineage": list(METRIC_PAIR_LINEAGE_FIELDS),
    }


def _operator_next_steps(status: str, inventory: dict[str, Any]) -> list[str]:
    if status == "materialized_real_metric_artifacts_ready":
        return [
            "Run PipelineControlRealMetricEvidenceRun with the materialized model evaluation and feature stability report.",
            "Treat a clear run as manual proposal-review input only; it still does not write config, learning memory, recommendations, allocation, paper trades, or live trades.",
        ]
    return [
        "Instrument or supply a locked model-evaluation artifact that records max_drawdown, train_score, validation_score/test_score, and sample_count from the same evaluation window.",
        "Instrument or supply a feature-stability report that records feature importances and feature_stability_score, unstable_feature_count, or unstable_features.",
        "Keep matching ticker, model, target_name, timeframe, and context_fingerprint lineage on both locked artifacts.",
        "Keep current risk, validation, and feature_stability cautions visible until those artifacts exist.",
        "Use existing replay, backtest, lineage, and selected-feature artifacts only as supporting history, not as substitutes.",
    ]


def _next_runner_inputs(materialized_paths: dict[str, Any]) -> dict[str, Any]:
    model_path = materialized_paths.get("model_evaluation_json", {}).get("latest_json")
    feature_path = materialized_paths.get("feature_stability_report", {}).get("latest_json")
    command = None
    if model_path and feature_path:
        command = (
            "python run_agent_pipeline_control_real_metric_evidence_run.py "
            f"--model-evaluation-json {model_path} "
            f"--feature-stability-report {feature_path} "
            "--replay-batch-json reports\\dean_os\\historical_replay_batch_repaired_expanded\\latest.json "
            "--data-quality-json diagnostic_reports\\feature_lineage_report_current_cache.json "
            "--output-dir reports\\dean_os\\pipeline_control_real_metric_evidence_run_current"
        )
    return {
        "model_evaluation_json": model_path,
        "feature_stability_report": feature_path,
        "can_invoke_pipeline_control_real_metric_evidence_run": bool(model_path and feature_path),
        "command_preview": command,
    }


def _artifact_preview(artifact: dict[str, Any] | None, saved_paths: dict[str, Any] | None) -> dict[str, Any]:
    if not artifact:
        return {"available": False, "saved_paths": saved_paths or {}}
    return {
        "available": True,
        "artifact_class": artifact.get("artifact_class"),
        "source_path": artifact.get("source_artifact", {}).get("path"),
        "saved_paths": saved_paths or {},
        "metrics": artifact.get("metrics"),
        "feature_count": len(artifact.get("feature_importance", {})),
        "feature_stability_score": artifact.get("feature_stability_score"),
        "unstable_feature_count": artifact.get("unstable_feature_count"),
        "lineage": artifact.get("joined_lineage") or artifact.get("training_lineage"),
    }


def _write_materialized_artifacts(
    *,
    output_dir: Path,
    run_id: str,
    model_artifact: dict[str, Any] | None,
    feature_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    if not model_artifact or not feature_artifact:
        return {}
    model_paths = ReviewArtifactWriter(output_dir / "model_evaluation").write(
        payload=model_artifact,
        markdown=_render_materialized_model_markdown(model_artifact),
        run_id=f"{run_id}_model_evaluation",
    )
    feature_paths = ReviewArtifactWriter(output_dir / "feature_stability").write(
        payload=feature_artifact,
        markdown=_render_materialized_feature_markdown(feature_artifact),
        run_id=f"{run_id}_feature_stability",
    )
    return {
        "model_evaluation_json": model_paths,
        "feature_stability_report": feature_paths,
    }


def _render_materialized_model_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "# Materialized Locked Model Evaluation",
        "",
        f"- Artifact class: `{payload.get('artifact_class')}`",
        f"- Source path: `{payload.get('source_artifact', {}).get('path')}`",
    ]
    for key in [*MODEL_REQUIRED_METRICS, *MODEL_OPTIONAL_METRICS]:
        if key in metrics:
            lines.append(f"- `{key}`: {metrics[key]}")
    lines.extend(["", "## Safety", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _render_materialized_feature_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Materialized Feature Stability Report",
        "",
        f"- Artifact class: `{payload.get('artifact_class')}`",
        f"- Source path: `{payload.get('source_artifact', {}).get('path')}`",
        f"- Feature count: {len(payload.get('feature_importance', {}))}",
        f"- Feature stability score: {payload.get('feature_stability_score')}",
        f"- Unstable feature count: {payload.get('unstable_feature_count')}",
        "",
        "## Safety",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _build_model_evaluation_artifact(record: dict[str, Any], *, run_id: str) -> dict[str, Any] | None:
    payload = _load_json(record.get("path"))
    if (
        not payload
        or _is_synthetic_or_fixture(payload)
        or not verify_locked_model_evaluation(payload)["valid"]
    ):
        return None
    source_metrics = payload.get("metrics")
    if not isinstance(source_metrics, dict):
        return None
    metrics: dict[str, float] = {}
    for canonical, aliases in MODEL_REQUIRED_METRICS.items():
        value = _first_number(source_metrics, aliases)
        if value is None:
            return None
        metrics[canonical] = value
    for canonical, aliases in MODEL_OPTIONAL_METRICS.items():
        value = _first_number(source_metrics, aliases)
        if value is not None:
            metrics[canonical] = value
    joined_lineage = _extract_candidate_lineage(
        payload,
        lineage_container="joined_lineage",
    )
    evaluated_at = _evaluation_window_end(
        joined_lineage.get("evaluation_window")
    )
    return {
        "run_id": f"{run_id}_model_evaluation",
        "created_at": utc_now_iso(),
        "artifact_class": "locked_model_evaluation",
        "evidence_class": "materialized_from_saved_locked_metric_artifact",
        "source_artifact": _source_artifact(record),
        "metrics": metrics,
        "joined_lineage": joined_lineage,
        "evaluated_at": evaluated_at,
        "as_of_contract": {
            "evaluation_data_through": evaluated_at,
            "materialized_at": utc_now_iso(),
            "source_sha256": record.get("source_sha256"),
        },
        "materialization_contract": {
            "required_metrics": list(MODEL_REQUIRED_METRICS),
            "same_window_required": True,
            "source_locked_artifact_verified": True,
            "source_provenance_proof": verify_locked_model_evaluation(
                payload
            ).get("proof"),
            "matching_feature_stability_lineage_required": list(METRIC_PAIR_LINEAGE_FIELDS),
            "synthetic_or_fixture_allowed": False,
            "supporting_artifacts_allowed_as_substitutes": False,
        },
        "explicit_non_actions": _explicit_non_actions(),
    }


def _build_feature_stability_artifact(record: dict[str, Any], *, run_id: str) -> dict[str, Any] | None:
    payload = _load_json(record.get("path"))
    if (
        not payload
        or _is_synthetic_or_fixture(payload)
        or not verify_locked_feature_stability(payload)["valid"]
    ):
        return None
    importances = _extract_feature_importances(payload)
    if not importances:
        return None
    stability_score = _first_number(payload, ("feature_stability_score", "stability_score"))
    unstable_features = _find_key(payload, "unstable_features")
    unstable_count = _first_number(payload, ("unstable_feature_count", "unstable_features_count"))
    if unstable_count is None and isinstance(unstable_features, list):
        unstable_count = float(len(unstable_features))
    if stability_score is None and unstable_count is None and not isinstance(unstable_features, list):
        return None
    return {
        "run_id": f"{run_id}_feature_stability",
        "created_at": utc_now_iso(),
        "artifact_class": "locked_feature_stability_report",
        "evidence_class": "materialized_from_saved_locked_feature_stability_artifact",
        "source_artifact": _source_artifact(record),
        "training_lineage": _extract_candidate_lineage(payload, lineage_container="training_lineage"),
        "feature_importance": importances,
        "feature_stability_score": stability_score,
        "unstable_feature_count": int(unstable_count) if unstable_count is not None else None,
        "unstable_features": unstable_features if isinstance(unstable_features, list) else [],
        "materialization_contract": {
            "importances_required": True,
            "stability_signal_required": True,
            "source_locked_artifact_verified": True,
            "source_provenance_proof": verify_locked_feature_stability(
                payload
            ).get("proof"),
            "matching_model_evaluation_lineage_required": list(METRIC_PAIR_LINEAGE_FIELDS),
            "synthetic_or_fixture_allowed": False,
            "selected_feature_manifest_allowed_as_substitute": False,
        },
        "explicit_non_actions": _explicit_non_actions(),
    }


def _select_compatible_metric_pair(
    model_records: list[dict[str, Any]],
    feature_records: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for model_record in model_records:
        model_payload = _load_json(model_record.get("path"))
        if not model_payload:
            continue
        model_lineage = _extract_candidate_lineage(model_payload, lineage_container="joined_lineage")
        if not _lineage_complete(model_lineage):
            continue
        for feature_record in feature_records:
            feature_payload = _load_json(feature_record.get("path"))
            if not feature_payload:
                continue
            feature_lineage = _extract_candidate_lineage(
                feature_payload,
                lineage_container="training_lineage",
            )
            if _lineages_match(model_lineage, feature_lineage):
                return model_record, feature_record
    return None, None


def _extract_candidate_lineage(payload: dict[str, Any], *, lineage_container: str) -> dict[str, Any]:
    nested = payload.get(lineage_container)
    nested = nested if isinstance(nested, dict) else {}
    return {
        "ticker": nested.get("ticker") or _find_key(payload, "ticker") or _find_key(payload, "symbol"),
        "model": nested.get("model")
        or _find_key(payload, "model_id")
        or _find_key(payload, "model_name")
        or _find_key(payload, "model_type")
        or _find_key(payload, "selected_primary_model"),
        "target_name": nested.get("target_name")
        or _find_key(payload, "target_name")
        or _find_key(payload, "target")
        or _find_key(payload, "target_column"),
        "timeframe": nested.get("timeframe")
        or _find_key(payload, "timeframe")
        or _find_key(payload, "horizon")
        or _find_key(payload, "prediction_horizon"),
        "context_fingerprint": nested.get("context_fingerprint")
        or _find_key(payload, "context_fingerprint")
        or _find_key(payload, "context_key")
        or _find_key(payload, "regime_fingerprint"),
        "evaluation_window": nested.get("evaluation_window"),
    }


def _lineage_complete(lineage: dict[str, Any]) -> bool:
    return all(not _lineage_value_missing(lineage.get(field)) for field in METRIC_PAIR_LINEAGE_FIELDS)


def _lineages_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _lineage_complete(left) and _lineage_complete(right) and all(
        _normalize_lineage_value(left[field]) == _normalize_lineage_value(right[field])
        for field in METRIC_PAIR_LINEAGE_FIELDS
    )


def _lineage_value_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _normalize_lineage_value(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _source_artifact(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": record.get("artifact_id"),
        "path": record.get("path"),
        "classification": record.get("classification"),
        "supporting_artifact_type": record.get("supporting_artifact_type"),
        "sha256": record.get("source_sha256"),
    }


def _evaluation_window_end(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    evaluation = value.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else value
    end = evaluation.get("end")
    return str(end) if end is not None else None


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training, evaluation, replay, backtest, or pickle/model loading is executed.",
        "No synthetic metric artifact is generated.",
        "No partial artifact is promoted to locked evidence.",
        "No production config, learning memory, recommendation, allocation, paper order, broker call, or live trade is generated.",
    ]


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _first_number(payload: Any, aliases: tuple[str, ...]) -> float | None:
    normalized_aliases = {_normalize_key(alias) for alias in aliases}
    for key, value in _walk_items(payload):
        normalized_key = _normalize_key(key)
        if normalized_key in normalized_aliases or any(normalized_key.endswith(f".{alias}") for alias in normalized_aliases):
            number = _number(value)
            if number is not None:
                return number
    return None


def _extract_feature_importances(payload: Any) -> dict[str, float]:
    for key in ("feature_importance", "feature_importances", "feature_weights", "importances"):
        value = _find_key(payload, key)
        parsed = _parse_feature_importances(value)
        if parsed:
            return parsed
    return {}


def _parse_feature_importances(value: Any) -> dict[str, float]:
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
            if not isinstance(item, dict):
                continue
            name = item.get("feature") or item.get("name") or item.get("column")
            weight = _number(item.get("importance", item.get("weight", item.get("value"))))
            if name and weight is not None:
                result[str(name)] = weight
        return result
    return {}


def _find_key(payload: Any, target_key: str) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _normalize_key(str(key)) == _normalize_key(target_key):
                return value
            found = _find_key(value, target_key)
            if found is not None:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_key(item, target_key)
            if found is not None:
                return found
    return None


def _walk_items(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(payload, dict):
        items: list[tuple[str, Any]] = []
        for key, value in payload.items():
            key_text = _normalize_key(str(key))
            full_key = f"{prefix}.{key_text}" if prefix else key_text
            items.append((full_key, value))
            items.extend(_walk_items(value, full_key))
        return items
    if isinstance(payload, list):
        items = []
        for item in payload:
            items.extend(_walk_items(item, prefix))
        return items
    return []


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
        if value is None:
            continue
        text = str(value).lower()
        if "synthetic" in text or "fixture" in text:
            return True
    return False


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
