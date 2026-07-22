from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_TRAINING_CANDIDATE_JSON = "data/colab/accumulated/main_database/pipeline_control_metric_artifacts_manifest.json"
DEFAULT_EVALUATION_CANDIDATE_JSON = "data/results/pipeline_control_evaluation_metric_artifacts_manifest.json"

TRAINING_REQUIRED_METRICS = {
    "train_score": ("train_score", "training_score", "in_sample_score"),
    "validation_score": ("validation_score", "val_score", "test_score", "out_of_sample_score"),
    "sample_count": ("sample_count", "n_samples", "test_samples", "validation_samples", "observations"),
}
EVALUATION_REQUIRED_METRICS = {
    "max_drawdown": ("max_drawdown", "maximum_drawdown", "mdd", "drawdown"),
}
OPTIONAL_EVALUATION_METRICS = {
    "total_return": ("total_return", "return", "realized_return", "pnl_pct"),
    "pnl": ("pnl", "profit", "net_profit"),
    "sharpe": ("sharpe", "sharpe_ratio"),
}
JOIN_REQUIRED_FIELDS = ("ticker", "model", "target_name", "timeframe", "context_fingerprint", "evaluation_window")


class PipelineControlLockedEvaluationAssembler:
    """Assemble a locked model-evaluation artifact only from joined real candidates.

    Training and Stage 7 evaluation artifacts are useful, but dangerous to merge
    casually. This packet writes a locked model-evaluation JSON only when the
    two candidates prove same model/target/context/window lineage.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_control_locked_evaluation_assembler_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        training_candidate_json: str | Path | None = DEFAULT_TRAINING_CANDIDATE_JSON,
        evaluation_candidate_json: str | Path | None = DEFAULT_EVALUATION_CANDIDATE_JSON,
        write_artifact: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        run_id = _run_id("pipeline_control_locked_evaluation_assembler")
        resolved_training_candidate_json = _resolve_manifest_candidate(
            training_candidate_json,
            artifact_types=("model_evaluation_json", "model_evaluation_candidate"),
        )
        resolved_evaluation_candidate_json = _resolve_manifest_candidate(
            evaluation_candidate_json,
            artifact_types=("evaluation_metric_candidate", "evaluation_metric_json"),
        )
        training = _load_json_artifact(resolved_training_candidate_json)
        evaluation = _load_json_artifact(resolved_evaluation_candidate_json)
        checks = _assembly_checks(training, evaluation)
        status = _assembly_status(checks)
        locked_artifact = (
            _build_locked_model_evaluation_artifact(training, evaluation, run_id=run_id)
            if status == "locked_model_evaluation_assembled"
            else None
        )
        materialized_paths: dict[str, str] = {}
        if locked_artifact and write_artifact:
            materialized_paths = ReviewArtifactWriter(self.output_dir / "locked_model_evaluation").write(
                payload=locked_artifact,
                markdown=render_locked_model_evaluation_markdown(locked_artifact),
                run_id=locked_artifact["run_id"],
            )

        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_locked_evaluation_assembler",
            "inputs": {
                "training_candidate_json": str(training_candidate_json) if training_candidate_json else None,
                "evaluation_candidate_json": str(evaluation_candidate_json) if evaluation_candidate_json else None,
                "resolved_training_candidate_json": str(resolved_training_candidate_json) if resolved_training_candidate_json else None,
                "resolved_evaluation_candidate_json": str(resolved_evaluation_candidate_json) if resolved_evaluation_candidate_json else None,
                "write_artifact": write_artifact,
            },
            "summary": _summary(status, checks, materialized_paths),
            "same_window_join_contract": _same_window_join_contract(),
            "input_artifacts": {
                "training_candidate": _artifact_preview(training),
                "evaluation_candidate": _artifact_preview(evaluation),
            },
            "assembly_checks": checks,
            "locked_model_evaluation_artifact": _artifact_output_preview(locked_artifact, materialized_paths),
            "next_runner_inputs": _next_runner_inputs(materialized_paths),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(status, checks),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_control_locked_evaluation_assembler_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_control_locked_evaluation_assembler_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Pipeline Control Locked Evaluation Assembler",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Assembly status: `{summary.get('assembly_status')}`",
        f"- Training candidate available: {summary.get('training_candidate_available')}",
        f"- Evaluation candidate available: {summary.get('evaluation_candidate_available')}",
        f"- Same-window lineage proven: {summary.get('same_window_lineage_proven')}",
        f"- Locked model evaluation written: {summary.get('locked_model_evaluation_written')}",
        f"- Can supply model evaluation to real runner: {summary.get('can_supply_model_evaluation_to_real_runner')}",
        f"- Can run real metric evidence now: {summary.get('can_run_real_metric_evidence_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Assembly Checks",
        "",
    ]
    for check in payload.get("assembly_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")

    lines.extend(["", "## Next Runner Inputs", ""])
    next_inputs = payload.get("next_runner_inputs", {})
    lines.append(f"- Model evaluation JSON: `{next_inputs.get('model_evaluation_json')}`")
    lines.append(f"- Feature stability report required separately: {next_inputs.get('feature_stability_report_required_separately')}")
    lines.append(f"- Can invoke real metric evidence run now: {next_inputs.get('can_invoke_pipeline_control_real_metric_evidence_run')}")

    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def render_locked_model_evaluation_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "# Locked Model Evaluation",
        "",
        f"- Artifact class: `{payload.get('artifact_class')}`",
        f"- Evidence class: `{payload.get('evidence_class')}`",
        f"- Join status: `{payload.get('join_contract', {}).get('join_status')}`",
    ]
    for key in [*EVALUATION_REQUIRED_METRICS, *TRAINING_REQUIRED_METRICS, *OPTIONAL_EVALUATION_METRICS]:
        if key in metrics:
            lines.append(f"- `{key}`: {metrics[key]}")
    lines.extend(["", "## Lineage", ""])
    for key, value in payload.get("joined_lineage", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Safety", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _assembly_checks(training: dict[str, Any], evaluation: dict[str, Any]) -> list[dict[str, str]]:
    checks = []
    checks.append(_availability_check("training_candidate", training))
    checks.append(_availability_check("evaluation_candidate", evaluation))
    if not training["available"] or not evaluation["available"]:
        return checks
    training_payload = training["payload"]
    evaluation_payload = evaluation["payload"]
    checks.append(_not_synthetic_check("training_candidate", training_payload))
    checks.append(_not_synthetic_check("evaluation_candidate", evaluation_payload))
    checks.append(
        _artifact_class_check(
            "training_candidate",
            training_payload,
            "pipeline_control_model_evaluation_candidate",
        )
    )
    checks.append(
        _artifact_class_check(
            "evaluation_candidate",
            evaluation_payload,
            "pipeline_control_evaluation_metric_candidate",
        )
    )
    checks.extend(_required_metric_checks("training", training_payload, TRAINING_REQUIRED_METRICS))
    checks.extend(_required_metric_checks("evaluation", evaluation_payload, EVALUATION_REQUIRED_METRICS))
    checks.extend(_lineage_checks(training_payload, evaluation_payload))
    return checks


def _availability_check(artifact_id: str, artifact: dict[str, Any]) -> dict[str, str]:
    if artifact["available"]:
        return _check("pass", f"{artifact_id}_available", f"Loaded {artifact['path']}.")
    return _check("fail", f"{artifact_id}_available", artifact["message"])


def _not_synthetic_check(artifact_id: str, payload: dict[str, Any]) -> dict[str, str]:
    return _check(
        "fail" if _is_synthetic_or_fixture(payload) else "pass",
        f"{artifact_id}_not_synthetic",
        "Synthetic or fixture artifact rejected." if _is_synthetic_or_fixture(payload) else "No synthetic/fixture marker found.",
    )


def _artifact_class_check(
    artifact_id: str,
    payload: dict[str, Any],
    expected: str,
) -> dict[str, str]:
    actual = str(payload.get("artifact_class") or "")
    return _check(
        "pass" if actual == expected else "fail",
        f"{artifact_id}_class_valid",
        (
            f"Artifact class is {expected}."
            if actual == expected
            else f"Expected artifact_class={expected}, received {actual or 'missing'}."
        ),
    )


def _required_metric_checks(
    artifact_id: str,
    payload: dict[str, Any],
    required: dict[str, tuple[str, ...]],
) -> list[dict[str, str]]:
    checks = []
    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    for canonical, aliases in required.items():
        value = _first_number(metrics, aliases)
        checks.append(
            _check(
                "pass" if value is not None else "fail",
                f"{artifact_id}_{canonical}_present",
                f"{canonical}={value}." if value is not None else f"Missing required metric: {canonical}.",
            )
        )
    return checks


def _lineage_checks(training: dict[str, Any], evaluation: dict[str, Any]) -> list[dict[str, str]]:
    checks = []
    training_lineage = _extract_lineage(training, side="training")
    evaluation_lineage = _extract_lineage(evaluation, side="evaluation")
    for field in JOIN_REQUIRED_FIELDS:
        left = training_lineage.get(field)
        right = evaluation_lineage.get(field)
        if _is_missing(left):
            checks.append(_check("fail", f"training_{field}_present", f"Missing training lineage field: {field}."))
            continue
        checks.append(_check("pass", f"training_{field}_present", f"Training {field}={_display(left)}."))
        if _is_missing(right):
            checks.append(_check("fail", f"evaluation_{field}_present", f"Missing evaluation lineage field: {field}."))
            continue
        checks.append(_check("pass", f"evaluation_{field}_present", f"Evaluation {field}={_display(right)}."))
        checks.append(_match_check(field, left, right))
    return checks


def _match_check(field: str, left: Any, right: Any) -> dict[str, str]:
    if field == "evaluation_window":
        matched = _window_key(left) == _window_key(right) and _window_key(left) is not None
    elif field == "ticker":
        matched = _string_match_or_contains(left, right)
    elif field == "model":
        matched = _model_match(left, right)
    else:
        matched = _normalize_text(left) == _normalize_text(right)
    return _check(
        "pass" if matched else "fail",
        f"{field}_matches",
        f"{field} lineage matches." if matched else f"{field} lineage mismatch: training={_display(left)} evaluation={_display(right)}.",
    )


def _assembly_status(checks: list[dict[str, str]]) -> str:
    if any(check.get("status") == "fail" for check in checks):
        return "blocked_missing_same_window_lineage"
    return "locked_model_evaluation_assembled"


def _summary(status: str, checks: list[dict[str, str]], materialized_paths: dict[str, str]) -> dict[str, Any]:
    locked_path = materialized_paths.get("latest_json")
    return {
        "assembly_status": status,
        "training_candidate_available": _check_passed(checks, "training_candidate_available"),
        "evaluation_candidate_available": _check_passed(checks, "evaluation_candidate_available"),
        "same_window_lineage_proven": status == "locked_model_evaluation_assembled",
        "failed_check_count": sum(1 for check in checks if check.get("status") == "fail"),
        "blocked_check_codes": [check.get("code") for check in checks if check.get("status") == "fail"],
        "locked_model_evaluation_written": bool(locked_path),
        "can_supply_model_evaluation_to_real_runner": bool(locked_path),
        "can_run_real_metric_evidence_now": False,
        "feature_stability_report_required_separately": True,
        "can_clear_current_real_cautions": False,
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _build_locked_model_evaluation_artifact(
    training: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    training_payload = training["payload"]
    evaluation_payload = evaluation["payload"]
    training_metrics = training_payload.get("metrics")
    training_metrics = (
        training_metrics if isinstance(training_metrics, dict) else {}
    )
    evaluation_metrics = evaluation_payload.get("metrics")
    evaluation_metrics = (
        evaluation_metrics if isinstance(evaluation_metrics, dict) else {}
    )
    metrics: dict[str, float] = {}
    for canonical, aliases in EVALUATION_REQUIRED_METRICS.items():
        metrics[canonical] = _first_number(evaluation_metrics, aliases)  # type: ignore[assignment]
    for canonical, aliases in TRAINING_REQUIRED_METRICS.items():
        metrics[canonical] = _first_number(training_metrics, aliases)  # type: ignore[assignment]
    for canonical, aliases in OPTIONAL_EVALUATION_METRICS.items():
        value = _first_number(evaluation_metrics, aliases)
        if value is not None:
            metrics[canonical] = value
    test_score = _first_number(
        training_metrics,
        ("test_score", "out_of_sample_score"),
    )
    if test_score is not None:
        metrics["test_score"] = test_score
    joined_lineage = _joined_lineage(training_payload, evaluation_payload)
    evaluated_at = _evaluation_window_end(
        joined_lineage.get("evaluation_window")
    )
    return {
        "run_id": f"{run_id}_locked_model_evaluation",
        "created_at": utc_now_iso(),
        "artifact_class": "locked_model_evaluation",
        "evidence_class": "assembled_from_joined_training_and_stage_7_evaluation_candidates",
        "source_artifacts": {
            "training_candidate_json": training["path"],
            "evaluation_candidate_json": evaluation["path"],
        },
        "source_artifact_hashes": {
            "training_candidate_sha256": _file_sha256(training["path"]),
            "evaluation_candidate_sha256": _file_sha256(evaluation["path"]),
        },
        "metrics": metrics,
        "joined_lineage": joined_lineage,
        "evaluated_at": evaluated_at,
        "as_of_contract": {
            "evaluation_data_through": evaluated_at,
            "assembled_at": utc_now_iso(),
        },
        "join_contract": {
            "join_status": "same_window_lineage_proven",
            "required_training_metrics": list(TRAINING_REQUIRED_METRICS),
            "required_evaluation_metrics": list(EVALUATION_REQUIRED_METRICS),
            "required_lineage_fields": list(JOIN_REQUIRED_FIELDS),
            "synthetic_or_fixture_allowed": False,
            "partial_artifact_promotion_allowed": False,
        },
        "explicit_non_actions": _explicit_non_actions(),
    }


def _joined_lineage(training: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    training_lineage = _extract_lineage(training, side="training")
    evaluation_lineage = _extract_lineage(evaluation, side="evaluation")
    return {
        key: _display(training_lineage.get(key))
        if key != "evaluation_window"
        else {
            "training": training_lineage.get(key),
            "evaluation": evaluation_lineage.get(key),
        }
        for key in JOIN_REQUIRED_FIELDS
    }


def _evaluation_window_end(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    evaluation = value.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else value
    end = evaluation.get("end")
    return str(end) if end is not None else None


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _same_window_join_contract() -> dict[str, Any]:
    return {
        "required_training_metrics": list(TRAINING_REQUIRED_METRICS),
        "required_evaluation_metrics": list(EVALUATION_REQUIRED_METRICS),
        "required_lineage_fields": list(JOIN_REQUIRED_FIELDS),
        "accepted_output": "locked_model_evaluation",
        "feature_stability_is_not_assembled_here": True,
        "synthetic_or_fixture_allowed": False,
        "supporting_drawdown_only_allowed_as_substitute": False,
    }


def _artifact_preview(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact["available"]:
        return {"available": False, "path": artifact.get("path"), "message": artifact.get("message")}
    payload = artifact["payload"]
    return {
        "available": True,
        "path": artifact.get("path"),
        "artifact_class": _find_key(payload, "artifact_class"),
        "evidence_class": _find_key(payload, "evidence_class"),
        "contract_status": _find_key(payload, "contract_status"),
        "is_synthetic_or_fixture": _is_synthetic_or_fixture(payload),
        "lineage": _extract_lineage(payload, side="preview"),
    }


def _artifact_output_preview(artifact: dict[str, Any] | None, paths: dict[str, str]) -> dict[str, Any]:
    if not artifact:
        return {"available": False, "saved_paths": paths}
    return {
        "available": True,
        "artifact_class": artifact.get("artifact_class"),
        "evidence_class": artifact.get("evidence_class"),
        "metrics": artifact.get("metrics"),
        "joined_lineage": artifact.get("joined_lineage"),
        "saved_paths": paths,
    }


def _next_runner_inputs(materialized_paths: dict[str, str]) -> dict[str, Any]:
    model_path = materialized_paths.get("latest_json")
    return {
        "model_evaluation_json": model_path,
        "feature_stability_report": None,
        "feature_stability_report_required_separately": True,
        "can_invoke_pipeline_control_real_metric_evidence_run": False,
        "command_preview": None,
    }


def _operator_next_steps(status: str, checks: list[dict[str, str]]) -> list[str]:
    if status == "locked_model_evaluation_assembled":
        return [
            "Supply the assembled locked model-evaluation JSON to PipelineControlRealMetricEvidenceRun only after a separate locked feature-stability report exists.",
            "Keep risk, validation, and feature-stability cautions visible until the real metric runner clears the full chain.",
        ]
    failed = [check.get("code") for check in checks if check.get("status") == "fail"]
    return [
        "Do not merge the training candidate and Stage 7 evaluation candidate until all failed join checks are resolved.",
        "Add same-window lineage fields to the pipeline artifacts: ticker, model, target_name, timeframe, context_fingerprint, and evaluation_window.",
        "Current failed checks: " + (", ".join(str(item) for item in failed) if failed else "none"),
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training, evaluation, replay, backtest, or model loading is executed.",
        "No synthetic metric artifact is generated.",
        "No partial training or drawdown artifact is promoted without same-window lineage.",
        "No feature-stability artifact is invented by this assembler.",
        "No production config, learning memory, recommendation, allocation, paper order, broker call, or live trade is generated.",
    ]


def _extract_lineage(payload: dict[str, Any], *, side: str) -> dict[str, Any]:
    return {
        "ticker": _lineage_ticker(payload, side=side),
        "model": _lineage_model(payload, side=side),
        "target_name": _first_value(payload, ("target_name", "target", "target_column")),
        "timeframe": _first_value(payload, ("timeframe", "horizon", "prediction_horizon")),
        "context_fingerprint": _first_value(payload, ("context_fingerprint", "context_key", "regime_fingerprint")),
        "evaluation_window": _lineage_window(payload),
    }


def _lineage_ticker(payload: dict[str, Any], *, side: str) -> Any:
    value = _first_value(payload, ("ticker", "symbol"))
    if value is not None:
        return value
    tickers = _find_key(payload, "tickers")
    if isinstance(tickers, list) and tickers:
        return tickers if side == "evaluation" else tickers[0]
    return None


def _lineage_model(payload: dict[str, Any], *, side: str) -> Any:
    value = _first_value(payload, ("model_id", "model_name", "model_type", "selected_primary_model"))
    if value is not None:
        return value
    selected = _find_key(payload, "selected_primary_models")
    if isinstance(selected, list) and selected:
        return selected if side == "evaluation" else selected[0]
    return None


def _lineage_window(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("evaluation_window", "locked_evaluation_window", "same_window", "window"):
        value = _find_key(payload, key)
        parsed = _parse_window(value)
        if parsed:
            return parsed
    start = _first_value(payload, ("evaluation_start", "window_start", "start_date"))
    end = _first_value(payload, ("evaluation_end", "window_end", "end_date"))
    if start is not None and end is not None:
        return {"start": str(start), "end": str(end)}
    return None


def _parse_window(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        start = value.get("start") or value.get("start_date") or value.get("from")
        end = value.get("end") or value.get("end_date") or value.get("to")
        if start is not None and end is not None:
            return {
                "start": str(start),
                "end": str(end),
                "sample_count": value.get("sample_count"),
                "source": value.get("source"),
            }
    return None


def _window_key(value: Any) -> tuple[str, str] | None:
    parsed = _parse_window(value)
    if not parsed:
        return None
    return (_normalize_text(parsed.get("start")), _normalize_text(parsed.get("end")))


def _model_match(left: Any, right: Any) -> bool:
    return _string_match_or_contains(left, right)


def _string_match_or_contains(left: Any, right: Any) -> bool:
    left_values = _normalized_values(left)
    right_values = _normalized_values(right)
    for left_value in left_values:
        for right_value in right_values:
            if left_value == right_value:
                return True
            if left_value and right_value and (left_value in right_value or right_value in left_value):
                return True
    return False


def _normalized_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_normalize_text(item) for item in value if not _is_missing(item)]
    if _is_missing(value):
        return []
    return [_normalize_text(value)]


def _first_value(payload: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _find_key(payload, key)
        if value is not None:
            return value
    return None


def _first_number(payload: Any, aliases: tuple[str, ...]) -> float | None:
    normalized_aliases = {_normalize_key(alias) for alias in aliases}
    for key, value in _walk_items(payload):
        normalized_key = _normalize_key(key)
        if normalized_key in normalized_aliases or any(normalized_key.endswith(f".{alias}") for alias in normalized_aliases):
            number = _number(value)
            if number is not None:
                return number
    return None


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


def _resolve_manifest_candidate(path: str | Path | None, *, artifact_types: tuple[str, ...]) -> str | Path | None:
    if not path:
        return path
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
        return path
    if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), list):
        return path
    candidates = []
    for item in payload.get("artifacts", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("artifact_type")) not in artifact_types:
            continue
        item_path = item.get("path")
        if item_path:
            candidates.append(Path(str(item_path)))
    if not candidates:
        return path
    existing = [candidate for candidate in candidates if candidate.exists()]
    return existing[-1] if existing else candidates[-1]


def _load_json_artifact(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"available": False, "path": None, "payload": {}, "message": "No path supplied."}
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"available": False, "path": str(artifact_path), "payload": {}, "message": f"Missing file: {artifact_path}."}
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return {"available": False, "path": str(artifact_path), "payload": {}, "message": f"Could not load JSON: {exc}."}
    if not isinstance(payload, dict):
        return {"available": False, "path": str(artifact_path), "payload": {}, "message": "Expected a JSON object."}
    return {"available": True, "path": str(artifact_path), "payload": payload, "message": "loaded"}


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


def _check_passed(checks: list[dict[str, str]], code: str) -> bool:
    return any(check.get("code") == code and check.get("status") == "pass" for check in checks)


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _display(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
