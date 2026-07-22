from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_FEATURE_STABILITY_CANDIDATE_JSON = "data/colab/accumulated/main_database/pipeline_control_metric_artifacts_manifest.json"

LINEAGE_REQUIRED_FIELDS = ("ticker", "model", "target_name", "timeframe", "context_fingerprint")
ACCEPTED_READY_STATUSES = {"ready_feature_stability_candidate", "locked_feature_stability_report"}


class PipelineControlLockedFeatureStabilityAssembler:
    """Assemble a locked feature-stability report from a measured candidate.

    This is intentionally separate from model-evaluation assembly. Feature
    importances alone are not enough; a real stability signal must be present.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_control_locked_feature_stability_assembler_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        feature_stability_candidate_json: str | Path | None = DEFAULT_FEATURE_STABILITY_CANDIDATE_JSON,
        write_artifact: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        run_id = _run_id("pipeline_control_locked_feature_stability_assembler")
        resolved_candidate_json = _resolve_manifest_candidate(
            feature_stability_candidate_json,
            artifact_types=("feature_stability_report", "feature_stability_candidate"),
        )
        candidate = _load_json_artifact(resolved_candidate_json)
        checks = _assembly_checks(candidate)
        status = _assembly_status(checks)
        locked_artifact = (
            _build_locked_feature_stability_artifact(candidate, run_id=run_id)
            if status == "locked_feature_stability_assembled"
            else None
        )
        materialized_paths: dict[str, str] = {}
        if locked_artifact and write_artifact:
            materialized_paths = ReviewArtifactWriter(self.output_dir / "locked_feature_stability").write(
                payload=locked_artifact,
                markdown=render_locked_feature_stability_markdown(locked_artifact),
                run_id=locked_artifact["run_id"],
            )

        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_locked_feature_stability_assembler",
            "inputs": {
                "feature_stability_candidate_json": str(feature_stability_candidate_json)
                if feature_stability_candidate_json
                else None,
                "resolved_feature_stability_candidate_json": str(resolved_candidate_json)
                if resolved_candidate_json
                else None,
                "write_artifact": write_artifact,
            },
            "summary": _summary(status, checks, materialized_paths),
            "locked_feature_stability_contract": _locked_feature_stability_contract(),
            "input_artifact": _artifact_preview(candidate),
            "assembly_checks": checks,
            "locked_feature_stability_artifact": _artifact_output_preview(locked_artifact, materialized_paths),
            "next_runner_inputs": _next_runner_inputs(materialized_paths),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(status, checks),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_control_locked_feature_stability_assembler_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_control_locked_feature_stability_assembler_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Pipeline Control Locked Feature Stability Assembler",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Assembly status: `{summary.get('assembly_status')}`",
        f"- Candidate available: {summary.get('feature_stability_candidate_available')}",
        f"- Importances present: {summary.get('feature_importances_present')}",
        f"- Stability signal present: {summary.get('stability_signal_present')}",
        f"- Lineage present: {summary.get('lineage_present')}",
        f"- Locked feature stability written: {summary.get('locked_feature_stability_written')}",
        f"- Can supply feature stability to real runner: {summary.get('can_supply_feature_stability_to_real_runner')}",
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
    lines.append(f"- Feature stability report: `{next_inputs.get('feature_stability_report')}`")
    lines.append(f"- Model evaluation JSON required separately: {next_inputs.get('model_evaluation_json_required_separately')}")
    lines.append(f"- Can invoke real metric evidence run now: {next_inputs.get('can_invoke_pipeline_control_real_metric_evidence_run')}")

    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def render_locked_feature_stability_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Locked Feature Stability Report",
        "",
        f"- Artifact class: `{payload.get('artifact_class')}`",
        f"- Evidence class: `{payload.get('evidence_class')}`",
        f"- Feature count: {payload.get('feature_importance_count')}",
        f"- Stability score: {payload.get('feature_stability_score')}",
        f"- Unstable feature count: {payload.get('unstable_feature_count')}",
        "",
        "## Lineage",
        "",
    ]
    for key, value in payload.get("training_lineage", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Safety", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _assembly_checks(candidate: dict[str, Any]) -> list[dict[str, str]]:
    checks = [_availability_check("feature_stability_candidate", candidate)]
    if not candidate["available"]:
        return checks
    payload = candidate["payload"]
    importances = _extract_feature_importances(payload)
    signal = _stability_signal(payload)
    checks.extend(
        [
            _not_synthetic_check("feature_stability_candidate", payload),
            _not_partial_contract_check(payload),
            _check(
                "pass" if importances else "fail",
                "feature_importances_present",
                f"{len(importances)} importances recognized." if importances else "Feature importances or weights are missing.",
            ),
            _check(
                "pass" if signal["present"] else "fail",
                "feature_stability_signal_present",
                signal["message"],
            ),
        ]
    )
    checks.extend(_lineage_checks(payload))
    return checks


def _availability_check(artifact_id: str, artifact: dict[str, Any]) -> dict[str, str]:
    if artifact["available"]:
        return _check("pass", f"{artifact_id}_available", f"Loaded {artifact['path']}.")
    return _check("fail", f"{artifact_id}_available", artifact["message"])


def _not_synthetic_check(artifact_id: str, payload: dict[str, Any]) -> dict[str, str]:
    synthetic = _is_synthetic_or_fixture(payload)
    return _check(
        "fail" if synthetic else "pass",
        f"{artifact_id}_not_synthetic",
        "Synthetic or fixture artifact rejected." if synthetic else "No synthetic/fixture marker found.",
    )


def _not_partial_contract_check(payload: dict[str, Any]) -> dict[str, str]:
    contract_status = _find_key(payload, "contract_status")
    if str(contract_status or "").startswith("partial_"):
        return _check("fail", "feature_stability_candidate_not_partial", f"contract_status={contract_status}.")
    if contract_status and str(contract_status) not in ACCEPTED_READY_STATUSES:
        return _check("warn", "feature_stability_candidate_contract_status_unknown", f"contract_status={contract_status}.")
    return _check("pass", "feature_stability_candidate_not_partial", f"contract_status={contract_status or 'not_declared'}.")


def _lineage_checks(payload: dict[str, Any]) -> list[dict[str, str]]:
    lineage = _extract_lineage(payload)
    checks = []
    for field in LINEAGE_REQUIRED_FIELDS:
        value = lineage.get(field)
        checks.append(
            _check(
                "pass" if not _is_missing(value) else "fail",
                f"{field}_present",
                f"{field}={_display(value)}." if not _is_missing(value) else f"Missing lineage field: {field}.",
            )
        )
    return checks


def _assembly_status(checks: list[dict[str, str]]) -> str:
    if any(check.get("status") == "fail" for check in checks):
        return "blocked_missing_measured_feature_stability"
    return "locked_feature_stability_assembled"


def _summary(status: str, checks: list[dict[str, str]], materialized_paths: dict[str, str]) -> dict[str, Any]:
    locked_path = materialized_paths.get("latest_json")
    return {
        "assembly_status": status,
        "feature_stability_candidate_available": _check_passed(checks, "feature_stability_candidate_available"),
        "feature_importances_present": _check_passed(checks, "feature_importances_present"),
        "stability_signal_present": _check_passed(checks, "feature_stability_signal_present"),
        "lineage_present": all(_check_passed(checks, f"{field}_present") for field in LINEAGE_REQUIRED_FIELDS),
        "failed_check_count": sum(1 for check in checks if check.get("status") == "fail"),
        "warning_check_count": sum(1 for check in checks if check.get("status") == "warn"),
        "blocked_check_codes": [check.get("code") for check in checks if check.get("status") == "fail"],
        "locked_feature_stability_written": bool(locked_path),
        "can_supply_feature_stability_to_real_runner": bool(locked_path),
        "can_run_real_metric_evidence_now": False,
        "model_evaluation_json_required_separately": True,
        "can_clear_current_real_cautions": False,
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _build_locked_feature_stability_artifact(candidate: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    payload = candidate["payload"]
    importances = _extract_feature_importances(payload)
    signal = _stability_signal(payload)
    unstable_features = signal.get("unstable_features") or []
    unstable_count = signal.get("unstable_feature_count")
    if unstable_count is None and isinstance(unstable_features, list):
        unstable_count = len(unstable_features)
    return {
        "run_id": f"{run_id}_locked_feature_stability",
        "created_at": utc_now_iso(),
        "artifact_class": "locked_feature_stability_report",
        "evidence_class": "assembled_from_measured_feature_stability_candidate",
        "source_artifacts": {
            "feature_stability_candidate_json": candidate["path"],
        },
        "training_lineage": _extract_lineage(payload),
        "feature_importance": importances,
        "feature_importance_count": len(importances),
        "feature_stability_score": signal.get("feature_stability_score"),
        "unstable_feature_count": int(unstable_count) if unstable_count is not None else None,
        "unstable_features": [str(item) for item in unstable_features] if isinstance(unstable_features, list) else [],
        "stability_signal": {
            "present": True,
            "source_fields": signal.get("source_fields", []),
            "raw_status": signal.get("raw_status"),
        },
        "assembly_contract": {
            "importances_required": True,
            "measured_stability_signal_required": True,
            "required_lineage_fields": list(LINEAGE_REQUIRED_FIELDS),
            "synthetic_or_fixture_allowed": False,
            "partial_artifact_promotion_allowed": False,
            "selected_feature_manifest_allowed_as_substitute": False,
        },
        "explicit_non_actions": _explicit_non_actions(),
    }


def _locked_feature_stability_contract() -> dict[str, Any]:
    return {
        "accepted_output": "locked_feature_stability_report",
        "required_fields": [
            "feature_importance or feature_importances or feature_weights",
            "feature_stability_score or stability_score or unstable_feature_count or unstable_features or stability_signal",
        ],
        "required_lineage_fields": list(LINEAGE_REQUIRED_FIELDS),
        "model_evaluation_is_not_assembled_here": True,
        "synthetic_or_fixture_allowed": False,
        "selected_feature_manifest_allowed_as_substitute": False,
        "partial_artifact_promotion_allowed": False,
    }


def _artifact_preview(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact["available"]:
        return {"available": False, "path": artifact.get("path"), "message": artifact.get("message")}
    payload = artifact["payload"]
    signal = _stability_signal(payload)
    return {
        "available": True,
        "path": artifact.get("path"),
        "artifact_class": _find_key(payload, "artifact_class"),
        "evidence_class": _find_key(payload, "evidence_class"),
        "contract_status": _find_key(payload, "contract_status"),
        "is_synthetic_or_fixture": _is_synthetic_or_fixture(payload),
        "feature_importance_count": len(_extract_feature_importances(payload)),
        "stability_signal_present": signal["present"],
        "lineage": _extract_lineage(payload),
    }


def _artifact_output_preview(artifact: dict[str, Any] | None, paths: dict[str, str]) -> dict[str, Any]:
    if not artifact:
        return {"available": False, "saved_paths": paths}
    return {
        "available": True,
        "artifact_class": artifact.get("artifact_class"),
        "evidence_class": artifact.get("evidence_class"),
        "training_lineage": artifact.get("training_lineage"),
        "feature_importance_count": artifact.get("feature_importance_count"),
        "feature_stability_score": artifact.get("feature_stability_score"),
        "unstable_feature_count": artifact.get("unstable_feature_count"),
        "saved_paths": paths,
    }


def _next_runner_inputs(materialized_paths: dict[str, str]) -> dict[str, Any]:
    feature_path = materialized_paths.get("latest_json")
    return {
        "model_evaluation_json": None,
        "model_evaluation_json_required_separately": True,
        "feature_stability_report": feature_path,
        "can_invoke_pipeline_control_real_metric_evidence_run": False,
        "command_preview": None,
    }


def _operator_next_steps(status: str, checks: list[dict[str, str]]) -> list[str]:
    if status == "locked_feature_stability_assembled":
        return [
            "Supply the assembled locked feature-stability report to PipelineControlRealMetricEvidenceRun only after a separate locked model-evaluation JSON exists.",
            "Keep risk, validation, and feature-stability cautions visible until the real metric runner clears the full chain.",
        ]
    failed = [check.get("code") for check in checks if check.get("status") == "fail"]
    return [
        "Do not promote feature importances alone to feature-stability evidence.",
        "Add a measured stability signal: feature_stability_score, unstable_feature_count, unstable_features, or stability_signal.",
        "Keep lineage fields on the candidate: ticker, model, target_name, timeframe, and context_fingerprint.",
        "Current failed checks: " + (", ".join(str(item) for item in failed) if failed else "none"),
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training, evaluation, replay, backtest, or model loading is executed.",
        "No synthetic feature-stability artifact is generated.",
        "No selected-feature manifest or feature-importance-only artifact is promoted to locked stability evidence.",
        "No production config, learning memory, recommendation, allocation, paper order, broker call, or live trade is generated.",
    ]


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


def _stability_signal(payload: dict[str, Any]) -> dict[str, Any]:
    score = _first_number(payload, ("feature_stability_score", "stability_score"))
    unstable_features = _find_key(payload, "unstable_features")
    unstable_count = _first_number(payload, ("unstable_feature_count", "unstable_features_count"))
    raw_status = _find_key(payload, "stability_signal")
    source_fields = []
    if score is not None:
        source_fields.append("feature_stability_score")
    if unstable_count is not None:
        source_fields.append("unstable_feature_count")
    if isinstance(unstable_features, list):
        source_fields.append("unstable_features")
    if raw_status is not None:
        source_fields.append("stability_signal")
    present = bool(source_fields)
    return {
        "present": present,
        "message": "Stability signal present via " + ", ".join(source_fields) + "." if present else "Missing measured stability signal.",
        "feature_stability_score": score,
        "unstable_feature_count": unstable_count,
        "unstable_features": unstable_features if isinstance(unstable_features, list) else [],
        "raw_status": raw_status,
        "source_fields": source_fields,
    }


def _extract_lineage(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": _first_value(payload, ("ticker", "symbol")),
        "model": _first_value(payload, ("model_id", "model_name", "model_type", "selected_primary_model")),
        "target_name": _first_value(payload, ("target_name", "target", "target_column")),
        "timeframe": _first_value(payload, ("timeframe", "horizon", "prediction_horizon")),
        "context_fingerprint": _first_value(payload, ("context_fingerprint", "context_key", "regime_fingerprint")),
    }


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
    if _is_missing(value):
        return "missing"
    return str(value)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
