from __future__ import annotations

import datetime as _dt
import json
import math
import re
from pathlib import Path
from typing import Any

MODEL_EVALUATION_REQUIRED_FIELDS = (
    "max_drawdown",
    "train_score",
    "validation_score",
    "sample_count",
)
FEATURE_STABILITY_REQUIRED_FIELDS = (
    "feature_importance",
    "stability_signal",
)


def build_model_evaluation_candidate(
    *,
    ticker: str,
    target_name: str,
    model_type: str,
    timeframe: str,
    context_fingerprint: str,
    market_regime: str,
    volatility_regime: str,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    train_sample_count: int,
    validation_sample_count: int,
    test_metrics: dict[str, Any] | None = None,
    test_sample_count: int = 0,
    max_drawdown: float | None = None,
    evaluation_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a saved training evidence candidate without inventing missing risk metrics."""
    train_score = _number(train_metrics.get("score"))
    validation_score = _number(validation_metrics.get("score"))
    test_score = _number((test_metrics or {}).get("score"))
    sample_count = int(train_sample_count) + int(validation_sample_count) + int(test_sample_count)
    parsed_window = _parse_evaluation_window(evaluation_window)

    metrics: dict[str, Any] = {
        "train_score": train_score,
        "validation_score": validation_score,
        "test_score": test_score,
        "sample_count": sample_count,
        "train_sample_count": int(train_sample_count),
        "validation_sample_count": int(validation_sample_count),
        "test_sample_count": int(test_sample_count),
    }
    if max_drawdown is not None:
        metrics["max_drawdown"] = _number(max_drawdown)

    missing = [field for field in MODEL_EVALUATION_REQUIRED_FIELDS if metrics.get(field) is None]
    payload = {
        "artifact_class": "pipeline_control_model_evaluation_candidate",
        "evidence_class": "pipeline_training_output",
        "created_at": _utc_now_iso(),
        "ticker": ticker,
        "target_name": target_name,
        "model_type": model_type,
        "timeframe": timeframe,
        "context_fingerprint": context_fingerprint,
        "market_regime": market_regime,
        "volatility_regime": volatility_regime,
        "metrics": metrics,
        "split_metrics": {
            "train": _json_ready(train_metrics),
            "validation": _json_ready(validation_metrics),
            "test": _json_ready(test_metrics or {}),
        },
        "contract_status": "ready_locked_model_evaluation_candidate" if not missing else "partial_model_evaluation_candidate",
        "missing_for_locked_model_evaluation": missing,
        "same_window_contract": {
            "same_model_and_target_required": True,
            "same_locked_evaluation_window_required": True,
            "max_drawdown_source": "same_window_supplied" if max_drawdown is not None else "not_supplied_by_training_stage",
            "evaluation_window_source": parsed_window.get("source") if parsed_window else "not_supplied_by_training_stage",
        },
        "explicit_non_actions": _explicit_non_actions(),
    }
    if parsed_window:
        payload["evaluation_window"] = parsed_window
    return payload


def build_feature_stability_candidate(
    *,
    ticker: str,
    target_name: str,
    model_type: str,
    timeframe: str,
    context_fingerprint: str,
    market_regime: str,
    volatility_regime: str,
    feature_importance: dict[str, float],
    stability_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a feature evidence candidate and keep stability absent unless measured."""
    payload: dict[str, Any] = {
        "artifact_class": "pipeline_control_feature_stability_candidate",
        "evidence_class": "pipeline_training_output",
        "created_at": _utc_now_iso(),
        "ticker": ticker,
        "target_name": target_name,
        "model_type": model_type,
        "timeframe": timeframe,
        "context_fingerprint": context_fingerprint,
        "market_regime": market_regime,
        "volatility_regime": volatility_regime,
        "feature_importance": _json_ready(feature_importance),
        "feature_importance_status": "measured_from_trained_model" if feature_importance else "not_available_from_model",
        "feature_importance_count": len(feature_importance),
        "feature_importance_normalized": bool(feature_importance),
        "stability_signal_status": "not_measured",
        "contract_status": "partial_feature_stability_candidate",
        "missing_for_locked_feature_stability": ["stability_signal"],
        "explicit_non_actions": _explicit_non_actions(),
    }

    if stability_analysis:
        payload["feature_stability_analysis"] = _json_ready(stability_analysis)
        stability_score = _number(
            stability_analysis.get("feature_stability_score", stability_analysis.get("stability_score"))
        )
        unstable_features = stability_analysis.get("unstable_features")
        unstable_count = stability_analysis.get("unstable_feature_count")
        if unstable_count is None and isinstance(unstable_features, list):
            unstable_count = len(unstable_features)
        if stability_score is not None:
            payload["feature_stability_score"] = stability_score
        if unstable_count is not None:
            payload["unstable_feature_count"] = int(unstable_count)
        if isinstance(unstable_features, list):
            payload["unstable_features"] = [str(item) for item in unstable_features]
        if stability_score is not None or unstable_count is not None or isinstance(unstable_features, list):
            payload["stability_signal_status"] = "measured"
            payload["stability_signal"] = {
                "status": "measured",
                "source": stability_analysis.get("analysis_method", "supplied_stability_analysis"),
            }
            payload["contract_status"] = (
                "ready_feature_stability_candidate" if feature_importance else "partial_feature_stability_candidate"
            )
            payload["missing_for_locked_feature_stability"] = [] if feature_importance else ["feature_importance"]

    return payload


def build_feature_distribution_stability_analysis(
    train_features: Any,
    validation_features: Any,
    feature_names: list[str],
    *,
    drift_threshold: float = 1.0,
    min_samples_per_split: int = 2,
) -> dict[str, Any]:
    """Measure selected-feature distribution drift from already-split train/validation data."""
    features = [str(feature) for feature in feature_names if str(feature)]
    threshold = float(drift_threshold) if _number(drift_threshold) and float(drift_threshold) > 0 else 1.0
    min_samples = max(1, int(min_samples_per_split))
    if not features:
        return {
            "analysis_method": "train_validation_distribution_drift_v1",
            "measurement_status": "not_measured_no_selected_features",
            "feature_count": 0,
            "explicit_non_actions": _explicit_non_actions(),
        }

    drift_by_feature: dict[str, dict[str, Any]] = {}
    skipped_features: dict[str, dict[str, Any]] = {}
    stable_features: list[str] = []
    unstable_features: list[str] = []
    bounded_drift_total = 0.0

    for feature_index, feature in enumerate(features):
        train_values = _finite_feature_values(train_features, feature, feature_index)
        validation_values = _finite_feature_values(validation_features, feature, feature_index)
        if len(train_values) < min_samples or len(validation_values) < min_samples:
            skipped_features[feature] = {
                "train_sample_count": len(train_values),
                "validation_sample_count": len(validation_values),
                "reason": "insufficient_finite_samples",
            }
            continue

        train_stats = _distribution_stats(train_values)
        validation_stats = _distribution_stats(validation_values)
        mean_shift = _scaled_abs_difference(
            validation_stats["mean"],
            train_stats["mean"],
            max(abs(train_stats["mean"]), train_stats["std"]),
        )
        std_shift = _scaled_abs_difference(validation_stats["std"], train_stats["std"], train_stats["std"])
        drift_score = max(mean_shift, std_shift)
        is_unstable = drift_score > threshold
        if is_unstable:
            unstable_features.append(feature)
        else:
            stable_features.append(feature)
        bounded_drift_total += min(drift_score / threshold, 1.0)
        drift_by_feature[feature] = {
            "train_sample_count": len(train_values),
            "validation_sample_count": len(validation_values),
            "train_mean": round(train_stats["mean"], 10),
            "validation_mean": round(validation_stats["mean"], 10),
            "train_std": round(train_stats["std"], 10),
            "validation_std": round(validation_stats["std"], 10),
            "mean_shift_ratio": round(mean_shift, 10),
            "std_shift_ratio": round(std_shift, 10),
            "drift_score": round(drift_score, 10),
            "status": "unstable" if is_unstable else "stable",
        }

    measured_count = len(drift_by_feature)
    if skipped_features or measured_count != len(features):
        return {
            "analysis_method": "train_validation_distribution_drift_v1",
            "measurement_status": "not_measured_incomplete_feature_coverage",
            "feature_count": len(features),
            "measured_feature_count": measured_count,
            "skipped_feature_count": len(skipped_features),
            "skipped_features": skipped_features,
            "feature_distribution_drift": drift_by_feature,
            "drift_threshold": threshold,
            "min_samples_per_split": min_samples,
            "explicit_non_actions": _explicit_non_actions(),
        }

    stability_score = 1.0 - (bounded_drift_total / measured_count)
    return {
        "analysis_method": "train_validation_distribution_drift_v1",
        "measurement_status": "measured",
        "feature_stability_score": round(max(0.0, min(1.0, stability_score)), 6),
        "stability_score": round(max(0.0, min(1.0, stability_score)), 6),
        "stable_features": stable_features,
        "unstable_features": unstable_features,
        "unstable_feature_count": len(unstable_features),
        "feature_count": len(features),
        "measured_feature_count": measured_count,
        "drift_threshold": threshold,
        "min_samples_per_split": min_samples,
        "feature_distribution_drift": drift_by_feature,
        "explicit_non_actions": _explicit_non_actions(),
    }


def build_split_evaluation_window(split_features: Any, *, source: str = "validation_feature_index") -> dict[str, Any] | None:
    """Read the held-out split window from an existing feature frame index."""
    index = getattr(split_features, "index", None)
    if index is None:
        return None
    try:
        count = int(len(index))
    except TypeError:
        return None
    if count <= 0:
        return None
    return {
        "start": str(index[0]),
        "end": str(index[count - 1]),
        "sample_count": count,
        "source": source,
    }


def extract_native_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Extract native importances/coefs from an already-trained model wrapper."""
    for candidate in _model_candidates(model):
        values = _read_importance_values(candidate)
        parsed = _importance_dict(values, feature_names)
        if parsed:
            return _normalize_importance(parsed)
    return {}


def write_pipeline_control_metric_artifact_candidates(
    *,
    batch_dir: str | Path,
    context_key: str,
    model_evaluation: dict[str, Any],
    feature_stability: dict[str, Any],
) -> dict[str, Any]:
    """Write candidate artifacts and refresh a manifest that inventory/materializer can expand."""
    base_dir = Path(batch_dir) / "pipeline_control_metric_artifacts"
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_key = _safe_key(context_key)
    model_path = base_dir / f"model_evaluation_{safe_key}.json"
    feature_path = base_dir / f"feature_stability_{safe_key}.json"
    manifest_path = Path(batch_dir) / "pipeline_control_metric_artifacts_manifest.json"

    _write_json(model_path, model_evaluation)
    _write_json(feature_path, feature_stability)
    manifest = _load_manifest(manifest_path)
    manifest["artifact_class"] = "pipeline_control_metric_artifacts_manifest"
    manifest["evidence_class"] = "pipeline_training_output_manifest"
    manifest["last_updated_at"] = _utc_now_iso()
    manifest["explicit_non_actions"] = _explicit_non_actions()

    artifacts = [
        item for item in manifest.get("artifacts", [])
        if item.get("context_key") != context_key
    ]
    artifacts.extend(
        [
            {
                "artifact_type": "model_evaluation_json",
                "context_key": context_key,
                "path": str(model_path),
                "contract_status": model_evaluation.get("contract_status"),
                "missing_for_locked_evidence": model_evaluation.get("missing_for_locked_model_evaluation", []),
            },
            {
                "artifact_type": "feature_stability_report",
                "context_key": context_key,
                "path": str(feature_path),
                "contract_status": feature_stability.get("contract_status"),
                "missing_for_locked_evidence": feature_stability.get("missing_for_locked_feature_stability", []),
            },
        ]
    )
    manifest["artifacts"] = artifacts
    _write_json(manifest_path, manifest)

    return {
        "manifest": str(manifest_path),
        "model_evaluation_json": str(model_path),
        "feature_stability_report": str(feature_path),
        "model_evaluation_contract_status": model_evaluation.get("contract_status"),
        "feature_stability_contract_status": feature_stability.get("contract_status"),
    }


def _model_candidates(model: Any) -> list[Any]:
    candidates = []
    for candidate in (model, getattr(model, "model", None), getattr(model, "estimator", None)):
        if candidate is not None and all(candidate is not seen for seen in candidates):
            candidates.append(candidate)
    return candidates


def _read_importance_values(model: Any) -> Any:
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        return model.coef_
    getter = getattr(model, "get_feature_importance", None)
    if callable(getter):
        try:
            return getter()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError):
            return None
    return None


def _importance_dict(values: Any, feature_names: list[str]) -> dict[str, float]:
    if values is None:
        return {}
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list | tuple):
        return {}
    flattened = _flatten_numbers(values)
    if len(flattened) != len(feature_names):
        return {}
    return {
        str(feature): abs(float(value))
        for feature, value in zip(feature_names, flattened, strict=False)
        if _number(value) is not None
    }


def _flatten_numbers(values: Any) -> list[float]:
    if hasattr(values, "tolist") and not isinstance(values, str | bytes):
        try:
            values = values.tolist()
        except (ValueError, TypeError, AttributeError):
            pass
    if isinstance(values, list | tuple):
        result: list[float] = []
        for item in values:
            result.extend(_flatten_numbers(item))
        return result
    number = _number(values)
    return [] if number is None else [number]


def _normalize_importance(values: dict[str, float]) -> dict[str, float]:
    total = sum(abs(value) for value in values.values())
    if total <= 0:
        return {}
    normalized = {feature: abs(float(value)) / total for feature, value in values.items()}
    return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))


def _finite_feature_values(frame: Any, feature_name: str, feature_index: int) -> list[float]:
    return [
        value for value in _flatten_numbers(_feature_values(frame, feature_name, feature_index))
        if math.isfinite(value)
    ]


def _feature_values(frame: Any, feature_name: str, feature_index: int) -> Any:
    if frame is None:
        return []
    try:
        return frame[feature_name]
    except (KeyError, TypeError, AttributeError, IndexError, ValueError):
        pass

    rows = frame.tolist() if hasattr(frame, "tolist") else frame
    if isinstance(rows, dict):
        return rows.get(feature_name, [])
    if not isinstance(rows, list | tuple):
        try:
            rows = list(rows)
        except TypeError:
            return []

    values = []
    for row in rows:
        if isinstance(row, dict):
            if feature_name in row:
                values.append(row[feature_name])
            continue
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, list | tuple) and feature_index < len(row):
            values.append(row[feature_index])
    return values


def _distribution_stats(values: list[float]) -> dict[str, float]:
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {"mean": mean, "std": math.sqrt(variance)}


def _scaled_abs_difference(current: float, baseline: float, scale: float) -> float:
    epsilon = 1e-12
    return abs(current - baseline) / max(abs(scale), epsilon)


def _parse_evaluation_window(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    start = value.get("start") or value.get("start_date") or value.get("from")
    end = value.get("end") or value.get("end_date") or value.get("to")
    if start is None or end is None:
        return None
    return {
        "start": str(start),
        "end": str(end),
        "sample_count": value.get("sample_count"),
        "source": value.get("source"),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"created_at": _utc_now_iso(), "artifacts": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {"created_at": _utc_now_iso(), "artifacts": []}
    if not isinstance(payload, dict):
        return {"created_at": _utc_now_iso(), "artifacts": []}
    if not isinstance(payload.get("artifacts"), list):
        payload["artifacts"] = []
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError, AttributeError):
            return str(value)
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No external API call is made.",
        "No recommendation, allocation, order, broker call, paper trade, or live trade is generated.",
        "No missing drawdown or feature-stability signal is synthesized.",
        "Partial candidates do not clear pipeline-control cautions.",
    ]
