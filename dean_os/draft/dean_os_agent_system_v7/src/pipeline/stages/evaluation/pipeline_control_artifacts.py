from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

COMMON_MODEL_TYPES = (
    "random_forest",
    "linear_regression",
    "logistic_regression",
    "gradient_boosting",
    "extra_trees",
    "lightgbm",
    "catboost",
    "xgboost",
    "autoencoder",
    "transformer",
    "linear",
    "forest",
    "tabnet",
    "lstm",
    "gru",
    "svm",
    "knn",
    "mlp",
)
TIMEFRAME_PATTERN = re.compile(r"(?:^|_)(\d{1,3}(?:m|h|d|w)|1mo|3mo|6mo|1y)(?:_|$)", re.IGNORECASE)


def build_evaluation_metric_candidate(
    *,
    financial_metrics: dict[str, Any],
    backtest_results: dict[str, Any],
    evaluation_summary: dict[str, Any],
    signals_df: Any = None,
    portfolio_history: Any = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a Stage 7 metric candidate without joining it to model training evidence."""
    metrics = {
        "max_drawdown": _first_number(financial_metrics, ["max_drawdown", "max_drawdown_pct", "drawdown"]),
        "total_return": _first_number(financial_metrics, ["total_return", "total_return_pct"]),
        "sharpe": _first_number(financial_metrics, ["sharpe", "sharpe_ratio"]),
        "volatility": _first_number(financial_metrics, ["volatility"]),
        "cagr": _first_number(financial_metrics, ["cagr"]),
    }
    window = _window(portfolio_history, signals_df)
    lineage = _lineage(signals_df, portfolio_history, backtest_results, summary_path)
    missing = [key for key in ("max_drawdown",) if metrics.get(key) is None]
    payload = {
        "artifact_class": "pipeline_control_evaluation_metric_candidate",
        "evidence_class": "pipeline_stage_7_evaluation_output",
        "created_at": _utc_now_iso(),
        "metrics": {key: value for key, value in metrics.items() if value is not None},
        "evaluation_window": window,
        "lineage": lineage,
        "summary_keys": sorted(str(key) for key in evaluation_summary.keys()),
        "contract_status": "evaluation_risk_metrics_available" if not missing else "partial_evaluation_metric_candidate",
        "missing_for_evaluation_metric_candidate": missing,
        "same_window_join_status": "requires_matching_training_candidate_before_locked_model_evaluation",
        "same_window_join_contract": {
            "requires_training_context_key_match": True,
            "requires_evaluation_window_match": True,
            "requires_model_or_run_manifest_match": True,
            "partial_artifacts_must_not_be_promoted": True,
        },
        "explicit_non_actions": _explicit_non_actions(),
    }
    payload.update(_single_context_identity(lineage))
    return payload


def write_evaluation_metric_artifact_candidate(
    *,
    output_dir: str | Path,
    candidate: dict[str, Any],
    context_key: str = "stage_7_evaluation",
) -> dict[str, Any]:
    base_dir = Path(output_dir) / "pipeline_control_evaluation_metric_artifacts"
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_key = _safe_key(context_key)
    candidate_path = base_dir / f"evaluation_metric_{safe_key}.json"
    manifest_path = Path(output_dir) / "pipeline_control_evaluation_metric_artifacts_manifest.json"

    _write_json(candidate_path, candidate)
    manifest = _load_manifest(manifest_path)
    manifest["artifact_class"] = "pipeline_control_evaluation_metric_artifacts_manifest"
    manifest["evidence_class"] = "pipeline_stage_7_evaluation_output_manifest"
    manifest["last_updated_at"] = _utc_now_iso()
    manifest["explicit_non_actions"] = _explicit_non_actions()
    artifacts = [
        item for item in manifest.get("artifacts", [])
        if item.get("context_key") != context_key
    ]
    artifacts.append(
        {
            "artifact_type": "evaluation_metric_candidate",
            "context_key": context_key,
            "path": str(candidate_path),
            "contract_status": candidate.get("contract_status"),
            "same_window_join_status": candidate.get("same_window_join_status"),
            "missing_for_locked_evidence": candidate.get("missing_for_evaluation_metric_candidate", []),
        }
    )
    manifest["artifacts"] = artifacts
    _write_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "evaluation_metric_candidate": str(candidate_path),
        "contract_status": candidate.get("contract_status"),
        "same_window_join_status": candidate.get("same_window_join_status"),
    }


def _lineage(signals_df: Any, portfolio_history: Any, backtest_results: dict[str, Any], summary_path: str | Path | None) -> dict[str, Any]:
    tickers = _unique_column_values(signals_df, "ticker")
    selected_models = _unique_column_values(signals_df, "selected_primary_model")
    model_context_ids = _unique_any_column_values(signals_df, ("model_context_id", "context_id", "model_context"))
    target_names = _unique_any_column_values(signals_df, ("target_name", "target", "target_col", "target_column"))
    model_types = _unique_any_column_values(signals_df, ("model_type", "model_name", "selected_primary_model"))
    timeframes = _unique_any_column_values(signals_df, ("timeframe", "tf", "interval"))
    context_fingerprints = _unique_any_column_values(
        signals_df,
        ("context_fingerprint", "context_pattern_id", "regime_fingerprint"),
    )
    parsed_contexts = _parse_model_contexts(model_context_ids, model_types)
    target_names = _merge_unique(target_names, [item["target_name"] for item in parsed_contexts if item.get("target_name")])
    model_types = _merge_unique(model_types, [item["model_type"] for item in parsed_contexts if item.get("model_type")])
    tickers = _merge_unique(tickers, [item["ticker"] for item in parsed_contexts if item.get("ticker")])
    timeframes = _merge_unique(timeframes, [_timeframe_from_target(target) for target in target_names])
    return {
        "summary_path": str(summary_path) if summary_path else None,
        "signal_count": _row_count(signals_df),
        "portfolio_history_count": _row_count(portfolio_history),
        "ticker_count": len(tickers),
        "tickers": tickers,
        "selected_primary_models": selected_models,
        "model_context_ids": model_context_ids,
        "parsed_model_contexts": parsed_contexts,
        "target_names": target_names,
        "model_types": model_types,
        "timeframes": timeframes,
        "context_fingerprints": context_fingerprints,
        "single_context_join_candidate": _has_single_context_join_fields(
            tickers,
            model_types,
            target_names,
            timeframes,
            context_fingerprints,
        ),
        "backtest_result_keys": sorted(str(key) for key in backtest_results.keys()),
        "backtest_performance_present": isinstance(backtest_results.get("performance"), dict),
    }


def _window(portfolio_history: Any, signals_df: Any) -> dict[str, Any]:
    source = portfolio_history if _row_count(portfolio_history) else signals_df
    index = getattr(source, "index", None)
    if index is not None and len(index) > 0:
        return {
            "start": str(index[0]),
            "end": str(index[-1]),
            "sample_count": int(len(index)),
            "source": "portfolio_history_index" if source is portfolio_history else "signals_index",
        }
    return {"start": None, "end": None, "sample_count": 0, "source": "unavailable"}


def _unique_column_values(frame: Any, column: str) -> list[str]:
    columns = getattr(frame, "columns", [])
    if column not in columns:
        return []
    try:
        values = frame[column].dropna().unique().tolist()
    except (ValueError, TypeError, AttributeError, KeyError):
        return []
    return sorted(str(value) for value in values)


def _unique_any_column_values(frame: Any, columns: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for column in columns:
        values.extend(_unique_column_values(frame, column))
    return _dedupe_sorted(values)


def _parse_model_contexts(context_ids: list[str], model_types: list[str]) -> list[dict[str, str]]:
    parsed = []
    known_models = _dedupe_sorted([*model_types, *COMMON_MODEL_TYPES])
    for context_id in context_ids:
        normalized = str(context_id).strip()
        if normalized.startswith("model_"):
            normalized = normalized.removeprefix("model_")
        for model_type in sorted(known_models, key=len, reverse=True):
            suffix = f"_{model_type}"
            if not normalized.lower().endswith(suffix.lower()):
                continue
            rest = normalized[: -len(suffix)]
            if "_" not in rest:
                continue
            ticker, target_name = rest.split("_", 1)
            parsed.append(
                {
                    "context_id": str(context_id),
                    "ticker": ticker,
                    "target_name": target_name,
                    "model_type": model_type,
                }
            )
            break
    return parsed


def _timeframe_from_target(target_name: str | None) -> str | None:
    if not target_name:
        return None
    match = TIMEFRAME_PATTERN.search(str(target_name))
    return match.group(1) if match else None


def _single_context_identity(lineage: dict[str, Any]) -> dict[str, str]:
    identity: dict[str, str] = {}
    field_map = {
        "ticker": "tickers",
        "model_type": "model_types",
        "target_name": "target_names",
        "timeframe": "timeframes",
        "context_fingerprint": "context_fingerprints",
    }
    for output_field, lineage_field in field_map.items():
        values = lineage.get(lineage_field, [])
        if isinstance(values, list) and len(values) == 1 and values[0] not in (None, ""):
            identity[output_field] = str(values[0])
    return identity


def _has_single_context_join_fields(
    tickers: list[str],
    model_types: list[str],
    target_names: list[str],
    timeframes: list[str],
    context_fingerprints: list[str],
) -> bool:
    return all(len(values) == 1 for values in (tickers, model_types, target_names, timeframes, context_fingerprints))


def _merge_unique(left: list[str], right: list[str | None]) -> list[str]:
    return _dedupe_sorted([*left, *[item for item in right if item]])


def _dedupe_sorted(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def _row_count(frame: Any) -> int:
    try:
        return int(len(frame))
    except TypeError:
        return 0


def _first_number(payload: dict[str, Any], keys: list[str]) -> float | None:
    normalized = {_normalize_key(key) for key in keys}
    for key, value in payload.items():
        if _normalize_key(str(key)) in normalized:
            number = _number(value)
            if number is not None:
                return number
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


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


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training or replay is started by this artifact writer.",
        "No evaluation metric is joined to model-training evidence without a matching same-window contract.",
        "No recommendation, allocation, order, broker call, paper trade, or live trade is generated.",
    ]
