from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport
from dean_os.utils import clamp


METRIC_ALIASES = {
    "validation_score": (
        "validation_score",
        "val_score",
        "out_of_sample_score",
        "walk_forward_score",
        "wf_score",
        "model_score",
        "score",
        "test_score",
        "f1",
        "auc",
        "accuracy",
        "r2",
    ),
    "sharpe": ("sharpe", "sharpe_ratio", "test_sharpe", "validation_sharpe"),
    "max_drawdown": ("max_drawdown", "maximum_drawdown", "mdd", "drawdown"),
    "win_rate": ("win_rate", "hit_rate", "directional_accuracy"),
    "sample_count": ("sample_count", "n_samples", "test_samples", "validation_samples", "observations", "row_count"),
}

TIMESTAMP_ALIASES = ("evaluated_at", "created_at", "timestamp", "completed_at", "run_at")


class ModelPerformanceAgent(BaseAgent):
    """Reads local evaluation/backtest metrics and reports model readiness."""

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        metrics = inspect_model_performance(
            performance_path=self.config.get("performance_path"),
            pipeline_result=context.pipeline_result,
            min_validation_score=float(self.config.get("min_validation_score", 0.55)),
            min_sharpe=float(self.config.get("min_sharpe", 0.0)),
            max_drawdown=float(self.config.get("max_drawdown", 0.25)),
            min_sample_count=int(self.config.get("min_sample_count", 50)),
            max_age_hours=float(self.config.get("max_age_hours", 24 * 30)),
            as_of=_parse_datetime(self.config.get("as_of")) if self.config.get("as_of") else datetime.now(UTC),
        )
        context.metadata["model_performance"] = metrics

        verdict = metrics["verdict"]
        if verdict == "clear":
            reasons = ["Model performance checks passed against configured thresholds."]
            risks = []
            quality_score = 0.9
            confidence = 0.82
        else:
            reasons = metrics.get("reasons") or ["Model performance is unavailable or below review thresholds."]
            risks = [
                "Model promotion, tuning, or paper-trading use should remain gated until reviewed evaluation evidence is available."
            ]
            quality_score = 0.45 if metrics.get("status") == "ok" else 0.25
            confidence = 0.75

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=confidence,
            data_quality_score=quality_score,
            signal_strength=metrics.get("signal_strength", 0.0),
            reasons=reasons,
            risks=risks,
            blind_spots=[
                "This agent reads supplied evaluation metrics only; it does not rerun backtests, verify leakage, or retrain models."
            ],
            evidence=[
                self.evidence("file", str(metrics.get("performance_path")), "performance_path", metrics.get("performance_path")),
                self.evidence("metric", "model_performance", "performance_score", metrics.get("performance_score")),
                self.evidence("metric", "model_performance", "threshold_failures", metrics.get("threshold_failures", [])),
                self.evidence("metric", "model_performance", "metric_count", len(metrics.get("metrics", {}))),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot=metrics,
        )


def inspect_model_performance(
    performance_path: str | Path | None = None,
    pipeline_result: dict[str, Any] | None = None,
    min_validation_score: float = 0.55,
    min_sharpe: float = 0.0,
    max_drawdown: float = 0.25,
    min_sample_count: int = 50,
    max_age_hours: float = 24 * 30,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    as_of = as_of or datetime.now(UTC)
    raw_payload, source_path, unavailable_reason = _load_metric_payload(performance_path, pipeline_result)
    if unavailable_reason:
        return {
            "status": "unavailable",
            "verdict": "caution",
            "performance_path": str(source_path) if source_path else None,
            "reason": unavailable_reason,
            "reasons": [unavailable_reason],
            "metrics": {},
            "thresholds": _thresholds(min_validation_score, min_sharpe, max_drawdown, min_sample_count, max_age_hours),
            "threshold_failures": ["missing_evaluation_metrics"],
            "performance_score": 0.0,
            "signal_strength": 0.0,
            "as_of": as_of.isoformat(),
        }

    flat_payload = _flatten_payload(raw_payload)
    metrics = _extract_metrics(flat_payload)
    evaluated_at = _extract_timestamp(flat_payload)
    threshold_failures, reasons = _threshold_failures(
        metrics=metrics,
        evaluated_at=evaluated_at,
        as_of=as_of,
        min_validation_score=min_validation_score,
        min_sharpe=min_sharpe,
        max_drawdown=max_drawdown,
        min_sample_count=min_sample_count,
        max_age_hours=max_age_hours,
    )
    if not metrics:
        threshold_failures.append("missing_recognized_metrics")
        reasons.append("No recognized model performance metrics were found.")

    performance_score = _performance_score(metrics)
    signal_strength = clamp(performance_score * 2 - 1, -1.0, 1.0)
    verdict = "clear" if metrics and not threshold_failures else "caution"

    return {
        "status": "ok",
        "verdict": verdict,
        "performance_path": str(source_path) if source_path else None,
        "metrics": metrics,
        "evaluated_at": evaluated_at.isoformat() if evaluated_at else None,
        "thresholds": _thresholds(min_validation_score, min_sharpe, max_drawdown, min_sample_count, max_age_hours),
        "threshold_failures": threshold_failures,
        "reasons": reasons,
        "performance_score": round(performance_score, 4),
        "signal_strength": round(signal_strength, 4),
        "as_of": as_of.isoformat(),
    }


def _load_metric_payload(
    performance_path: str | Path | None,
    pipeline_result: dict[str, Any] | None,
) -> tuple[Any | None, Path | None, str | None]:
    if performance_path:
        path = Path(performance_path)
        if not path.exists():
            return None, path, f"Model performance artifact does not exist: {path}"
        try:
            return _read_metric_file(path), path, None
        except Exception as exc:
            return None, path, f"Could not read model performance artifact: {type(exc).__name__}: {exc}"
    if pipeline_result:
        return pipeline_result, None, None
    return None, None, "No model performance artifact or pipeline_result metrics were supplied."


def _read_metric_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError("CSV metric artifact is empty.")
        return frame.iloc[-1].to_dict()
    raise ValueError(f"Unsupported model performance artifact type: {path.suffix}")


def _flatten_payload(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, list):
        if not payload:
            return {}
        return _flatten_payload(payload[-1], prefix=prefix)
    if not isinstance(payload, dict):
        return {}
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_payload(value, key_text))
        else:
            flattened[_normalize_key(key_text)] = value
            flattened[_normalize_key(str(key))] = value
    return flattened


def _extract_metrics(flat_payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        value = _first_numeric(flat_payload, aliases)
        if value is None:
            continue
        if canonical in {"validation_score", "win_rate"}:
            value = _normalize_ratio(value)
        if canonical == "max_drawdown":
            value = abs(value)
        metrics[canonical] = round(value, 6)
    return metrics


def _threshold_failures(
    metrics: dict[str, float],
    evaluated_at: datetime | None,
    as_of: datetime,
    min_validation_score: float,
    min_sharpe: float,
    max_drawdown: float,
    min_sample_count: int,
    max_age_hours: float,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    reasons: list[str] = []
    validation_score = metrics.get("validation_score")
    sharpe = metrics.get("sharpe")
    drawdown = metrics.get("max_drawdown")
    sample_count = metrics.get("sample_count")

    if validation_score is not None and validation_score < min_validation_score:
        failures.append("validation_score_below_threshold")
        reasons.append(f"Validation score {validation_score:.3f} is below threshold {min_validation_score:.3f}.")
    if sharpe is not None and sharpe < min_sharpe:
        failures.append("sharpe_below_threshold")
        reasons.append(f"Sharpe {sharpe:.3f} is below threshold {min_sharpe:.3f}.")
    if drawdown is not None and drawdown > max_drawdown:
        failures.append("drawdown_above_threshold")
        reasons.append(f"Max drawdown {drawdown:.3f} exceeds threshold {max_drawdown:.3f}.")
    if sample_count is not None and sample_count < min_sample_count:
        failures.append("sample_count_below_threshold")
        reasons.append(f"Sample count {sample_count:.0f} is below threshold {min_sample_count}.")
    if evaluated_at is None:
        failures.append("missing_evaluation_timestamp")
        reasons.append("No evaluation timestamp was supplied.")
    else:
        age_hours = (as_of - evaluated_at).total_seconds() / 3600
        if age_hours > max_age_hours:
            failures.append("evaluation_artifact_stale")
            reasons.append(f"Evaluation artifact is stale: age {age_hours:.1f}h exceeds {max_age_hours:.1f}h.")
        if age_hours < 0:
            failures.append("evaluation_timestamp_in_future")
            reasons.append("Evaluation timestamp is in the future relative to the evaluation clock.")
    return failures, reasons


def _performance_score(metrics: dict[str, float]) -> float:
    components: list[float] = []
    if "validation_score" in metrics:
        components.append(metrics["validation_score"])
    if "win_rate" in metrics:
        components.append(metrics["win_rate"])
    if "sharpe" in metrics:
        components.append(clamp((metrics["sharpe"] + 1.0) / 3.0, 0.0, 1.0))
    if "max_drawdown" in metrics:
        components.append(clamp(1.0 - metrics["max_drawdown"], 0.0, 1.0))
    if not components:
        return 0.0
    return clamp(sum(components) / len(components), 0.0, 1.0)


def _extract_timestamp(flat_payload: dict[str, Any]) -> datetime | None:
    for alias in TIMESTAMP_ALIASES:
        value = flat_payload.get(_normalize_key(alias))
        if value is None:
            continue
        try:
            return _parse_datetime(str(value))
        except ValueError:
            continue
    return None


def _first_numeric(flat_payload: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        value = flat_payload.get(_normalize_key(alias))
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _normalize_key(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _normalize_ratio(value: float) -> float:
    if value > 1.0:
        value = value / 100.0
    return clamp(value, 0.0, 1.0)


def _thresholds(
    min_validation_score: float,
    min_sharpe: float,
    max_drawdown: float,
    min_sample_count: int,
    max_age_hours: float,
) -> dict[str, Any]:
    return {
        "min_validation_score": min_validation_score,
        "min_sharpe": min_sharpe,
        "max_drawdown": max_drawdown,
        "min_sample_count": min_sample_count,
        "max_age_hours": max_age_hours,
    }


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
