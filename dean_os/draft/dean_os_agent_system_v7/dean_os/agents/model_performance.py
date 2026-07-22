from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_evidence_inventory import (
    verify_locked_model_evaluation,
)
from dean_os.packets.pipeline_model_case_packet import inspect_pipeline_model_case
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
        analyzer_review = (
            context.metadata
            .get("pipeline_review_contract", {})
            .get("stage7_analyzer_review", {})
        )
        if isinstance(analyzer_review, dict):
            metrics["stage7_analyzer_review"] = analyzer_review
        evidence_chain = inspect_real_metric_evidence_chain(
            self.config.get("evidence_chain_path"),
            expected_model_evaluation_path=self.config.get(
                "performance_path"
            ),
        )
        metrics["real_metric_evidence_chain"] = evidence_chain
        model_case = inspect_pipeline_model_case(
            self.config.get("model_case_path"),
            expected_model_evaluation_path=self.config.get(
                "performance_path"
            ),
            expected_evidence_chain_path=self.config.get(
                "evidence_chain_path"
            ),
        )
        metrics["pipeline_model_case"] = model_case
        if (
            self.config.get("performance_path")
            and evidence_chain.get("status")
            != "real_metric_evidence_chain_ready"
        ):
            metrics["verdict"] = "caution"
            metrics["signal_strength"] = 0.0
            metrics.setdefault("threshold_failures", []).append(
                "real_metric_evidence_chain_not_ready"
            )
            metrics.setdefault("reasons", []).append(
                "The locked model artifact is not backed by a ready full "
                "metric-evidence chain."
            )
        if (
            self.config.get("model_case_path")
            and model_case.get("usable_for_review") is not True
        ):
            metrics["verdict"] = "caution"
            metrics["signal_strength"] = 0.0
            metrics.setdefault("threshold_failures", []).append(
                "pipeline_model_case_binding_invalid"
            )
            metrics.setdefault("reasons", []).append(
                "The configured pipeline model case is missing, stale, or "
                "not bound to the current locked model and evidence chain."
            )
        elif model_case.get("status") == "evaluation_block_case_ready":
            metrics["verdict"] = "caution"
            metrics["signal_strength"] = 0.0
            metrics.setdefault("threshold_failures", []).append(
                "pipeline_model_negative_evaluation_case"
            )
            metrics.setdefault("reasons", []).append(
                "A structured negative evaluation case is active for: "
                + ", ".join(
                    model_case.get("blocked_metric_planes", [])
                )
                + "."
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
                self.evidence(
                    "metric",
                    "stage7_analyzer_review",
                    "status",
                    analyzer_review.get("status", "not_reported")
                    if isinstance(analyzer_review, dict)
                    else "not_reported",
                ),
                self.evidence(
                    "metric",
                    "real_metric_evidence_chain",
                    "status",
                    evidence_chain.get("status"),
                ),
                self.evidence(
                    "metric",
                    "pipeline_model_case",
                    "status",
                    model_case.get("status"),
                ),
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
    (
        raw_payload,
        source_path,
        source_contract,
        unavailable_reason,
    ) = _load_metric_payload(performance_path, pipeline_result)
    if unavailable_reason:
        return {
            "status": "unavailable",
            "verdict": "caution",
            "performance_path": str(source_path) if source_path else None,
            "source_contract": source_contract,
            "reason": unavailable_reason,
            "reasons": [unavailable_reason],
            "metrics": {},
            "thresholds": _thresholds(min_validation_score, min_sharpe, max_drawdown, min_sample_count, max_age_hours),
            "threshold_failures": ["missing_evaluation_metrics"],
            "performance_score": 0.0,
            "evaluation_scope": {},
            "evaluation_scope_complete": False,
            "signal_strength": 0.0,
            "as_of": as_of.isoformat(),
            "evidence_chain_required_for_promotion": True,
            "can_promote_model": False,
            "can_write_production_config": False,
            "can_trade": False,
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
    if source_contract == "pipeline_stage7_evaluation_summary_metrics":
        threshold_failures.append(
            "pipeline_metrics_not_locked_model_evidence"
        )
        reasons.append(
            "Stage 7 summary metrics remain supporting pipeline output until "
            "a same-window locked model-evaluation artifact is assembled."
        )
    if not metrics:
        threshold_failures.append("missing_recognized_metrics")
        reasons.append("No recognized model performance metrics were found.")

    performance_score = _performance_score(metrics)
    signal_strength = clamp(performance_score * 2 - 1, -1.0, 1.0)
    verdict = "clear" if metrics and not threshold_failures else "caution"
    evaluation_scope = _extract_evaluation_scope(raw_payload)

    return {
        "status": "ok",
        "verdict": verdict,
        "performance_path": str(source_path) if source_path else None,
        "source_contract": source_contract,
        "evidence_provenance": raw_payload.get(
            "_source_provenance",
            {},
        ),
        "metrics": metrics,
        "evaluated_at": evaluated_at.isoformat() if evaluated_at else None,
        "thresholds": _thresholds(min_validation_score, min_sharpe, max_drawdown, min_sample_count, max_age_hours),
        "threshold_failures": threshold_failures,
        "reasons": reasons,
        "performance_score": round(performance_score, 4),
        "evaluation_scope": evaluation_scope,
        "evaluation_scope_complete": all(
            evaluation_scope.get(field)
            for field in (
                "ticker",
                "model",
                "target_name",
                "timeframe",
                "context_fingerprint",
            )
        ),
        "signal_strength": round(signal_strength, 4),
        "as_of": as_of.isoformat(),
        "evidence_chain_required_for_promotion": True,
        "can_promote_model": False,
        "can_write_production_config": False,
        "can_trade": False,
    }


def _extract_evaluation_scope(
    payload: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        payload.get("joined_lineage"),
        payload.get("lineage"),
        payload.get("model_context"),
        (
            payload.get("case", {}).get("lineage")
            if isinstance(payload.get("case"), dict)
            else None
        ),
        payload,
    ]
    fields = (
        "ticker",
        "model",
        "target_name",
        "timeframe",
        "context_fingerprint",
    )
    scope: dict[str, Any] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for field in fields:
            value = candidate.get(field)
            if field not in scope and value not in {None, ""}:
                scope[field] = value
    return {
        field: scope.get(field) for field in fields if scope.get(field)
    }


def _load_metric_payload(
    performance_path: str | Path | None,
    pipeline_result: dict[str, Any] | None,
) -> tuple[Any | None, Path | None, str, str | None]:
    if performance_path:
        path = Path(performance_path)
        if not path.exists():
            return (
                None,
                path,
                "explicit_performance_artifact",
                f"Model performance artifact does not exist: {path}",
            )
        try:
            payload = _read_metric_file(path)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            provenance = verify_locked_model_evaluation(payload)
            if not provenance["valid"]:
                return (
                    None,
                    path,
                    "verified_locked_model_evaluation_artifact",
                    "Performance artifact is not a verified locked model "
                    "evaluation: "
                    + ", ".join(provenance["failures"]),
                )
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                return (
                    None,
                    path,
                    "verified_locked_model_evaluation_artifact",
                    "Locked model evaluation has no canonical metrics object.",
                )
            joined_lineage = payload.get("joined_lineage")
            joined_lineage = (
                joined_lineage
                if isinstance(joined_lineage, dict)
                else {}
            )
            evaluated_at = (
                payload.get("evaluated_at")
                or _evaluation_window_end(
                    joined_lineage.get("evaluation_window")
                )
            )
            return (
                {
                    "metrics": metrics,
                    "evaluated_at": evaluated_at,
                    "_source_provenance": provenance,
                },
                path,
                "verified_locked_model_evaluation_artifact",
                None,
            )
        except Exception as exc:
            return (
                None,
                path,
                "verified_locked_model_evaluation_artifact",
                "Could not read model performance artifact: "
                f"{type(exc).__name__}: {exc}",
            )
    if pipeline_result:
        evaluation_summary = _find_nested_mapping(
            pipeline_result,
            "evaluation_summary",
        )
        metrics = (
            evaluation_summary.get("metrics")
            if isinstance(evaluation_summary, dict)
            else None
        )
        if isinstance(metrics, dict):
            return (
                {
                    "metrics": metrics,
                    "timestamp": evaluation_summary.get("timestamp"),
                },
                None,
                "pipeline_stage7_evaluation_summary_metrics",
                None,
            )
        return (
            None,
            None,
            "pipeline_stage7_evaluation_summary_metrics",
            "Pipeline result has no canonical evaluation_summary.metrics "
            "payload; arbitrary nested scores are not accepted.",
        )
    return (
        None,
        None,
        "none",
        "No model performance artifact or pipeline_result metrics were supplied.",
    )


def _find_nested_mapping(
    value: Any,
    key: str,
    depth: int = 0,
) -> dict[str, Any] | None:
    if depth > 5 or not isinstance(value, dict):
        return None
    candidate = value.get(key)
    if isinstance(candidate, dict):
        return candidate
    for child in value.values():
        found = _find_nested_mapping(child, key, depth + 1)
        if found is not None:
            return found
    return None


def _evaluation_window_end(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    evaluation = value.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else value
    end = evaluation.get("end")
    return str(end) if end is not None else None


def inspect_real_metric_evidence_chain(
    evidence_chain_path: str | Path | None,
    *,
    expected_model_evaluation_path: str | Path | None = None,
) -> dict[str, Any]:
    if not evidence_chain_path:
        return {
            "status": "not_configured",
            "path": None,
            "model_evaluation_path_matches": False,
            "can_clear_current_real_cautions": False,
            "can_promote_model": False,
            "can_trade": False,
        }
    path = Path(evidence_chain_path)
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "model_evaluation_path_matches": False,
            "can_clear_current_real_cautions": False,
            "can_promote_model": False,
            "can_trade": False,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return {
            "status": "unreadable",
            "path": str(path),
            "error_type": type(exc).__name__,
            "model_evaluation_path_matches": False,
            "can_clear_current_real_cautions": False,
            "can_promote_model": False,
            "can_trade": False,
        }
    if not isinstance(payload, dict):
        return {
            "status": "invalid_shape",
            "path": str(path),
            "model_evaluation_path_matches": False,
            "can_clear_current_real_cautions": False,
            "can_promote_model": False,
            "can_trade": False,
        }
    summary = payload.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    inputs = payload.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    referenced_model_path = inputs.get("model_evaluation_json")
    referenced_model_sha256 = inputs.get("model_evaluation_sha256")
    path_matches = _same_path(
        referenced_model_path,
        expected_model_evaluation_path,
    )
    current_model_sha256 = _file_sha256(
        expected_model_evaluation_path
    )
    sha_matches = bool(
        referenced_model_sha256
        and current_model_sha256
        and referenced_model_sha256 == current_model_sha256
    )
    chain_status = summary.get(
        "real_metric_evidence_status",
        "not_reported",
    )
    ready = (
        payload.get("mode") == "pipeline_control_real_metric_evidence_run"
        and chain_status == "real_metric_evidence_chain_ready"
        and summary.get("can_use_as_metric_evidence") is True
        and summary.get("can_clear_current_real_cautions") is True
        and path_matches
        and sha_matches
    )
    return {
        "status": (
            "real_metric_evidence_chain_ready"
            if ready
            else str(chain_status)
        ),
        "path": str(path),
        "model_evaluation_path": referenced_model_path,
        "model_evaluation_path_matches": path_matches,
        "model_evaluation_sha256": referenced_model_sha256,
        "model_evaluation_sha256_matches": sha_matches,
        "blocked_metric_planes": summary.get(
            "blocked_metric_planes",
            [],
        ),
        "caution_metric_planes": summary.get(
            "caution_metric_planes",
            [],
        ),
        "can_clear_current_real_cautions": ready,
        "can_promote_model": False,
        "can_trade": False,
    }


def _same_path(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except OSError:
        return False


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


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

    required_metrics = {
        "validation_score": validation_score,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "sample_count": sample_count,
    }
    for metric_name, metric_value in required_metrics.items():
        if metric_value is None:
            failures.append(f"missing_{metric_name}")
            reasons.append(
                f"Required model-performance metric is missing: {metric_name}."
            )

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
