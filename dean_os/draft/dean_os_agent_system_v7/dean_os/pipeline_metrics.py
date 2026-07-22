from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

MetricStatus = Literal["observed", "missing", "invalid", "not_applicable"]
SnapshotStatus = Literal["ready", "partial", "empty", "skipped", "failed"]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class PipelineRunIdentity(BaseModel):
    """Stable identity and execution lineage for one pipeline metric snapshot."""

    run_id: str = Field(default_factory=lambda: f"pipeline_metrics_{uuid4().hex}")
    pipeline_status: str = "unknown"
    adapter_mode: str | None = None
    batch_name: str | None = None
    as_of: str | None = None
    tickers: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    requested_stages: list[int] = Field(default_factory=list)
    completed_stages: list[int] = Field(default_factory=list)
    model_name: str | None = None
    target_name: str | None = None
    context_fingerprint: str | None = None
    source: str = "pipeline_result"

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, value: list[str]) -> list[str]:
        return sorted({str(item).upper().strip() for item in value if str(item).strip()})

    @field_validator("timeframes")
    @classmethod
    def _normalize_timeframes(cls, value: list[str]) -> list[str]:
        return sorted({str(item).lower().strip() for item in value if str(item).strip()})


class ProfitabilityMetrics(BaseModel):
    pnl: float | None = None
    total_return: float | None = None
    sharpe: float | None = None
    win_rate: float | None = None
    total_trades: int | None = None


class RiskMetrics(BaseModel):
    max_drawdown: float | None = None
    volatility: float | None = None
    var_95: float | None = None
    expected_shortfall_95: float | None = None
    gross_exposure: float | None = None


class ValidationMetrics(BaseModel):
    train_score: float | None = None
    validation_score: float | None = None
    test_score: float | None = None
    train_test_gap: float | None = None
    sample_count: int | None = None
    train_sample_count: int | None = None
    validation_sample_count: int | None = None
    test_sample_count: int | None = None
    walk_forward_fold_count: int | None = None


class FeatureStabilityMetrics(BaseModel):
    feature_importance: dict[str, float] = Field(default_factory=dict)
    feature_count: int | None = None
    feature_concentration: float | None = None
    max_feature_weight_abs: float | None = None
    feature_stability_score: float | None = None
    unstable_feature_count: int | None = None
    unstable_features: list[str] = Field(default_factory=list)


class DataQualityMetrics(BaseModel):
    warning_count: int = 0
    leakage_flag_count: int = 0
    missing_ratio: float | None = None
    duplicate_ratio: float | None = None
    freshness_hours: float | None = None
    warnings: list[str] = Field(default_factory=list)
    leakage_flags: list[str] = Field(default_factory=list)


class ReplayMetrics(BaseModel):
    clear_hit_rate: float | None = None
    clear_evaluated_runs: int | None = None
    quality_blocked_runs: int | None = None
    average_realized_return: float | None = None
    replay_window_count: int | None = None


class PipelineMetricCompleteness(BaseModel):
    observed_fields: int = 0
    expected_fields: int = 0
    coverage_ratio: float = 0.0
    observed_groups: list[str] = Field(default_factory=list)
    missing_groups: list[str] = Field(default_factory=list)


class PipelineMetricSnapshot(BaseModel):
    """Versioned, JSON-safe metric contract between ``src`` and DEAN-OS.

    The snapshot deliberately separates metric families. Missing values remain
    missing; the normalizer does not invent risk, validation, replay, or feature
    evidence from unrelated fields.
    """

    schema_version: str = "dean_pipeline_metric_snapshot_v1"
    created_at: str = Field(default_factory=_utc_now_iso)
    status: SnapshotStatus = "empty"
    identity: PipelineRunIdentity = Field(default_factory=PipelineRunIdentity)
    profitability: ProfitabilityMetrics = Field(default_factory=ProfitabilityMetrics)
    risk: RiskMetrics = Field(default_factory=RiskMetrics)
    validation: ValidationMetrics = Field(default_factory=ValidationMetrics)
    feature_stability: FeatureStabilityMetrics = Field(default_factory=FeatureStabilityMetrics)
    data_quality: DataQualityMetrics = Field(default_factory=DataQualityMetrics)
    replay: ReplayMetrics = Field(default_factory=ReplayMetrics)
    completeness: PipelineMetricCompleteness = Field(default_factory=PipelineMetricCompleteness)
    evidence_availability: dict[str, bool] = Field(default_factory=dict)
    source_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raw_inventory: dict[str, Any] = Field(default_factory=dict)
    authority_boundary: dict[str, bool] = Field(
        default_factory=lambda: {
            "review_only": True,
            "can_write_production_config": False,
            "can_promote_model": False,
            "can_write_learning_memory": False,
            "can_trade": False,
        }
    )

    def to_agent_metadata(self) -> dict[str, Any]:
        """Compatibility projection for existing DEAN-OS pipeline agents.

        New code should consume the versioned snapshot directly. This projection
        keeps older ModelPerformance/Tuning agents operational while Codex
        gradually migrates them to the canonical contract.
        """

        failures: list[str] = []
        if self.validation.validation_score is not None and self.validation.validation_score < 0.55:
            failures.append("validation_score_below_threshold")
        if self.profitability.sharpe is not None and self.profitability.sharpe < 0.0:
            failures.append("sharpe_below_threshold")
        if self.risk.max_drawdown is not None and abs(self.risk.max_drawdown) > 0.25:
            failures.append("drawdown_above_threshold")
        if self.validation.sample_count is not None and self.validation.sample_count < 50:
            failures.append("sample_count_below_threshold")
        if not self.evidence_availability.get("model_performance", False):
            failures.append("missing_evaluation_metrics")
        elif self.status == "partial":
            failures.append("pipeline_metric_snapshot_incomplete")

        evaluation_scope = {
            "ticker": self.identity.tickers[0] if len(self.identity.tickers) == 1 else None,
            "tickers": self.identity.tickers,
            "model": self.identity.model_name,
            "target_name": self.identity.target_name,
            "timeframe": self.identity.timeframes[0] if len(self.identity.timeframes) == 1 else None,
            "timeframes": self.identity.timeframes,
            "context_fingerprint": self.identity.context_fingerprint,
        }
        model_performance = {
            **self.profitability.model_dump(mode="json"),
            **self.risk.model_dump(mode="json"),
            **self.validation.model_dump(mode="json"),
            "threshold_failures": failures,
            "verdict": "caution" if failures else "clear",
            "performance_score": self.validation.validation_score,
            "evaluation_scope": evaluation_scope,
            "source_contract": self.schema_version,
            "snapshot_status": self.status,
        }
        data_freshness = {
            "pipeline_metric_snapshot": {
                "stale": False if self.data_quality.freshness_hours is None else self.data_quality.freshness_hours > 24.0,
                "age_hours": self.data_quality.freshness_hours,
                "as_of": self.identity.as_of,
            }
        }
        return {
            "model_performance": model_performance,
            "data_freshness": data_freshness,
            "feature_stability": self.feature_stability.model_dump(mode="json"),
            "data_quality": self.data_quality.model_dump(mode="json"),
            "replay_summary": self.replay.model_dump(mode="json"),
        }

    def to_control_surface_payloads(self) -> dict[str, dict[str, Any]]:
        """Return in-memory payloads compatible with ``PipelineControlSurface``."""

        metrics = {
            **self.profitability.model_dump(mode="json"),
            **self.risk.model_dump(mode="json"),
            **self.validation.model_dump(mode="json"),
        }
        model_payload = {
            "artifact_class": "dean_pipeline_metric_snapshot_model_performance_v1",
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "metrics": {key: value for key, value in metrics.items() if value is not None},
            "evaluation_scope": {
                "ticker": self.identity.tickers[0] if len(self.identity.tickers) == 1 else None,
                "tickers": self.identity.tickers,
                "timeframe": self.identity.timeframes[0] if len(self.identity.timeframes) == 1 else None,
                "timeframes": self.identity.timeframes,
                "model": self.identity.model_name,
                "target_name": self.identity.target_name,
                "context_fingerprint": self.identity.context_fingerprint,
                "as_of": self.identity.as_of,
            },
            "pipeline_status": self.identity.pipeline_status,
            "snapshot_status": self.status,
        }
        feature_payload = {
            "artifact_class": "dean_pipeline_metric_snapshot_feature_stability_v1",
            "schema_version": self.schema_version,
            **self.feature_stability.model_dump(mode="json"),
            "snapshot_status": self.status,
        }
        quality_payload = {
            "artifact_class": "dean_pipeline_metric_snapshot_data_quality_v1",
            "schema_version": self.schema_version,
            **self.data_quality.model_dump(mode="json"),
            "snapshot_status": self.status,
        }
        replay_payload = {
            "artifact_class": "dean_pipeline_metric_snapshot_replay_v1",
            "schema_version": self.schema_version,
            "summary": self.replay.model_dump(mode="json"),
            "snapshot_status": self.status,
        }
        return {
            "model_performance": (
                model_payload if self.evidence_availability.get("model_performance", False) else {}
            ),
            "feature_report": (
                feature_payload if self.evidence_availability.get("feature_stability", False) else {}
            ),
            "data_quality": (
                quality_payload if self.evidence_availability.get("data_quality", False) else {}
            ),
            "replay_batch": (
                replay_payload if self.evidence_availability.get("replay", False) else {}
            ),
        }


class PipelineMetricNormalizer:
    """Normalize heterogeneous pipeline outputs into ``PipelineMetricSnapshot``.

    The current pipeline has several legitimate output shapes: local-stage
    aggregates, final-stage summaries, evaluation summaries, and dedicated
    pipeline-control artifacts. This class is the single compatibility layer.
    """

    _PROFITABILITY_ALIASES = {
        "pnl": ("pnl", "profit", "net_profit"),
        "total_return": ("total_return", "total_return_pct", "return", "realized_return", "pnl_pct"),
        "sharpe": ("sharpe", "sharpe_ratio"),
        "win_rate": ("win_rate",),
        "total_trades": ("total_trades", "trade_count", "observed_trade_count"),
    }
    _RISK_ALIASES = {
        "max_drawdown": ("max_drawdown", "max_drawdown_pct", "maximum_drawdown", "mdd", "drawdown"),
        "volatility": ("volatility", "realized_volatility", "realized_vol_daily"),
        "var_95": ("var_95", "daily_var_95", "value_at_risk_95"),
        "expected_shortfall_95": ("expected_shortfall_95", "cvar_95", "conditional_var_95"),
        "gross_exposure": ("gross_exposure", "max_gross_exposure"),
    }
    _VALIDATION_ALIASES = {
        "train_score": ("train_score", "training_score", "in_sample_score"),
        "validation_score": ("validation_score", "val_score", "out_of_sample_score"),
        "test_score": ("test_score",),
        "sample_count": ("sample_count", "n_samples", "observations"),
        "train_sample_count": ("train_sample_count",),
        "validation_sample_count": ("validation_sample_count", "val_sample_count"),
        "test_sample_count": ("test_sample_count",),
        "walk_forward_fold_count": ("walk_forward_fold_count", "fold_count", "n_folds"),
    }
    _REPLAY_ALIASES = {
        "clear_hit_rate": ("clear_hit_rate", "hit_rate"),
        "clear_evaluated_runs": ("clear_evaluated_runs", "evaluated_runs", "sample_count"),
        "quality_blocked_runs": ("quality_blocked_runs", "blocked_runs"),
        "average_realized_return": ("clear_average_realized_return", "average_realized_return"),
        "replay_window_count": ("replay_window_count", "window_count"),
    }

    def __init__(self, project_root: str | Path = ".", *, load_json_artifacts: bool = True):
        self.project_root = Path(project_root).resolve()
        self.load_json_artifacts = load_json_artifacts

    def from_pipeline_result(
        self,
        result: dict[str, Any] | None,
        *,
        as_of: str | None = None,
        tickers: Iterable[str] | None = None,
        timeframes: Iterable[str] | None = None,
    ) -> PipelineMetricSnapshot:
        raw = result if isinstance(result, dict) else {}
        artifacts = self._collect_artifacts(raw)
        search_roots = self._search_roots(raw, artifacts)

        identity = PipelineRunIdentity(
            pipeline_status=str(raw.get("status") or self._find_first(search_roots, ("status",)) or "unknown"),
            adapter_mode=_string_or_none(raw.get("adapter_mode")),
            batch_name=_string_or_none(raw.get("batch_name") or self._find_first(search_roots, ("batch_name",))),
            as_of=_string_or_none(as_of or raw.get("as_of") or raw.get("timestamp") or self._find_first(search_roots, ("evaluated_at", "created_at", "timestamp"))),
            tickers=list(tickers or _as_string_list(raw.get("tickers") or self._find_first(search_roots, ("tickers",)))),
            timeframes=list(timeframes or _as_string_list(raw.get("timeframes") or raw.get("timeframe") or self._find_first(search_roots, ("timeframes", "timeframe")))),
            requested_stages=_as_int_list(raw.get("requested_stages") or raw.get("stages_to_run")),
            completed_stages=_as_int_list(raw.get("completed_stages") or self._find_first(search_roots, ("completed_stages", "executed_stages"))),
            model_name=_string_or_none(self._find_first(search_roots, ("model_name", "best_model", "model_type"))),
            target_name=_string_or_none(self._find_first(search_roots, ("target_name", "target"))),
            context_fingerprint=_string_or_none(self._find_first(search_roots, ("context_fingerprint",))),
        )

        profitability = ProfitabilityMetrics(
            **self._extract_alias_group(search_roots, self._PROFITABILITY_ALIASES, integer_fields={"total_trades"})
        )
        risk = RiskMetrics(**self._extract_alias_group(search_roots, self._RISK_ALIASES))
        validation_values = self._extract_alias_group(
            search_roots,
            self._VALIDATION_ALIASES,
            integer_fields={
                "sample_count",
                "train_sample_count",
                "validation_sample_count",
                "test_sample_count",
                "walk_forward_fold_count",
            },
        )
        if validation_values.get("train_score") is not None and validation_values.get("validation_score") is not None:
            validation_values["train_test_gap"] = abs(
                float(validation_values["train_score"]) - float(validation_values["validation_score"])
            )
        validation = ValidationMetrics(**validation_values)

        feature_importance = self._extract_feature_importance(search_roots)
        concentration = _feature_concentration(feature_importance)
        feature_stability = FeatureStabilityMetrics(
            feature_importance=feature_importance,
            feature_count=len(feature_importance) if feature_importance else _int_or_none(self._find_first(search_roots, ("feature_count", "feature_importance_count"))),
            feature_concentration=concentration,
            max_feature_weight_abs=max((abs(value) for value in feature_importance.values()), default=None),
            feature_stability_score=_number_or_none(self._find_first(search_roots, ("feature_stability_score", "stability_score"))),
            unstable_feature_count=_int_or_none(self._find_first(search_roots, ("unstable_feature_count",))),
            unstable_features=_as_string_list(self._find_first(search_roots, ("unstable_features",))),
        )
        if feature_stability.unstable_feature_count is None and feature_stability.unstable_features:
            feature_stability.unstable_feature_count = len(feature_stability.unstable_features)

        quality_keys = ("data_quality_warnings", "warnings", "audit_warnings", "leakage_flags", "leakage_warnings", "missing_ratio", "missingness_ratio", "duplicate_ratio")
        data_quality_present = any(_contains_any_key(root, quality_keys) for root in search_roots)
        warnings = self._collect_named_list(search_roots, ("data_quality_warnings", "warnings", "audit_warnings"))
        leakage_flags = self._collect_named_list(search_roots, ("leakage_flags", "leakage_warnings"))
        data_quality = DataQualityMetrics(
            warning_count=len(warnings),
            leakage_flag_count=len(leakage_flags),
            missing_ratio=_number_or_none(self._find_first(search_roots, ("missing_ratio", "missingness_ratio"))),
            duplicate_ratio=_number_or_none(self._find_first(search_roots, ("duplicate_ratio",))),
            freshness_hours=_number_or_none(self._find_first(search_roots, ("evaluation_age_hours", "freshness_hours", "age_hours"))),
            warnings=warnings,
            leakage_flags=leakage_flags,
        )

        replay_summary_roots = self._replay_roots(search_roots)
        replay_present = bool(replay_summary_roots)
        replay = ReplayMetrics(
            **self._extract_alias_group(
                replay_summary_roots,
                self._REPLAY_ALIASES,
                integer_fields={"clear_evaluated_runs", "quality_blocked_runs", "replay_window_count"},
            )
        )

        evidence_availability = {
            "model_performance": any(
                value is not None
                for value in [
                    profitability.pnl,
                    profitability.total_return,
                    profitability.sharpe,
                    risk.max_drawdown,
                    risk.volatility,
                    validation.train_score,
                    validation.validation_score,
                    validation.sample_count,
                ]
            ),
            "feature_stability": bool(feature_importance)
            or feature_stability.feature_stability_score is not None
            or feature_stability.unstable_feature_count is not None,
            "data_quality": data_quality_present,
            "replay": replay_present,
        }
        completeness = self._completeness(
            profitability=profitability,
            risk=risk,
            validation=validation,
            feature_stability=feature_stability,
            data_quality=data_quality,
            replay=replay,
            evidence_availability=evidence_availability,
        )
        status = self._snapshot_status(raw, completeness)
        snapshot_warnings: list[str] = []
        if status in {"empty", "partial"}:
            snapshot_warnings.append(
                "Pipeline metric evidence is incomplete; absent metric families remain cautionary and cannot be inferred."
            )
        if raw.get("pipeline_skipped"):
            snapshot_warnings.append(str(raw.get("skip_reason") or "pipeline was skipped"))

        return PipelineMetricSnapshot(
            status=status,
            identity=identity,
            profitability=profitability,
            risk=risk,
            validation=validation,
            feature_stability=feature_stability,
            data_quality=data_quality,
            replay=replay,
            completeness=completeness,
            evidence_availability=evidence_availability,
            source_artifacts=artifacts,
            warnings=_dedupe(snapshot_warnings),
            raw_inventory={
                "top_level_keys": sorted(str(key) for key in raw.keys()),
                "search_root_count": len(search_roots),
                "artifact_count": len(artifacts),
            },
        )

    def _search_roots(self, raw: dict[str, Any], artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        roots: list[dict[str, Any]] = []
        preferred_paths = (
            ("results", "model_metrics"),
            ("results", "evaluation_summary"),
            ("evaluation_summary",),
            ("financial_metrics",),
            ("training_metrics",),
            ("validation_metrics",),
            ("metrics",),
            ("model_performance",),
            ("replay_batch",),
        )
        for path in preferred_paths:
            value = _get_path(raw, path)
            if isinstance(value, dict):
                roots.append(value)
        roots.append(raw)
        roots.extend(item["payload"] for item in artifacts if isinstance(item.get("payload"), dict))

        unique: list[dict[str, Any]] = []
        seen: set[int] = set()
        for root in roots:
            if id(root) in seen:
                continue
            seen.add(id(root))
            unique.append(root)
        return unique

    def _collect_artifacts(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[Any] = []
        for key in (
            "pipeline_control_evaluation_metric_artifacts",
            "pipeline_control_metric_artifact_manifests",
            "stage7_metric_artifacts",
            "stage4_metric_artifact_manifests",
        ):
            found = _find_nested_values(raw, key)
            candidates.extend(found)

        artifacts: list[dict[str, Any]] = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                # A direct evidence payload should be preserved as-is.
                if candidate.get("artifact_class") or candidate.get("metrics") or candidate.get("feature_importance"):
                    artifacts.append(
                        {
                            "source": "inline",
                            "artifact_class": candidate.get("artifact_class"),
                            "payload": candidate,
                        }
                    )
                    continue
                for label, value in candidate.items():
                    loaded = self._artifact_from_value(value, label=str(label))
                    if loaded:
                        artifacts.append(loaded)
            elif isinstance(candidate, list):
                for item in candidate:
                    loaded = self._artifact_from_value(item)
                    if loaded:
                        artifacts.append(loaded)
            else:
                loaded = self._artifact_from_value(candidate)
                if loaded:
                    artifacts.append(loaded)

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for artifact in artifacts:
            fingerprint = json.dumps(
                {
                    "path": artifact.get("path"),
                    "artifact_class": artifact.get("artifact_class"),
                    "payload": artifact.get("payload"),
                },
                sort_keys=True,
                default=str,
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(artifact)
        return deduped

    def _artifact_from_value(self, value: Any, *, label: str | None = None) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return {
                "source": "inline",
                "label": label,
                "artifact_class": value.get("artifact_class"),
                "payload": value,
            }
        if not isinstance(value, (str, Path)) or not str(value).strip():
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        resolved = path.resolve()
        record: dict[str, Any] = {"source": "path", "label": label, "path": str(resolved)}
        if not self.load_json_artifacts or resolved.suffix.lower() != ".json" or not resolved.exists():
            return record
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record["load_error"] = f"{type(exc).__name__}: {exc}"
            return record
        if isinstance(payload, dict):
            record["payload"] = payload
            record["artifact_class"] = payload.get("artifact_class")
        return record

    def _extract_alias_group(
        self,
        roots: list[dict[str, Any]],
        aliases: dict[str, tuple[str, ...]],
        *,
        integer_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        integer_fields = integer_fields or set()
        output: dict[str, Any] = {}
        for canonical, candidates in aliases.items():
            raw = self._find_first(roots, candidates)
            output[canonical] = _int_or_none(raw) if canonical in integer_fields else _number_or_none(raw)
        return output

    def _find_first(self, roots: list[dict[str, Any]], keys: tuple[str, ...]) -> Any:
        for root in roots:
            value = _find_first_recursive(root, keys)
            if value is not None:
                return value
        return None

    def _extract_feature_importance(self, roots: list[dict[str, Any]]) -> dict[str, float]:
        for root in roots:
            raw = _find_first_recursive(root, ("feature_importance", "feature_importances", "feature_weights"))
            parsed = _normalize_feature_importance(raw)
            if parsed:
                return parsed
        return {}

    def _collect_named_list(self, roots: list[dict[str, Any]], keys: tuple[str, ...]) -> list[str]:
        values: list[str] = []
        for root in roots:
            for key in keys:
                for item in _find_nested_values(root, key):
                    values.extend(_as_string_list(item))
        return _dedupe(values)

    def _replay_roots(self, roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        replay_roots: list[dict[str, Any]] = []
        for root in roots:
            for key in ("replay_batch", "historical_replay", "replay_summary", "summary"):
                value = _find_first_recursive(root, (key,))
                if isinstance(value, dict):
                    replay_roots.append(value)
        return replay_roots

    def _completeness(
        self,
        *,
        profitability: ProfitabilityMetrics,
        risk: RiskMetrics,
        validation: ValidationMetrics,
        feature_stability: FeatureStabilityMetrics,
        data_quality: DataQualityMetrics,
        replay: ReplayMetrics,
        evidence_availability: dict[str, bool],
    ) -> PipelineMetricCompleteness:
        groups: dict[str, list[Any]] = {
            "profitability": [profitability.pnl, profitability.total_return, profitability.sharpe],
            "risk": [risk.max_drawdown, risk.volatility, risk.var_95],
            "validation": [validation.train_score, validation.validation_score, validation.sample_count],
            "feature_stability": [
                feature_stability.feature_stability_score,
                feature_stability.feature_concentration,
                feature_stability.feature_count,
            ],
            "data_quality": [
                data_quality.missing_ratio,
                data_quality.duplicate_ratio,
                data_quality.warning_count if evidence_availability.get("data_quality") else None,
                data_quality.leakage_flag_count if evidence_availability.get("data_quality") else None,
            ],
            "replay": [replay.clear_hit_rate, replay.clear_evaluated_runs, replay.quality_blocked_runs],
        }
        observed_fields = sum(value is not None for values in groups.values() for value in values)
        expected_fields = sum(len(values) for values in groups.values())
        group_availability = {
            "profitability": evidence_availability.get("model_performance", False),
            "risk": evidence_availability.get("model_performance", False),
            "validation": evidence_availability.get("model_performance", False),
            "feature_stability": evidence_availability.get("feature_stability", False),
            "data_quality": evidence_availability.get("data_quality", False),
            "replay": evidence_availability.get("replay", False),
        }
        observed_groups = [
            name
            for name, values in groups.items()
            if group_availability.get(name, False) and any(value is not None for value in values)
        ]
        missing_groups = [name for name in groups if name not in observed_groups]
        return PipelineMetricCompleteness(
            observed_fields=observed_fields,
            expected_fields=expected_fields,
            coverage_ratio=round(observed_fields / expected_fields if expected_fields else 0.0, 6),
            observed_groups=observed_groups,
            missing_groups=missing_groups,
        )

    @staticmethod
    def _snapshot_status(raw: dict[str, Any], completeness: PipelineMetricCompleteness) -> SnapshotStatus:
        if raw.get("pipeline_skipped") or str(raw.get("status", "")).lower() == "pipeline_skipped":
            return "skipped"
        if str(raw.get("status", "")).lower() in {"failed", "error"}:
            return "failed"
        if completeness.observed_fields == 0:
            return "empty"
        if not completeness.missing_groups and completeness.coverage_ratio >= 0.65:
            return "ready"
        return "partial"


def _get_path(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _find_first_recursive(payload: Any, keys: tuple[str, ...], *, max_depth: int = 7) -> Any:
    normalized_keys = {str(key).lower() for key in keys}

    def walk(value: Any, depth: int) -> Any:
        if depth > max_depth:
            return None
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in normalized_keys and item not in (None, "", [], {}):
                    return item
            for item in value.values():
                found = walk(item, depth + 1)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = walk(item, depth + 1)
                if found is not None:
                    return found
        return None

    return walk(payload, 0)


def _find_nested_values(payload: Any, target_key: str, *, max_depth: int = 8) -> list[Any]:
    target = target_key.lower()
    results: list[Any] = []

    def walk(value: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() == target:
                    results.append(item)
                walk(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    walk(payload, 0)
    return results


def _contains_any_key(payload: Any, keys: tuple[str, ...], *, max_depth: int = 7) -> bool:
    targets = {str(key).lower() for key in keys}

    def walk(value: Any, depth: int) -> bool:
        if depth > max_depth:
            return False
        if isinstance(value, dict):
            if any(str(key).lower() in targets for key in value):
                return True
            return any(walk(item, depth + 1) for item in value.values())
        if isinstance(value, list):
            return any(walk(item, depth + 1) for item in value)
        return False

    return walk(payload, 0)


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _int_or_none(value: Any) -> int | None:
    number = _number_or_none(value)
    return int(number) if number is not None else None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        values = list(value.keys())
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return _dedupe([str(item).strip() for item in values if str(item).strip()])


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    output: list[int] = []
    for item in values:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number not in output:
            output.append(number)
    return output


def _normalize_feature_importance(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        output: dict[str, float] = {}
        for key, raw in value.items():
            number = _number_or_none(raw)
            if number is not None:
                output[str(key)] = number
        return output
    if isinstance(value, list):
        output = {}
        for index, item in enumerate(value):
            if isinstance(item, dict):
                name = item.get("feature") or item.get("name") or item.get("feature_name")
                number = _number_or_none(item.get("importance") or item.get("weight") or item.get("value"))
                if name is not None and number is not None:
                    output[str(name)] = number
            else:
                number = _number_or_none(item)
                if number is not None:
                    output[f"feature_{index}"] = number
        return output
    return {}


def _feature_concentration(importances: dict[str, float]) -> float | None:
    if not importances:
        return None
    absolute = sorted((abs(value) for value in importances.values()), reverse=True)
    total = sum(absolute)
    if total <= 0:
        return None
    return absolute[0] / total


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


__all__ = [
    "DataQualityMetrics",
    "FeatureStabilityMetrics",
    "PipelineMetricCompleteness",
    "PipelineMetricNormalizer",
    "PipelineMetricSnapshot",
    "PipelineRunIdentity",
    "ProfitabilityMetrics",
    "ReplayMetrics",
    "RiskMetrics",
    "ValidationMetrics",
]
