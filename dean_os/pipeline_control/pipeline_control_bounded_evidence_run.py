from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.pipeline_control.pipeline_control_locked_evaluation_assembler import PipelineControlLockedEvaluationAssembler
from dean_os.pipeline_control.pipeline_control_locked_feature_stability_assembler import (
    PipelineControlLockedFeatureStabilityAssembler,
)
from dean_os.pipeline_control.pipeline_control_real_metric_evidence_run import (
    DEFAULT_ARCHITECTURE_MAP_JSON,
    DEFAULT_DATA_QUALITY_JSON,
    DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON,
    DEFAULT_REPLAY_BATCH_JSON,
    PipelineControlRealMetricEvidenceRun,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from src.pipeline.stages.evaluation.pipeline_control_artifacts import (
    build_evaluation_metric_candidate,
    write_evaluation_metric_artifact_candidate,
)
from src.pipeline.stages.modeling.pipeline_control_artifacts import (
    build_feature_distribution_stability_analysis,
    build_feature_stability_candidate,
    build_model_evaluation_candidate,
    build_split_evaluation_window,
    extract_native_feature_importance,
    write_pipeline_control_metric_artifact_candidates,
)
from src.pipeline.target_column_utils import is_target_like_column


class PipelineControlBoundedEvidenceRun:
    """Produce one real, offline, review-only metric evidence slice."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/pipeline_control_bounded_evidence_run_current",
    ):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        *,
        source_path: str | Path,
        ticker: str,
        timeframe: str,
        target_name: str,
        macro_source_path: str | Path | None = None,
        start: str | None = None,
        end: str | None = None,
        max_rows: int = 600,
        max_features: int = 40,
        gap_size: int = 5,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        min_rows: int = 180,
        input_is_enriched: bool = False,
        transaction_cost_per_turn: float = 0.0025,
        run_real_metric_review: bool = True,
        replay_batch_json: str | Path | None = DEFAULT_REPLAY_BATCH_JSON,
        data_quality_json: str | Path | None = DEFAULT_DATA_QUALITY_JSON,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        domain_instance_contract_json: str | Path | None = DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        replay_batch_json = replay_batch_json or DEFAULT_REPLAY_BATCH_JSON
        data_quality_json = data_quality_json or DEFAULT_DATA_QUALITY_JSON
        architecture_map_json = architecture_map_json or DEFAULT_ARCHITECTURE_MAP_JSON
        domain_instance_contract_json = (
            domain_instance_contract_json or DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON
        )
        run_id = _run_id("pipeline_control_bounded_evidence_run")
        work_dir = self.output_dir / run_id
        source = Path(source_path)
        macro_source = Path(macro_source_path) if macro_source_path else None
        source_frame = _load_source_frame(source)
        bounded_frame = _bound_source_frame(
            source_frame,
            ticker=ticker,
            timeframe=timeframe,
            start=start,
            end=end,
            max_rows=max_rows,
        )
        source_checks = _source_quality_checks(bounded_frame, min_rows=min_rows)
        macro_frame, macro_provenance, macro_checks = _prepare_macro_input(
            macro_source,
            bounded_frame=bounded_frame,
        )
        source_checks.extend(macro_checks)
        if any(check["status"] == "fail" for check in source_checks):
            return self._blocked_payload(
                run_id=run_id,
                source=source,
                macro_source=macro_source,
                macro_provenance=macro_provenance,
                ticker=ticker,
                timeframe=timeframe,
                target_name=target_name,
                source_checks=source_checks,
                save=save,
            )

        enriched = (
            bounded_frame.copy()
            if input_is_enriched
            else await _run_stage_3_enrichment(
                bounded_frame,
                timeframe=timeframe,
                macro_frame=macro_frame,
            )
        )
        prepared = _prepare_model_frame(enriched, target_name=target_name)
        split = _chronological_split(
            prepared,
            gap_size=gap_size,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        selected_features = _select_train_features(
            split["train"],
            target_name=target_name,
            max_features=max_features,
        )
        if not selected_features:
            raise ValueError("No finite numeric features remain after target/leakage filtering.")

        matrices = _impute_split_features(split, selected_features)
        targets = {
            split_id: split[split_id][target_name].astype(int)
            for split_id in ("train", "validation", "test")
        }
        if targets["train"].nunique() < 2:
            raise ValueError("Training target contains fewer than two classes.")

        model = RandomForestClassifier(
            n_estimators=128,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )
        model.fit(matrices["train"], targets["train"])
        predictions = {
            split_id: pd.Series(
                model.predict(matrices[split_id]),
                index=matrices[split_id].index,
                name="prediction",
            )
            for split_id in ("train", "validation", "test")
        }
        split_metrics = {
            split_id: _classification_metrics(targets[split_id], predictions[split_id])
            for split_id in ("train", "validation", "test")
        }

        source_fingerprint = _sha256_file(source)
        context_fingerprint = _context_fingerprint(
            source_fingerprint=source_fingerprint,
            macro_fingerprint=(
                macro_provenance.get("sha256")
                if macro_frame is not None and not input_is_enriched
                else None
            ),
            ticker=ticker,
            timeframe=timeframe,
            target_name=target_name,
            selected_features=selected_features,
            split=split,
        )
        model_type = "random_forest"
        context_key = f"{ticker}_{target_name}_{model_type}"
        feature_importance = extract_native_feature_importance(model, selected_features)
        stability_analysis = build_feature_distribution_stability_analysis(
            matrices["train"],
            matrices["validation"],
            selected_features,
        )
        evaluation_window = build_split_evaluation_window(
            matrices["test"],
            source="bounded_test_feature_index",
        )
        heldout = _heldout_evaluation(
            test_frame=split["test"],
            predictions=predictions["test"],
            ticker=ticker,
            timeframe=timeframe,
            target_name=target_name,
            model_type=model_type,
            context_fingerprint=context_fingerprint,
            transaction_cost_per_turn=transaction_cost_per_turn,
        )

        training_candidate = build_model_evaluation_candidate(
            ticker=ticker,
            target_name=target_name,
            model_type=model_type,
            timeframe=timeframe,
            context_fingerprint=context_fingerprint,
            market_regime="offline_observed",
            volatility_regime="measured_from_saved_window",
            train_metrics=split_metrics["train"],
            validation_metrics=split_metrics["validation"],
            test_metrics=split_metrics["test"],
            train_sample_count=len(split["train"]),
            validation_sample_count=len(split["validation"]),
            test_sample_count=len(split["test"]),
            max_drawdown=None,
            evaluation_window=evaluation_window,
        )
        feature_candidate = build_feature_stability_candidate(
            ticker=ticker,
            target_name=target_name,
            model_type=model_type,
            timeframe=timeframe,
            context_fingerprint=context_fingerprint,
            market_regime="offline_observed",
            volatility_regime="measured_from_saved_window",
            feature_importance=feature_importance,
            stability_analysis=stability_analysis,
        )
        training_candidate["bounded_run_provenance"] = _bounded_provenance(
            source=source,
            source_fingerprint=source_fingerprint,
            macro_provenance=macro_provenance,
            macro_used=bool(macro_frame is not None and not input_is_enriched),
            run_id=run_id,
        )
        feature_candidate["bounded_run_provenance"] = training_candidate["bounded_run_provenance"]
        training_paths = write_pipeline_control_metric_artifact_candidates(
            batch_dir=work_dir,
            context_key=context_key,
            model_evaluation=training_candidate,
            feature_stability=feature_candidate,
        )

        evaluation_candidate = build_evaluation_metric_candidate(
            financial_metrics=heldout["financial_metrics"],
            backtest_results=heldout["backtest_results"],
            evaluation_summary=heldout["evaluation_summary"],
            signals_df=heldout["signals"],
            portfolio_history=heldout["portfolio_history"],
            summary_path=work_dir / "bounded_evaluation_summary.json",
        )
        evaluation_candidate["bounded_run_provenance"] = training_candidate["bounded_run_provenance"]
        evaluation_paths = write_evaluation_metric_artifact_candidate(
            output_dir=work_dir,
            candidate=evaluation_candidate,
            context_key=context_key,
        )

        model_path = work_dir / "models" / f"{context_key}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        locked_evaluation = PipelineControlLockedEvaluationAssembler(
            work_dir / "locked_evaluation"
        ).build(
            training_candidate_json=training_paths["model_evaluation_json"],
            evaluation_candidate_json=evaluation_paths["evaluation_metric_candidate"],
        )
        locked_feature = PipelineControlLockedFeatureStabilityAssembler(
            work_dir / "locked_feature_stability"
        ).build(
            feature_stability_candidate_json=training_paths["feature_stability_report"],
        )
        locked_model_path = locked_evaluation.get("next_runner_inputs", {}).get("model_evaluation_json")
        locked_feature_path = locked_feature.get("next_runner_inputs", {}).get("feature_stability_report")
        real_metric_review = _run_real_metric_review(
            output_dir=work_dir / "real_metric_review",
            locked_model_path=locked_model_path,
            locked_feature_path=locked_feature_path,
            run_real_metric_review=run_real_metric_review,
            replay_batch_json=replay_batch_json,
            data_quality_json=data_quality_json,
            architecture_map_json=architecture_map_json,
            domain_instance_contract_json=domain_instance_contract_json,
        )

        summary = _summary(
            source_checks=source_checks,
            split=split,
            selected_features=selected_features,
            split_metrics=split_metrics,
            stability_analysis=stability_analysis,
            heldout=heldout,
            locked_model_path=locked_model_path,
            locked_feature_path=locked_feature_path,
            real_metric_review=real_metric_review,
            macro_provenance=macro_provenance,
            macro_used=bool(macro_frame is not None and not input_is_enriched),
        )
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_bounded_evidence_run",
            "inputs": {
                "source_path": str(source),
                "macro_source_path": str(macro_source) if macro_source else None,
                "ticker": ticker,
                "timeframe": timeframe,
                "target_name": target_name,
                "start": start,
                "end": end,
                "max_rows": max_rows,
                "max_features": max_features,
                "gap_size": gap_size,
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "input_is_enriched": input_is_enriched,
                "transaction_cost_per_turn": transaction_cost_per_turn,
                "run_real_metric_review": run_real_metric_review,
            },
            "summary": summary,
            "source_provenance": {
                "path": str(source),
                "sha256": source_fingerprint,
                "source_evidence_class": "saved_offline_market_data",
                "synthetic": False,
                "bounded_row_count": len(bounded_frame),
                "bounded_start": str(bounded_frame.index[0]),
                "bounded_end": str(bounded_frame.index[-1]),
            },
            "macro_provenance": {
                **macro_provenance,
                "used_in_stage_3": bool(macro_frame is not None and not input_is_enriched),
            },
            "source_quality_checks": source_checks,
            "split_windows": _split_windows(split),
            "selected_features": selected_features,
            "split_metrics": split_metrics,
            "feature_stability_analysis": stability_analysis,
            "heldout_evaluation": {
                "financial_metrics": heldout["financial_metrics"],
                "evaluation_window": heldout["evaluation_window"],
                "diagnostics": heldout["diagnostics"],
            },
            "artifacts": {
                "model_path": str(model_path),
                "training_candidates": training_paths,
                "evaluation_candidate": evaluation_paths,
                "locked_model_evaluation": locked_model_path,
                "locked_feature_stability": locked_feature_path,
                "locked_evaluation_report": locked_evaluation.get("saved_paths", {}),
                "locked_feature_report": locked_feature.get("saved_paths", {}),
                "real_metric_review": _saved_paths(real_metric_review),
            },
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_bounded_evidence_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)

    def _blocked_payload(
        self,
        *,
        run_id: str,
        source: Path,
        macro_source: Path | None,
        macro_provenance: dict[str, Any],
        ticker: str,
        timeframe: str,
        target_name: str,
        source_checks: list[dict[str, Any]],
        save: bool,
    ) -> dict[str, Any]:
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_bounded_evidence_run",
            "inputs": {
                "source_path": str(source),
                "macro_source_path": str(macro_source) if macro_source else None,
                "ticker": ticker,
                "timeframe": timeframe,
                "target_name": target_name,
            },
            "summary": {
                "bounded_evidence_status": "blocked_source_quality",
                "failed_source_check_count": sum(
                    1 for check in source_checks if check["status"] == "fail"
                ),
                "locked_model_evaluation_ready": False,
                "locked_feature_stability_ready": False,
                "can_use_as_metric_evidence": False,
                "can_clear_current_real_cautions": False,
                "can_trade": False,
            },
            "macro_provenance": macro_provenance,
            "source_quality_checks": source_checks,
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_bounded_evidence_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_bounded_evidence_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Pipeline Control Bounded Evidence Run",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('bounded_evidence_status')}`",
        f"- Rows: {summary.get('model_row_count')}",
        f"- Features: {summary.get('selected_feature_count')}",
        f"- Train score: {summary.get('train_score')}",
        f"- Validation score: {summary.get('validation_score')}",
        f"- Test score: {summary.get('test_score')}",
        f"- Test balanced score: {summary.get('test_balanced_accuracy')}",
        f"- Max drawdown: {summary.get('max_drawdown')}",
        f"- Feature stability: {summary.get('feature_stability_score')}",
        f"- Macro artifact provided: {summary.get('macro_artifact_provided')}",
        f"- Macro artifact used in Stage 3: {summary.get('macro_used_in_stage_3')}",
        f"- Selected macro features: {summary.get('selected_macro_feature_count')}",
        f"- Locked model evaluation ready: {summary.get('locked_model_evaluation_ready')}",
        f"- Locked feature stability ready: {summary.get('locked_feature_stability_ready')}",
        f"- Can use as metric evidence: {summary.get('can_use_as_metric_evidence')}",
        f"- Can clear current real cautions: {summary.get('can_clear_current_real_cautions')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Source Checks",
        "",
    ]
    for check in payload.get("source_quality_checks", []):
        lines.append(f"- {check.get('status', '').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


async def _run_stage_3_enrichment(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    macro_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from src.config.unified_config_manager import UnifiedConfigManager
    from src.core.error_handling.error_handler import ErrorHandler
    from src.pipeline.stages.stage_3_feature_engineering import FeatureEngineeringStage

    config_manager = UnifiedConfigManager()
    stage = FeatureEngineeringStage(
        config_manager,
        ErrorHandler(config_manager),
        mode="bounded_evidence",
    )
    cleaned_data: dict[str, Any] = {
        "prices": {timeframe: frame.reset_index(drop=True)},
    }
    if macro_frame is not None:
        cleaned_data["macro_data"] = macro_frame.reset_index(drop=True)
    result = await stage.run(
        cleaned_data=cleaned_data,
        target_column="__bounded_skip_feature_selection__",
        context_id="bounded_pipeline_control_evidence",
        offline_only=True,
    )
    enriched = result.get("enriched_data")
    if not isinstance(enriched, pd.DataFrame) or enriched.empty:
        raise ValueError("Stage 3 did not produce enriched data.")
    return enriched


def _load_source_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source artifact does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Bounded evidence source must be a real parquet or CSV table.")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Source artifact is not a tabular DataFrame.")
    return frame


def _prepare_macro_input(
    path: Path | None,
    *,
    bounded_frame: pd.DataFrame,
) -> tuple[pd.DataFrame | None, dict[str, Any], list[dict[str, Any]]]:
    if path is None:
        return None, {
            "provided": False,
            "source_evidence_class": "not_provided",
            "synthetic": False,
            "live_collection_performed": False,
        }, []

    base_provenance = {
        "provided": True,
        "path": str(path),
        "source_evidence_class": "saved_offline_macro_data",
        "synthetic": False,
        "live_collection_performed": False,
    }
    if not path.exists():
        return None, base_provenance, [
            _check("fail", "macro_source_exists", f"Macro artifact does not exist: {path}")
        ]

    frame = _load_source_frame(path)
    source_sha256 = _sha256_file(path)
    captured_at = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    provenance = {
        **base_provenance,
        "sha256": source_sha256,
        "source_row_count": len(frame),
        "captured_at": captured_at.isoformat(),
    }
    if frame.empty:
        return None, provenance, [
            _check("fail", "macro_non_empty", "Macro artifact contains zero rows.")
        ]

    date_column = next(
        (column for column in ("datetime", "date", "timestamp") if column in frame.columns),
        None,
    )
    series_column = next(
        (column for column in ("series_id", "series") if column in frame.columns),
        None,
    )
    missing = [
        label
        for label, column in (
            ("datetime/date/timestamp", date_column),
            ("series_id/series", series_column),
            ("value", "value" if "value" in frame.columns else None),
        )
        if column is None
    ]
    if missing:
        provenance["missing_required_columns"] = missing
        return None, provenance, [
            _check(
                "fail",
                "macro_schema",
                f"Macro artifact is missing required columns: {', '.join(missing)}.",
            )
        ]

    normalized = frame.copy()
    normalized["datetime"] = pd.to_datetime(
        normalized[date_column],
        errors="coerce",
        utc=True,
    )
    normalized["series_id"] = normalized[series_column].astype("string")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")

    availability_column = next(
        (
            column
            for column in ("available_at", "published_at", "realtime_start")
            if column in normalized.columns
        ),
        None,
    )
    if availability_column:
        available_at = pd.to_datetime(
            normalized[availability_column],
            errors="coerce",
            utc=True,
        ).fillna(captured_at)
        availability_basis = availability_column
    else:
        available_at = pd.Series(captured_at, index=normalized.index)
        availability_basis = "artifact_mtime_conservative"
    normalized["_available_at"] = available_at

    if bounded_frame.empty:
        provenance.update(
            {
                "availability_basis": availability_basis,
                "usable_row_count": 0,
            }
        )
        return None, provenance, [
            _check("fail", "macro_price_window", "Price window is empty; macro timing cannot be checked.")
        ]

    bounded_start = pd.Timestamp(bounded_frame.index.min())
    bounded_end = pd.Timestamp(bounded_frame.index.max())
    if bounded_start.tzinfo is None:
        bounded_start = bounded_start.tz_localize("UTC")
    else:
        bounded_start = bounded_start.tz_convert("UTC")
    if bounded_end.tzinfo is None:
        bounded_end = bounded_end.tz_localize("UTC")
    else:
        bounded_end = bounded_end.tz_convert("UTC")

    normalized = normalized.loc[
        normalized["datetime"].notna()
        & normalized["series_id"].notna()
        & normalized["value"].notna()
        & np.isfinite(normalized["value"])
        & (normalized["datetime"] <= bounded_end)
        & (normalized["_available_at"] <= bounded_start)
    ].copy()
    normalized = normalized.sort_values(["datetime", "series_id"]).drop_duplicates(
        ["datetime", "series_id"],
        keep="last",
    )
    usable = normalized[["datetime", "series_id", "value"]].reset_index(drop=True)
    latest_observation = usable["datetime"].max() if not usable.empty else None
    staleness_days = (
        float((bounded_end - latest_observation).total_seconds() / 86400.0)
        if latest_observation is not None
        else None
    )
    provenance.update(
        {
            "availability_basis": availability_basis,
            "usable_row_count": len(usable),
            "series_count": int(usable["series_id"].nunique()) if not usable.empty else 0,
            "observation_start": (
                usable["datetime"].min().isoformat() if not usable.empty else None
            ),
            "observation_end": latest_observation.isoformat() if latest_observation is not None else None,
            "price_window_start": bounded_start.isoformat(),
            "price_window_end": bounded_end.isoformat(),
            "latest_observation_age_days": staleness_days,
            "future_rows_excluded": int(
                (pd.to_datetime(frame[date_column], errors="coerce", utc=True) > bounded_end).sum()
            ),
            "not_yet_available_rows_excluded": int(
                (available_at > bounded_start).sum()
            ),
        }
    )
    checks = [
        _check(
            "pass" if not usable.empty else "fail",
            "macro_temporal_availability",
            (
                f"usable_rows={len(usable)}, availability_basis={availability_basis}, "
                f"price_window_start={bounded_start.isoformat()}."
            ),
        ),
        _check(
            "pass" if not usable.empty else "fail",
            "macro_finite_values",
            f"finite_usable_rows={len(usable)}.",
        ),
    ]
    if staleness_days is not None and staleness_days > 14:
        checks.append(
            _check(
                "warn",
                "macro_freshness",
                f"Latest macro observation is {staleness_days:.1f} days before the price-window end.",
            )
        )
    else:
        checks.append(
            _check(
                "pass",
                "macro_freshness",
                f"Latest macro observation age is {staleness_days:.1f} days."
                if staleness_days is not None
                else "Macro freshness is unavailable.",
            )
        )
    return usable if not usable.empty else None, provenance, checks


def _bound_source_frame(
    frame: pd.DataFrame,
    *,
    ticker: str,
    timeframe: str,
    start: str | None,
    end: str | None,
    max_rows: int,
) -> pd.DataFrame:
    result = frame.copy()
    ticker_column = next((column for column in ("ticker", "symbol") if column in result.columns), None)
    timeframe_column = next((column for column in ("interval", "timeframe", "tf") if column in result.columns), None)
    datetime_column = next((column for column in ("datetime", "timestamp", "date") if column in result.columns), None)
    if not datetime_column:
        datetime_column = next(
            (
                column
                for column in result.columns
                if str(column).lower().startswith(("datetime_", "timestamp_", "date_"))
            ),
            None,
        )
    if not ticker_column or not timeframe_column or not datetime_column:
        raise ValueError("Source requires ticker, timeframe/interval, and datetime columns.")
    result = result.loc[result[ticker_column].astype(str).str.upper().eq(ticker.upper())]
    result = result.loc[result[timeframe_column].astype(str).str.lower().eq(timeframe.lower())]
    timestamps = pd.to_datetime(result[datetime_column], errors="coerce", utc=True)
    result = result.assign(_bounded_datetime=timestamps).dropna(subset=["_bounded_datetime"])
    if start:
        result = result.loc[result["_bounded_datetime"] >= pd.to_datetime(start, utc=True)]
    if end:
        result = result.loc[result["_bounded_datetime"] <= pd.to_datetime(end, utc=True)]
    result = result.sort_values("_bounded_datetime").drop_duplicates("_bounded_datetime", keep="last")
    if max_rows > 0:
        result = result.tail(int(max_rows))
    result[datetime_column] = result["_bounded_datetime"]
    result = result.drop(columns=["_bounded_datetime"])
    return result.set_index(datetime_column, drop=False)


def _source_quality_checks(frame: pd.DataFrame, *, min_rows: int) -> list[dict[str, Any]]:
    row_count = len(frame)
    checks = [
        _check(
            "pass" if row_count >= min_rows else "fail",
            "minimum_rows",
            f"rows={row_count}, required={min_rows}.",
        ),
        _check(
            "pass" if frame.index.is_monotonic_increasing else "fail",
            "chronological_order",
            "Datetime index is chronological.",
        ),
        _check(
            "pass" if not frame.index.has_duplicates else "fail",
            "unique_timestamps",
            "Datetime index is unique.",
        ),
    ]
    close = (
        pd.to_numeric(frame["close"], errors="coerce")
        if "close" in frame.columns
        else pd.Series(dtype=float)
    )
    finite_close_count = int(np.isfinite(close).sum())
    checks.append(
        _check(
            "pass" if finite_close_count == row_count else "fail",
            "finite_close_prices",
            f"finite_close_rows={finite_close_count}/{row_count}.",
        )
    )
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    max_abs_return = float(returns.abs().max()) if not returns.empty else math.inf
    checks.append(
        _check(
            "pass" if max_abs_return <= 0.25 else "fail",
            "extreme_return_boundary",
            f"max_abs_one_step_return={max_abs_return:.6f}, limit=0.25.",
        )
    )
    return checks


def _prepare_model_frame(enriched: pd.DataFrame, *, target_name: str) -> pd.DataFrame:
    if target_name not in enriched.columns:
        raise ValueError(f"Target column is missing after enrichment: {target_name}")
    frame = enriched.copy()
    frame["_bounded_datetime"] = _resolve_enriched_datetime(frame)
    frame["__forward_return"] = pd.to_numeric(frame["close"], errors="coerce").shift(-1) / pd.to_numeric(
        frame["close"], errors="coerce"
    ) - 1.0
    frame[target_name] = pd.to_numeric(frame[target_name], errors="coerce")
    frame = frame.dropna(subset=["_bounded_datetime", target_name, "__forward_return"])
    frame = frame.sort_values("_bounded_datetime").drop_duplicates("_bounded_datetime", keep="last")
    frame = frame.set_index("_bounded_datetime", drop=True)
    return frame


def _resolve_enriched_datetime(frame: pd.DataFrame) -> pd.Series:
    exact = [column for column in ("datetime", "timestamp", "date") if column in frame.columns]
    suffixed = [
        column
        for column in frame.columns
        if str(column).lower().startswith(("datetime_", "timestamp_", "date_"))
    ]
    for columns in (exact, suffixed):
        candidates = []
        for column in columns:
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
            valid_count = int(parsed.notna().sum())
            if valid_count:
                candidates.append((valid_count, str(column), parsed))
        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return candidates[0][2]
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(
            pd.to_datetime(frame.index, errors="coerce", utc=True),
            index=frame.index,
        )
    raise ValueError("Enriched frame has no usable datetime column or DatetimeIndex.")


def _chronological_split(
    frame: pd.DataFrame,
    *,
    gap_size: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, pd.DataFrame]:
    total = len(frame)
    gap = max(1, int(gap_size))
    validation_count = max(20, int(total * validation_fraction))
    test_count = max(20, int(total * test_fraction))
    train_end = total - validation_count - test_count - (2 * gap)
    validation_start = train_end + gap
    validation_end = validation_start + validation_count
    test_start = validation_end + gap
    if train_end < 60 or test_start >= total:
        raise ValueError(
            f"Insufficient rows for bounded purged split: total={total}, train={train_end}, "
            f"validation={validation_count}, test={test_count}, gap={gap}."
        )
    return {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[validation_start:validation_end].copy(),
        "test": frame.iloc[test_start:].copy(),
    }


def _select_train_features(
    train: pd.DataFrame,
    *,
    target_name: str,
    max_features: int,
) -> list[str]:
    excluded = {
        target_name,
        "__forward_return",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    numeric = train.select_dtypes(include=[np.number, "bool"])
    candidates = [
        str(column)
        for column in numeric.columns
        if column not in excluded and not is_target_like_column(column)
    ]
    ranked = []
    for column in candidates:
        values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        coverage = float(values.notna().mean())
        variance = float(values.var()) if coverage else 0.0
        if coverage < 0.8 or not math.isfinite(variance) or variance <= 0:
            continue
        ranked.append((column, coverage, math.log1p(abs(variance))))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    limit = max(1, int(max_features))
    return [item[0] for item in ranked[:limit]]


def _impute_split_features(
    split: dict[str, pd.DataFrame],
    selected_features: list[str],
) -> dict[str, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    train_values = split["train"][selected_features].replace([np.inf, -np.inf], np.nan)
    imputer.fit(train_values)
    result = {}
    for split_id in ("train", "validation", "test"):
        values = split[split_id][selected_features].replace([np.inf, -np.inf], np.nan)
        result[split_id] = pd.DataFrame(
            imputer.transform(values),
            index=values.index,
            columns=selected_features,
        )
    return result


def _classification_metrics(target: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    accuracy = float(accuracy_score(target, prediction))
    balanced_accuracy = float(balanced_accuracy_score(target, prediction))
    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        prediction,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(target, prediction, labels=[0, 1]).tolist()
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "score": accuracy,
        "sample_count": int(len(target)),
        "actual_positive_rate": float(target.mean()),
        "predicted_positive_rate": float(prediction.mean()),
        "majority_class_baseline": float(target.value_counts(normalize=True).max()),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": matrix,
    }


def _heldout_evaluation(
    *,
    test_frame: pd.DataFrame,
    predictions: pd.Series,
    ticker: str,
    timeframe: str,
    target_name: str,
    model_type: str,
    context_fingerprint: str,
    transaction_cost_per_turn: float,
) -> dict[str, Any]:
    positions = predictions.astype(float).clip(0.0, 1.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    forward_returns = test_frame["__forward_return"].astype(float)
    gross_strategy_returns = positions * forward_returns
    transaction_costs = turnover * max(0.0, float(transaction_cost_per_turn))
    strategy_returns = (gross_strategy_returns - transaction_costs).clip(lower=-0.99)
    equity = (1.0 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    total_return = float(equity.iloc[-1] - 1.0)
    max_drawdown = float(abs(drawdown.min()))
    periods_per_year = _periods_per_year(timeframe)
    return_std = float(strategy_returns.std(ddof=1))
    sharpe = (
        float(strategy_returns.mean() / return_std * math.sqrt(periods_per_year))
        if return_std > 0
        else 0.0
    )
    signals = pd.DataFrame(
        {
            "ticker": ticker,
            "selected_primary_model": model_type,
            "model_context_id": f"{ticker}_{target_name}_{model_type}",
            "target_name": target_name,
            "model_type": model_type,
            "timeframe": timeframe,
            "context_fingerprint": context_fingerprint,
            "prediction": predictions.astype(int),
            "position": positions,
            "price": pd.to_numeric(test_frame["close"], errors="coerce"),
        },
        index=test_frame.index,
    )
    portfolio_history = pd.DataFrame(
        {
            "strategy_return": strategy_returns,
            "equity": equity,
            "drawdown": drawdown,
        },
        index=test_frame.index,
    )
    financial_metrics = {
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "sharpe": sharpe,
        "volatility": float(return_std * math.sqrt(periods_per_year)),
    }
    evaluation_window = {
        "start": str(test_frame.index[0]),
        "end": str(test_frame.index[-1]),
        "sample_count": len(test_frame),
        "source": "bounded_heldout_test_index",
    }
    active_returns = gross_strategy_returns.loc[positions > 0]
    diagnostics = {
        "active_bar_count": int((positions > 0).sum()),
        "flat_bar_count": int((positions <= 0).sum()),
        "position_change_count": int((turnover > 0).sum()),
        "total_turnover": float(turnover.sum()),
        "transaction_cost_total": float(transaction_costs.sum()),
        "gross_total_return": float((1.0 + gross_strategy_returns.clip(lower=-0.99)).prod() - 1.0),
        "net_total_return": total_return,
        "active_bar_positive_return_rate": (
            float((active_returns > 0).mean()) if not active_returns.empty else None
        ),
        "annualization_periods": periods_per_year,
        "annualized_sharpe_sample_count": int(len(strategy_returns)),
        "annualized_sharpe_caution": "Short or irregular windows can make annualized Sharpe unstable.",
    }
    return {
        "financial_metrics": financial_metrics,
        "backtest_results": {
            "performance": financial_metrics,
            "evaluation_window": evaluation_window,
            "transaction_cost_per_turn": float(transaction_cost_per_turn),
            "execution_mode": "offline_long_flat_metric_evaluation",
        },
        "evaluation_summary": {
            "status": "bounded_heldout_evaluation_complete",
            "metrics": financial_metrics,
            "evaluation_window": evaluation_window,
        },
        "signals": signals,
        "portfolio_history": portfolio_history,
        "evaluation_window": evaluation_window,
        "diagnostics": diagnostics,
    }


def _run_real_metric_review(
    *,
    output_dir: Path,
    locked_model_path: str | None,
    locked_feature_path: str | None,
    run_real_metric_review: bool,
    replay_batch_json: str | Path | None,
    data_quality_json: str | Path | None,
    architecture_map_json: str | Path | None,
    domain_instance_contract_json: str | Path | None,
) -> dict[str, Any]:
    if not run_real_metric_review:
        return {"invoked": False, "skip_reason": "disabled_by_operator"}
    if not locked_model_path or not locked_feature_path:
        return {"invoked": False, "skip_reason": "locked_metric_pair_unavailable"}
    payload = PipelineControlRealMetricEvidenceRun(output_dir).build(
        model_evaluation_json=locked_model_path,
        feature_stability_report=locked_feature_path,
        replay_batch_json=replay_batch_json,
        data_quality_json=data_quality_json,
        architecture_map_json=architecture_map_json,
        domain_instance_contract_json=domain_instance_contract_json,
    )
    return {"invoked": True, "payload": payload}


def _summary(
    *,
    source_checks: list[dict[str, Any]],
    split: dict[str, pd.DataFrame],
    selected_features: list[str],
    split_metrics: dict[str, dict[str, float]],
    stability_analysis: dict[str, Any],
    heldout: dict[str, Any],
    locked_model_path: str | None,
    locked_feature_path: str | None,
    real_metric_review: dict[str, Any],
    macro_provenance: dict[str, Any],
    macro_used: bool,
) -> dict[str, Any]:
    real_summary = (
        real_metric_review.get("payload", {}).get("summary", {})
        if real_metric_review.get("invoked")
        else {}
    )
    pair_ready = bool(locked_model_path and locked_feature_path)
    status = "bounded_locked_metric_pair_ready"
    if real_metric_review.get("invoked"):
        status = real_summary.get("real_metric_evidence_status", "bounded_real_metric_review_complete")
    elif not pair_ready:
        status = "bounded_metric_pair_blocked"
    return {
        "bounded_evidence_status": status,
        "source_quality_passed": all(check["status"] != "fail" for check in source_checks),
        "model_row_count": sum(len(split[split_id]) for split_id in split),
        "train_sample_count": len(split["train"]),
        "validation_sample_count": len(split["validation"]),
        "test_sample_count": len(split["test"]),
        "selected_feature_count": len(selected_features),
        "selected_macro_feature_count": sum(
            1 for feature in selected_features if str(feature).upper().startswith("FRED_")
        ),
        "macro_artifact_provided": bool(macro_provenance.get("provided")),
        "macro_used_in_stage_3": macro_used,
        "macro_series_count": int(macro_provenance.get("series_count", 0) or 0),
        "macro_latest_observation_age_days": macro_provenance.get(
            "latest_observation_age_days"
        ),
        "train_score": split_metrics["train"]["score"],
        "validation_score": split_metrics["validation"]["score"],
        "test_score": split_metrics["test"]["score"],
        "test_balanced_accuracy": split_metrics["test"]["balanced_accuracy"],
        "test_majority_class_baseline": split_metrics["test"]["majority_class_baseline"],
        "max_drawdown": heldout["financial_metrics"]["max_drawdown"],
        "total_return": heldout["financial_metrics"]["total_return"],
        "sharpe": heldout["financial_metrics"]["sharpe"],
        "feature_stability_score": stability_analysis.get("feature_stability_score"),
        "unstable_feature_count": stability_analysis.get("unstable_feature_count"),
        "locked_model_evaluation_ready": bool(locked_model_path),
        "locked_feature_stability_ready": bool(locked_feature_path),
        "real_metric_review_invoked": bool(real_metric_review.get("invoked")),
        "blocked_metric_planes": real_summary.get("blocked_metric_planes", []),
        "caution_metric_planes": real_summary.get("caution_metric_planes", []),
        "can_use_as_metric_evidence": bool(real_summary.get("can_use_as_metric_evidence", False)),
        "can_clear_current_real_cautions": bool(
            real_summary.get("can_clear_current_real_cautions", False)
        ),
        "can_write_learning_memory": False,
        "can_write_production_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _split_windows(split: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    return {
        split_id: {
            "start": str(frame.index[0]),
            "end": str(frame.index[-1]),
            "sample_count": len(frame),
        }
        for split_id, frame in split.items()
    }


def _context_fingerprint(
    *,
    source_fingerprint: str,
    macro_fingerprint: str | None,
    ticker: str,
    timeframe: str,
    target_name: str,
    selected_features: list[str],
    split: dict[str, pd.DataFrame],
) -> str:
    payload = {
        "source_sha256": source_fingerprint,
        "macro_source_sha256": macro_fingerprint,
        "ticker": ticker,
        "timeframe": timeframe,
        "target_name": target_name,
        "model_type": "random_forest",
        "selected_features": selected_features,
        "split_windows": _split_windows(split),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded_provenance(
    *,
    source: Path,
    source_fingerprint: str,
    macro_provenance: dict[str, Any],
    macro_used: bool,
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "source_path": str(source),
        "source_sha256": source_fingerprint,
        "source_evidence_class": "saved_offline_market_data",
        "synthetic": False,
        "live_collection_performed": False,
        "macro_source": macro_provenance,
        "macro_used_in_stage_3": macro_used,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _periods_per_year(timeframe: str) -> float:
    normalized = timeframe.strip().lower()
    return {
        "15m": 252.0 * 26.0,
        "30m": 252.0 * 13.0,
        "60m": 252.0 * 6.5,
        "1h": 252.0 * 6.5,
        "1d": 252.0,
    }.get(normalized, 252.0)


def _saved_paths(real_metric_review: dict[str, Any]) -> dict[str, Any]:
    if not real_metric_review.get("invoked"):
        return {}
    return real_metric_review.get("payload", {}).get("saved_paths", {})


def _check(status: str, code: str, message: str) -> dict[str, Any]:
    return {"status": status, "code": code, "message": message}


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector or external API is called.",
        "No Stage 6 execution, order, broker route, paper trade, or live trade is created.",
        "The trained model is stored only under this review run and is not promoted to production.",
        "No autonomous tuning, learning-memory write, production-config write, or model promotion occurs.",
        "Held-out positions are metric-only long/flat observations, not recommendations.",
    ]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
