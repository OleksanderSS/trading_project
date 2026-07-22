from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from src.pipeline.stages.modeling.walk_forward_validation import (
    PipelineWalkForwardValidationEvaluator,
    WalkForwardValidationConfig,
)

_REQUIRED_CONTEXT_TIMEFRAMES = {
    "15m": ("15m", "60m", "1d"),
    "60m": ("60m", "1d"),
    "1d": ("1d",),
}


class PipelineControlWalkForwardValidationRun:
    """Run development-only Stage 3 and purged walk-forward validation."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_control_walk_forward_validation_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        *,
        historical_recovery_json: str | Path,
        ticker: str,
        timeframe: str,
        target_name: str,
        acknowledge_development_only: bool,
        macro_source_path: str | Path | None = None,
        forward_accrual_gate_json: str | Path | None = None,
        min_train_rows: int = 360,
        validation_rows: int = 120,
        step_rows: int = 120,
        purge_rows: int = 5,
        max_folds: int = 4,
        max_features: int = 40,
        save: bool = True,
    ) -> dict[str, Any]:
        if not acknowledge_development_only:
            raise ValueError(
                "Explicit development-only acknowledgement is required."
            )
        normalized_timeframe = str(timeframe).strip().lower()
        if normalized_timeframe not in _REQUIRED_CONTEXT_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}.")

        recovery_path = Path(historical_recovery_json)
        recovery = _load_json(recovery_path)
        recovery_status = recovery.get("summary", {}).get("recovery_status")
        if recovery_status != "historical_context_partitions_ready":
            raise ValueError(
                "Historical recovery artifact is not ready for development use."
            )
        development_frames, source_lineage = _load_development_frames(
            recovery,
            ticker=ticker,
            base_timeframe=normalized_timeframe,
        )
        forward_lineage: dict[str, Any] | None = None
        forward_base_rows = 0
        if forward_accrual_gate_json is not None:
            forward_frames, forward_lineage = (
                _load_forward_development_frames(
                    Path(forward_accrual_gate_json),
                    ticker=ticker,
                    base_timeframe=normalized_timeframe,
                    target_name=target_name,
                )
            )
            forward_base_rows = len(forward_frames[normalized_timeframe])
            development_frames = _merge_development_frames(
                development_frames,
                forward_frames,
            )
        macro_frame, macro_lineage = _load_macro_frame(
            Path(macro_source_path) if macro_source_path else None
        )
        enriched, context_report = await _run_active_stage_3(
            development_frames,
            macro_frame=macro_frame,
        )
        context_summary = context_report.get("summary", {})
        if context_report.get("join_direction") != "backward":
            raise ValueError("Stage 3 context lineage is not backward-only.")
        if int(context_summary.get("future_context_violations", -1)) != 0:
            raise ValueError("Stage 3 context lineage contains future matches.")
        if context_summary.get("row_identity_preserved") is not True:
            raise ValueError("Stage 3 context assembly did not preserve row identity.")

        config = WalkForwardValidationConfig(
            min_train_rows=min_train_rows,
            validation_rows=validation_rows,
            step_rows=step_rows,
            purge_rows=purge_rows,
            max_folds=max_folds,
            max_features=max_features,
        )
        full_lineage = {
            "historical_recovery_json": {
                "path": str(recovery_path),
                "sha256": _sha256_file(recovery_path),
            },
            "development_artifacts": source_lineage,
            "forward_development": forward_lineage,
            "macro": macro_lineage,
        }
        candidate = PipelineWalkForwardValidationEvaluator(config).evaluate(
            enriched,
            ticker=ticker,
            timeframe=normalized_timeframe,
            target_name=target_name,
            timeframe_context_report=context_report,
            source_lineage=full_lineage,
        )
        metrics = candidate["metrics"]
        run_id = _run_id("pipeline_control_walk_forward_validation")
        summary = {
            "validation_status": candidate["contract_status"],
            "ticker": ticker.upper(),
            "timeframe": normalized_timeframe,
            "target_name": target_name,
            "development_rows_loaded": int(
                sum(len(frame) for frame in development_frames.values())
            ),
            "forward_development_base_rows_loaded": forward_base_rows,
            "fold_count": metrics["fold_count"],
            "mean_validation_balanced_accuracy": metrics[
                "mean_validation_balanced_accuracy"
            ],
            "minimum_validation_balanced_accuracy": metrics[
                "minimum_validation_balanced_accuracy"
            ],
            "mean_train_validation_gap": metrics[
                "mean_train_validation_gap"
            ],
            "mean_feature_stability_score": metrics[
                "mean_feature_stability_score"
            ],
            "test_rows_loaded": 0,
            "test_metrics_read": False,
            "past_evaluation_rows_loaded": 0,
            "frozen_test_windows_accessed": False,
            "contract_passed": metrics["contract_passed"],
            "can_freeze_candidate_for_new_holdout": candidate[
                "promotion_contract"
            ]["can_freeze_candidate_for_new_holdout"],
            "can_promote_model": False,
            "can_write_production_config": False,
            "can_trade": False,
        }
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_walk_forward_validation_run",
            "summary": summary,
            "inputs": {
                "historical_recovery_json": str(recovery_path),
                "ticker": ticker.upper(),
                "timeframe": normalized_timeframe,
                "target_name": target_name,
                "macro_source_path": (
                    str(macro_source_path) if macro_source_path else None
                ),
                "forward_accrual_gate_json": (
                    str(forward_accrual_gate_json)
                    if forward_accrual_gate_json
                    else None
                ),
                "acknowledge_development_only": (
                    acknowledge_development_only
                ),
                "validation_config": asdict(config),
            },
            "source_lineage": full_lineage,
            "timeframe_context_report": context_report,
            "walk_forward_candidate": candidate,
            "next_step": (
                "Freeze this feature/model contract and accumulate a new forward "
                "holdout without inspecting it during development."
                if metrics["contract_passed"]
                else (
                    "Keep the candidate blocked. Review fold failures and data coverage; "
                    "do not inspect frozen tests or launch a variant loop."
                )
            ),
            "explicit_non_actions": [
                "Only historical development_* artifacts and optional gate-approved forward-development rows were loaded.",
                "Past-evaluation and frozen test artifacts were not opened.",
                "Stage 3 ran offline against saved local data.",
                "The fixed RandomForest fold models were not persisted.",
                "No hyperparameter search, production promotion, config write, recommendation, order, or trade ran.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_walk_forward_validation_markdown(payload),
                run_id=run_id,
            )
        return json_ready(payload)


def render_walk_forward_validation_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Pipeline Control Walk-Forward Validation",
        "",
        f"- Status: `{summary.get('validation_status')}`",
        f"- Context: `{summary.get('ticker')}/{summary.get('timeframe')}/{summary.get('target_name')}`",
        f"- Development rows loaded: {summary.get('development_rows_loaded')}",
        f"- Fold count: {summary.get('fold_count')}",
        f"- Mean validation balanced accuracy: {summary.get('mean_validation_balanced_accuracy')}",
        f"- Minimum validation balanced accuracy: {summary.get('minimum_validation_balanced_accuracy')}",
        f"- Mean train-validation gap: {summary.get('mean_train_validation_gap')}",
        f"- Mean feature stability: {summary.get('mean_feature_stability_score')}",
        f"- Test rows loaded: {summary.get('test_rows_loaded')}",
        f"- Past-evaluation rows loaded: {summary.get('past_evaluation_rows_loaded')}",
        f"- Contract passed: {summary.get('contract_passed')}",
        f"- Can promote model: {summary.get('can_promote_model')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Folds",
        "",
    ]
    candidate = payload.get("walk_forward_candidate", {})
    for fold in candidate.get("folds", []):
        validation = fold.get("validation_metrics", {})
        lines.append(
            f"- Fold {fold.get('fold')}: validation_balanced="
            f"{validation.get('balanced_accuracy')} accuracy="
            f"{validation.get('accuracy')} majority="
            f"{validation.get('majority_class_baseline')} gap="
            f"{fold.get('train_validation_balanced_accuracy_gap')} stability="
            f"{fold.get('feature_stability', {}).get('feature_stability_score')}"
        )
    lines.extend(["", "## Next Step", "", str(payload.get("next_step", ""))])
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(
        f"- {item}" for item in payload.get("explicit_non_actions", [])
    )
    return "\n".join(lines).strip() + "\n"


async def _run_active_stage_3(
    frames: dict[str, pd.DataFrame],
    *,
    macro_frame: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from src.config.unified_config_manager import UnifiedConfigManager
    from src.core.error_handling.error_handler import ErrorHandler
    from src.pipeline.stages.stage_3_feature_engineering import (
        FeatureEngineeringStage,
    )

    config_manager = UnifiedConfigManager()
    stage = FeatureEngineeringStage(
        config_manager,
        ErrorHandler(config_manager),
        mode="walk_forward_review",
    )
    cleaned_data: dict[str, Any] = {
        "prices": {
            timeframe: frame.reset_index(drop=True)
            for timeframe, frame in frames.items()
        }
    }
    if macro_frame is not None:
        cleaned_data["macro_data"] = macro_frame.reset_index(drop=True)
    result = await stage.run(
        cleaned_data=cleaned_data,
        target_column="__walk_forward_skip_stage3_selection__",
        context_id="pipeline_control_walk_forward_validation",
        offline_only=True,
    )
    enriched = result.get("enriched_data")
    context_report = result.get("timeframe_context_report")
    if not isinstance(enriched, pd.DataFrame) or enriched.empty:
        raise ValueError("Active Stage 3 produced no enriched data.")
    if not isinstance(context_report, dict):
        raise ValueError("Active Stage 3 produced no timeframe context report.")
    return enriched, context_report


def _load_development_frames(
    recovery: dict[str, Any],
    *,
    ticker: str,
    base_timeframe: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    artifacts = recovery.get("artifacts", {})
    frames: dict[str, pd.DataFrame] = {}
    lineage: dict[str, Any] = {}
    for timeframe in _REQUIRED_CONTEXT_TIMEFRAMES[base_timeframe]:
        artifact_id = f"development_{timeframe}"
        artifact = artifacts.get(artifact_id)
        if not isinstance(artifact, dict):
            raise ValueError(f"Missing recovery artifact: {artifact_id}.")
        if artifact.get("synthetic") is not False:
            raise ValueError(f"Development artifact is not real: {artifact_id}.")
        path = Path(str(artifact.get("path", "")))
        if not path.exists():
            raise FileNotFoundError(f"Development artifact is missing: {path}.")
        if path.suffix.lower() != ".parquet" or not _has_parquet_magic(path):
            raise ValueError(f"Development artifact is not real Parquet: {path}.")
        actual_sha256 = _sha256_file(path)
        expected_sha256 = artifact.get("sha256")
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(f"Development artifact hash mismatch: {artifact_id}.")
        frame = pd.read_parquet(path)
        if "ticker" not in frame.columns:
            raise ValueError(f"Development artifact lacks ticker: {artifact_id}.")
        frame = frame.loc[
            frame["ticker"].astype(str).str.upper().eq(ticker.upper())
        ].copy()
        if frame.empty:
            raise ValueError(
                f"Development artifact has no rows for {ticker}: {artifact_id}."
            )
        if "partition_id" in frame.columns:
            partitions = set(
                frame["partition_id"].dropna().astype(str).str.lower().unique()
            )
            if partitions and partitions != {"development"}:
                raise ValueError(
                    f"Development artifact has unexpected partitions: {partitions}."
                )
        frame["partition_id"] = "development"
        frames[timeframe] = frame
        lineage[artifact_id] = {
            "path": str(path),
            "sha256": actual_sha256,
            "row_count_for_ticker": int(len(frame)),
            "synthetic": False,
            "partition": "development",
        }
    return frames, lineage


def _load_forward_development_frames(
    gate_path: Path,
    *,
    ticker: str,
    base_timeframe: str,
    target_name: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    gate = _load_json(gate_path)
    summary = gate.get("summary", {})
    if (
        gate.get("mode")
        != "pipeline_control_forward_data_accrual_gate"
        or summary.get("gate_status")
        != "forward_development_artifact_ready"
        or summary.get("can_supply_next_development_run") is not True
    ):
        raise ValueError(
            "Forward accrual gate is not ready to supply a development run."
        )
    artifact = gate.get("eligible_development_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Forward accrual gate has no eligible artifact.")
    if (
        artifact.get("artifact_class")
        != "pipeline_control_forward_development_artifact"
        or artifact.get("evidence_class")
        != "validated_forward_development_source"
    ):
        raise ValueError(
            "Forward accrual gate artifact class is not accepted."
        )
    expected_context = (
        f"{ticker.upper()}/{base_timeframe}/{target_name}"
    )
    if artifact.get("context_key") != expected_context:
        raise ValueError(
            "Forward accrual gate context does not match requested run."
        )
    if (
        artifact.get("lane") != "development_refresh_only"
        or artifact.get("may_be_used_as_locked_test_evidence") is not False
        or artifact.get("may_be_called_virgin_holdout") is not False
    ):
        raise ValueError(
            "Forward accrual gate does not preserve development-only safety."
        )
    source_path = Path(str(artifact.get("source_path", "")))
    if (
        not source_path.exists()
        or source_path.suffix.lower() != ".parquet"
        or not _has_parquet_magic(source_path)
    ):
        raise ValueError(
            f"Forward development source is not real Parquet: {source_path}."
        )
    actual_sha256 = _sha256_file(source_path)
    if actual_sha256 != artifact.get("source_sha256"):
        raise ValueError("Forward development source hash mismatch.")
    frame = pd.read_parquet(source_path)
    required = {
        "datetime",
        "ticker",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Forward development source lacks columns: {missing}."
        )
    target_like = [
        str(column)
        for column in frame.columns
        if _is_target_like_column(column)
    ]
    if target_like:
        raise ValueError(
            f"Forward development source contains target columns: {target_like}."
        )
    frame["datetime"] = pd.to_datetime(
        frame["datetime"], errors="coerce", utc=True
    ).astype("datetime64[ns, UTC]")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["interval"] = frame["interval"].astype(str).str.lower()
    watermark = pd.to_datetime(
        artifact.get("start_exclusive"),
        errors="coerce",
        utc=True,
    )
    if pd.isna(watermark):
        raise ValueError("Forward accrual gate has no valid watermark.")
    base = frame.loc[
        frame["ticker"].eq(ticker.upper())
        & frame["interval"].eq(base_timeframe)
        & frame["datetime"].gt(watermark)
    ].copy()
    base = (
        base.sort_values("datetime", kind="mergesort")
        .drop_duplicates(
            ["ticker", "interval", "datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )
    recorded_rows = int(artifact.get("eligible_new_row_count", -1))
    if len(base) != recorded_rows:
        raise ValueError(
            "Forward development row count no longer matches accrual gate."
        )
    if base.empty:
        raise ValueError("Forward accrual gate supplies no new rows.")
    base["partition_id"] = "forward_development"
    frames = {base_timeframe: base}
    for timeframe in _REQUIRED_CONTEXT_TIMEFRAMES[base_timeframe]:
        if timeframe == base_timeframe:
            continue
        frames[timeframe] = _aggregate_forward_ohlcv(
            base,
            timeframe=timeframe,
        )
    lineage = {
        "accrual_gate_json": {
            "path": str(gate_path),
            "sha256": _sha256_file(gate_path),
        },
        "source_path": str(source_path),
        "source_sha256": actual_sha256,
        "context_key": expected_context,
        "start_exclusive": watermark.isoformat(),
        "base_row_count": len(base),
        "derived_context_row_counts": {
            timeframe: len(context)
            for timeframe, context in frames.items()
            if timeframe != base_timeframe
        },
        "partition": "forward_development",
        "test_rows_loaded": 0,
        "past_evaluation_rows_loaded": 0,
    }
    return frames, lineage


def _aggregate_forward_ohlcv(
    frame: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.DataFrame:
    if timeframe not in {"60m", "1d"}:
        raise ValueError(
            f"Unsupported derived forward timeframe: {timeframe}."
        )
    result = frame[
        [
            "datetime",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()
    result["_bucket"] = (
        result["datetime"].dt.floor("60min")
        if timeframe == "60m"
        else result["datetime"].dt.normalize()
    )
    result = (
        result.sort_values("datetime", kind="mergesort")
        .groupby(["ticker", "_bucket"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .rename(columns={"_bucket": "datetime"})
    )
    result["interval"] = timeframe
    result["partition_id"] = "forward_development"
    return result[
        [
            "datetime",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "interval",
            "partition_id",
        ]
    ]


def _merge_development_frames(
    historical: dict[str, pd.DataFrame],
    forward: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for timeframe, historical_frame in historical.items():
        forward_frame = forward.get(timeframe)
        if forward_frame is None:
            result[timeframe] = historical_frame.copy()
            continue
        combined = pd.concat(
            [historical_frame, forward_frame],
            ignore_index=True,
            sort=False,
        )
        combined["datetime"] = pd.to_datetime(
            combined["datetime"], errors="coerce", utc=True
        ).astype("datetime64[ns, UTC]")
        combined["ticker"] = combined["ticker"].astype(str).str.upper()
        combined["interval"] = (
            combined["interval"].astype(str).str.lower()
        )
        combined["partition_id"] = combined["partition_id"].astype(str)
        result[timeframe] = (
            combined.sort_values(
                ["ticker", "partition_id", "datetime"],
                kind="mergesort",
            )
            .drop_duplicates(
                ["ticker", "interval", "partition_id", "datetime"],
                keep="last",
            )
            .reset_index(drop=True)
        )
    return result


def _is_target_like_column(column: Any) -> bool:
    normalized = str(column).strip().lower()
    return (
        normalized.startswith("target")
        or "_target" in normalized
        or normalized.startswith("label")
        or normalized in {"y", "prediction", "predicted_target"}
    )


def _load_macro_frame(
    path: Path | None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if path is None:
        return None, {
            "provided": False,
            "offline": True,
            "external_fetch_performed": False,
        }
    if not path.exists():
        raise FileNotFoundError(f"Macro source does not exist: {path}.")
    if path.suffix.lower() == ".parquet":
        if not _has_parquet_magic(path):
            raise ValueError(f"Macro source is not real Parquet: {path}.")
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Macro source must be Parquet or CSV.")
    if frame.empty:
        raise ValueError("Macro source is empty.")
    frame = _normalize_macro_long_form(frame)
    return frame, {
        "provided": True,
        "path": str(path),
        "sha256": _sha256_file(path),
        "row_count": int(len(frame)),
        "series_count": int(frame["series_id"].nunique()),
        "observation_start": frame["datetime"].min().isoformat(),
        "observation_end": frame["datetime"].max().isoformat(),
        "offline": True,
        "external_fetch_performed": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact does not exist: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}.")
    return payload


def _normalize_macro_long_form(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename_map: dict[str, str] = {}
    if "series_id" not in result.columns and "series" in result.columns:
        rename_map["series"] = "series_id"
    if "datetime" not in result.columns:
        date_column = next(
            (
                column
                for column in ("date", "timestamp")
                if column in result.columns
            ),
            None,
        )
        if date_column:
            rename_map[date_column] = "datetime"
    if rename_map:
        result = result.rename(columns=rename_map)
    required = {"datetime", "series_id", "value"}
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(
            f"Macro source is missing long-form columns: {missing}."
        )
    result["datetime"] = pd.to_datetime(
        result["datetime"],
        errors="coerce",
        utc=True,
    ).astype("datetime64[ns, UTC]")
    result["series_id"] = result["series_id"].astype("string").str.strip()
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["datetime", "series_id", "value"])
    result = result.loc[result["series_id"].ne("")]
    result = (
        result.sort_values(["datetime", "series_id"], kind="mergesort")
        .drop_duplicates(["datetime", "series_id"], keep="last")
        .reset_index(drop=True)
    )
    if result.empty:
        raise ValueError("Macro source has no usable long-form observations.")
    return result


def _has_parquet_magic(path: Path) -> bool:
    with path.open("rb") as handle:
        if path.stat().st_size < 8:
            return False
        prefix = handle.read(4)
        handle.seek(-4, 2)
        suffix = handle.read(4)
    return prefix == b"PAR1" and suffix == b"PAR1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    return (
        f"{prefix}_"
        f"{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
    )
