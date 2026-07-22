from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_feature_timeframe_audit import (
    PipelineFeatureTimeframeAudit,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_target_readiness_audit import (
    PipelineTargetReadinessAudit,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from src.pipeline.timeframe_lineage import (
    normalize_timeframe,
    timeframe_lineage_report,
)

STAGE23_SCHEMA_VERSION = "dean_pipeline_stage23_regeneration_v1"
STAGE3_CACHE_SCHEMA_VERSION = "dean_pipeline_stage23_stage3_cache_v1"


class PipelineStage23Regeneration:
    """Regenerate one bounded Stage 2/3 batch from a saved Stage 1 source."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_stage23_regeneration"
        ),
    ):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        *,
        source_path: str | Path,
        batch_dir: str | Path,
        tickers: list[str],
        timeframe: str,
        max_rows_per_ticker: int = 600,
        shard_cache_dir: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        source = Path(source_path)
        requested_tickers = sorted(
            {
                str(ticker).strip().upper()
                for ticker in tickers
                if str(ticker).strip()
            }
        )
        resolved_timeframe = normalize_timeframe(timeframe)
        if not requested_tickers:
            raise ValueError("At least one ticker is required")
        if not resolved_timeframe:
            raise ValueError("A supported timeframe is required")
        if max_rows_per_ticker < 5:
            raise ValueError("max_rows_per_ticker must be at least 5")

        total_start = time.perf_counter()
        timings_seconds: dict[str, float] = {}
        started = time.perf_counter()
        raw_market, source_format = _load_saved_stage1_market(source)
        timings_seconds["source_load"] = _elapsed(started)
        started = time.perf_counter()
        source_sha256 = _sha256(source)
        timings_seconds["source_sha256"] = _elapsed(started)
        started = time.perf_counter()
        selected = _select_bounded_market_frame(
            raw_market,
            tickers=requested_tickers,
            timeframe=resolved_timeframe,
            max_rows_per_ticker=max_rows_per_ticker,
        )
        timings_seconds["source_select"] = _elapsed(started)
        started = time.perf_counter()
        source_checks = _source_checks(
            selected,
            tickers=requested_tickers,
            timeframe=resolved_timeframe,
        )
        timings_seconds["source_checks"] = _elapsed(started)
        blockers = [
            check["code"]
            for check in source_checks
            if check["status"] == "fail"
        ]
        run_id = _run_id("pipeline_stage23_regeneration")
        base_payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_stage23_regeneration",
            "schema_version": "dean_pipeline_stage23_regeneration_v1",
            "source_artifact": {
                "path": str(source),
                "sha256": source_sha256,
                "format": source_format,
                "source_market_row_count": len(raw_market),
            },
            "scope": {
                "tickers": requested_tickers,
                "timeframe": resolved_timeframe,
                "max_rows_per_ticker": max_rows_per_ticker,
                "selected_row_count": len(selected),
                "shard_cache_dir": str(shard_cache_dir)
                if shard_cache_dir
                else str(self.output_dir / "stage3_shard_cache"),
            },
            "source_checks": source_checks,
            "runtime_profile": {
                "timings_seconds": timings_seconds,
                "total_elapsed_seconds": _elapsed(total_start),
            },
            "safety": _safety(),
        }
        if blockers:
            base_payload["runtime_profile"][
                "total_elapsed_seconds"
            ] = _elapsed(total_start)
            payload = {
                **base_payload,
                "status": "stage23_regeneration_blocked_source",
                "blocking_reasons": blockers,
                "batch_artifacts": {},
                "feature_timeframe_audit": {},
            }
            return self._save(payload, save=save)

        started = time.perf_counter()
        stage2_frame, stage2_quality = _run_bounded_stage2(
            selected,
            timeframe=resolved_timeframe,
        )
        timings_seconds["stage2"] = _elapsed(started)
        from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_bounded_evidence_run import (
            _run_stage_3_enrichment,
        )

        started = time.perf_counter()
        enriched = await _run_stage_3_enrichment(
            stage2_frame,
            timeframe=resolved_timeframe,
        )
        timings_seconds["stage3_enrichment"] = _elapsed(started)
        from src.pipeline.hybrid.colab_manager import (
            BatchPreparationConfig,
            ColabManager,
        )
        from src.pipeline.hybrid.feature_processor import FeatureProcessor

        started = time.perf_counter()
        processed = _build_or_load_stage3_shard_cache(
            enriched,
            source_sha256=source_sha256,
            source_path=source,
            tickers=requested_tickers,
            timeframe=resolved_timeframe,
            max_rows_per_ticker=max_rows_per_ticker,
            cache_dir=Path(shard_cache_dir) if shard_cache_dir else self.output_dir / "stage3_shard_cache",
            feature_processor=FeatureProcessor(),
            save=save,
        )
        timings_seconds["stage3_shard_cache_or_feature_split"] = _elapsed(
            started
        )
        if not processed:
            raise ValueError("Stage 3 output could not be split")

        started = time.perf_counter()
        batch_path = Path(batch_dir)
        manager = ColabManager(
            output_dir=batch_path,
            batch_name=batch_path.name,
        )
        preparation = manager.prepare_colab_batch(
            processed["features"],
            processed["targets"],
            BatchPreparationConfig(
                tickers=requested_tickers,
                timeframes=[resolved_timeframe],
                batch_name=batch_path.name,
                accumulate=False,
                check_feature_selection=False,
            ),
        )
        timings_seconds["colab_batch_preparation"] = _elapsed(started)
        metadata_path = Path(preparation["metadata_path"])
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        feature_path = Path(preparation["files"]["features"])
        target_path = Path(preparation["files"]["targets"])
        started = time.perf_counter()
        feature_audit = PipelineFeatureTimeframeAudit(
            self.output_dir / "feature_timeframe_audit"
        ).build(
            features_path=feature_path,
            tickers=requested_tickers,
            save=save,
        )
        timings_seconds["feature_timeframe_audit"] = _elapsed(started)
        started = time.perf_counter()
        target_audit = PipelineTargetReadinessAudit(
            self.output_dir / "target_readiness_audit"
        ).build(
            targets_path=target_path,
            features_path=feature_path,
            batch_metadata_path=metadata_path,
            tickers=requested_tickers,
            timeframe=resolved_timeframe,
            save=save,
        )
        timings_seconds["target_readiness_audit"] = _elapsed(started)
        audit_summary = feature_audit.get("summary") or {}
        target_summary = target_audit.get("summary") or {}
        ready = bool(
            audit_summary.get("can_use_for_stage4")
            and audit_summary.get("timezone_aware_ticker_count")
            == len(requested_tickers)
            and target_summary.get("can_use_for_stage4")
        )
        post_audit_blockers = []
        if not audit_summary.get("can_use_for_stage4"):
            post_audit_blockers.append(
                "regenerated_feature_timeframe_audit_failed"
            )
        if audit_summary.get("timezone_aware_ticker_count") != len(
            requested_tickers
        ):
            post_audit_blockers.append(
                "regenerated_feature_timezone_audit_failed"
            )
        if not target_summary.get("can_use_for_stage4"):
            post_audit_blockers.append(
                "regenerated_target_readiness_audit_failed"
            )
        payload = {
            **base_payload,
            "status": (
                "stage23_regeneration_review_ready"
                if ready
                else "stage23_regeneration_blocked_post_audit"
            ),
            "blocking_reasons": (
                []
                if ready
                else post_audit_blockers
            ),
            "stage2": {
                "row_count": len(stage2_frame),
                "quality": stage2_quality,
            },
            "stage3": {
                "enriched_row_count": len(enriched),
                "feature_column_count": len(
                    processed["features"].columns
                ),
                "target_column_count": len(
                    [
                        column
                        for column in processed["targets"].columns
                        if str(column).startswith("target_")
                    ]
                ),
                "target_table_column_count": len(
                    processed["targets"].columns
                ),
            },
            "batch_artifacts": {
                "batch_dir": str(batch_path),
                "metadata_path": str(metadata_path),
                "features_path": str(feature_path),
                "targets_path": str(target_path),
                "features_sha256": _sha256(feature_path),
                "targets_sha256": _sha256(target_path),
                "lineage": metadata.get("lineage") or {},
            },
            "stage3_cache": processed.get("cache", {}),
            "feature_timeframe_audit": {
                "status": feature_audit.get("status"),
                "summary": audit_summary,
                "saved_paths": feature_audit.get("saved_paths") or {},
            },
            "target_readiness_audit": {
                "status": target_audit.get("status"),
                "summary": target_summary,
                "saved_paths": target_audit.get(
                    "saved_paths"
                )
                or {},
            },
            "stage4_eligible_targets": list(
                target_summary.get("ready_target_names") or []
            ),
            "stage4_excluded_targets": list(
                target_summary.get("blocked_target_names") or []
            ),
            "can_use_for_stage4": ready,
            "can_use_for_stage5": False,
            "can_train": ready,
            "can_promote_model": False,
            "can_create_ticker_forecast": False,
            "can_trade": False,
        }
        payload["runtime_profile"]["total_elapsed_seconds"] = _elapsed(
            total_start
        )
        return self._save(payload, save=save)

    def _save(
        self,
        payload: dict[str, Any],
        *,
        save: bool,
    ) -> dict[str, Any]:
        payload = json_ready(payload)
        if not save:
            return payload
        paths = ReviewArtifactWriter(self.output_dir).write(
            payload=payload,
            markdown=render_pipeline_stage23_regeneration(payload),
            run_id=payload["run_id"],
        )
        return {**payload, "saved_paths": paths}


def _load_saved_stage1_market(
    path: Path,
) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    prefix = path.read_bytes()[:4]
    if prefix == b"PAR1":
        payload: Any = pd.read_parquet(path)
        source_format = "parquet"
    elif prefix[:1] == b"\x80":
        payload = pd.read_pickle(path)
        source_format = "legacy_pickle_with_parquet_extension"
    else:
        raise ValueError(
            "Saved Stage 1 source is neither parquet nor recognized "
            "legacy pickle"
        )
    if isinstance(payload, dict):
        payload = payload.get("market_data")
    if not isinstance(payload, pd.DataFrame):
        raise ValueError(
            "Saved Stage 1 source has no market_data DataFrame"
        )
    return payload, source_format


def _select_bounded_market_frame(
    frame: pd.DataFrame,
    *,
    tickers: list[str],
    timeframe: str,
    max_rows_per_ticker: int,
) -> pd.DataFrame:
    required = {
        "ticker",
        "datetime",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Saved Stage 1 market data missing columns: "
            + ", ".join(missing)
        )
    selected = frame.loc[
        frame["ticker"].astype(str).str.upper().isin(tickers)
        & frame["interval"]
        .map(normalize_timeframe)
        .eq(timeframe)
    ].copy()
    selected["ticker"] = selected["ticker"].astype(str).str.upper()
    selected = selected.sort_values(["ticker", "datetime"])
    selected = (
        selected.groupby("ticker", group_keys=False, sort=True)
        .tail(max_rows_per_ticker)
        .reset_index(drop=True)
    )
    return selected


def _source_checks(
    frame: pd.DataFrame,
    *,
    tickers: list[str],
    timeframe: str,
) -> list[dict[str, Any]]:
    checks = []
    present = sorted(set(frame.get("ticker", pd.Series(dtype=str))))
    checks.append(
        _check(
            "pass" if present == tickers else "fail",
            "ticker_coverage",
            f"present={present}, requested={tickers}",
        )
    )
    timezone = getattr(
        pd.to_datetime(frame["datetime"], errors="coerce").dt,
        "tz",
        None,
    )
    checks.append(
        _check(
            "pass" if timezone is not None else "fail",
            "datetime_timezone",
            f"timezone={timezone}",
        )
    )
    lineage = timeframe_lineage_report(
        frame[["ticker", "datetime", "interval"]],
        declared_timeframe=timeframe,
    )
    checks.append(
        _check(
            (
                "pass"
                if lineage.get("status")
                == "timeframe_cadence_verified"
                else "fail"
            ),
            "timeframe_cadence",
            (
                f"declared={timeframe}, "
                f"observed={lineage.get('observed_timeframe')}, "
                f"status={lineage.get('status')}"
            ),
            details=lineage,
        )
    )
    duplicate_count = int(
        frame.duplicated(
            ["ticker", "datetime", "interval"],
            keep=False,
        ).sum()
    )
    checks.append(
        _check(
            "pass" if duplicate_count == 0 else "fail",
            "unique_row_identity",
            f"duplicate_rows={duplicate_count}",
        )
    )
    cross_ticker_count = _cross_ticker_duplicate_count(frame)
    checks.append(
        _check(
            "pass" if cross_ticker_count == 0 else "fail",
            "cross_ticker_ohlcv_identity",
            f"duplicate_rows={cross_ticker_count}",
        )
    )
    numeric = frame[
        ["open", "high", "low", "close", "volume"]
    ].apply(pd.to_numeric, errors="coerce")
    finite = bool(np.isfinite(numeric.to_numpy()).all())
    positive = bool(
        (numeric[["open", "high", "low", "close"]] > 0)
        .all()
        .all()
    )
    checks.append(
        _check(
            "pass" if finite and positive else "fail",
            "finite_positive_ohlcv",
            f"finite={finite}, positive_prices={positive}",
        )
    )
    return checks


def _run_bounded_stage2(
    frame: pd.DataFrame,
    *,
    timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from src.pipeline.stages.processing.data_handler import (
        ProcessingDataHandler,
    )
    from src.processing.filters.price_filter import PriceFilter

    handler = ProcessingDataHandler(
        normalization_manager=None,
        data_filter=None,
    )
    cleaned = handler.clean_and_normalize_market_data(frame)
    grouped = handler.group_by_timeframes(cleaned)
    filtered, quality = PriceFilter({}).filter_price_data(grouped)
    payload = filtered.get(timeframe)
    if not isinstance(payload, dict):
        reasons = (quality.get(timeframe) or {}).get(
            "hard_failures", []
        )
        raise ValueError(
            "Bounded Stage 2 rejected selected market data: "
            + ", ".join(reasons)
        )
    result = payload.get("data")
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise ValueError("Bounded Stage 2 produced no market rows")
    return result, quality.get(timeframe) or {}


def _build_or_load_stage3_shard_cache(
    enriched: pd.DataFrame,
    *,
    source_sha256: str,
    source_path: Path,
    tickers: list[str],
    timeframe: str,
    max_rows_per_ticker: int,
    cache_dir: Path,
    feature_processor: Any,
    save: bool,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_results: list[dict[str, Any]] = []
    feature_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        shard_frame = enriched.loc[
            enriched["ticker"].astype(str).str.upper().eq(ticker)
        ].copy()
        if shard_frame.empty:
            raise ValueError(f"Stage 3 shard for {ticker} is empty")
        shard_key = _stage3_cache_key(
            source_sha256=source_sha256,
            ticker=ticker,
            timeframe=timeframe,
            max_rows_per_ticker=max_rows_per_ticker,
        )
        shard_dir = cache_dir / shard_key
        shard_result = _load_or_create_stage3_shard(
            shard_frame,
            shard_dir=shard_dir,
            ticker=ticker,
            timeframe=timeframe,
            source_sha256=source_sha256,
            source_path=source_path,
            max_rows_per_ticker=max_rows_per_ticker,
            feature_processor=feature_processor,
            save=save,
        )
        shard_results.append(shard_result["cache"])
        feature_frames.append(shard_result["features"])
        target_frames.append(shard_result["targets"])
    return {
        "features": pd.concat(feature_frames, ignore_index=True),
        "targets": pd.concat(target_frames, ignore_index=True),
        "cache": {
            "schema_version": STAGE3_CACHE_SCHEMA_VERSION,
            "cache_dir": str(cache_dir),
            "shard_count": len(shard_results),
            "shards": shard_results,
        },
    }


def _load_or_create_stage3_shard(
    shard_frame: pd.DataFrame,
    *,
    shard_dir: Path,
    ticker: str,
    timeframe: str,
    source_sha256: str,
    source_path: Path,
    max_rows_per_ticker: int,
    feature_processor: Any,
    save: bool,
) -> dict[str, Any]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = shard_dir / "shard_metadata.json"
    features_path = shard_dir / "features.parquet"
    targets_path = shard_dir / "targets.parquet"
    selected_sha256 = _frame_sha256(shard_frame)
    expected_key = _stage3_cache_key(
        source_sha256=source_sha256,
        ticker=ticker,
        timeframe=timeframe,
        max_rows_per_ticker=max_rows_per_ticker,
        selected_sha256=selected_sha256,
    )
    if metadata_path.is_file() and features_path.is_file() and targets_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") == expected_key:
            if (
                metadata.get("features_sha256") == _sha256(features_path)
                and metadata.get("targets_sha256") == _sha256(targets_path)
            ):
                return {
                    "features": pd.read_parquet(features_path),
                    "targets": pd.read_parquet(targets_path),
                    "cache": metadata,
                }
    processed = feature_processor.process_enriched_data(shard_frame)
    if not processed:
        raise ValueError(f"Stage 3 shard for {ticker} could not be split")
    features = processed["features"].copy()
    targets = processed["targets"].copy()
    metadata = {
        "schema_version": STAGE3_CACHE_SCHEMA_VERSION,
        "cache_key": expected_key,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "selected_sha256": selected_sha256,
        "ticker": ticker,
        "timeframe": timeframe,
        "max_rows_per_ticker": max_rows_per_ticker,
        "source_row_count": len(shard_frame),
        "feature_row_count": len(features),
        "target_row_count": len(targets),
    }
    if save:
        features.to_parquet(features_path, index=False)
        targets.to_parquet(targets_path, index=False)
        metadata["features_sha256"] = _sha256(features_path)
        metadata["targets_sha256"] = _sha256(targets_path)
        metadata_path.write_text(
            json.dumps(json_ready(metadata), indent=2),
            encoding="utf-8",
        )
    return {
        "features": features,
        "targets": targets,
        "cache": metadata,
    }


def _stage3_cache_key(
    *,
    source_sha256: str,
    ticker: str,
    timeframe: str,
    max_rows_per_ticker: int,
    selected_sha256: str | None = None,
) -> str:
    payload = {
        "schema_version": STAGE3_CACHE_SCHEMA_VERSION,
        "stage23_schema_version": STAGE23_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "ticker": ticker,
        "timeframe": timeframe,
        "max_rows_per_ticker": max_rows_per_ticker,
        "selected_sha256": selected_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    for column in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = pd.to_datetime(canonical[column], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%f%z"
            )
    blob = json.dumps(
        canonical.to_dict(orient="records"),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cross_ticker_duplicate_count(frame: pd.DataFrame) -> int:
    columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    duplicates = frame.loc[frame.duplicated(columns, keep=False)]
    if duplicates.empty:
        return 0
    cross = duplicates.groupby(
        columns,
        dropna=False,
    )["ticker"].transform("nunique") > 1
    return int(cross.sum())


def _check(
    status: str,
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "status": status,
        "code": code,
        "message": message,
    }
    if details is not None:
        result["details"] = details
    return result


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "saved_source_only": True,
        "live_collection_performed": False,
        "stage4_run_performed": False,
        "stage5_run_performed": False,
        "model_promotion_performed": False,
        "learning_write_performed": False,
        "decision_influence": False,
        "can_trade": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    return (
        f"{prefix}_"
        f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def render_pipeline_stage23_regeneration(
    payload: dict[str, Any],
) -> str:
    scope = payload.get("scope") or {}
    lines = [
        "# Bounded Pipeline Stage 2/3 Regeneration",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Tickers: `{', '.join(scope.get('tickers') or [])}`",
        f"- Timeframe: `{scope.get('timeframe')}`",
        f"- Selected source rows: {scope.get('selected_row_count')}",
        f"- Can use for Stage 4: `{payload.get('can_use_for_stage4', False)}`",
        "- Can trade: `False`",
        "",
        "## Source Checks",
        "",
    ]
    lines.extend(
        (
            f"- {item.get('status', '').upper()}: "
            f"`{item.get('code')}` - {item.get('message')}"
        )
        for item in payload.get("source_checks", [])
    )
    return "\n".join(lines).strip() + "\n"


__all__ = [
    "PipelineStage23Regeneration",
    "render_pipeline_stage23_regeneration",
]
