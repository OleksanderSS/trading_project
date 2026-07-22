from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_stage23_regeneration import (
    _load_saved_stage1_market,
    _select_bounded_market_frame,
    _source_checks,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from dean_os.world_model.world_model_pipeline_context import (
    DEFAULT_WORLD_MODEL_TIMEFRAMES,
    WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT,
    WorldModelPipelineContextDiscovery,
)
from src.pipeline.timeframe_lineage import normalize_timeframe

PIPELINE_TIMEFRAME_LANE_READINESS_CONTRACT = (
    "dean_pipeline_timeframe_lane_readiness_v1"
)


class PipelineTimeframeLaneReadinessPlan:
    """Review-only plan for exact pipeline timeframe-lane coverage.

    This component checks whether the saved Stage 1 source contains requested
    timeframe/ticker rows and compares that source coverage with existing
    Stage23/Stage4/world-model context artifacts. It does not run Stage23,
    Stage4, Stage5, training, tuning, learning, or trading.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/pipeline_timeframe_lane_readiness_current",
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        source_path: str | Path,
        tickers: list[str],
        timeframes: list[str] | None = None,
        max_rows_per_ticker: int = 200,
        pipeline_context_json: str | Path | dict[str, Any] | None = None,
        pipeline_context_base: str | Path = "reports/dean_os",
        save: bool = True,
    ) -> dict[str, Any]:
        source = Path(source_path)
        requested_tickers = _normalize_tickers(tickers)
        requested_timeframes = _normalize_timeframes(
            timeframes or list(DEFAULT_WORLD_MODEL_TIMEFRAMES)
        )
        source_sha256 = _file_sha256(source)
        market_frame, source_format = _load_saved_stage1_market(source)
        source_lanes = _source_lanes(
            market_frame,
            tickers=requested_tickers,
            timeframes=requested_timeframes,
        )
        source_validations = _source_validations(
            market_frame,
            tickers=requested_tickers,
            timeframes=requested_timeframes,
            max_rows_per_ticker=max_rows_per_ticker,
        )
        context = _load_or_discover_context(
            pipeline_context_json,
            tickers=requested_tickers,
            timeframes=requested_timeframes,
            base_path=pipeline_context_base,
        )
        context_lanes = {
            str(lane.get("timeframe")): lane
            for lane in context.get("timeframe_lanes", []) or []
            if isinstance(lane, dict)
        }
        lanes = [
            _lane_plan(
                timeframe,
                source_lane=source_lanes.get(timeframe, {}),
                source_validation=source_validations.get(timeframe, {}),
                context_lane=context_lanes.get(timeframe, {}),
                source_path=source,
                source_sha256=source_sha256,
                tickers=requested_tickers,
                max_rows_per_ticker=max_rows_per_ticker,
            )
            for timeframe in requested_timeframes
        ]
        summary = _summary(lanes)
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "pipeline_timeframe_lane_readiness",
            "contract": PIPELINE_TIMEFRAME_LANE_READINESS_CONTRACT,
            "inputs": {
                "source_path": str(source),
                "source_sha256": source_sha256,
                "source_format": source_format,
                "tickers": requested_tickers,
                "timeframes": requested_timeframes,
                "max_rows_per_ticker": max_rows_per_ticker,
                "pipeline_context_contract": context.get("contract")
                or context.get("schema_version"),
                "pipeline_context_run_id": context.get("run_id"),
                "pipeline_context_status": context.get("summary", {}).get("status")
                or context.get("pipeline_context", {}).get("status"),
            },
            "summary": summary,
            "timeframe_lanes": lanes,
            "operator_next_steps": _operator_next_steps(summary, lanes),
            "safety": _safety(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_timeframe_lane_readiness_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_timeframe_lane_readiness_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Pipeline Timeframe Lane Readiness",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Source-available lanes: {summary.get('source_available_lane_count')}",
        f"- Source-valid lanes: {summary.get('source_valid_lane_count')}",
        f"- Source-invalid lanes: {summary.get('source_invalid_lane_count')}",
        f"- Exact-context lanes: {summary.get('exact_context_lane_count')}",
        f"- Missing artifact lanes: {summary.get('artifact_missing_lane_count')}",
        f"- Ready lanes missing Stage3 cache: {summary.get('stage3_cache_missing_ready_lane_count')}",
        f"- Batch artifact lanes: {summary.get('batch_artifact_lane_count')}",
        f"- Can condition world model: {summary.get('can_condition_world_model')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Lanes",
        "",
    ]
    for lane in payload.get("timeframe_lanes", []):
        lines.extend(
            [
                f"- `{lane.get('timeframe')}`: `{lane.get('lane_status')}`",
                f"  - source rows: {lane.get('source', {}).get('total_row_count')}",
                f"  - source valid: `{(lane.get('source', {}).get('validation') or {}).get('valid')}`",
                f"  - context status: `{lane.get('context', {}).get('status')}`",
                f"  - stage3 cache: `{lane.get('context', {}).get('stage3_cache_status')}`",
                f"  - batch artifact: `{lane.get('batch_artifact_status')}`",
                f"  - next action: {lane.get('recommended_next_action')}",
            ]
        )
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _load_or_discover_context(
    pipeline_context_json: str | Path | dict[str, Any] | None,
    *,
    tickers: list[str],
    timeframes: list[str],
    base_path: str | Path,
) -> dict[str, Any]:
    if isinstance(pipeline_context_json, dict):
        return dict(pipeline_context_json)
    if pipeline_context_json is not None:
        path = Path(pipeline_context_json)
        return json.loads(path.read_text(encoding="utf-8"))
    return WorldModelPipelineContextDiscovery(base_path=base_path).build(
        tickers=tickers,
        timeframes=timeframes,
        save=False,
    )


def _source_lanes(
    frame: pd.DataFrame,
    *,
    tickers: list[str],
    timeframes: list[str],
) -> dict[str, dict[str, Any]]:
    required = {"ticker", "datetime", "interval"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("market source missing columns: " + ", ".join(missing))
    working = frame.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    working["resolved_timeframe"] = working["interval"].map(normalize_timeframe)
    working = working.loc[working["ticker"].isin(tickers)].copy()
    result: dict[str, dict[str, Any]] = {}
    for timeframe in timeframes:
        lane = working.loc[working["resolved_timeframe"].eq(timeframe)].copy()
        by_ticker: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            ticker_frame = lane.loc[lane["ticker"].eq(ticker)]
            by_ticker[ticker] = {
                "row_count": int(len(ticker_frame)),
                "first_datetime": _dt_min(ticker_frame),
                "last_datetime": _dt_max(ticker_frame),
            }
        total = int(len(lane))
        result[timeframe] = {
            "timeframe": timeframe,
            "available": total > 0,
            "total_row_count": total,
            "ticker_row_counts": by_ticker,
            "all_requested_tickers_available": all(
                item["row_count"] > 0 for item in by_ticker.values()
            ),
        }
    return result


def _source_validations(
    frame: pd.DataFrame,
    *,
    tickers: list[str],
    timeframes: list[str],
    max_rows_per_ticker: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for timeframe in timeframes:
        try:
            selected = _select_bounded_market_frame(
                frame,
                tickers=tickers,
                timeframe=timeframe,
                max_rows_per_ticker=max_rows_per_ticker,
            )
            if selected.empty:
                result[timeframe] = {
                    "valid": False,
                    "selected_row_count": 0,
                    "blocking_reasons": ["source_rows_missing"],
                    "source_checks": [],
                }
                continue
            checks = _source_checks(
                selected,
                tickers=tickers,
                timeframe=timeframe,
            )
            blockers = [
                item["code"]
                for item in checks
                if item.get("status") == "fail"
            ]
            result[timeframe] = {
                "valid": not blockers,
                "selected_row_count": len(selected),
                "blocking_reasons": blockers,
                "source_checks": checks,
            }
        except Exception as exc:
            result[timeframe] = {
                "valid": False,
                "selected_row_count": 0,
                "blocking_reasons": ["source_validation_error"],
                "source_checks": [
                    {
                        "status": "fail",
                        "code": "source_validation_error",
                        "message": str(exc),
                    }
                ],
            }
    return result


def _lane_plan(
    timeframe: str,
    *,
    source_lane: dict[str, Any],
    source_validation: dict[str, Any],
    context_lane: dict[str, Any],
    source_path: Path,
    source_sha256: str,
    tickers: list[str],
    max_rows_per_ticker: int,
) -> dict[str, Any]:
    stage23 = (context_lane.get("artifacts") or {}).get("stage23_regeneration") or {}
    stage23_path = stage23.get("path")
    stage23_payload = _load_optional_json(stage23_path)
    compatibility = _stage23_source_compatibility(
        stage23_payload,
        source_sha256=source_sha256,
        tickers=tickers,
        timeframe=timeframe,
    )
    candidate_available = bool(stage23.get("available") and stage23_payload)
    stage23_compatible = bool(
        candidate_available and compatibility["compatible"]
    )
    batch_status = (
        _batch_artifact_status(stage23_payload)
        if stage23_compatible
        else {
            "status": (
                "stage23_artifact_ignored_incompatible_source"
                if candidate_available
                else "stage23_artifact_missing"
            )
        }
    )
    source_available = bool(source_lane.get("available"))
    source_valid = bool(source_validation.get("valid"))
    context_status = str(context_lane.get("status") or "pipeline_lane_missing")
    cache_status = str(
        context_lane.get("stage3_cache_status") or "stage23_artifact_missing"
    )
    if candidate_available and not stage23_compatible:
        context_status = "pipeline_lane_incompatible_source_artifact"
        cache_status = "stage23_artifact_incompatible_source"
    if not source_available:
        lane_status = "source_rows_missing"
        next_action = "Backfill source rows or explicitly mark this lane absent."
    elif not source_valid:
        lane_status = "source_rows_invalid"
        next_action = (
            "Repair source timeframe cadence/lineage before running Stage23 "
            "for this lane."
        )
    elif context_status == "pipeline_lane_exact_context_available" and cache_status == "stage3_cache_materialized_in_stage23_artifact":
        lane_status = "exact_lane_ready_with_stage3_cache"
        next_action = "Use as world-model context; no Stage23 action needed."
    elif context_status == "pipeline_lane_exact_context_available":
        lane_status = "exact_lane_ready_but_stage3_cache_missing"
        next_action = (
            "Materialize true Stage3 shard-cache with a scheduled/optimized "
            "Stage23 run; do not treat batch artifacts as reusable shard cache."
        )
    elif stage23_compatible:
        lane_status = "stage23_available_but_exact_context_missing"
        next_action = "Run or repair Stage4 exact-context review for this lane."
    else:
        lane_status = "source_available_but_stage23_artifact_missing"
        next_action = (
            "Run bounded Stage23 for this lane, preferably in a scheduled job "
            "or after profiling Stage3 runtime."
        )
    return {
        "timeframe": timeframe,
        "lane_status": lane_status,
        "source": {
            **source_lane,
            "validation": source_validation,
        },
        "context": {
            "status": context_status,
            "stage23_ready": bool(
                stage23_compatible and context_lane.get("stage23_ready")
            ),
            "stage3_cache_status": cache_status,
            "stage3_shard_count": int(
                context_lane.get("stage3_shard_count") or 0
            )
            if stage23_compatible
            else 0,
            "stage4_exact_context_count": int(
                context_lane.get("stage4_exact_context_count") or 0
            )
            if stage23_compatible
            else 0,
            "warnings": (
                list(context_lane.get("warnings") or [])
                if stage23_compatible
                else list(context_lane.get("warnings") or [])
                + compatibility.get("blocking_reasons", [])
            ),
        },
        "stage23_artifact": {
            **stage23,
            "effective_available": stage23_compatible,
            "source_compatibility": compatibility,
        },
        "batch_artifact_status": batch_status["status"],
        "batch_artifacts": batch_status,
        "recommended_next_action": next_action,
        "suggested_stage23_command": _stage23_command(
            source_path=source_path,
            tickers=tickers,
            timeframe=timeframe,
            max_rows_per_ticker=max_rows_per_ticker,
        )
        if source_available and source_valid
        else None,
    }


def _stage23_source_compatibility(
    stage23_payload: dict[str, Any] | None,
    *,
    source_sha256: str,
    tickers: list[str],
    timeframe: str,
) -> dict[str, Any]:
    if not stage23_payload:
        return {
            "compatible": False,
            "status": "stage23_artifact_missing",
            "blocking_reasons": ["stage23_artifact_missing"],
        }

    reasons: list[str] = []
    source_artifact = stage23_payload.get("source_artifact") or {}
    artifact_sha256 = str(source_artifact.get("sha256") or "")
    if not artifact_sha256:
        reasons.append("stage23_source_sha256_missing")
    elif artifact_sha256 != source_sha256:
        reasons.append("stage23_source_sha256_mismatch")

    scope = stage23_payload.get("scope") or {}
    artifact_tickers = {
        str(value).strip().upper()
        for value in scope.get("tickers", [])
        if str(value).strip()
    }
    missing_tickers = sorted(set(tickers) - artifact_tickers)
    if missing_tickers:
        reasons.append("stage23_ticker_scope_mismatch")

    artifact_timeframe = normalize_timeframe(scope.get("timeframe"))
    if artifact_timeframe != timeframe:
        reasons.append("stage23_timeframe_scope_mismatch")

    return {
        "compatible": not reasons,
        "status": (
            "stage23_source_compatible"
            if not reasons
            else "stage23_source_incompatible"
        ),
        "blocking_reasons": reasons,
        "expected_source_sha256": source_sha256,
        "artifact_source_sha256": artifact_sha256 or None,
        "requested_tickers": tickers,
        "artifact_tickers": sorted(artifact_tickers),
        "missing_tickers": missing_tickers,
        "requested_timeframe": timeframe,
        "artifact_timeframe": artifact_timeframe,
    }


def _batch_artifact_status(stage23_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not stage23_payload:
        return {"status": "stage23_artifact_missing"}
    batch = stage23_payload.get("batch_artifacts") or {}
    features = Path(str(batch.get("features_path") or ""))
    targets = Path(str(batch.get("targets_path") or ""))
    metadata = Path(str(batch.get("metadata_path") or ""))
    checks = {
        "features_exists": features.is_file(),
        "targets_exists": targets.is_file(),
        "metadata_exists": metadata.is_file(),
        "features_sha256_matches": _sha_matches(features, batch.get("features_sha256")),
        "targets_sha256_matches": _sha_matches(targets, batch.get("targets_sha256")),
    }
    ready = all(checks.values())
    return {
        "status": (
            "batch_artifacts_verified_not_reusable_stage3_cache"
            if ready
            else "batch_artifacts_missing_or_unverified"
        ),
        **checks,
        "features_path": str(features) if str(features) != "." else None,
        "targets_path": str(targets) if str(targets) != "." else None,
        "metadata_path": str(metadata) if str(metadata) != "." else None,
        "note": (
            "Verified batch artifacts can support review lineage, but they are "
            "not the reusable Stage3 shard-cache expected by current Stage23."
        ),
    }


def _summary(lanes: list[dict[str, Any]]) -> dict[str, Any]:
    source_available = sum(lane["source"].get("available") is True for lane in lanes)
    source_valid = sum(
        (lane["source"].get("validation") or {}).get("valid") is True
        for lane in lanes
    )
    exact = sum(
        lane["context"].get("status") == "pipeline_lane_exact_context_available"
        for lane in lanes
    )
    missing_artifact = sum(
        lane["lane_status"] == "source_available_but_stage23_artifact_missing"
        for lane in lanes
    )
    invalid_source = sum(lane["lane_status"] == "source_rows_invalid" for lane in lanes)
    cache_missing = sum(
        lane["lane_status"] == "exact_lane_ready_but_stage3_cache_missing"
        for lane in lanes
    )
    batch = sum(
        lane["batch_artifact_status"]
        == "batch_artifacts_verified_not_reusable_stage3_cache"
        for lane in lanes
    )
    status = (
        "pipeline_timeframe_lanes_ready"
        if exact == len(lanes) and cache_missing == 0 and missing_artifact == 0
        else "pipeline_timeframe_lanes_ready_with_gaps"
        if exact
        else "pipeline_timeframe_lanes_source_available_artifacts_missing"
        if source_available
        else "pipeline_timeframe_lanes_source_missing"
    )
    return {
        "status": status,
        "requested_lane_count": len(lanes),
        "source_available_lane_count": source_available,
        "source_valid_lane_count": source_valid,
        "source_invalid_lane_count": invalid_source,
        "exact_context_lane_count": exact,
        "artifact_missing_lane_count": missing_artifact,
        "stage3_cache_missing_ready_lane_count": cache_missing,
        "batch_artifact_lane_count": batch,
        "can_condition_world_model": exact > 0,
        "can_register_replay_tasks": False,
        "can_write_learning_memory": False,
        "can_trade": False,
    }


def _operator_next_steps(summary: dict[str, Any], lanes: list[dict[str, Any]]) -> list[str]:
    steps: list[str] = []
    if summary.get("stage3_cache_missing_ready_lane_count"):
        missing = [
            lane["timeframe"]
            for lane in lanes
            if lane["lane_status"] == "exact_lane_ready_but_stage3_cache_missing"
        ]
        steps.append(
            "Materialize true Stage3 shard-cache for ready lanes: "
            + ", ".join(missing)
            + "."
        )
    if summary.get("artifact_missing_lane_count"):
        missing = [
            lane["timeframe"]
            for lane in lanes
            if lane["lane_status"] == "source_available_but_stage23_artifact_missing"
        ]
        steps.append(
            "Create bounded Stage23/Stage4 artifacts for source-backed missing lanes: "
            + ", ".join(missing)
            + "."
        )
    if summary.get("source_invalid_lane_count"):
        invalid = [
            lane["timeframe"]
            for lane in lanes
            if lane["lane_status"] == "source_rows_invalid"
        ]
        steps.append(
            "Repair source cadence/lineage before Stage23 for lanes: "
            + ", ".join(invalid)
            + "."
        )
    if not steps:
        steps.append("No missing lane actions were detected.")
    steps.append("Do not treat this readiness plan as a prediction or trading signal.")
    return steps


def _stage23_command(
    *,
    source_path: Path,
    tickers: list[str],
    timeframe: str,
    max_rows_per_ticker: int,
) -> str:
    ticker_args = " ".join(f"--ticker {ticker}" for ticker in tickers)
    safe_timeframe = timeframe.replace("/", "_")
    return (
        f"python run_agent_pipeline_stage23_regeneration.py {source_path} "
        f"{ticker_args} --timeframe {timeframe} "
        f"--max-rows-per-ticker {max_rows_per_ticker} "
        f"--batch-dir data\\colab\\regenerated\\lane_{safe_timeframe}_stage23_review "
        f"--output-dir reports\\dean_os\\pipeline_stage23_regeneration_lane_{safe_timeframe}_review "
        f"--shard-cache-dir data\\colab\\stage3_shard_cache\\dean_review"
    )


def _normalize_tickers(tickers: list[str]) -> list[str]:
    result = sorted({str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()})
    if not result:
        raise ValueError("At least one ticker is required")
    return result


def _normalize_timeframes(timeframes: list[str]) -> list[str]:
    result: list[str] = []
    for raw in timeframes:
        normalized = normalize_timeframe(str(raw))
        if not normalized:
            raise ValueError(f"Unsupported timeframe: {raw}")
        if normalized not in result:
            result.append(normalized)
    return result


def _load_optional_json(path: Any) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha_matches(path: Path, expected: Any) -> bool:
    if not path.is_file() or not expected:
        return False
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == str(expected)


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dt_min(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    value = pd.to_datetime(frame["datetime"], errors="coerce").min()
    return value.isoformat() if pd.notna(value) else None


def _dt_max(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    value = pd.to_datetime(frame["datetime"], errors="coerce").max()
    return value.isoformat() if pd.notna(value) else None


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "stage23_run_performed": False,
        "stage4_run_performed": False,
        "stage5_run_performed": False,
        "training_performed": False,
        "production_config_write_performed": False,
        "learning_memory_write_performed": False,
        "broker_access_performed": False,
        "live_execution_performed": False,
    }


def _run_id() -> str:
    return f"pipeline_timeframe_lane_readiness_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "PIPELINE_TIMEFRAME_LANE_READINESS_CONTRACT",
    "PipelineTimeframeLaneReadinessPlan",
    "render_pipeline_timeframe_lane_readiness_markdown",
]
