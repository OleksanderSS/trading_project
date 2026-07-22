from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT = "dean_world_model_pipeline_context_v1"
DEFAULT_WORLD_MODEL_TIMEFRAMES = ("15m", "60m", "1d")


class WorldModelPipelineContextDiscovery:
    """Build a review-only pipeline context bundle for world-model packets.

    This reads existing DEAN-OS/pipeline review artifacts and summarizes exact
    timeframe-lane coverage. It does not regenerate pipeline data, run Stage 4,
    run Stage 5, register replay tasks, write learning memory, or trade.
    """

    def __init__(
        self,
        *,
        base_path: str | Path = "reports/dean_os",
        output_dir: str | Path = "reports/dean_os/world_model_pipeline_context_current",
    ):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        tickers: list[str] | None = None,
        timeframes: list[str] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        requested_tickers = _normalize_tickers(tickers)
        requested_timeframes = _normalize_timeframes(
            timeframes or list(DEFAULT_WORLD_MODEL_TIMEFRAMES)
        )
        artifacts = _discover_artifacts(self.base_path)
        lanes = [
            _timeframe_lane(
                timeframe,
                artifacts=artifacts,
                requested_tickers=requested_tickers,
            )
            for timeframe in requested_timeframes
        ]
        metric_readiness = _latest_payload(
            artifacts["metric_input_readiness"]
        )
        stage5_review = _latest_payload(artifacts["stage5_prediction_review"])
        stage5_review_fragment = _stage5_review_fragment(stage5_review)
        stage7_review = _latest_payload(artifacts["stage7_regime_review"])
        metrics = _bundle_metrics(
            lanes,
            metric_readiness=metric_readiness,
            stage5_review=stage5_review,
        )
        context_tags = _bundle_context_tags(
            lanes,
            metric_readiness=metric_readiness,
            stage5_review=stage5_review,
        )
        status = _bundle_status(lanes)
        pipeline_context = {
            "schema_version": WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT,
            "status": status,
            "timeframes": requested_timeframes,
            "tickers": requested_tickers,
            "context_tags": context_tags,
            "metrics": metrics,
            "timeframe_lane_status": {
                lane["timeframe"]: lane["status"] for lane in lanes
            },
            "review_note": (
                "This is pipeline context for world-model interpretation only; "
                "it is not a prediction, evaluation clearance, recommendation, "
                "or trading signal."
            ),
        }
        indicator_state_grid = {
            "schema_version": WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT,
            "status": (
                "indicator_state_grid_ready_with_gaps"
                if metrics["pipeline_lane_available_count"] > 0
                else "indicator_state_grid_missing"
            ),
            "metrics": metrics,
            "context_tags": context_tags,
            "timeframe_lanes": lanes,
        }
        payload = {
            "run_id": _run_id("world_model_pipeline_context"),
            "created_at": utc_now_iso(),
            "mode": "world_model_pipeline_context_discovery",
            "contract": WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT,
            "base_path": str(self.base_path),
            "requested": {
                "tickers": requested_tickers,
                "timeframes": requested_timeframes,
            },
            "summary": {
                "status": status,
                "requested_timeframe_count": len(requested_timeframes),
                "available_lane_count": metrics["pipeline_lane_available_count"],
                "exact_context_lane_count": metrics[
                    "pipeline_lane_exact_context_count"
                ],
                "missing_lane_count": metrics["pipeline_lane_missing_count"],
                "stage3_shard_count": metrics["stage3_shard_count"],
                "stage3_cache_materialized_lane_count": metrics[
                    "stage3_cache_materialized_lane_count"
                ],
                "stage3_cache_missing_ready_lane_count": metrics[
                    "stage3_cache_missing_ready_lane_count"
                ],
                "stage3_cache_status_counts": metrics[
                    "stage3_cache_status_counts"
                ],
                "stage4_exact_context_count": metrics[
                    "stage4_exact_context_count"
                ],
                "stage5_context_count": metrics["stage5_context_count"],
                "stage5_complete_context_count": metrics[
                    "stage5_complete_context_count"
                ],
                "can_condition_world_model": (
                    metrics["pipeline_lane_available_count"] > 0
                ),
                "can_register_replay_tasks": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "pipeline_context": pipeline_context,
            "indicator_state_grid": indicator_state_grid,
            "timeframe_lanes": lanes,
            "artifact_inventory": _artifact_inventory(artifacts),
            "pipeline_metric_input_readiness": metric_readiness,
            "stage5_prediction_review": stage5_review_fragment,
            "stage7_regime_review": stage7_review,
            "operator_next_steps": _operator_next_steps(status, lanes),
            "safety": _safety(),
        }
        if save:
            paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_world_model_pipeline_context_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = paths
        return json_ready(payload)


def metadata_from_pipeline_context_bundle(
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Return MarketContext.metadata additions for a discovered bundle."""

    metadata: dict[str, Any] = {}
    pipeline_context = bundle.get("pipeline_context")
    if isinstance(pipeline_context, dict):
        metadata["pipeline_context"] = pipeline_context
    indicator_state_grid = bundle.get("indicator_state_grid")
    if isinstance(indicator_state_grid, dict):
        metadata["indicator_state_grid"] = indicator_state_grid
    stage5 = bundle.get("stage5_prediction_review")
    if isinstance(stage5, dict) and stage5:
        metadata["stage5_prediction_review"] = stage5
    stage7 = bundle.get("stage7_regime_review")
    if isinstance(stage7, dict) and stage7:
        metadata["stage7_regime_review"] = stage7
    metric_readiness = bundle.get("pipeline_metric_input_readiness")
    if isinstance(metric_readiness, dict) and metric_readiness:
        metadata["pipeline_metric_input_readiness"] = metric_readiness
    lanes = bundle.get("timeframe_lanes")
    if isinstance(lanes, list):
        metadata["pipeline_timeframe_lanes"] = lanes
    metadata["world_model_pipeline_context_bundle"] = {
        "run_id": bundle.get("run_id"),
        "contract": bundle.get("contract"),
        "status": bundle.get("summary", {}).get("status"),
        "saved_paths": bundle.get("saved_paths", {}),
    }
    return metadata


def render_world_model_pipeline_context_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS World Model Pipeline Context",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Available lanes: {summary.get('available_lane_count')}",
        f"- Exact context lanes: {summary.get('exact_context_lane_count')}",
        f"- Missing lanes: {summary.get('missing_lane_count')}",
        f"- Stage 3 shards: {summary.get('stage3_shard_count')}",
        f"- Stage 3 cache materialized lanes: {summary.get('stage3_cache_materialized_lane_count')}",
        f"- Stage 3 cache missing on ready lanes: {summary.get('stage3_cache_missing_ready_lane_count')}",
        f"- Stage 5 contexts: {summary.get('stage5_context_count')}",
        f"- Can condition world model: {summary.get('can_condition_world_model')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Timeframe Lanes",
        "",
    ]
    for lane in payload.get("timeframe_lanes", []):
        lines.extend(
            [
                f"- `{lane.get('timeframe')}`: `{lane.get('status')}`",
                f"  - exact_context_ready: {lane.get('exact_context_ready')}",
                f"  - stage3_cache_status: `{lane.get('stage3_cache_status')}`",
                f"  - stage3_shards: {lane.get('stage3_shard_count')}",
                f"  - stage4_reviews: {lane.get('stage4_exact_context_count')}",
                f"  - stage5_contexts: {lane.get('stage5_context_count')}",
            ]
        )
    if not payload.get("timeframe_lanes"):
        lines.append("- none")
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _discover_artifacts(base_path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "stage23_regeneration": _load_pattern(
            base_path,
            "pipeline_stage23_regeneration*/latest.json",
            mode="pipeline_stage23_regeneration",
        ),
        "stage4_exact_context": _load_pattern(
            base_path,
            "pipeline_stage4_exact_context_review*/latest.json",
            mode="pipeline_stage4_exact_context_review",
        ),
        "stage5_prediction_review": _load_pattern(
            base_path,
            "pipeline_prediction_review_packet*/latest.json",
            schema_version="dean_stage5_prediction_review_v1",
        ),
        "stage7_regime_review": _load_pattern(
            base_path,
            "*stage7*regime*/latest.json",
            schema_version="dean_stage7_regime_review_v1",
        ),
        "metric_input_readiness": _load_pattern(
            base_path,
            "pipeline_metric_input_readiness_gate*/latest.json",
            mode="pipeline_metric_input_readiness_gate",
        ),
    }


def _load_pattern(
    base_path: Path,
    pattern: str,
    *,
    mode: str | None = None,
    schema_version: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not base_path.exists():
        return results
    for path in sorted(base_path.glob(pattern)):
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if mode and payload.get("mode") != mode:
            continue
        if schema_version and payload.get("schema_version") != schema_version:
            continue
        results.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "last_write_time": path.stat().st_mtime,
                "payload": payload,
            }
        )
    results.sort(key=lambda item: item["last_write_time"], reverse=True)
    return results


def _timeframe_lane(
    timeframe: str,
    *,
    artifacts: dict[str, list[dict[str, Any]]],
    requested_tickers: list[str],
) -> dict[str, Any]:
    stage23 = [
        item
        for item in artifacts["stage23_regeneration"]
        if _payload_timeframe(item["payload"]) == timeframe
    ]
    stage4_candidates = [
        item
        for item in artifacts["stage4_exact_context"]
        if _payload_timeframe(item["payload"]) == timeframe
        and _ticker_matches(item["payload"], requested_tickers)
    ]
    stage5_contexts = _stage5_contexts(
        artifacts["stage5_prediction_review"],
        timeframe=timeframe,
        requested_tickers=requested_tickers,
    )
    best_stage23 = stage23[0] if stage23 else None
    stage4 = [
        item
        for item in stage4_candidates
        if best_stage23
        and _stage4_matches_stage23(
            item["payload"],
            best_stage23["payload"],
        )
    ]
    incompatible_stage4_count = len(stage4_candidates) - len(stage4)
    stage3_cache = (
        best_stage23["payload"].get("stage3_cache", {}) if best_stage23 else {}
    )
    stage3_shard_count = int(stage3_cache.get("shard_count") or 0)
    stage3_cache_status = _stage3_cache_status(
        best_stage23,
        stage3_cache=stage3_cache,
    )
    exact_context_ready = any(_stage4_exact_ready(item["payload"]) for item in stage4)
    stage23_ready = bool(
        best_stage23
        and best_stage23["payload"].get("status")
        == "stage23_regeneration_review_ready"
    )
    if exact_context_ready:
        status = "pipeline_lane_exact_context_available"
    elif stage23_ready:
        status = "pipeline_lane_stage23_context_available"
    elif stage5_contexts:
        status = "pipeline_lane_stage5_supporting_context_available"
    else:
        status = "pipeline_lane_missing"
    warnings = _lane_warnings(
        timeframe,
        exact_context_ready=exact_context_ready,
        stage23_ready=stage23_ready,
        stage3_cache_status=stage3_cache_status,
        stage5_context_count=len(stage5_contexts),
    )
    if incompatible_stage4_count:
        warnings.append(
            f"stage4_parent_hash_mismatch_count={incompatible_stage4_count}"
        )
    return {
        "timeframe": timeframe,
        "status": status,
        "exact_context_ready": exact_context_ready,
        "stage23_ready": stage23_ready,
        "stage3_cache_status": stage3_cache_status,
        "stage3_shard_count": stage3_shard_count,
        "stage4_exact_context_count": len(stage4),
        "stage4_incompatible_context_count": incompatible_stage4_count,
        "stage5_context_count": len(stage5_contexts),
        "stage5_complete_context_count": sum(
            1
            for item in stage5_contexts
            if item.get("lineage_status") == "complete"
            and not item.get("review_issues")
        ),
        "tickers": _lane_tickers(stage23, stage4, stage5_contexts),
        "artifacts": {
            "stage23_regeneration": _artifact_ref(best_stage23),
            "stage4_exact_context": [_artifact_ref(item) for item in stage4],
            "stage5_context_keys": [
                item.get("context_key") for item in stage5_contexts
            ],
        },
        "warnings": warnings,
    }


def _stage5_contexts(
    artifacts: list[dict[str, Any]],
    *,
    timeframe: str,
    requested_tickers: list[str],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for item in artifacts:
        payload = item["payload"]
        for context in payload.get("contexts", []) or []:
            if not isinstance(context, dict):
                continue
            if _normalize_timeframe(context.get("timeframe")) != timeframe:
                continue
            if requested_tickers and _upper(context.get("ticker")) not in requested_tickers:
                continue
            contexts.append(context)
    return contexts


def _bundle_metrics(
    lanes: list[dict[str, Any]],
    *,
    metric_readiness: dict[str, Any],
    stage5_review: dict[str, Any],
) -> dict[str, Any]:
    available = [
        lane for lane in lanes if lane["status"] != "pipeline_lane_missing"
    ]
    exact = [lane for lane in lanes if lane.get("exact_context_ready")]
    readiness_summary = metric_readiness.get("summary", {})
    axis_counts = readiness_summary.get("axis_status_counts", {})
    cache_status_counts: dict[str, int] = {}
    for lane in lanes:
        status = str(lane.get("stage3_cache_status") or "unknown")
        cache_status_counts[status] = cache_status_counts.get(status, 0) + 1
    return {
        "pipeline_lane_available_count": len(available),
        "pipeline_lane_exact_context_count": len(exact),
        "pipeline_lane_missing_count": len(lanes) - len(available),
        "stage3_shard_count": sum(
            int(lane.get("stage3_shard_count") or 0) for lane in lanes
        ),
        "stage3_cache_materialized_lane_count": sum(
            lane.get("stage3_cache_status")
            == "stage3_cache_materialized_in_stage23_artifact"
            for lane in lanes
        ),
        "stage3_cache_missing_ready_lane_count": sum(
            lane.get("stage3_cache_status")
            == "stage3_cache_missing_from_ready_stage23_artifact"
            for lane in lanes
        ),
        "stage3_cache_status_counts": dict(sorted(cache_status_counts.items())),
        "stage4_exact_context_count": sum(
            int(lane.get("stage4_exact_context_count") or 0) for lane in lanes
        ),
        "stage5_context_count": int(stage5_review.get("context_count") or 0),
        "stage5_complete_context_count": int(
            stage5_review.get("complete_context_count") or 0
        ),
        "metric_clear_plane_count": int(axis_counts.get("clear") or 0),
        "metric_caution_plane_count": int(axis_counts.get("caution") or 0),
        "metric_blocked_plane_count": len(
            readiness_summary.get("blocked_metric_planes") or []
        ),
    }


def _bundle_context_tags(
    lanes: list[dict[str, Any]],
    *,
    metric_readiness: dict[str, Any],
    stage5_review: dict[str, Any],
) -> list[str]:
    tags: set[str] = set()
    for lane in lanes:
        timeframe = str(lane["timeframe"])
        if lane["status"] == "pipeline_lane_missing":
            tags.add(f"pipeline_lane_{timeframe}_missing")
        elif lane.get("exact_context_ready"):
            tags.add(f"pipeline_lane_{timeframe}_exact_context")
        else:
            tags.add(f"pipeline_lane_{timeframe}_supporting_context")
        if lane.get("stage3_shard_count"):
            tags.add(f"pipeline_lane_{timeframe}_stage3_shard_cache")
        elif lane.get("stage3_cache_status") == (
            "stage3_cache_missing_from_ready_stage23_artifact"
        ):
            tags.add(f"pipeline_lane_{timeframe}_stage3_cache_missing")
    readiness_status = metric_readiness.get("summary", {}).get(
        "readiness_status"
    )
    if readiness_status:
        tags.add("pipeline_metric_" + _slug(readiness_status))
    stage5_status = stage5_review.get("status")
    if stage5_status:
        tags.add("pipeline_stage5_" + _slug(stage5_status))
    return sorted(tags)


def _bundle_status(lanes: list[dict[str, Any]]) -> str:
    available = sum(lane["status"] != "pipeline_lane_missing" for lane in lanes)
    if not available:
        return "pipeline_context_bundle_missing"
    if available == len(lanes):
        return "pipeline_context_bundle_ready"
    return "pipeline_context_bundle_ready_with_gaps"


def _operator_next_steps(status: str, lanes: list[dict[str, Any]]) -> list[str]:
    missing = [
        lane["timeframe"] for lane in lanes if lane["status"] == "pipeline_lane_missing"
    ]
    steps = [
        "Attach this bundle to the world-model event packet as context only.",
        "Do not use this bundle to clear Stage 5, tune models, or trade.",
    ]
    if missing:
        steps.append(
            "Materialize exact pipeline context lanes before treating coverage "
            f"as complete: {', '.join(missing)}."
        )
    cache_missing = [
        lane["timeframe"]
        for lane in lanes
        if lane.get("stage3_cache_status")
        == "stage3_cache_missing_from_ready_stage23_artifact"
    ]
    if cache_missing:
        steps.append(
            "Rerun bounded Stage23 regeneration with current shard-cache code "
            "so stage3_cache metadata is materialized for: "
            + ", ".join(cache_missing)
            + "."
        )
    if status == "pipeline_context_bundle_missing":
        steps.insert(0, "Refresh or locate pipeline review artifacts first.")
    return steps


def _artifact_inventory(
    artifacts: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [_artifact_ref(item) for item in values]
        for key, values in artifacts.items()
    }


def _latest_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(items[0]["payload"]) if items else {}


def _stage5_review_fragment(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    keep_keys = (
        "run_id",
        "created_at",
        "mode",
        "schema_version",
        "status",
        "source_path",
        "source_contract",
        "source_artifact",
        "requested_tickers",
        "requested_timeframes",
        "filter_to_requested_scope",
        "source_context_count",
        "excluded_by_scope_count",
        "context_count",
        "complete_context_count",
        "review_issue_counts",
        "missing_lineage_field_counts",
        "sector_context_overlay_summary",
        "evidence_class",
        "target_semantics_contract",
        "safety",
    )
    fragment = {key: payload.get(key) for key in keep_keys if key in payload}
    fragment["contexts_included"] = False
    fragment["review_note"] = (
        "World-model pipeline context stores Stage5 summary/binding only; "
        "full per-context prediction records remain in the source artifact."
    )
    return fragment


def _artifact_ref(item: dict[str, Any] | None) -> dict[str, Any]:
    if not item:
        return {"available": False, "path": None, "sha256": None}
    payload = item.get("payload", {})
    return {
        "available": True,
        "path": item.get("path"),
        "sha256": item.get("sha256"),
        "run_id": payload.get("run_id"),
        "mode": payload.get("mode"),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
    }


def _payload_timeframe(payload: dict[str, Any]) -> str | None:
    scope = payload.get("scope", {}) if isinstance(payload, dict) else {}
    return _normalize_timeframe(
        scope.get("timeframe")
        or payload.get("timeframe")
        or payload.get("requested_timeframe")
    )


def _ticker_matches(payload: dict[str, Any], requested_tickers: list[str]) -> bool:
    if not requested_tickers:
        return True
    scope = payload.get("scope", {}) if isinstance(payload, dict) else {}
    payload_tickers = _normalize_tickers(
        scope.get("tickers")
        or [scope.get("ticker")]
        or payload.get("tickers")
        or []
    )
    return bool(set(payload_tickers).intersection(requested_tickers))


def _stage4_exact_ready(payload: dict[str, Any]) -> bool:
    lineage = payload.get("parent_lineage", {}) or {}
    timeframe = payload.get("timeframe_lineage", {}) or {}
    return bool(
        lineage.get("all_parent_hashes_verified")
        and timeframe.get("safe_for_prediction_lineage")
    )


def _stage4_matches_stage23(
    stage4_payload: dict[str, Any],
    stage23_payload: dict[str, Any],
) -> bool:
    lineage = stage4_payload.get("parent_lineage", {}) or {}
    batch = stage23_payload.get("batch_artifacts", {}) or {}
    stage4_feature_sha = str(
        (lineage.get("features") or {}).get("sha256") or ""
    )
    stage4_target_sha = str(
        (lineage.get("targets") or {}).get("sha256") or ""
    )
    stage23_feature_sha = str(batch.get("features_sha256") or "")
    stage23_target_sha = str(batch.get("targets_sha256") or "")
    return bool(
        lineage.get("all_parent_hashes_verified")
        and stage4_feature_sha
        and stage4_target_sha
        and stage4_feature_sha == stage23_feature_sha
        and stage4_target_sha == stage23_target_sha
    )


def _stage3_cache_status(
    stage23_artifact: dict[str, Any] | None,
    *,
    stage3_cache: dict[str, Any],
) -> str:
    if not stage23_artifact:
        return "stage23_artifact_missing"
    payload = stage23_artifact.get("payload", {})
    if payload.get("status") != "stage23_regeneration_review_ready":
        return "stage23_artifact_not_ready"
    if not isinstance(stage3_cache, dict) or not stage3_cache:
        return "stage3_cache_missing_from_ready_stage23_artifact"
    if int(stage3_cache.get("shard_count") or 0) > 0:
        return "stage3_cache_materialized_in_stage23_artifact"
    return "stage3_cache_empty_in_stage23_artifact"


def _lane_tickers(
    stage23: list[dict[str, Any]],
    stage4: list[dict[str, Any]],
    stage5_contexts: list[dict[str, Any]],
) -> list[str]:
    tickers: set[str] = set()
    for item in stage23:
        tickers.update(
            _normalize_tickers(item["payload"].get("scope", {}).get("tickers") or [])
        )
    for item in stage4:
        ticker = item["payload"].get("scope", {}).get("ticker")
        if ticker:
            tickers.add(_upper(ticker))
    for item in stage5_contexts:
        ticker = item.get("ticker")
        if ticker:
            tickers.add(_upper(ticker))
    return sorted(tickers)


def _lane_warnings(
    timeframe: str,
    *,
    exact_context_ready: bool,
    stage23_ready: bool,
    stage3_cache_status: str,
    stage5_context_count: int,
) -> list[str]:
    warnings: list[str] = []
    if not exact_context_ready:
        warnings.append(f"exact_context_missing_for_{timeframe}")
    if not stage23_ready:
        warnings.append(f"stage23_context_missing_for_{timeframe}")
    elif stage3_cache_status == "stage3_cache_missing_from_ready_stage23_artifact":
        warnings.append(f"stage3_cache_metadata_missing_for_{timeframe}")
    if not stage5_context_count:
        warnings.append(f"stage5_context_missing_for_{timeframe}")
    return warnings


def _normalize_tickers(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return sorted({_upper(value) for value in values if str(value).strip()})


def _normalize_timeframes(values: list[Any]) -> list[str]:
    normalized = [_normalize_timeframe(value) for value in values]
    return [value for value in normalized if value]


def _normalize_timeframe(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {
        "15min": "15m",
        "15": "15m",
        "1h": "60m",
        "60min": "60m",
        "60": "60m",
        "hourly": "60m",
        "daily": "1d",
        "day": "1d",
        "1day": "1d",
    }
    text = aliases.get(text, text)
    return text if text in {"15m", "60m", "1d"} else None


def _upper(value: Any) -> str:
    return str(value).strip().upper()


def _slug(value: Any) -> str:
    text = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in str(value)
    ).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "pipeline_regeneration_performed": False,
        "stage4_run_performed": False,
        "stage5_run_performed": False,
        "replay_task_registration_performed": False,
        "learning_memory_write_performed": False,
        "production_config_write_performed": False,
        "model_promotion_performed": False,
        "can_trade": False,
    }


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "DEFAULT_WORLD_MODEL_TIMEFRAMES",
    "WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT",
    "WorldModelPipelineContextDiscovery",
    "metadata_from_pipeline_context_bundle",
    "render_world_model_pipeline_context_markdown",
]
