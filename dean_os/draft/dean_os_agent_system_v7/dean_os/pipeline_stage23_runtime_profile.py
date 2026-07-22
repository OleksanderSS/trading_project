from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_stage23_regeneration import (
    _load_saved_stage1_market,
    _run_bounded_stage2,
    _select_bounded_market_frame,
    _source_checks,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from src.pipeline.timeframe_lineage import normalize_timeframe

PIPELINE_STAGE23_RUNTIME_PROFILE_CONTRACT = (
    "dean_pipeline_stage23_runtime_profile_v1"
)


class PipelineStage23RuntimeProfile:
    """Review-only runtime profile for bounded Stage23 lane generation.

    The profile is intentionally diagnostic. It reads saved Stage 1 data,
    validates source lanes, optionally runs Stage 2 on a bounded sample, and
    optionally runs a small Stage 3 sample. It does not create Stage23 batch artifacts,
    Stage4/Stage5 artifacts, learning memory, model promotions, predictions, or
    trades.
    """

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_stage23_runtime_profile_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    async def build(
        self,
        *,
        source_path: str | Path,
        tickers: list[str],
        timeframes: list[str],
        max_rows_per_ticker: int = 80,
        include_stage2: bool = False,
        include_stage3: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        if max_rows_per_ticker < 5:
            raise ValueError("max_rows_per_ticker must be at least 5")
        source = Path(source_path)
        requested_tickers = _normalize_tickers(tickers)
        requested_timeframes = _normalize_timeframes(timeframes)

        total_start = time.perf_counter()
        source_start = time.perf_counter()
        raw_market, source_format = _load_saved_stage1_market(source)
        source_load_seconds = _elapsed(source_start)

        lanes: list[dict[str, Any]] = []
        for timeframe in requested_timeframes:
            lanes.append(
                await self._profile_lane(
                    raw_market,
                    source_path=source,
                    tickers=requested_tickers,
                    timeframe=timeframe,
                    max_rows_per_ticker=max_rows_per_ticker,
                    include_stage2=include_stage2 or include_stage3,
                    include_stage3=include_stage3,
                )
            )

        summary = _summary(
            lanes,
            include_stage2=include_stage2 or include_stage3,
            include_stage3=include_stage3,
        )
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "pipeline_stage23_runtime_profile",
            "contract": PIPELINE_STAGE23_RUNTIME_PROFILE_CONTRACT,
            "inputs": {
                "source_path": str(source),
                "source_format": source_format,
                "source_market_row_count": len(raw_market),
                "tickers": requested_tickers,
                "timeframes": requested_timeframes,
                "max_rows_per_ticker": max_rows_per_ticker,
                "include_stage2": include_stage2 or include_stage3,
                "include_stage3": include_stage3,
            },
            "summary": {
                **summary,
                "source_load_seconds": source_load_seconds,
                "total_elapsed_seconds": _elapsed(total_start),
            },
            "timeframe_lanes": lanes,
            "operator_next_steps": _operator_next_steps(lanes),
            "safety": _safety(
                include_stage2=include_stage2 or include_stage3,
                include_stage3=include_stage3,
            ),
        }
        payload = json_ready(payload)
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_stage23_runtime_profile_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return payload

    async def _profile_lane(
        self,
        raw_market: pd.DataFrame,
        *,
        source_path: Path,
        tickers: list[str],
        timeframe: str,
        max_rows_per_ticker: int,
        include_stage2: bool,
        include_stage3: bool,
    ) -> dict[str, Any]:
        lane_start = time.perf_counter()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        selected = _select_bounded_market_frame(
            raw_market,
            tickers=tickers,
            timeframe=timeframe,
            max_rows_per_ticker=max_rows_per_ticker,
        )
        timings["source_select"] = _elapsed(started)

        started = time.perf_counter()
        checks = _source_checks(
            selected,
            tickers=tickers,
            timeframe=timeframe,
        )
        timings["source_checks"] = _elapsed(started)
        blockers = [
            item["code"]
            for item in checks
            if item.get("status") == "fail"
        ]
        if blockers:
            return _lane_payload(
                timeframe=timeframe,
                status="source_checks_blocked",
                source_path=source_path,
                tickers=tickers,
                max_rows_per_ticker=max_rows_per_ticker,
                selected_row_count=len(selected),
                stage2_row_count=0,
                enriched_row_count=0,
                feature_column_count=0,
                target_column_count=0,
                source_checks=checks,
                blocking_reasons=blockers,
                timings_seconds=timings,
                total_seconds=_elapsed(lane_start),
                include_stage2=include_stage2,
                include_stage3=include_stage3,
            )

        if not include_stage2:
            return _lane_payload(
                timeframe=timeframe,
                status="source_profile_ready",
                source_path=source_path,
                tickers=tickers,
                max_rows_per_ticker=max_rows_per_ticker,
                selected_row_count=len(selected),
                stage2_row_count=0,
                enriched_row_count=0,
                feature_column_count=0,
                target_column_count=0,
                source_checks=checks,
                blocking_reasons=[],
                timings_seconds=timings,
                total_seconds=_elapsed(lane_start),
                include_stage2=False,
                include_stage3=False,
            )

        started = time.perf_counter()
        stage2_frame, stage2_quality = _run_bounded_stage2(
            selected,
            timeframe=timeframe,
        )
        timings["stage2"] = _elapsed(started)

        enriched_row_count = 0
        feature_column_count = 0
        target_column_count = 0
        if include_stage3:
            from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_bounded_evidence_run import (
                _run_stage_3_enrichment,
            )
            from src.pipeline.hybrid.feature_processor import FeatureProcessor

            started = time.perf_counter()
            enriched = await _run_stage_3_enrichment(
                stage2_frame,
                timeframe=timeframe,
            )
            timings["stage3_enrichment"] = _elapsed(started)
            enriched_row_count = len(enriched)

            started = time.perf_counter()
            processed = FeatureProcessor().process_enriched_data(enriched)
            timings["feature_target_split"] = _elapsed(started)
            if not processed:
                raise ValueError("FeatureProcessor returned no profile output")
            features = processed["features"]
            targets = processed["targets"]
            feature_column_count = len(features.columns)
            target_column_count = len(
                [
                    column
                    for column in targets.columns
                    if str(column).startswith("target_")
                ]
            )

        return _lane_payload(
            timeframe=timeframe,
            status=(
                "stage3_profile_ready"
                if include_stage3
                else "stage2_profile_ready"
            ),
            source_path=source_path,
            tickers=tickers,
            max_rows_per_ticker=max_rows_per_ticker,
            selected_row_count=len(selected),
            stage2_row_count=len(stage2_frame),
            enriched_row_count=enriched_row_count,
            feature_column_count=feature_column_count,
            target_column_count=target_column_count,
            source_checks=checks,
            blocking_reasons=[],
            timings_seconds=timings,
            total_seconds=_elapsed(lane_start),
            include_stage2=include_stage2,
            include_stage3=include_stage3,
            stage2_quality=stage2_quality,
        )


def render_pipeline_stage23_runtime_profile_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# DEAN-OS Pipeline Stage23 Runtime Profile",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Profiled lanes: {summary.get('profiled_lane_count')}",
        f"- Stage2 included: `{summary.get('stage2_included')}`",
        f"- Stage3 included: `{summary.get('stage3_included')}`",
        f"- Slowest lane: `{summary.get('slowest_lane')}`",
        f"- Slowest step: `{summary.get('slowest_step')}`",
        f"- Total elapsed seconds: {summary.get('total_elapsed_seconds')}",
        f"- Can create Stage23 artifacts: `{summary.get('can_create_stage23_artifacts')}`",
        f"- Can trade: `{summary.get('can_trade')}`",
        "",
        "## Lanes",
        "",
    ]
    for lane in payload.get("timeframe_lanes", []):
        timings = lane.get("runtime_profile", {}).get("timings_seconds") or {}
        lines.extend(
            [
                f"- `{lane.get('timeframe')}`: `{lane.get('lane_status')}`",
                f"  - selected rows: {lane.get('row_counts', {}).get('selected')}",
                f"  - Stage2 rows: {lane.get('row_counts', {}).get('stage2')}",
                f"  - total seconds: {lane.get('runtime_profile', {}).get('total_elapsed_seconds')}",
                f"  - timings: `{timings}`",
                f"  - next action: {lane.get('recommended_next_action')}",
            ]
        )
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _lane_payload(
    *,
    timeframe: str,
    status: str,
    source_path: Path,
    tickers: list[str],
    max_rows_per_ticker: int,
    selected_row_count: int,
    stage2_row_count: int,
    enriched_row_count: int,
    feature_column_count: int,
    target_column_count: int,
    source_checks: list[dict[str, Any]],
    blocking_reasons: list[str],
    timings_seconds: dict[str, float],
    total_seconds: float,
    include_stage2: bool,
    include_stage3: bool,
    stage2_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "lane_status": status,
        "blocking_reasons": blocking_reasons,
        "row_counts": {
            "selected": selected_row_count,
            "stage2": stage2_row_count,
            "enriched": enriched_row_count,
        },
        "column_counts": {
            "features": feature_column_count,
            "targets": target_column_count,
        },
        "source_checks": source_checks,
        "stage2_quality": stage2_quality or {},
        "runtime_profile": {
            "timings_seconds": timings_seconds,
            "total_elapsed_seconds": total_seconds,
            "slowest_step": _slowest_step(timings_seconds),
        },
        "recommended_next_action": _lane_next_action(
            status,
            include_stage2=include_stage2,
            include_stage3=include_stage3,
        ),
        "suggested_stage23_command": _stage23_command(
            source_path=source_path,
            tickers=tickers,
            timeframe=timeframe,
            max_rows_per_ticker=max_rows_per_ticker,
        )
        if not blocking_reasons
        else None,
    }


def _summary(
    lanes: list[dict[str, Any]],
    *,
    include_stage2: bool,
    include_stage3: bool,
) -> dict[str, Any]:
    ready = [
        lane
        for lane in lanes
        if lane.get("lane_status")
        in {"source_profile_ready", "stage2_profile_ready", "stage3_profile_ready"}
    ]
    blocked = len(lanes) - len(ready)
    slowest_lane = None
    slowest_seconds = -1.0
    slowest_step = None
    for lane in lanes:
        total = float(
            (lane.get("runtime_profile") or {}).get(
                "total_elapsed_seconds",
                0.0,
            )
        )
        if total > slowest_seconds:
            slowest_lane = lane.get("timeframe")
            slowest_seconds = total
            slowest_step = (lane.get("runtime_profile") or {}).get(
                "slowest_step"
            )
    return {
        "status": (
            "pipeline_stage23_runtime_profile_ready"
            if blocked == 0
            else "pipeline_stage23_runtime_profile_ready_with_gaps"
            if ready
            else "pipeline_stage23_runtime_profile_blocked"
        ),
        "profiled_lane_count": len(lanes),
        "ready_lane_count": len(ready),
        "blocked_lane_count": blocked,
        "stage2_included": include_stage2,
        "stage3_included": include_stage3,
        "slowest_lane": slowest_lane,
        "slowest_step": slowest_step,
        "can_create_stage23_artifacts": False,
        "can_condition_world_model": False,
        "can_write_learning_memory": False,
        "can_trade": False,
    }


def _operator_next_steps(lanes: list[dict[str, Any]]) -> list[str]:
    steps = [
        "Use this as a runtime diagnostic only; it does not replace Stage23 artifacts.",
    ]
    for lane in lanes:
        if lane.get("suggested_stage23_command"):
            steps.append(
                f"{lane.get('timeframe')}: if runtime is acceptable, run "
                "the suggested Stage23 command with shared shard-cache."
            )
    steps.append(
        "After Stage23 cache/artifacts exist, rerun Stage4 exact-context review "
        "and then WorldModelPipelineContextDiscovery."
    )
    return steps


def _lane_next_action(
    status: str,
    *,
    include_stage2: bool,
    include_stage3: bool,
) -> str:
    if status == "source_checks_blocked":
        return "Repair source lane coverage/cadence before running Stage23."
    if include_stage3:
        return "Use timings to decide whether to schedule full Stage23 cache materialization."
    if include_stage2:
        return "Stage2 sample passed; run with --include-stage3 only when an interactive budget is acceptable."
    return "Run with --include-stage2 on a small sample before scheduling full Stage23."


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
    result = sorted(
        {
            str(ticker).strip().upper()
            for ticker in tickers
            if str(ticker).strip()
        }
    )
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


def _slowest_step(timings: dict[str, float]) -> str | None:
    if not timings:
        return None
    return max(timings.items(), key=lambda item: item[1])[0]


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def _safety(*, include_stage2: bool, include_stage3: bool) -> dict[str, bool]:
    return {
        "review_only": True,
        "saved_source_only": True,
        "stage2_profile_performed": bool(include_stage2),
        "stage3_profile_performed": bool(include_stage3),
        "stage23_batch_write_performed": False,
        "stage3_cache_write_performed": False,
        "stage4_run_performed": False,
        "stage5_run_performed": False,
        "training_performed": False,
        "prediction_performed": False,
        "production_config_write_performed": False,
        "learning_memory_write_performed": False,
        "broker_access_performed": False,
        "live_execution_performed": False,
        "can_trade": False,
    }


def _run_id() -> str:
    return f"pipeline_stage23_runtime_profile_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "PIPELINE_STAGE23_RUNTIME_PROFILE_CONTRACT",
    "PipelineStage23RuntimeProfile",
    "render_pipeline_stage23_runtime_profile_markdown",
]
