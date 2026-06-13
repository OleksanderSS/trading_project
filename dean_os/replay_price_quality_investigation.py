from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.outcome_evaluation import _prepare_market_frame, _read_market_frame
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


DEFAULT_REPORT_PATHS = (
    "reports/dean_os/replay_price_normalizer/latest.json",
    "reports/dean_os/historical_replay_batch/latest.json",
    "reports/dean_os/historical_research_replay_batch_202603_202604/latest.json",
    "reports/dean_os/historical_research_replay_20260301_filtered/latest.json",
)


class ReplayPriceQualityInvestigationPlan:
    """Read-only diagnosis plan for replay price-quality blockers."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/replay_price_quality_investigation"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        report_paths: list[str | Path] | None = None,
        price_data_paths: list[str | Path] | None = None,
        benchmark_ticker: str = "SPY",
        close_col: str = "close",
        datetime_col: str = "datetime",
        large_step_threshold: float = 0.15,
        save: bool = True,
    ) -> dict[str, Any]:
        resolved_report_paths = DEFAULT_REPORT_PATHS if report_paths is None else report_paths
        reports = _load_reports(resolved_report_paths)
        benchmark = benchmark_ticker.upper()
        discovered_price_paths = _discover_price_paths(reports)
        all_price_paths = _unique_paths([*(price_data_paths or []), *discovered_price_paths])
        warnings = _collect_warning_records(reports)
        windows = _replay_windows(reports, benchmark=benchmark)
        artifact_diagnostics = [
            _diagnose_artifact(
                path=path,
                benchmark=benchmark,
                close_col=close_col,
                datetime_col=datetime_col,
                large_step_threshold=large_step_threshold,
            )
            for path in all_price_paths
        ]
        window_diagnostics = [
            _diagnose_window(
                path=window["price_data_path"],
                benchmark=benchmark,
                as_of=window["as_of"],
                lookback_days=int(window.get("lookback_days") or 180),
                close_col=close_col,
                datetime_col=datetime_col,
                large_step_threshold=large_step_threshold,
            )
            for window in windows
            if window.get("price_data_path")
        ]
        hypotheses = _hypotheses(warnings, artifact_diagnostics, window_diagnostics)
        summary = _summary(reports, warnings, artifact_diagnostics, window_diagnostics, hypotheses)
        payload = {
            "run_id": _run_id("replay_price_quality_investigation"),
            "created_at": utc_now_iso(),
            "mode": "replay_price_quality_investigation",
            "inputs": {
                "report_paths": [str(path) for path in resolved_report_paths],
                "price_data_paths": [str(path) for path in price_data_paths or []],
                "benchmark_ticker": benchmark,
                "close_col": close_col,
                "datetime_col": datetime_col,
                "large_step_threshold": large_step_threshold,
            },
            "summary": summary,
            "warning_summary": _warning_summary(warnings),
            "warning_records": warnings,
            "artifact_diagnostics": artifact_diagnostics,
            "window_diagnostics": window_diagnostics,
            "hypotheses": hypotheses,
            "operator_tasks": _operator_tasks(summary, hypotheses),
            "commands": _commands(all_price_paths, benchmark),
            "safety": {
                "read_only": True,
                "data_mutation_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "pipeline_run_performed": False,
                "learning_write_performed": False,
                "operation_proposal_created": False,
                "config_write_performed": False,
                "broker_access_performed": False,
            },
            "recommendations": _recommendations(summary, hypotheses),
        }
        if save:
            self.save(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_replay_price_quality_investigation_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_replay_price_quality_investigation_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Replay Price Quality Investigation",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('investigation_status')}`",
        f"- Reports loaded: {summary.get('reports_loaded')}",
        f"- Price artifacts inspected: {summary.get('price_artifacts_inspected')}",
        f"- Warning records: {summary.get('warning_record_count')}",
        f"- Extreme benchmark warnings: {summary.get('extreme_benchmark_warning_count')}",
        f"- Window diagnostics: {summary.get('window_diagnostic_count')}",
        "",
        "## Hypotheses",
        "",
    ]
    for hypothesis in payload.get("hypotheses", []):
        lines.append(
            f"- `{hypothesis.get('hypothesis')}` severity=`{hypothesis.get('severity')}` "
            f"confidence={hypothesis.get('confidence')}: {hypothesis.get('reason')}"
        )
    lines.extend(["", "## Window Diagnostics", ""])
    for window in payload.get("window_diagnostics", [])[:10]:
        lines.append(
            f"- `{window.get('path')}` as_of=`{window.get('as_of')}` "
            f"return={window.get('lookback_return')} rows={window.get('row_count')} "
            f"largest_step={window.get('largest_abs_step_return')}"
        )
    lines.extend(["", "## Operator Tasks", ""])
    for task in payload.get("operator_tasks", []):
        lines.append(f"- `{task.get('priority')}` {task.get('task_id')}: {task.get('description')}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _load_reports(paths: list[str | Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        resolved = Path(path)
        if not resolved.exists():
            reports.append({"path": str(resolved), "loaded": False, "error": "missing"})
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            reports.append({"path": str(resolved), "loaded": False, "error": f"invalid_json: {exc}"})
            continue
        reports.append({"path": str(resolved), "loaded": True, "payload": payload, "mode": payload.get("mode")})
    return reports


def _collect_warning_records(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for report in reports:
        if not report.get("loaded"):
            continue
        payload = report.get("payload", {})
        for location, warning in _walk_warnings(payload):
            records.append(
                {
                    "report_path": report.get("path"),
                    "report_mode": payload.get("mode"),
                    "location": location,
                    "warning": warning,
                    "category": _warning_category(warning),
                    "benchmark_return": _extract_warning_return(warning),
                }
            )
    return records


def _walk_warnings(payload: Any, path: str = "$") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_path = f"{path}.{key}"
            if key in {"warnings", "quality_warnings", "price_warnings", "comparison_window_warnings"} and isinstance(value, list):
                found.extend((next_path, str(item)) for item in value if str(item).strip())
            elif key == "quality_warnings" and isinstance(value, dict):
                found.extend((f"{next_path}.{item_key}", str(item_key)) for item_key in value)
            else:
                found.extend(_walk_warnings(value, next_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_walk_warnings(value, f"{path}[{index}]"))
    return found


def _discover_price_paths(reports: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for report in reports:
        if not report.get("loaded"):
            continue
        payload = report.get("payload", {})
        paths.extend(_walk_values_for_keys(payload, {"price_data_path", "market_data_path", "path"}))
        artifact_path = payload.get("artifact", {}).get("path") if isinstance(payload.get("artifact"), dict) else None
        if artifact_path:
            paths.append(str(artifact_path))
    return _unique_paths(paths)


def _replay_windows(reports: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for report in reports:
        if not report.get("loaded"):
            continue
        payload = report.get("payload", {})
        inputs = payload.get("inputs", {}) if isinstance(payload.get("inputs"), dict) else {}
        price_path = inputs.get("price_data_path")
        lookback = inputs.get("lookback_days", 180)
        if inputs.get("as_of"):
            windows.append(
                {
                    "source_report": report.get("path"),
                    "price_data_path": price_path,
                    "as_of": inputs.get("as_of"),
                    "lookback_days": lookback,
                    "benchmark_ticker": benchmark,
                }
            )
        for run in payload.get("runs", []) if isinstance(payload.get("runs"), list) else []:
            if run.get("as_of"):
                windows.append(
                    {
                        "source_report": report.get("path"),
                        "price_data_path": price_path,
                        "as_of": run.get("as_of"),
                        "lookback_days": lookback,
                        "benchmark_ticker": benchmark,
                    }
                )
    return _dedupe_windows(windows)


def _diagnose_artifact(
    path: str | Path,
    benchmark: str,
    close_col: str,
    datetime_col: str,
    large_step_threshold: float,
) -> dict[str, Any]:
    resolved = Path(path)
    base = {
        "path": str(resolved),
        "exists": resolved.exists(),
        "status": "unreadable",
        "benchmark_ticker": benchmark,
        "issues": [],
        "warnings": [],
    }
    if not resolved.exists():
        base["issues"].append("Price artifact does not exist.")
        return base
    try:
        import pandas as pd

        raw = _read_market_frame(pd, resolved)
        frame = _prepare_market_frame(pd, raw, close_col=close_col, datetime_col=datetime_col)
    except Exception as exc:
        base["issues"].append(f"Could not read/prepare price artifact: {type(exc).__name__}: {exc}")
        return base
    diagnostics = _price_frame_diagnostics(frame, benchmark=benchmark, large_step_threshold=large_step_threshold)
    return {**base, **diagnostics, "status": "inspected"}


def _diagnose_window(
    path: str | Path,
    benchmark: str,
    as_of: str,
    lookback_days: int,
    close_col: str,
    datetime_col: str,
    large_step_threshold: float,
) -> dict[str, Any]:
    as_of_dt = _parse_datetime(as_of)
    start = as_of_dt - timedelta(days=lookback_days)
    resolved = Path(path)
    base = {
        "path": str(resolved),
        "as_of": as_of_dt.isoformat(),
        "lookback_start": start.isoformat(),
        "lookback_days": lookback_days,
        "benchmark_ticker": benchmark,
        "exists": resolved.exists(),
        "status": "unreadable",
        "issues": [],
        "warnings": [],
    }
    if not resolved.exists():
        base["issues"].append("Price artifact does not exist.")
        return base
    try:
        import pandas as pd

        raw = _read_market_frame(pd, resolved)
        frame = _prepare_market_frame(pd, raw, close_col=close_col, datetime_col=datetime_col)
    except Exception as exc:
        base["issues"].append(f"Could not read/prepare price artifact: {type(exc).__name__}: {exc}")
        return base
    window = frame[(frame["_dean_datetime"] >= start) & (frame["_dean_datetime"] <= as_of_dt)]
    diagnostics = _price_frame_diagnostics(window, benchmark=benchmark, large_step_threshold=large_step_threshold)
    return {**base, **diagnostics, "status": "inspected"}


def _price_frame_diagnostics(frame: Any, benchmark: str, large_step_threshold: float) -> dict[str, Any]:
    if frame.empty:
        return {"row_count": 0, "benchmark_row_count": 0, "issues": ["No rows in inspected price frame."], "warnings": []}
    benchmark_frame = frame[frame["_dean_ticker"] == benchmark].sort_values("_dean_datetime").reset_index(drop=True).copy()
    working = frame.copy()
    working["_dean_date"] = working["_dean_datetime"].dt.date
    rows_per_day = working.groupby(["_dean_ticker", "_dean_date"]).size()
    interval_counts = working["interval"].astype(str).value_counts().to_dict() if "interval" in working.columns else {}
    base = {
        "row_count": int(len(frame)),
        "ticker_count": int(working["_dean_ticker"].nunique()),
        "start": working["_dean_datetime"].min().isoformat(),
        "end": working["_dean_datetime"].max().isoformat(),
        "interval_counts": interval_counts,
        "max_rows_per_ticker_day": int(rows_per_day.max()) if len(rows_per_day) else 0,
        "multi_row_ticker_day_count": int((rows_per_day > 1).sum()) if len(rows_per_day) else 0,
        "benchmark_row_count": int(len(benchmark_frame)),
        "issues": [],
        "warnings": [],
    }
    if benchmark_frame.empty:
        base["issues"].append(f"Benchmark {benchmark} has no rows.")
        return base
    close = benchmark_frame["_dean_close"].astype(float)
    step_returns = close.pct_change().replace([float("inf"), float("-inf")], None).dropna()
    largest_step = float(step_returns.abs().max()) if len(step_returns) else None
    largest_step_index = int(step_returns.abs().idxmax()) if len(step_returns) else None
    previous_step_index = largest_step_index - 1 if largest_step_index is not None and largest_step_index > 0 else None
    largest_step_at = benchmark_frame.loc[largest_step_index, "_dean_datetime"].isoformat() if largest_step_index is not None else None
    largest_step_from_at = (
        benchmark_frame.loc[previous_step_index, "_dean_datetime"].isoformat() if previous_step_index is not None else None
    )
    largest_step_signed = float(step_returns.loc[largest_step_index]) if largest_step_index is not None else None
    lookback_return = float(close.iloc[-1] / close.iloc[0] - 1.0) if len(close) >= 2 and close.iloc[0] else None
    base.update(
        {
            "benchmark_start": benchmark_frame["_dean_datetime"].min().isoformat(),
            "benchmark_end": benchmark_frame["_dean_datetime"].max().isoformat(),
            "benchmark_start_close": float(close.iloc[0]),
            "benchmark_end_close": float(close.iloc[-1]),
            "benchmark_min_close": float(close.min()),
            "benchmark_max_close": float(close.max()),
            "lookback_return": lookback_return,
            "largest_abs_step_return": largest_step,
            "largest_step_return": largest_step_signed,
            "largest_step_from_at": largest_step_from_at,
            "largest_abs_step_at": largest_step_at,
            "largest_step_from_close": float(close.iloc[previous_step_index]) if previous_step_index is not None else None,
            "largest_step_to_close": float(close.iloc[largest_step_index]) if largest_step_index is not None else None,
        }
    )
    if lookback_return is not None and abs(lookback_return) > 0.5:
        base["warnings"].append(f"Benchmark {benchmark} return is extreme in inspected frame: {lookback_return:.3f}.")
    if largest_step is not None and largest_step > large_step_threshold:
        base["warnings"].append(
            f"Benchmark {benchmark} has a large one-step move: {largest_step:.3f} "
            f"({base.get('largest_step_from_close')} -> {base.get('largest_step_to_close')})."
        )
    if interval_counts.get("1d") and base["max_rows_per_ticker_day"] > 1:
        base["warnings"].append("Rows are labelled 1d but multiple rows per ticker/day exist.")
    return base


def _hypotheses(
    warnings: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    if any(item.get("category") == "interval_mixing" for item in warnings) or any(
        artifact.get("multi_row_ticker_day_count", 0) for artifact in artifacts
    ):
        hypotheses.append(
            {
                "hypothesis": "interval_mixing_or_daily_label_issue",
                "severity": "high",
                "confidence": 0.8,
                "reason": "At least one artifact/report shows multiple rows per ticker/day while rows are labelled daily.",
            }
        )
    extreme_windows = [window for window in windows if window.get("lookback_return") is not None and abs(float(window["lookback_return"])) > 0.5]
    if extreme_windows:
        largest_steps = [window for window in extreme_windows if window.get("largest_abs_step_return") and float(window["largest_abs_step_return"]) > 0.15]
        hypotheses.append(
            {
                "hypothesis": "benchmark_window_price_anomaly",
                "severity": "high",
                "confidence": 0.85 if largest_steps else 0.65,
                "reason": "Benchmark lookback returns are extreme inside replay windows; inspect SPY adjusted prices and split/dividend handling.",
                "affected_windows": len(extreme_windows),
            }
        )
    if any(report_warning.get("category") == "duplicate_datetime" for report_warning in warnings):
        hypotheses.append(
            {
                "hypothesis": "duplicate_timestamp_rows",
                "severity": "medium",
                "confidence": 0.7,
                "reason": "Replay reports mention duplicate ticker/datetime rows.",
            }
        )
    if not hypotheses:
        hypotheses.append(
            {
                "hypothesis": "no_repeated_price_quality_blocker_detected",
                "severity": "low",
                "confidence": 0.5,
                "reason": "Loaded reports do not contain repeated price-quality warnings.",
            }
        )
    return hypotheses


def _summary(
    reports: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
) -> dict[str, Any]:
    loaded = sum(1 for report in reports if report.get("loaded"))
    extreme = sum(1 for warning in warnings if warning.get("category") == "extreme_benchmark")
    status = "investigation_required" if warnings or any(window.get("warnings") for window in windows) else "clear"
    if any(hypothesis.get("severity") == "high" for hypothesis in hypotheses):
        status = "blocked_price_quality"
    return {
        "investigation_status": status,
        "reports_loaded": loaded,
        "reports_missing_or_invalid": len(reports) - loaded,
        "price_artifacts_inspected": sum(1 for artifact in artifacts if artifact.get("status") == "inspected"),
        "warning_record_count": len(warnings),
        "extreme_benchmark_warning_count": extreme,
        "window_diagnostic_count": len(windows),
        "high_severity_hypothesis_count": sum(1 for hypothesis in hypotheses if hypothesis.get("severity") == "high"),
        "can_use_replay_hit_miss_for_learning": False if status != "clear" else False,
    }


def _operator_tasks(summary: dict[str, Any], hypotheses: list[dict[str, Any]]) -> list[dict[str, str]]:
    tasks = [
        {
            "task_id": "inspect_benchmark_window_prices",
            "priority": "P0",
            "description": "Open the SPY rows for blocked replay windows and verify start/end close, largest step, and adjusted-price semantics.",
        }
    ]
    if any(item.get("hypothesis") == "interval_mixing_or_daily_label_issue" for item in hypotheses):
        tasks.append(
            {
                "task_id": "verify_daily_normalization_source",
                "priority": "P0",
                "description": "Confirm whether raw 1d rows are actually intraday snapshots and whether normalized artifact should use first/last source timestamps.",
            }
        )
    if summary.get("extreme_benchmark_warning_count"):
        tasks.append(
            {
                "task_id": "compare_benchmark_against_external_or_fresh_artifact",
                "priority": "P1",
                "description": "Compare SPY adjusted close for the same lookback windows against a refreshed local artifact before clearing replay gates.",
            }
        )
    tasks.append(
        {
            "task_id": "rerun_replay_batch_after_price_fix",
            "priority": "P1",
            "description": "After price-quality is fixed, rerun historical replay batch and historical research replay batch before any learning promotion.",
        }
    )
    return tasks


def _commands(price_paths: list[str], benchmark: str) -> dict[str, str]:
    first_path = price_paths[0] if price_paths else "PATH_TO_PRICE_ARTIFACT"
    return {
        "rerun_normalizer": (
            "python run_agent_replay_price_normalizer.py "
            f"{first_path} --tickers AMD NVDA MSFT AAPL TSM QQQ {benchmark} --compare-replay "
            "--as-of 2026-03-01T00:00:00+00:00 --lookback-days 180 --horizon-days 60"
        ),
        "rerun_research_batch_after_fix": (
            "python run_agent_historical_research_replay_batch.py "
            f"{first_path} --tickers AMD NVDA MSFT AAPL TSM QQQ {benchmark} "
            "--as-of 2026-03-01T00:00:00+00:00 2026-04-01T00:00:00+00:00 "
            "--lookback-days 180 --horizon-days 30 --normalize-daily-bars"
        ),
    }


def _recommendations(summary: dict[str, Any], hypotheses: list[dict[str, Any]]) -> list[str]:
    recommendations = [
        "Keep replay hit/miss results diagnostic until price-quality blockers are cleared.",
        "Do not change analyst weights, tuning bounds, or learning memory from blocked replay windows.",
    ]
    if summary.get("extreme_benchmark_warning_count"):
        recommendations.append("Prioritize SPY benchmark inspection; benchmark distortion can corrupt all relative-strength rankings.")
    if any(item.get("hypothesis") == "interval_mixing_or_daily_label_issue" for item in hypotheses):
        recommendations.append("Keep using normalized artifacts, but inspect the blocked window because normalization alone did not clear it.")
    recommendations.append("After fixing price quality, rerun both price-only and research replay batches and compare clear_hit_rate.")
    return recommendations


def _warning_summary(warnings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_category": dict(Counter(warning["category"] for warning in warnings).most_common()),
        "by_warning": dict(Counter(warning["warning"] for warning in warnings).most_common()),
    }


def _warning_category(warning: str) -> str:
    text = warning.lower()
    if "benchmark" in text and "extreme" in text:
        return "extreme_benchmark"
    if "multiple rows per ticker/day" in text or "labelled 1d" in text:
        return "interval_mixing"
    if "duplicate" in text and "datetime" in text:
        return "duplicate_datetime"
    return "other"


def _extract_warning_return(warning: str) -> float | None:
    match = re.search(r"\((-?\d+(?:\.\d+)?)\)", warning)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _walk_values_for_keys(payload: Any, keys: set[str]) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and isinstance(value, str) and _looks_like_price_path(value):
                values.append(value)
            values.extend(_walk_values_for_keys(value, keys))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_walk_values_for_keys(value, keys))
    return values


def _looks_like_price_path(value: str) -> bool:
    lowered = value.lower()
    return (lowered.endswith(".parquet") or lowered.endswith(".csv")) and ("price" in lowered or "stage2_prices" in lowered)


def _unique_paths(values: list[str | Path]) -> list[str]:
    result: list[str] = []
    seen = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _dedupe_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for window in windows:
        key = (window.get("price_data_path"), window.get("as_of"), window.get("lookback_days"))
        if key in seen:
            continue
        seen.add(key)
        result.append(window)
    return result


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _run_id(prefix: str) -> str:
    return prefix + "_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_")
