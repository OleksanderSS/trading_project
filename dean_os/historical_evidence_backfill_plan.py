from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_READINESS_REPORT = "reports/dean_os/replay_calibration_readiness_gate_after_step14_research/latest.json"
DEFAULT_RESEARCH_BATCH = "reports/dean_os/historical_research_replay_batch_repaired_expanded_step14/latest.json"

DATE_COLUMNS = (
    "published_date",
    "publication_date",
    "publishedAt",
    "pub_date",
    "time_published",
    "timestamp",
    "datetime",
    "date",
    "realtime_start",
)
TEXT_COLUMNS = ("title", "headline", "content", "description", "summary", "search_term")


class HistoricalEvidenceBackfillPlan:
    """Read-only plan for improving weak historical research replay evidence."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/historical_evidence_backfill_plan"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        readiness_report_path: str | Path | None = DEFAULT_READINESS_REPORT,
        research_batch_path: str | Path | None = DEFAULT_RESEARCH_BATCH,
        news_data_paths: list[str | Path] | None = None,
        macro_data_paths: list[str | Path] | None = None,
        materials_paths: list[str | Path] | None = None,
        tickers: list[str] | None = None,
        lookback_days: int = 180,
        min_documents_per_run: int = 5,
        save: bool = True,
    ) -> dict[str, Any]:
        readiness = _load_optional_json(readiness_report_path)
        research_batch = _load_optional_json(research_batch_path)
        runs = _research_runs(research_batch.get("payload", {}), min_documents_per_run=min_documents_per_run)
        requested_tickers = _requested_tickers(research_batch.get("payload", {}), tickers)
        weak_runs = [run for run in runs if run["needs_backfill"]]
        source_audits = _source_audits(
            runs=weak_runs or runs,
            news_data_paths=news_data_paths or [],
            macro_data_paths=macro_data_paths or [],
            tickers=requested_tickers,
            lookback_days=lookback_days,
        )
        tasks = _tasks(
            weak_runs=weak_runs,
            source_audits=source_audits,
            materials_paths=materials_paths or [],
            requested_tickers=requested_tickers,
        )
        summary = _summary(readiness, research_batch, runs, weak_runs, tasks, requested_tickers)
        payload = {
            "run_id": _run_id("historical_evidence_backfill_plan"),
            "created_at": utc_now_iso(),
            "mode": "historical_evidence_backfill_plan",
            "inputs": {
                "readiness_report_path": str(readiness_report_path) if readiness_report_path else None,
                "research_batch_path": str(research_batch_path) if research_batch_path else None,
                "news_data_paths": [str(path) for path in news_data_paths or []],
                "macro_data_paths": [str(path) for path in macro_data_paths or []],
                "materials_paths": [str(path) for path in materials_paths or []],
                "tickers": requested_tickers,
                "lookback_days": lookback_days,
                "min_documents_per_run": min_documents_per_run,
            },
            "summary": summary,
            "readiness_context": _readiness_context(readiness.get("payload", {})),
            "weak_runs": weak_runs,
            "coverage_gaps": _coverage_gaps(weak_runs, requested_tickers),
            "source_audits": source_audits,
            "backfill_tasks": tasks,
            "commands": _commands(research_batch.get("payload", {}), news_data_paths or [], macro_data_paths or []),
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
            "recommendations": _recommendations(summary, tasks),
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
        rendered_md = render_historical_evidence_backfill_plan_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_historical_evidence_backfill_plan_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Historical Evidence Backfill Plan",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('backfill_status')}`",
        f"- Weak runs: {summary.get('weak_run_count')}",
        f"- Missing ticker count: {summary.get('missing_ticker_count')}",
        f"- Task count: {summary.get('task_count')}",
        "",
        "## Coverage Gaps",
        "",
    ]
    gaps = payload.get("coverage_gaps", {})
    lines.append(f"- Missing tickers: `{gaps.get('missing_tickers', [])}`")
    lines.append(f"- Weak dates sample: `{gaps.get('weak_dates_sample', [])}`")
    lines.extend(["", "## Tasks", ""])
    for task in payload.get("backfill_tasks", []):
        lines.append(f"- `{task.get('priority')}` {task.get('task_id')}: {task.get('description')}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _research_runs(payload: dict[str, Any], min_documents_per_run: int) -> list[dict[str, Any]]:
    runs = payload.get("runs", []) if isinstance(payload.get("runs"), list) else []
    result: list[dict[str, Any]] = []
    for run in runs:
        document_count = int(run.get("evidence_document_count") or 0)
        quality = str(run.get("evidence_data_quality") or "unknown")
        missing = sorted(str(ticker).upper() for ticker in run.get("evidence_missing_tickers", []) if str(ticker).strip())
        result.append(
            {
                "run_id": run.get("run_id"),
                "as_of": run.get("as_of"),
                "horizon_days": run.get("horizon_days"),
                "evidence_document_count": document_count,
                "evidence_data_quality": quality,
                "evidence_tickers": sorted(str(ticker).upper() for ticker in run.get("evidence_tickers", []) if str(ticker).strip()),
                "evidence_missing_tickers": missing,
                "research_stance": run.get("research_stance"),
                "research_expected_direction": run.get("research_expected_direction"),
                "exam_verdict": run.get("exam_verdict"),
                "needs_backfill": quality != "strong" or bool(missing) or document_count < min_documents_per_run,
                "backfill_reasons": _backfill_reasons(quality, missing, document_count, min_documents_per_run),
            }
        )
    return result


def _backfill_reasons(quality: str, missing: list[str], document_count: int, min_documents_per_run: int) -> list[str]:
    reasons: list[str] = []
    if quality != "strong":
        reasons.append(f"evidence quality is {quality}")
    if missing:
        reasons.append(f"missing tickers: {', '.join(missing)}")
    if document_count < min_documents_per_run:
        reasons.append(f"document count {document_count} below floor {min_documents_per_run}")
    return reasons


def _source_audits(
    runs: list[dict[str, Any]],
    news_data_paths: list[str | Path],
    macro_data_paths: list[str | Path],
    tickers: list[str],
    lookback_days: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "news": [_audit_table(path, runs, tickers, lookback_days, source_type="news") for path in news_data_paths],
        "macro": [_audit_table(path, runs, tickers, lookback_days, source_type="macro") for path in macro_data_paths],
    }


def _audit_table(path: str | Path, runs: list[dict[str, Any]], tickers: list[str], lookback_days: int, source_type: str) -> dict[str, Any]:
    resolved = Path(path)
    base = {"path": str(resolved), "exists": resolved.exists(), "source_type": source_type, "status": "unreadable"}
    if not resolved.exists():
        return {**base, "error": "missing"}
    try:
        import pandas as pd

        frame = _read_table(pd, resolved)
    except Exception as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"}
    date_col = _first_column(frame, DATE_COLUMNS)
    if date_col is None:
        return {**base, "status": "no_timestamp_column", "row_count": int(len(frame)), "date_column": None}
    working = frame.copy()
    working["_dean_datetime"] = pd.to_datetime(working[date_col], utc=True, errors="coerce")
    working = working.dropna(subset=["_dean_datetime"])
    windows = [_window_audit(working, run, tickers, lookback_days) for run in runs if run.get("as_of")]
    return {
        **base,
        "status": "inspected",
        "row_count": int(len(frame)),
        "timestamped_row_count": int(len(working)),
        "date_column": str(date_col),
        "start": working["_dean_datetime"].min().isoformat() if len(working) else None,
        "end": working["_dean_datetime"].max().isoformat() if len(working) else None,
        "window_count": len(windows),
        "windows_with_rows": sum(1 for window in windows if window["row_count"] > 0),
        "ticker_hits": _ticker_hits(working, tickers),
        "window_sample": windows[:10],
    }


def _window_audit(frame: Any, run: dict[str, Any], tickers: list[str], lookback_days: int) -> dict[str, Any]:
    as_of = _parse_datetime(str(run["as_of"]))
    start = as_of - timedelta(days=lookback_days)
    window = frame[(frame["_dean_datetime"] >= start) & (frame["_dean_datetime"] <= as_of)]
    return {
        "as_of": as_of.isoformat(),
        "lookback_start": start.isoformat(),
        "row_count": int(len(window)),
        "ticker_hits": _ticker_hits(window, tickers),
    }


def _ticker_hits(frame: Any, tickers: list[str]) -> dict[str, int]:
    if frame.empty:
        return dict.fromkeys(tickers, 0)
    text_cols = [column for column in TEXT_COLUMNS if column in frame.columns]
    if not text_cols:
        return dict.fromkeys(tickers, 0)
    combined = frame[text_cols].fillna("").astype(str).apply(lambda row: " ".join(row), axis=1).str.lower()
    return {ticker: int(combined.str.contains(ticker.lower(), regex=False).sum()) for ticker in tickers}


def _tasks(
    weak_runs: list[dict[str, Any]],
    source_audits: dict[str, list[dict[str, Any]]],
    materials_paths: list[str | Path],
    requested_tickers: list[str],
) -> list[dict[str, Any]]:
    if not weak_runs:
        return []
    missing_counter = Counter(ticker for run in weak_runs for ticker in run.get("evidence_missing_tickers", []))
    tasks: list[dict[str, Any]] = []
    tasks.append(
        {
            "task_id": "backfill_historical_news_evidence",
            "priority": "P0",
            "description": "Provide or regenerate timestamped news evidence before each weak replay as_of date.",
            "tickers": sorted(missing_counter) or requested_tickers,
            "reason": "Historical research replay has weak evidence coverage.",
        }
    )
    tasks.append(
        {
            "task_id": "add_long_form_research_materials",
            "priority": "P1",
            "description": "Add filings, transcripts, sector reports, or dated local research materials for missing/inconclusive windows.",
            "existing_material_paths": [str(path) for path in materials_paths],
            "reason": "Research replay is neutral/inconclusive even when price replay forms candidates.",
        }
    )
    if not _has_inspected_rows(source_audits.get("macro", [])):
        tasks.append(
            {
                "task_id": "verify_macro_history",
                "priority": "P1",
                "description": "Provide timestamped macro history for the replay windows or verify why macro rows are absent.",
                "reason": "Macro audit did not find usable rows for weak replay windows.",
            }
        )
    if _source_has_rows_but_low_ticker_hits(source_audits.get("news", []), requested_tickers):
        tasks.append(
            {
                "task_id": "improve_news_ticker_matching",
                "priority": "P1",
                "description": "Review ticker/entity matching; source rows exist but ticker hits are sparse.",
                "reason": "Cached news may exist but not map cleanly to requested tickers.",
            }
        )
    tasks.append(
        {
            "task_id": "rerun_historical_research_replay_after_backfill",
            "priority": "P2",
            "description": "After source backfill, rerun historical research replay batch on the repaired price artifact.",
            "reason": "Calibration readiness must be rechecked after evidence quality improves.",
        }
    )
    return tasks


def _summary(
    readiness: dict[str, Any],
    research_batch: dict[str, Any],
    runs: list[dict[str, Any]],
    weak_runs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    requested_tickers: list[str],
) -> dict[str, Any]:
    missing = sorted({ticker for run in weak_runs for ticker in run.get("evidence_missing_tickers", [])})
    readiness_status = readiness.get("payload", {}).get("summary", {}).get("readiness_status")
    if weak_runs:
        status = "backfill_required"
    elif not research_batch.get("loaded"):
        status = "missing_research_batch"
    else:
        status = "evidence_ready"
    return {
        "backfill_status": status,
        "readiness_status": readiness_status,
        "run_count": len(runs),
        "weak_run_count": len(weak_runs),
        "missing_ticker_count": len(missing),
        "missing_tickers": missing,
        "requested_tickers": requested_tickers,
        "task_count": len(tasks),
        "can_run_calibration": False,
    }


def _coverage_gaps(weak_runs: list[dict[str, Any]], requested_tickers: list[str]) -> dict[str, Any]:
    missing_counter = Counter(ticker for run in weak_runs for ticker in run.get("evidence_missing_tickers", []))
    reason_counter = Counter(reason for run in weak_runs for reason in run.get("backfill_reasons", []))
    return {
        "missing_tickers": sorted(missing_counter),
        "missing_ticker_counts": dict(missing_counter.most_common()),
        "reason_counts": dict(reason_counter.most_common()),
        "weak_dates_sample": [run.get("as_of") for run in weak_runs[:20]],
        "requested_tickers": requested_tickers,
    }


def _commands(payload: dict[str, Any], news_paths: list[str | Path], macro_paths: list[str | Path]) -> dict[str, str | None]:
    inputs = payload.get("inputs", {}) if isinstance(payload.get("inputs"), dict) else {}
    price_path = inputs.get("price_data_path") or "data\\dean_os\\replay_prices\\replay_prices_1d_repaired_20260613_135839.parquet"
    tickers = inputs.get("tickers") or ["AMD", "NVDA", "MSFT", "AAPL", "TSM", "QQQ", "SPY"]
    ticker_args = " ".join(str(ticker) for ticker in tickers)
    news_args = " ".join(str(path) for path in news_paths) if news_paths else "data\\colab\\backup_20260510_153551\\stage2_news_20260505_151233.parquet"
    macro_args = " ".join(str(path) for path in macro_paths) if macro_paths else "data\\colab\\backup_20260510_153551\\stage2_macro_20260507_191104.parquet"
    return {
        "rerun_research_replay_after_backfill": (
            f"python run_agent_historical_research_replay_batch.py {price_path} --tickers {ticker_args} "
            "--start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 "
            "--step-days 14 --lookback-days 180 --horizon-days 30 "
            f"--news-data {news_args} --macro-data {macro_args} "
            "--tags historical_replay ai_cycle repaired_price_artifact evidence_backfilled "
            "--output-dir reports\\dean_os\\historical_research_replay_batch_repaired_after_backfill"
        ),
        "rerun_readiness_gate_after_backfill": (
            "python run_agent_replay_calibration_readiness.py "
            "--replay-batch-json reports\\dean_os\\historical_replay_batch_repaired_expanded\\latest.json "
            "--research-batch-json reports\\dean_os\\historical_research_replay_batch_repaired_after_backfill\\latest.json"
        ),
    }


def _recommendations(summary: dict[str, Any], tasks: list[dict[str, Any]]) -> list[str]:
    if summary["backfill_status"] == "evidence_ready":
        return ["Evidence coverage is ready; rerun replay calibration readiness before creating a manual calibration packet."]
    recommendations = [
        "Do not calibrate analyst weights from weak historical research replay evidence.",
        "Backfill dated historical evidence first, then rerun historical research replay and readiness gate.",
    ]
    if any(task["task_id"] == "improve_news_ticker_matching" for task in tasks):
        recommendations.append("Inspect ticker/entity matching because cached news may exist but fail to map to requested symbols.")
    return recommendations


def _readiness_context(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    gate = payload.get("gate", {}) if isinstance(payload.get("gate"), dict) else {}
    return {
        "readiness_status": summary.get("readiness_status"),
        "next_action": summary.get("next_action"),
        "blocker_count": summary.get("blocker_count"),
        "caution_count": summary.get("caution_count"),
        "passed_checks": gate.get("passed_checks", []),
    }


def _requested_tickers(payload: dict[str, Any], tickers: list[str] | None) -> list[str]:
    if tickers:
        return sorted({str(ticker).upper() for ticker in tickers if str(ticker).strip()})
    inputs = payload.get("inputs", {}) if isinstance(payload.get("inputs"), dict) else {}
    return sorted({str(ticker).upper() for ticker in inputs.get("tickers", []) if str(ticker).strip()})


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    if path is None:
        return {"path": None, "loaded": False, "payload": {}, "error": "not_provided"}
    try:
        payload = DeanPaths.load_json(path)
        return {"path": str(path), "loaded": True, "payload": payload if isinstance(payload, dict) else {"items": payload}}
    except Exception as exc:
        return {"path": str(path), "loaded": False, "payload": {}, "error": str(exc)}


def _read_table(pd: Any, path: Path) -> Any:
    from dean_os.dean_paths import DeanPaths

    try:
        return DeanPaths.load_data_file(path)
    except Exception as exc:
        raise ValueError(f"Failed to load table from {path}: {exc}")


def _first_column(frame: Any, candidates: tuple[str, ...]) -> str | None:
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _has_inspected_rows(audits: list[dict[str, Any]]) -> bool:
    return any(audit.get("status") == "inspected" and audit.get("windows_with_rows", 0) > 0 for audit in audits)


def _source_has_rows_but_low_ticker_hits(audits: list[dict[str, Any]], tickers: list[str]) -> bool:
    for audit in audits:
        if audit.get("status") != "inspected" or audit.get("windows_with_rows", 0) <= 0:
            continue
        hits = audit.get("ticker_hits", {})
        if tickers and sum(int(hits.get(ticker, 0)) for ticker in tickers) == 0:
            return True
    return False


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
