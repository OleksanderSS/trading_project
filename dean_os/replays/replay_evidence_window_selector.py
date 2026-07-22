from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.market_data_api import prepare_market_frame, read_market_frame
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_PRICE_ARTIFACT = "data/dean_os/replay_prices/replay_prices_1d_repaired_20260613_135839.parquet"
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


class ReplayEvidenceWindowSelector:
    """Selects replay windows where price outcomes and pre-as_of evidence overlap."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/replay_evidence_window_selector"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        price_data_path: str | Path = DEFAULT_PRICE_ARTIFACT,
        news_data_paths: list[str | Path] | None = None,
        macro_data_paths: list[str | Path] | None = None,
        materials_paths: list[str | Path] | None = None,
        tickers: list[str] | None = None,
        lookback_days: int = 180,
        horizon_days: list[int] | None = None,
        step_days: int = 7,
        start_as_of: str | None = None,
        end_as_of: str | None = None,
        min_evidence_rows: int = 1,
        min_source_count: int = 1,
        price_tolerance_days: int = 3,
        max_candidate_dates: int = 50,
        save: bool = True,
    ) -> dict[str, Any]:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(f"pandas is required for replay evidence window selection: {exc}") from exc

        horizons = _normalize_horizons(horizon_days or [30])
        price_path = Path(price_data_path)
        price_frame = _load_price_frame(pd, price_path)
        requested_tickers = _requested_tickers(price_frame, tickers)
        sources = _load_sources(
            pd=pd,
            news_data_paths=news_data_paths or [],
            macro_data_paths=macro_data_paths or [],
            materials_paths=materials_paths or [],
        )
        candidates = _candidate_windows(
            price_frame=price_frame,
            sources=sources,
            tickers=requested_tickers,
            lookback_days=lookback_days,
            horizons=horizons,
            step_days=step_days,
            start_as_of=start_as_of,
            end_as_of=end_as_of,
            min_evidence_rows=min_evidence_rows,
            min_source_count=min_source_count,
            price_tolerance_days=price_tolerance_days,
            max_candidate_dates=max_candidate_dates,
        )
        eligible = [item for item in candidates if item["eligible"]]
        summary = _summary(price_frame, sources, candidates, eligible)
        payload = {
            "run_id": _run_id("replay_evidence_window_selector"),
            "created_at": utc_now_iso(),
            "mode": "replay_evidence_window_selector",
            "inputs": {
                "price_data_path": str(price_path),
                "news_data_paths": [str(path) for path in news_data_paths or []],
                "macro_data_paths": [str(path) for path in macro_data_paths or []],
                "materials_paths": [str(path) for path in materials_paths or []],
                "tickers": requested_tickers,
                "lookback_days": lookback_days,
                "horizon_days": horizons,
                "step_days": step_days,
                "start_as_of": start_as_of,
                "end_as_of": end_as_of,
                "min_evidence_rows": min_evidence_rows,
                "min_source_count": min_source_count,
                "price_tolerance_days": price_tolerance_days,
                "max_candidate_dates": max_candidate_dates,
            },
            "summary": summary,
            "price_coverage": _price_coverage(price_frame, requested_tickers),
            "source_coverage": _source_coverage(sources),
            "eligible_windows": eligible,
            "rejected_windows_sample": [item for item in candidates if not item["eligible"]][:25],
            "commands": _commands(
                price_path=price_path,
                tickers=requested_tickers,
                eligible=eligible,
                horizons=horizons,
                lookback_days=lookback_days,
                news_paths=news_data_paths or [],
                macro_paths=macro_data_paths or [],
                materials_paths=materials_paths or [],
            ),
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
            "recommendations": _recommendations(summary),
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
        rendered_md = render_replay_evidence_window_selector_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_replay_evidence_window_selector_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Replay Evidence Window Selector",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('selection_status')}`",
        f"- Candidate windows: {summary.get('candidate_window_count')}",
        f"- Eligible windows: {summary.get('eligible_window_count')}",
        f"- Recommended start: `{summary.get('recommended_start_as_of')}`",
        f"- Recommended end: `{summary.get('recommended_end_as_of')}`",
        "",
        "## Eligible Windows",
        "",
    ]
    for item in payload.get("eligible_windows", [])[:15]:
        lines.append(
            f"- as_of=`{item.get('as_of')}` horizon={item.get('horizon_days')} "
            f"evidence_rows={item.get('evidence_rows')} source_count={item.get('source_count')}"
        )
    lines.extend(["", "## Commands", ""])
    for key, command in payload.get("commands", {}).items():
        if command:
            lines.append(f"- {key}: `{command}`")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _load_price_frame(pd: Any, path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Price artifact does not exist: {path}")
    return prepare_market_frame(pd, read_market_frame(pd, path), close_col="close", datetime_col="datetime")


def _load_sources(
    pd: Any,
    news_data_paths: list[str | Path],
    macro_data_paths: list[str | Path],
    materials_paths: list[str | Path],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for source_type, paths in [("news", news_data_paths), ("macro", macro_data_paths)]:
        for path in paths:
            sources.append(_load_table_source(pd, Path(path), source_type=source_type))
    for path in materials_paths:
        sources.append(_load_material_source(Path(path)))
    return sources


def _load_table_source(pd: Any, path: Path, source_type: str) -> dict[str, Any]:
    base = {"source_type": source_type, "path": str(path), "exists": path.exists(), "loaded": False, "rows": None}
    if not path.exists():
        return {**base, "status": "missing"}
    try:
        frame = _read_table(pd, path)
    except Exception as exc:
        return {**base, "status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
    date_col = _first_column(frame, DATE_COLUMNS)
    if date_col is None:
        return {**base, "status": "no_timestamp_column", "row_count": int(len(frame))}
    working = frame.copy()
    working["_dean_datetime"] = pd.to_datetime(working[date_col], utc=True, errors="coerce")
    working = working.dropna(subset=["_dean_datetime"])
    return {
        **base,
        "loaded": True,
        "status": "loaded",
        "row_count": int(len(frame)),
        "timestamped_row_count": int(len(working)),
        "date_column": str(date_col),
        "start": working["_dean_datetime"].min().to_pydatetime() if len(working) else None,
        "end": working["_dean_datetime"].max().to_pydatetime() if len(working) else None,
        "rows": working,
    }


def _load_material_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"source_type": "materials", "path": str(path), "exists": False, "loaded": False, "status": "missing"}
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    datetimes = [datetime.fromtimestamp(item.stat().st_mtime, tz=UTC) for item in files]
    return {
        "source_type": "materials",
        "path": str(path),
        "exists": True,
        "loaded": True,
        "status": "loaded_metadata_only",
        "row_count": len(files),
        "timestamped_row_count": len(datetimes),
        "start": min(datetimes) if datetimes else None,
        "end": max(datetimes) if datetimes else None,
        "rows": None,
    }


def _candidate_windows(
    price_frame: Any,
    sources: list[dict[str, Any]],
    tickers: list[str],
    lookback_days: int,
    horizons: list[int],
    step_days: int,
    start_as_of: str | None,
    end_as_of: str | None,
    min_evidence_rows: int,
    min_source_count: int,
    price_tolerance_days: int,
    max_candidate_dates: int,
) -> list[dict[str, Any]]:
    price_start = _to_datetime(price_frame["_dean_datetime"].min())
    price_end = _to_datetime(price_frame["_dean_datetime"].max())
    source_start = _earliest_source_start(sources)
    start = parse_datetime(start_as_of) if start_as_of else max(price_start + timedelta(days=lookback_days), source_start or price_start)
    end_limit = price_end - timedelta(days=min(horizons))
    end = parse_datetime(end_as_of) if end_as_of else end_limit
    if start > end:
        return []
    current = _midnight(start)
    end = _midnight(end)
    step = timedelta(days=max(int(step_days), 1))
    candidates: list[dict[str, Any]] = []
    while current <= end and len({item["as_of"] for item in candidates}) < max_candidate_dates:
        for horizon in horizons:
            candidates.append(
                _evaluate_candidate(
                    as_of=current,
                    horizon_days=horizon,
                    price_frame=price_frame,
                    sources=sources,
                    tickers=tickers,
                    lookback_days=lookback_days,
                    min_evidence_rows=min_evidence_rows,
                    min_source_count=min_source_count,
                    price_tolerance_days=price_tolerance_days,
                )
            )
        current += step
    return candidates


def _evaluate_candidate(
    as_of: datetime,
    horizon_days: int,
    price_frame: Any,
    sources: list[dict[str, Any]],
    tickers: list[str],
    lookback_days: int,
    min_evidence_rows: int,
    min_source_count: int,
    price_tolerance_days: int,
) -> dict[str, Any]:
    target_at = as_of + timedelta(days=horizon_days)
    evidence = _evidence_window(sources, as_of, lookback_days, tickers)
    price = _price_window(price_frame, as_of, target_at, tickers, price_tolerance_days)
    blockers: list[str] = []
    if evidence["evidence_rows"] < min_evidence_rows:
        blockers.append("not_enough_evidence_rows")
    if evidence["source_count"] < min_source_count:
        blockers.append("not_enough_sources")
    if price["missing_tickers"]:
        blockers.append("missing_price_tickers")
    eligible = not blockers
    return {
        "as_of": as_of.isoformat(),
        "horizon_days": horizon_days,
        "target_at": target_at.isoformat(),
        "eligible": eligible,
        "blockers": blockers,
        **evidence,
        **price,
    }


def _evidence_window(sources: list[dict[str, Any]], as_of: datetime, lookback_days: int, tickers: list[str]) -> dict[str, Any]:
    start = as_of - timedelta(days=lookback_days)
    evidence_rows = 0
    source_count = 0
    by_source: list[dict[str, Any]] = []
    ticker_hits = dict.fromkeys(tickers, 0)
    for source in sources:
        rows = source.get("rows")
        row_count = 0
        hits = dict.fromkeys(tickers, 0)
        if rows is not None:
            window = rows[(rows["_dean_datetime"] >= start) & (rows["_dean_datetime"] <= as_of)]
            row_count = int(len(window))
            hits = _ticker_hits(window, tickers)
        elif source.get("loaded") and source.get("start") and source.get("end"):
            if source["start"] <= as_of and source["end"] >= start:
                row_count = int(source.get("row_count") or 0)
        if row_count > 0:
            source_count += 1
        evidence_rows += row_count
        for ticker, count in hits.items():
            ticker_hits[ticker] = ticker_hits.get(ticker, 0) + count
        by_source.append({"source_type": source.get("source_type"), "path": source.get("path"), "row_count": row_count, "ticker_hits": hits})
    return {
        "evidence_rows": evidence_rows,
        "source_count": source_count,
        "ticker_hits": ticker_hits,
        "evidence_by_source": by_source,
    }


def _price_window(price_frame: Any, as_of: datetime, target_at: datetime, tickers: list[str], tolerance_days: int) -> dict[str, Any]:
    missing: list[str] = []
    per_ticker: dict[str, Any] = {}
    latest_required = target_at - timedelta(days=max(tolerance_days, 0))
    for ticker in tickers:
        frame = price_frame[price_frame["_dean_ticker"] == ticker]
        lookback = frame[frame["_dean_datetime"] <= as_of]
        future = frame[frame["_dean_datetime"] >= as_of]
        future_to_target = frame[(frame["_dean_datetime"] >= as_of) & (frame["_dean_datetime"] <= target_at)]
        status = "ok"
        if lookback.empty:
            status = "missing_lookback_price"
        elif future.empty:
            status = "missing_future_price"
        elif future_to_target.empty or future_to_target["_dean_datetime"].max().to_pydatetime() < latest_required:
            status = "future_price_horizon_too_short"
        if status != "ok":
            missing.append(ticker)
        per_ticker[ticker] = {
            "status": status,
            "snapshot_rows": int(len(lookback)),
            "future_rows": int(len(future_to_target)),
            "future_end": future_to_target["_dean_datetime"].max().isoformat() if len(future_to_target) else None,
        }
    return {"missing_tickers": missing, "price_by_ticker": per_ticker}


def _commands(
    price_path: Path,
    tickers: list[str],
    eligible: list[dict[str, Any]],
    horizons: list[int],
    lookback_days: int,
    news_paths: list[str | Path],
    macro_paths: list[str | Path],
    materials_paths: list[str | Path],
) -> dict[str, str | None]:
    if not eligible:
        return {"historical_replay_batch": None, "historical_research_replay_batch": None}
    dates = sorted({item["as_of"] for item in eligible})
    horizon_values = sorted({int(item["horizon_days"]) for item in eligible} or set(horizons))
    date_args = " ".join(dates[:25])
    ticker_args = " ".join(tickers)
    horizon_args = " ".join(str(item) for item in horizon_values)
    news_args = " ".join(str(path) for path in news_paths)
    macro_args = " ".join(str(path) for path in macro_paths)
    materials_args = " ".join(str(path) for path in materials_paths)
    common = f"{price_path} --tickers {ticker_args} --as-of {date_args} --lookback-days {lookback_days} --horizon-days {horizon_args}"
    news_macro = ""
    if news_args:
        news_macro += f" --news-data {news_args}"
    if macro_args:
        news_macro += f" --macro-data {macro_args}"
    research_sources = news_macro
    if materials_args:
        research_sources += f" --materials {materials_args}"
    return {
        "historical_replay_batch": (
            f"python run_agent_historical_replay_batch.py {common}{news_macro} "
            "--output-dir reports\\dean_os\\historical_replay_batch_evidence_window_selected"
        ),
        "historical_research_replay_batch": (
            f"python run_agent_historical_research_replay_batch.py {common}{research_sources} "
            "--tags historical_replay ai_cycle repaired_price_artifact evidence_window_selected "
            "--output-dir reports\\dean_os\\historical_research_replay_batch_evidence_window_selected"
        ),
    }


def _summary(price_frame: Any, sources: list[dict[str, Any]], candidates: list[dict[str, Any]], eligible: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selection_status": "windows_ready" if eligible else "no_eligible_windows",
        "candidate_window_count": len(candidates),
        "eligible_window_count": len(eligible),
        "recommended_start_as_of": eligible[0]["as_of"] if eligible else None,
        "recommended_end_as_of": eligible[-1]["as_of"] if eligible else None,
        "price_start": price_frame["_dean_datetime"].min().isoformat() if not price_frame.empty else None,
        "price_end": price_frame["_dean_datetime"].max().isoformat() if not price_frame.empty else None,
        "loaded_source_count": sum(1 for source in sources if source.get("loaded")),
        "source_status_counts": _counts(source.get("status") for source in sources),
    }


def _price_coverage(frame: Any, tickers: list[str]) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "ticker_count": int(frame["_dean_ticker"].nunique()) if not frame.empty else 0,
        "start": frame["_dean_datetime"].min().isoformat() if not frame.empty else None,
        "end": frame["_dean_datetime"].max().isoformat() if not frame.empty else None,
        "per_ticker": {
            ticker: {
                "rows": int(len(group)),
                "start": group["_dean_datetime"].min().isoformat() if len(group) else None,
                "end": group["_dean_datetime"].max().isoformat() if len(group) else None,
            }
            for ticker, group in ((ticker, frame[frame["_dean_ticker"] == ticker]) for ticker in tickers)
        },
    }


def _source_coverage(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source in sources:
        result.append(
            {
                "source_type": source.get("source_type"),
                "path": source.get("path"),
                "status": source.get("status"),
                "row_count": source.get("row_count"),
                "timestamped_row_count": source.get("timestamped_row_count"),
                "date_column": source.get("date_column"),
                "start": source.get("start").isoformat() if source.get("start") else None,
                "end": source.get("end").isoformat() if source.get("end") else None,
            }
        )
    return result


def _recommendations(summary: dict[str, Any]) -> list[str]:
    if summary["selection_status"] == "windows_ready":
        return [
            "Run the selected historical research replay batch before analyst calibration.",
            "If research evidence remains weak inside selected windows, provide richer dated materials rather than expanding earlier.",
        ]
    return [
        "No eligible overlap between evidence and future price windows was found.",
        "Provide older historical evidence or use shorter horizons / later as_of dates if future prices permit.",
    ]


def _read_table(pd: Any, path: Path) -> Any:
    from dean_os.dean_paths import DeanPaths

    try:
        return DeanPaths.load_data_file(path)
    except Exception as exc:
        raise ValueError(f"Failed to load table from {path}: {exc}")


def _ticker_hits(frame: Any, tickers: list[str]) -> dict[str, int]:
    if frame.empty:
        return dict.fromkeys(tickers, 0)
    text_cols = [column for column in TEXT_COLUMNS if column in frame.columns]
    if not text_cols:
        return dict.fromkeys(tickers, 0)
    combined = frame[text_cols].fillna("").astype(str).apply(lambda row: " ".join(row), axis=1).str.lower()
    return {ticker: int(combined.str.contains(ticker.lower(), regex=False).sum()) for ticker in tickers}


def _first_column(frame: Any, candidates: tuple[str, ...]) -> str | None:
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _requested_tickers(frame: Any, tickers: list[str] | None) -> list[str]:
    if tickers:
        return sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})
    return sorted(str(ticker).upper() for ticker in frame["_dean_ticker"].dropna().unique() if str(ticker).strip())


def _normalize_horizons(values: list[int]) -> list[int]:
    horizons = sorted({int(value) for value in values if int(value) > 0})
    if not horizons:
        raise ValueError("At least one positive horizon is required.")
    return horizons


def _earliest_source_start(sources: list[dict[str, Any]]) -> datetime | None:
    starts = [source["start"] for source in sources if source.get("loaded") and source.get("start")]
    return min(starts) if starts else None


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _to_datetime(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().astimezone(UTC)
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    return parse_datetime(str(value))


def _midnight(value: datetime) -> datetime:
    value = value.astimezone(UTC)
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
