# ARCHIVED (found during dean_os/agents audit, 2026-07-24): never
# instantiated anywhere except this file's own class definition -- absent
# from dean_os/config/agent_registry.yaml (the orchestrator's only agent
# source) and from every test in tests/dean_os/. Unlike MarketDataFreshnessAgent
# (also absent from the registry but wired directly in dean_os/paper_autonomy.py),
# no live call site instantiates CollectorHealthAgent at all. Archived with
# its sibling CollectorInventoryAgent rather than deleted, per this project's
# convention of keeping dead code inspectable.
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport


class CollectorHealthAgent(BaseAgent):
    """Checks whether one collector output is structurally usable in isolation."""

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        collector_type = str(self.config.get("collector_type", "rss"))
        output_path = self.config.get("output_path")
        expected_source_count = int(self.config.get("expected_source_count", 1))
        min_rows = int(self.config.get("min_rows", 1))
        max_duplicate_ratio = float(self.config.get("max_duplicate_ratio", 0.1))
        as_of = _parse_datetime(self.config.get("as_of")) if self.config.get("as_of") else datetime.now(UTC)

        metrics = inspect_collector_health(
            output_path=output_path,
            collector_type=collector_type,
            expected_source_count=expected_source_count,
            min_rows=min_rows,
            max_duplicate_ratio=max_duplicate_ratio,
            as_of=as_of,
        )
        context.metadata.setdefault("collector_health", {})[collector_type] = metrics

        verdict = metrics["verdict"]
        if verdict == "blocked":
            reasons = metrics["reasons"] or [f"{collector_type} collector output is not structurally usable."]
            risks = [
                "Collector output cannot be trusted for downstream news/sentiment alignment until schema, timestamps, and deduplication pass."
            ]
            signal_strength = -0.8
            data_quality_score = 0.15
        elif verdict == "caution":
            reasons = metrics["reasons"] or [f"{collector_type} collector output is usable but incomplete."]
            risks = [
                "Collector output may be usable for review, but downstream ingestion should stay gated until warnings clear."
            ]
            signal_strength = -0.2
            data_quality_score = 0.55
        else:
            reasons = metrics["reasons"] or [f"{collector_type} collector output passed isolated health checks."]
            risks = []
            signal_strength = 0.2
            data_quality_score = 0.9

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.9 if verdict == "clear" else 0.8,
            data_quality_score=data_quality_score,
            signal_strength=signal_strength,
            reasons=reasons,
            risks=risks,
            blind_spots=[
                "This health check validates local output shape and timing only; it does not confirm network reachability, API quota, or live source freshness."
            ],
            evidence=[
                self.evidence("file", str(metrics.get("output_path")), "output_path", metrics.get("output_path")),
                self.evidence("metric", "collector_health", "collector_type", collector_type),
                self.evidence("metric", "collector_health", "row_count", metrics.get("row_count", 0)),
                self.evidence("metric", "collector_health", "duplicate_ratio", metrics.get("duplicate_ratio")),
                self.evidence("metric", "collector_health", "raw_data_news_compatible", metrics.get("raw_data_news_compatible")),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot=metrics,
        )


def inspect_collector_health(
    output_path: str | Path | None,
    collector_type: str = "rss",
    expected_source_count: int = 1,
    min_rows: int = 1,
    max_duplicate_ratio: float = 0.1,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    as_of = as_of or datetime.now(UTC)
    path = _resolve_output_path(output_path)
    if path is None:
        return {
            "collector_type": collector_type,
            "status": "unavailable",
            "verdict": "blocked",
            "output_path": None,
            "reasons": ["No collector output path was provided."],
            "warnings": [],
            "row_count": 0,
            "duplicate_ratio": 1.0,
            "raw_data_news_compatible": False,
            "expected_source_count": expected_source_count,
            "min_rows": min_rows,
            "max_duplicate_ratio": max_duplicate_ratio,
            "as_of": as_of.isoformat(),
        }
    if not path.exists():
        return {
            "collector_type": collector_type,
            "status": "unavailable",
            "verdict": "blocked",
            "output_path": str(path),
            "reasons": [f"Collector output does not exist: {path}"],
            "warnings": [],
            "row_count": 0,
            "duplicate_ratio": 1.0,
            "raw_data_news_compatible": False,
            "expected_source_count": expected_source_count,
            "min_rows": min_rows,
            "max_duplicate_ratio": max_duplicate_ratio,
            "as_of": as_of.isoformat(),
        }

    try:
        payload = _read_payload(path)
        frame, raw_news = _extract_news_frame(payload)
        prepared = _prepare_news_frame(frame)
    except Exception as exc:
        return {
            "collector_type": collector_type,
            "status": "unavailable",
            "verdict": "blocked",
            "output_path": str(path),
            "reasons": [f"Could not inspect collector output: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "row_count": 0,
            "duplicate_ratio": 1.0,
            "raw_data_news_compatible": False,
            "expected_source_count": expected_source_count,
            "min_rows": min_rows,
            "max_duplicate_ratio": max_duplicate_ratio,
            "as_of": as_of.isoformat(),
        }

    row_count = int(len(prepared))
    source_count = _source_count(prepared)
    duplicate_count = _duplicate_count(prepared)
    duplicate_ratio = (duplicate_count / row_count) if row_count else 1.0
    latest_timestamp = _latest_timestamp(prepared)
    warnings: list[str] = []
    reasons: list[str] = []

    if row_count < min_rows:
        reasons.append(f"Only {row_count} usable news rows found; expected at least {min_rows}.")
    if source_count < expected_source_count:
        reasons.append(f"Only {source_count} source/feed values found; expected at least {expected_source_count}.")
    if duplicate_ratio > max_duplicate_ratio:
        reasons.append(
            f"Duplicate ratio {duplicate_ratio:.3f} exceeds threshold {max_duplicate_ratio:.3f}."
        )
    if latest_timestamp is None:
        reasons.append("No valid published timestamp could be parsed.")
    else:
        age_hours = (as_of - latest_timestamp).total_seconds() / 3600
        if age_hours < 0:
            warnings.append("Latest timestamp is in the future relative to the evaluation clock.")

    required_columns = _required_columns(prepared)
    missing_columns = [column for column in required_columns if column not in prepared.columns]
    if missing_columns:
        reasons.append(f"Missing required columns: {', '.join(missing_columns)}")

    raw_data_news_compatible = bool(raw_news) or _frame_to_news_records(prepared) is not None
    if not raw_data_news_compatible:
        reasons.append('Collector output cannot be normalized into raw_data["news"].')

    if reasons:
        verdict = "blocked" if row_count < min_rows or not raw_data_news_compatible else "caution"
    else:
        verdict = "clear"

    return {
        "collector_type": collector_type,
        "status": "ok" if verdict != "blocked" else "degraded",
        "verdict": verdict,
        "output_path": str(path),
        "row_count": row_count,
        "source_count": source_count,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "latest_timestamp": latest_timestamp.isoformat() if latest_timestamp else None,
        "required_columns": required_columns,
        "missing_columns": missing_columns,
        "warnings": warnings,
        "reasons": reasons,
        "raw_data_news_compatible": raw_data_news_compatible,
        "raw_data": {"news": _frame_to_news_records(prepared) if raw_data_news_compatible else []},
        "expected_source_count": expected_source_count,
        "min_rows": min_rows,
        "max_duplicate_ratio": max_duplicate_ratio,
        "as_of": as_of.isoformat(),
    }


def _resolve_output_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    return Path(raw_path)


def _read_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        return pd.read_parquet(path)
    raise ValueError(f"Unsupported collector output type: {path.suffix}")


def _extract_news_frame(payload: Any) -> tuple[Any, list[dict[str, Any]]]:
    if hasattr(payload, "columns"):
        return payload, []
    if isinstance(payload, dict):
        raw_news = payload.get("raw_data", {}).get("news", [])
        if isinstance(raw_news, list):
            try:
                import pandas as pd

                return pd.DataFrame(raw_news), raw_news
            except Exception:
                return raw_news, raw_news
        if "news" in payload and isinstance(payload["news"], list):
            try:
                import pandas as pd

                return pd.DataFrame(payload["news"]), payload["news"]
            except Exception:
                return payload["news"], payload["news"]
    if isinstance(payload, list):
        try:
            import pandas as pd

            return pd.DataFrame(payload), payload
        except Exception:
            return payload, []
    raise ValueError("Collector output did not contain news rows.")


def _prepare_news_frame(frame: Any) -> Any:
    if not hasattr(frame, "columns"):
        raise ValueError("Collector output is not tabular.")
    required = _required_columns(frame)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    prepared = frame.copy()
    prepared["_dean_title"] = prepared[_resolve_column(prepared, "title")].astype(str)
    prepared["_dean_url"] = prepared[_resolve_column(prepared, "url")].astype(str)
    prepared["_dean_source"] = prepared[_resolve_column(prepared, "source")].astype(str)
    prepared["_dean_published_at"] = _parse_series_datetime(prepared, _resolve_column(prepared, "published_at"))
    prepared = prepared.dropna(subset=["_dean_title", "_dean_url", "_dean_source", "_dean_published_at"])
    if prepared.empty:
        raise ValueError("No usable rows after parsing title/url/source/published_at.")
    return prepared


def _required_columns(frame: Any) -> list[str]:
    columns = [str(column).lower() for column in getattr(frame, "columns", [])]
    candidates = {
        "title": ("title", "headline"),
        "url": ("url", "link", "uri"),
        "published_at": ("published_at", "publishedat", "published", "date"),
        "source": ("source", "feed", "source_name"),
    }
    resolved: list[str] = []
    for canonical, options in candidates.items():
        for option in options:
            if option in columns:
                resolved.append(canonical)
                break
    return resolved


def _resolve_column(frame: Any, requested: str) -> str:
    if requested in frame.columns:
        return requested
    lowered = {str(column).lower(): column for column in frame.columns}
    return lowered.get(requested.lower(), requested)


def _parse_series_datetime(frame: Any, column: str):
    import pandas as pd

    return pd.to_datetime(frame[column], utc=True, errors="coerce")


def _source_count(frame: Any) -> int:
    source_column = _resolve_column(frame, "source")
    if source_column not in frame.columns:
        return 0
    return int(frame[source_column].astype(str).nunique())


def _duplicate_count(frame: Any) -> int:
    keys: list[str] = []
    for candidate in ("_dean_url", "_dean_title", "_dean_published_at"):
        if candidate in frame.columns:
            keys.append(candidate)
    if not keys:
        return 0
    duplicated = frame.duplicated(subset=keys, keep="first")
    return int(duplicated.sum())


def _latest_timestamp(frame: Any) -> datetime | None:
    if "_dean_published_at" not in frame.columns:
        return None
    series = frame["_dean_published_at"].dropna()
    if series.empty:
        return None
    return series.max().to_pydatetime()


def _frame_to_news_records(frame: Any) -> list[dict[str, Any]] | None:
    if not hasattr(frame, "iterrows"):
        return None
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        records.append(
            {
                "title": str(row.get(_resolve_column(frame, "title"), "")),
                "url": str(row.get(_resolve_column(frame, "url"), "")),
                "source": str(row.get(_resolve_column(frame, "source"), "")),
                "published_at": row.get(_resolve_column(frame, "published_at")),
            }
        )
    return records


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
