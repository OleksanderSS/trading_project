from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

TIMESTAMP_NAME_TOKENS = (
    "published",
    "timestamp",
    "datetime",
    "date",
    "created_at",
    "updated_at",
    "time",
)

PREFERRED_TIMESTAMP_COLUMNS = (
    "published_at",
    "timestamp",
    "datetime",
    "date",
    "created_at",
    "updated_at",
)


class EvidenceTimestampAudit:
    """Read-only audit for historical-replay source timestamps."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/evidence_timestamp_audit"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        source_paths: list[str | Path] | None = None,
        news_data_paths: list[str | Path] | None = None,
        macro_data_paths: list[str | Path] | None = None,
        evidence_pack_path: str | Path | None = None,
        as_of: str | None = None,
        start_at: str | None = None,
        min_parse_rate: float = 0.75,
        collapse_share_threshold: float = 0.95,
        collapse_min_rows: int = 10,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = _parse_optional_datetime(as_of)
        start_dt = _parse_optional_datetime(start_at)
        sources = [
            *_source_specs("source", source_paths or []),
            *_source_specs("news", news_data_paths or []),
            *_source_specs("macro", macro_data_paths or []),
        ]
        source_audits = [
            _audit_source(
                source=source,
                as_of=as_of_dt,
                start_at=start_dt,
                min_parse_rate=min_parse_rate,
                collapse_share_threshold=collapse_share_threshold,
                collapse_min_rows=collapse_min_rows,
            )
            for source in sources
        ]
        evidence_pack_audit = _audit_evidence_pack(
            evidence_pack_path=evidence_pack_path,
            as_of=as_of_dt,
            source_audits=source_audits,
            collapse_share_threshold=collapse_share_threshold,
            collapse_min_rows=collapse_min_rows,
        )
        summary = _summary(source_audits, evidence_pack_audit)
        payload = {
            "run_id": _run_id("evidence_timestamp_audit"),
            "created_at": utc_now_iso(),
            "mode": "evidence_timestamp_audit",
            "inputs": {
                "source_paths": [str(path) for path in source_paths or []],
                "news_data_paths": [str(path) for path in news_data_paths or []],
                "macro_data_paths": [str(path) for path in macro_data_paths or []],
                "evidence_pack_path": str(evidence_pack_path) if evidence_pack_path else None,
                "as_of": as_of_dt.isoformat() if as_of_dt else None,
                "start_at": start_dt.isoformat() if start_dt else None,
                "min_parse_rate": min_parse_rate,
                "collapse_share_threshold": collapse_share_threshold,
                "collapse_min_rows": collapse_min_rows,
            },
            "summary": summary,
            "source_audits": source_audits,
            "evidence_pack_audit": evidence_pack_audit,
            "safety": {
                "read_only": True,
                "collector_run_performed": False,
                "network_access_performed": False,
                "pipeline_run_performed": False,
                "learning_write_performed": False,
                "operation_proposal_created": False,
                "config_write_performed": False,
                "broker_access_performed": False,
            },
            "recommendations": _recommendations(summary, source_audits, evidence_pack_audit),
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
        rendered_md = render_evidence_timestamp_audit_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_evidence_timestamp_audit_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Evidence Timestamp Audit",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Audit status: `{summary.get('audit_status')}`",
        f"- Source count: {summary.get('source_count')}",
        f"- Ready: {summary.get('ready_count')}",
        f"- Suspicious: {summary.get('suspicious_count')}",
        f"- Blocked: {summary.get('blocked_count')}",
        f"- Evidence pack status: `{summary.get('evidence_pack_status')}`",
        "",
        "## Sources",
        "",
    ]
    for source in payload.get("source_audits", []):
        primary = source.get("primary_timestamp") or {}
        lines.append(
            f"- `{source.get('source_kind')}` `{source.get('path')}` status=`{source.get('status')}` "
            f"rows={source.get('row_count')} primary=`{primary.get('column')}` "
            f"range=`{primary.get('min')}` -> `{primary.get('max')}`"
        )
        for issue in source.get("issues", [])[:5]:
            lines.append(f"- Issue: {issue}")
        for warning in source.get("warnings", [])[:5]:
            lines.append(f"- Warning: {warning}")
        for note in source.get("notes", [])[:3]:
            lines.append(f"- Note: {note}")
    pack = payload.get("evidence_pack_audit", {})
    if pack.get("status") != "not_provided":
        lines.extend(
            [
                "",
                "## Evidence Pack",
                "",
                f"- Status: `{pack.get('status')}`",
                f"- Document count: {pack.get('document_count')}",
                f"- Date range: `{pack.get('date_range', {}).get('start')}` -> `{pack.get('date_range', {}).get('end')}`",
            ]
        )
        for issue in pack.get("issues", [])[:5]:
            lines.append(f"- Issue: {issue}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _audit_source(
    source: dict[str, Any],
    as_of: datetime | None,
    start_at: datetime | None,
    min_parse_rate: float,
    collapse_share_threshold: float,
    collapse_min_rows: int,
) -> dict[str, Any]:
    path = Path(source["path"])
    base = {
        "source_kind": source["source_kind"],
        "path": str(path),
        "exists": path.exists(),
        "status": "timestamp_blocked",
        "row_count": 0,
        "columns": [],
        "candidate_columns": [],
        "primary_timestamp": None,
        "issues": [],
        "warnings": [],
        "notes": [],
    }
    if not path.exists():
        base["issues"].append("Source file does not exist.")
        return base
    try:
        frame = _read_frame(path)
    except Exception as exc:
        base["issues"].append(f"Could not read source table: {type(exc).__name__}: {exc}")
        return base
    base["row_count"] = int(len(frame))
    base["columns"] = [str(column) for column in frame.columns]
    if frame.empty:
        base["issues"].append("Source table is empty.")
        return base

    candidates = _candidate_columns(frame)
    audits = [_audit_column(frame, column, as_of=as_of, start_at=start_at) for column in candidates]
    audits = [audit for audit in audits if audit["parsed_count"] > 0]
    base["candidate_columns"] = audits
    if not audits:
        base["issues"].append("No usable timestamp column was found.")
        return base

    primary = _select_primary_timestamp(audits)
    base["primary_timestamp"] = primary
    issues, warnings, notes = _classify_timestamp_issues(
        primary=primary,
        row_count=int(len(frame)),
        as_of=as_of,
        min_parse_rate=min_parse_rate,
        collapse_share_threshold=collapse_share_threshold,
        collapse_min_rows=collapse_min_rows,
    )
    base["issues"] = issues
    base["warnings"] = warnings
    base["notes"] = notes
    if issues:
        base["status"] = "timestamp_blocked"
    elif warnings:
        base["status"] = "timestamp_suspicious"
    else:
        base["status"] = "timestamp_ready"
    return base


def _audit_column(frame: Any, column: str, as_of: datetime | None, start_at: datetime | None) -> dict[str, Any]:
    import pandas as pd

    raw = frame[column]
    non_null = int(raw.notna().sum())
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    parsed_valid = parsed.dropna()
    parsed_count = int(parsed_valid.shape[0])
    row_count = int(len(frame))
    date_strings = parsed_valid.dt.date.astype(str) if parsed_count else []
    date_counts = Counter(date_strings)
    most_common_date, most_common_count = date_counts.most_common(1)[0] if date_counts else (None, 0)
    min_dt = parsed_valid.min().to_pydatetime() if parsed_count else None
    max_dt = parsed_valid.max().to_pydatetime() if parsed_count else None
    return {
        "column": str(column),
        "priority": _column_priority(str(column)),
        "non_null_count": non_null,
        "parsed_count": parsed_count,
        "row_count": row_count,
        "parse_rate": round(parsed_count / row_count, 6) if row_count else 0.0,
        "null_rate": round(1.0 - (non_null / row_count), 6) if row_count else 1.0,
        "min": min_dt.isoformat() if min_dt else None,
        "max": max_dt.isoformat() if max_dt else None,
        "unique_date_count": len(date_counts),
        "most_common_date": most_common_date,
        "most_common_date_count": int(most_common_count),
        "most_common_date_share": round(most_common_count / parsed_count, 6) if parsed_count else 0.0,
        "after_as_of_count": int((parsed_valid > as_of).sum()) if as_of and parsed_count else 0,
        "before_start_count": int((parsed_valid < start_at).sum()) if start_at and parsed_count else 0,
    }


def _audit_evidence_pack(
    evidence_pack_path: str | Path | None,
    as_of: datetime | None,
    source_audits: list[dict[str, Any]],
    collapse_share_threshold: float,
    collapse_min_rows: int,
) -> dict[str, Any]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    if not evidence_pack_path:
        return {"status": "not_provided", "issues": [], "warnings": []}
    try:
        payload = DeanPaths.load_json(evidence_pack_path)
    except Exception as exc:
        return {"status": "timestamp_blocked", "path": str(evidence_pack_path), "issues": [f"Evidence pack error: {exc}"], "warnings": []}
    coverage = payload.get("coverage", {})
    date_range = coverage.get("date_range", {}) if isinstance(coverage, dict) else {}
    document_count = int(coverage.get("document_count", 0) or 0) if isinstance(coverage, dict) else 0
    start = _parse_optional_datetime(date_range.get("start"))
    end = _parse_optional_datetime(date_range.get("end"))
    issues: list[str] = []
    warnings: list[str] = []
    if document_count and (start is None or end is None):
        issues.append("Evidence pack has documents but no parseable coverage date range.")
    if as_of and end and end > as_of:
        issues.append("Evidence pack date range extends after as_of.")
    if document_count >= collapse_min_rows and start and end and start.date() == end.date():
        warnings.append("Evidence pack date range is collapsed to one calendar day despite many documents.")
        if as_of and start.date() == as_of.date():
            warnings.append("Evidence pack date range is collapsed exactly to as_of; source timestamps may be batch/cutoff dates.")
    source_ready_ranges = [
        audit.get("primary_timestamp", {})
        for audit in source_audits
        if audit.get("primary_timestamp") and audit.get("status") == "timestamp_ready"
    ]
    if start and end and source_ready_ranges:
        source_min_values = [_parse_optional_datetime(item.get("min")) for item in source_ready_ranges]
        source_max_values = [_parse_optional_datetime(item.get("max")) for item in source_ready_ranges]
        source_min_values = [value for value in source_min_values if value]
        source_max_values = [value for value in source_max_values if value]
        if source_min_values and source_max_values:
            source_min = min(source_min_values)
            source_max = max(source_max_values)
            if as_of and source_max > as_of:
                source_max = as_of
            if start > source_min or end < source_max:
                warnings.append("Evidence pack date range is narrower than ready source timestamp coverage.")
    status = "timestamp_blocked" if issues else "timestamp_suspicious" if warnings else "timestamp_ready"
    return {
        "status": status,
        "path": str(evidence_pack_path),
        "document_count": document_count,
        "date_range": {
            "start": start.isoformat() if start else date_range.get("start"),
            "end": end.isoformat() if end else date_range.get("end"),
        },
        "issues": issues,
        "warnings": warnings,
    }


def _summary(source_audits: list[dict[str, Any]], evidence_pack_audit: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(audit.get("status") for audit in source_audits)
    evidence_status = evidence_pack_audit.get("status", "not_provided")
    blocked = int(status_counts.get("timestamp_blocked", 0))
    suspicious = int(status_counts.get("timestamp_suspicious", 0))
    if evidence_status == "timestamp_blocked":
        blocked += 1
    elif evidence_status == "timestamp_suspicious":
        suspicious += 1
    if blocked:
        status = "timestamp_blocked"
    elif suspicious:
        status = "timestamp_suspicious"
    elif source_audits:
        status = "timestamp_ready"
    else:
        status = "timestamp_blocked"
    return {
        "audit_status": status,
        "source_count": len(source_audits),
        "ready_count": int(status_counts.get("timestamp_ready", 0)),
        "suspicious_count": int(status_counts.get("timestamp_suspicious", 0)),
        "blocked_count": int(status_counts.get("timestamp_blocked", 0)),
        "evidence_pack_status": evidence_status,
        "can_run_historical_research_replay": status == "timestamp_ready",
        "can_promote_replay_to_learning": False,
    }


def _recommendations(
    summary: dict[str, Any],
    source_audits: list[dict[str, Any]],
    evidence_pack_audit: dict[str, Any],
) -> list[str]:
    recommendations = [
        "Use this audit before scaling old-data research replay across many as_of windows.",
        "Do not promote historical replay results into learning memory from sources with blocked or suspicious timestamps.",
    ]
    if not source_audits:
        recommendations.append("Pass at least one --news-data, --macro-data, or --source-data file to audit.")
    if summary.get("blocked_count"):
        recommendations.append("Fix missing, unparsable, or future-leaking timestamp columns before replay.")
    if summary.get("suspicious_count") or evidence_pack_audit.get("status") == "timestamp_suspicious":
        recommendations.append("Inspect whether date columns are true publish/event dates or batch/cutoff dates.")
    if evidence_pack_audit.get("status") == "not_provided":
        recommendations.append("Optionally pass --evidence-pack-json to compare source timestamps with evidence pack coverage.")
    if summary.get("audit_status") == "timestamp_ready":
        recommendations.append("Historical research replay can be run as a diagnostic exam; learning promotion still requires outcome and review gates.")
    return recommendations


def _classify_timestamp_issues(
    primary: dict[str, Any],
    row_count: int,
    as_of: datetime | None,
    min_parse_rate: float,
    collapse_share_threshold: float,
    collapse_min_rows: int,
) -> tuple[list[str], list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    if primary["parse_rate"] < min_parse_rate:
        issues.append(
            f"Primary timestamp parse rate is {primary['parse_rate']}, below required {min_parse_rate}."
        )
    if primary["after_as_of_count"]:
        notes.append(f"{primary['after_as_of_count']} raw rows are after as_of and must be filtered out by the replay/evidence-pack step.")
    if row_count >= collapse_min_rows and primary["most_common_date_share"] >= collapse_share_threshold:
        warnings.append(
            f"Timestamp distribution is collapsed: {primary['most_common_date_share']} of rows share {primary['most_common_date']}."
        )
        if as_of and primary["most_common_date"] == as_of.date().isoformat():
            warnings.append("Most rows share the as_of date; this may be a batch/cutoff timestamp, not a publish date.")
    if primary["unique_date_count"] == 1 and row_count > 1:
        warnings.append("Only one unique timestamp date is present.")
    return issues, warnings, notes


def _read_frame(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = _json_records(payload)
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported source extension: {path.suffix}")


def _json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"value": item} for item in payload]
    if isinstance(payload, dict):
        for key in ("records", "items", "data", "news", "macro", "documents"):
            if isinstance(payload.get(key), list):
                return [item if isinstance(item, dict) else {"value": item} for item in payload[key]]
        return [payload]
    return [{"value": payload}]


def _candidate_columns(frame: Any) -> list[str]:
    candidates: list[str] = []
    for column in frame.columns:
        name = str(column).lower()
        if name in PREFERRED_TIMESTAMP_COLUMNS or any(token in name for token in TIMESTAMP_NAME_TOKENS):
            candidates.append(column)
            continue
        if str(frame[column].dtype).startswith("datetime"):
            candidates.append(column)
    return sorted(set(candidates), key=lambda value: _column_priority(str(value)))


def _select_primary_timestamp(audits: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        audits,
        key=lambda item: (
            item["parse_rate"],
            item["parsed_count"],
            item["priority"],
            item["unique_date_count"],
        ),
        reverse=True,
    )[0]


def _column_priority(column: str) -> int:
    lowered = column.lower()
    if lowered in PREFERRED_TIMESTAMP_COLUMNS:
        return len(PREFERRED_TIMESTAMP_COLUMNS) - PREFERRED_TIMESTAMP_COLUMNS.index(lowered)
    if "published" in lowered:
        return 5
    if "timestamp" in lowered or "datetime" in lowered:
        return 4
    if lowered == "date" or lowered.endswith("_date"):
        return 3
    if "created" in lowered or "updated" in lowered:
        return 2
    return 1


def _source_specs(source_kind: str, paths: list[str | Path]) -> list[dict[str, str]]:
    return [{"source_kind": source_kind, "path": str(path)} for path in paths]


def _parse_optional_datetime(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _run_id(prefix: str) -> str:
    return prefix + "_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_")
