from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from dean_os.material_loaders import MaterialLoadError, load_research_directory
from dean_os.regime_context import normalize_context_tags
from dean_os.schemas import ResearchDocument, utc_now_iso
from dean_os.utils import json_ready


TEXT_COLUMNS = ("title", "headline", "summary", "description", "content", "body", "text")
TITLE_COLUMNS = ("title", "headline", "name", "event", "indicator", "series")
DATE_COLUMNS = (
    "published_at",
    "published_date",
    "publication_date",
    "pub_date",
    "publishedAt",
    "time_published",
    "timestamp",
    "datetime",
    "date",
    "created_at",
    "updated_at",
)
TICKER_COLUMNS = ("ticker", "tickers", "symbol", "symbols")
SECTOR_COLUMNS = ("sector", "sectors", "industry", "industries")


class AnalystEvidencePackRunner:
    """Builds a local-only evidence pack for analyst agents.

    The pack is intentionally a source-normalization layer, not a new analyst
    brain: it turns local materials, cached news, and macro tables into cited
    ResearchDocument payloads that Agent Lab can consume.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/analyst_evidence_pack"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        materials_paths: list[str | Path] | None = None,
        news_data_paths: list[str | Path] | None = None,
        macro_data_paths: list[str | Path] | None = None,
        source_routing_path: str | Path | None = None,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        tags: list[str] | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
        as_of: str | None = None,
        max_rows_per_table: int = 200,
        max_documents: int = 500,
        max_text_chars: int = 6000,
        include_routed_materials: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        tickers = _normalize_tickers(tickers or [])
        sectors = _normalize_strings(sectors or [])
        tags = normalize_context_tags(tags or [])
        start_dt = _parse_datetime(start_at) if start_at else None
        end_dt = _parse_datetime(end_at or as_of) if (end_at or as_of) else None
        source_routing = _load_source_routing(source_routing_path)
        material_paths = _expand_material_paths(
            materials_paths=materials_paths or [],
            source_routing=source_routing,
            include_routed_materials=include_routed_materials,
        )

        documents: list[ResearchDocument] = []
        warnings: list[str] = []
        dropped: list[dict[str, Any]] = []

        material_documents, material_warnings = self._load_material_documents(
            material_paths=material_paths,
            tickers=tickers,
            sectors=sectors,
            tags=tags,
            max_text_chars=max_text_chars,
        )
        documents.extend(material_documents)
        warnings.extend(material_warnings)

        table_documents, table_warnings, table_dropped = self._load_table_documents(
            paths=news_data_paths or [],
            source_kind="news",
            source_type="news",
            tickers=tickers,
            sectors=sectors,
            tags=[*tags, "news"],
            start_at=start_dt,
            end_at=end_dt,
            max_rows_per_table=max_rows_per_table,
            max_text_chars=max_text_chars,
        )
        documents.extend(table_documents)
        warnings.extend(table_warnings)
        dropped.extend(table_dropped)

        macro_documents, macro_warnings, macro_dropped = self._load_table_documents(
            paths=macro_data_paths or [],
            source_kind="macro",
            source_type="report",
            tickers=[],
            sectors=sectors,
            tags=[*tags, "macro"],
            start_at=start_dt,
            end_at=end_dt,
            max_rows_per_table=max_rows_per_table,
            max_text_chars=max_text_chars,
            keep_without_ticker=True,
        )
        documents.extend(macro_documents)
        warnings.extend(macro_warnings)
        dropped.extend(macro_dropped)

        documents, dedupe_count = _dedupe_documents(documents)
        if dedupe_count:
            dropped.append({"reason": "duplicate_document", "count": dedupe_count})
        documents = documents[:max_documents]

        coverage = _coverage(documents, warnings=warnings, dropped=dropped, requested_tickers=tickers)
        pack_id = _run_id("analyst_evidence_pack")
        payload = {
            "run_id": pack_id,
            "created_at": utc_now_iso(),
            "mode": "analyst_evidence_pack",
            "inputs": {
                "materials_paths": [str(path) for path in material_paths],
                "news_data_paths": [str(path) for path in news_data_paths or []],
                "macro_data_paths": [str(path) for path in macro_data_paths or []],
                "source_routing_path": str(source_routing_path) if source_routing_path else None,
                "tickers": tickers,
                "sectors": sectors,
                "tags": tags,
                "start_at": start_dt.isoformat() if start_dt else None,
                "end_at": end_dt.isoformat() if end_dt else None,
                "max_rows_per_table": max_rows_per_table,
                "max_documents": max_documents,
                "max_text_chars": max_text_chars,
            },
            "coverage": coverage,
            "source_routing": _source_routing_summary(source_routing),
            "analyst_inputs": _analyst_inputs(documents, coverage),
            "documents": [document.model_dump(mode="json") for document in documents],
            "warnings": warnings,
            "dropped": dropped,
            "recommendations": _recommendations(coverage, source_routing),
        }
        if save:
            json_path, md_path = self.save(payload)
            payload["saved_paths"] = {
                "json": str(json_path),
                "markdown": str(md_path),
                "latest_json": str(self.output_dir / "latest.json"),
                "latest_markdown": str(self.output_dir / "latest.md"),
            }
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_analyst_evidence_pack_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path

    def _load_material_documents(
        self,
        material_paths: list[str | Path],
        tickers: list[str],
        sectors: list[str],
        tags: list[str],
        max_text_chars: int,
    ) -> tuple[list[ResearchDocument], list[str]]:
        documents: list[ResearchDocument] = []
        warnings: list[str] = []
        for path in material_paths:
            try:
                loaded, errors = load_research_directory(
                    path=path,
                    tickers=tickers,
                    sectors=sectors,
                    tags=tags,
                    recursive=True,
                    ignore_errors=True,
                )
            except MaterialLoadError as exc:
                warnings.append(str(exc))
                continue
            warnings.extend(errors)
            for document in loaded:
                documents.append(_normalize_document(document, tickers, sectors, tags, max_text_chars))
        return documents, warnings

    def _load_table_documents(
        self,
        paths: list[str | Path],
        source_kind: str,
        source_type: str,
        tickers: list[str],
        sectors: list[str],
        tags: list[str],
        start_at: datetime | None,
        end_at: datetime | None,
        max_rows_per_table: int,
        max_text_chars: int,
        keep_without_ticker: bool = False,
    ) -> tuple[list[ResearchDocument], list[str], list[dict[str, Any]]]:
        documents: list[ResearchDocument] = []
        warnings: list[str] = []
        dropped: list[dict[str, Any]] = []
        for path in paths:
            try:
                rows = _read_table_rows(path)
            except Exception as exc:
                warnings.append(f"Could not read {source_kind} table {path}: {type(exc).__name__}: {exc}")
                continue
            used_rows = 0
            for index, row in enumerate(rows):
                if used_rows >= max_rows_per_table:
                    dropped.append({"path": str(path), "reason": "max_rows_per_table", "remaining_rows": len(rows) - index})
                    break
                document = _document_from_row(
                    row=row,
                    path=Path(path),
                    source_kind=source_kind,
                    source_type=source_type,
                    requested_tickers=tickers,
                    requested_sectors=sectors,
                    tags=tags,
                    start_at=start_at,
                    end_at=end_at,
                    max_text_chars=max_text_chars,
                    keep_without_ticker=keep_without_ticker,
                )
                if document is None:
                    continue
                documents.append(document)
                used_rows += 1
        return documents, warnings, dropped


def documents_from_evidence_pack(path: str | Path) -> list[ResearchDocument]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    return [ResearchDocument(**item) for item in documents if isinstance(item, dict)]


def render_analyst_evidence_pack_markdown(payload: dict[str, Any]) -> str:
    coverage = payload.get("coverage", {})
    analyst = payload.get("analyst_inputs", {}).get("base_analyst", {})
    manager = payload.get("analyst_inputs", {}).get("manager_plan", {})
    lines = [
        "# DEAN-OS Analyst Evidence Pack",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Quality: `{coverage.get('data_quality')}`",
        f"- Documents: {coverage.get('document_count', 0)}",
        f"- Source types: {coverage.get('by_source_type', {})}",
        f"- Tickers: {', '.join(coverage.get('tickers', [])) or 'none'}",
        f"- Sectors: {', '.join(coverage.get('sectors', [])) or 'none'}",
        f"- Warnings: {coverage.get('warning_count', 0)}",
        "",
        "## Base Analyst",
        "",
        f"- Ready: {analyst.get('ready')}",
        f"- Recommended profile: `{analyst.get('recommended_profile')}`",
        f"- Agent Lab command: `{analyst.get('agent_lab_command_preview')}`",
        f"- Manager mode: `{manager.get('mode')}`",
        f"- Active profiles: {', '.join(manager.get('active_profiles', [])) or 'none'}",
        f"- Candidate profiles: {', '.join(manager.get('candidate_profiles', [])) or 'none'}",
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    lines.extend(["", "## Sample Documents", ""])
    for document in payload.get("documents", [])[:10]:
        excerpt = str(document.get("text", ""))[:220].replace("\n", " ")
        lines.append(f"- `{document.get('source_type')}` {document.get('title')} :: {excerpt}")
    return "\n".join(lines).strip() + "\n"


def _document_from_row(
    row: dict[str, Any],
    path: Path,
    source_kind: str,
    source_type: str,
    requested_tickers: list[str],
    requested_sectors: list[str],
    tags: list[str],
    start_at: datetime | None,
    end_at: datetime | None,
    max_text_chars: int,
    keep_without_ticker: bool,
) -> ResearchDocument | None:
    published_at = _first_datetime(row)
    if published_at and start_at and published_at < start_at:
        return None
    if published_at and end_at and published_at > end_at:
        return None

    text = _row_text(row, source_kind=source_kind)
    if not text.strip():
        return None
    title = _first_text(row, TITLE_COLUMNS) or f"{source_kind} row from {path.name}"
    row_tickers = _normalize_tickers(_first_list(row, TICKER_COLUMNS))
    inferred = _infer_tickers(text, requested_tickers)
    doc_tickers = sorted(set(row_tickers or inferred))
    if requested_tickers and doc_tickers and not set(doc_tickers).intersection(requested_tickers):
        return None
    if requested_tickers and not doc_tickers and not keep_without_ticker:
        return None

    doc_sectors = _normalize_strings(_first_list(row, SECTOR_COLUMNS)) or requested_sectors
    doc_tags = normalize_context_tags([*tags, source_kind, path.stem])
    uri = str(row.get("url") or row.get("uri") or row.get("source_url") or path)
    metadata = {
        "path": str(path),
        "source_kind": source_kind,
        "row_keys": sorted(str(key) for key in row),
    }
    document = ResearchDocument(
        document_id=_document_id(uri, title, text),
        title=title[:180],
        source_type=source_type,
        text=text[:max_text_chars],
        uri=uri,
        published_at=published_at.isoformat() if published_at else _first_text(row, DATE_COLUMNS),
        tickers=doc_tickers or requested_tickers,
        sectors=doc_sectors,
        tags=doc_tags,
        metadata=metadata,
    )
    return document


def _normalize_document(
    document: ResearchDocument,
    requested_tickers: list[str],
    requested_sectors: list[str],
    tags: list[str],
    max_text_chars: int,
) -> ResearchDocument:
    text = document.text[:max_text_chars]
    tickers = _normalize_tickers(document.tickers) or _infer_tickers(text, requested_tickers) or requested_tickers
    sectors = _normalize_strings(document.sectors) or requested_sectors
    merged_tags = normalize_context_tags([*document.tags, *tags])
    return ResearchDocument(
        document_id=_document_id(document.uri or document.title, document.title, text),
        title=document.title,
        source_type=document.source_type,
        text=text,
        uri=document.uri,
        authors=document.authors,
        published_at=document.published_at,
        tickers=tickers,
        sectors=sectors,
        tags=merged_tags,
        metadata={**document.metadata, "evidence_pack_normalized": True},
    )


def _read_table_rows(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item if isinstance(item, dict) else {"value": item} for item in payload]
        if isinstance(payload, dict):
            for key in ("records", "items", "data", "news", "macro"):
                if isinstance(payload.get(key), list):
                    return [item if isinstance(item, dict) else {"value": item} for item in payload[key]]
            return [payload]
        return [{"value": payload}]
    import pandas as pd

    if suffix == ".csv":
        frame = pd.read_csv(resolved)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(resolved)
    else:
        raise ValueError(f"Unsupported table extension: {resolved.suffix}")
    frame = frame.where(frame.notna(), None)
    return frame.to_dict(orient="records")


def _row_text(row: dict[str, Any], source_kind: str) -> str:
    pieces = [_string(row.get(key)) for key in TEXT_COLUMNS if _string(row.get(key))]
    if pieces:
        return _clean_text(" ".join(pieces))
    visible = []
    for key, value in row.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        text = _string(value)
        if text:
            visible.append(f"{key}: {text}")
    prefix = "Macro observation" if source_kind == "macro" else "Source observation"
    return _clean_text(f"{prefix}. " + "; ".join(visible))


def _coverage(
    documents: list[ResearchDocument],
    warnings: list[str],
    dropped: list[dict[str, Any]],
    requested_tickers: list[str],
) -> dict[str, Any]:
    by_source_type = Counter(document.source_type for document in documents)
    by_ticker = Counter(ticker for document in documents for ticker in document.tickers)
    by_sector = Counter(sector for document in documents for sector in document.sectors)
    by_tag = Counter(tag for document in documents for tag in document.tags)
    dates = sorted(_parse_datetime(document.published_at) for document in documents if document.published_at and _parse_datetime(document.published_at))
    missing_requested_tickers = [ticker for ticker in requested_tickers if ticker not in by_ticker]
    data_quality = _data_quality(documents, by_source_type, missing_requested_tickers, warnings)
    return {
        "document_count": len(documents),
        "data_quality": data_quality,
        "research_ready": bool(documents),
        "agent_lab_ready": bool(documents),
        "by_source_type": dict(sorted(by_source_type.items())),
        "by_ticker": dict(sorted(by_ticker.items())),
        "by_sector": dict(sorted(by_sector.items())),
        "by_tag": dict(sorted(by_tag.items())),
        "tickers": sorted(by_ticker),
        "sectors": sorted(by_sector),
        "missing_requested_tickers": missing_requested_tickers,
        "date_range": {
            "start": dates[0].isoformat() if dates else None,
            "end": dates[-1].isoformat() if dates else None,
        },
        "warning_count": len(warnings),
        "dropped_count": sum(int(item.get("count", 1)) for item in dropped),
    }


def _data_quality(
    documents: list[ResearchDocument],
    by_source_type: Counter,
    missing_requested_tickers: list[str],
    warnings: list[str],
) -> str:
    if not documents:
        return "weak"
    if len(documents) >= 5 and len(by_source_type) >= 2 and not missing_requested_tickers and not warnings:
        return "strong"
    return "partial"


def _analyst_inputs(documents: list[ResearchDocument], coverage: dict[str, Any]) -> dict[str, Any]:
    profile = "generalist_base_analyst"
    if coverage.get("by_source_type", {}).get("report") and coverage.get("by_source_type", {}).get("news"):
        profile = "generalist_research_plus_macro"
    candidate_profiles = _candidate_profiles(coverage)
    return {
        "base_analyst": {
            "ready": bool(documents),
            "recommended_profile": profile,
            "document_count": len(documents),
            "source_types": sorted(coverage.get("by_source_type", {})),
            "tickers": coverage.get("tickers", []),
            "sectors": coverage.get("sectors", []),
            "agent_lab_command_preview": "python run_agent_lab.py --evidence-pack-json reports/dean_os/analyst_evidence_pack/latest.json --tickers "
            + (" ".join(coverage.get("tickers", [])) or "TICKER_HERE"),
        },
        "manager_plan": {
            "mode": "single_base_then_specialize",
            "active_profiles": ["generalist_base_analyst"] if documents else [],
            "candidate_profiles": candidate_profiles,
            "creation_policy": (
                "Start with one base analyst. Promote a specialized profile only when the evidence pack has "
                "matching source coverage, citations, and reviewed outcomes for that domain."
            ),
            "blocked_profiles": _blocked_profiles(candidate_profiles, coverage),
        },
    }


def _candidate_profiles(coverage: dict[str, Any]) -> list[str]:
    source_types = coverage.get("by_source_type", {})
    tags = set(coverage.get("by_tag", {}))
    candidates: list[str] = []
    if source_types.get("news"):
        candidates.append("news_catalyst")
    if source_types.get("report") or "macro" in tags:
        candidates.append("macro_policy")
    if coverage.get("sectors"):
        candidates.append("sector_cycle")
    if source_types.get("filing") or source_types.get("transcript"):
        candidates.append("value_screening")
    return candidates


def _blocked_profiles(candidate_profiles: list[str], coverage: dict[str, Any]) -> dict[str, str]:
    blocked: dict[str, str] = {}
    if "value_screening" not in candidate_profiles:
        blocked["value_screening"] = "Needs filings, transcripts, or fundamental reports."
    if "sector_cycle" not in candidate_profiles:
        blocked["sector_cycle"] = "Needs sector tags or sector-specific sources."
    if "macro_policy" not in candidate_profiles:
        blocked["macro_policy"] = "Needs macro reports/news or macro table input."
    if coverage.get("data_quality") == "weak":
        blocked["all_specialists"] = "Evidence pack quality is weak; keep only the base analyst active."
    return blocked


def _recommendations(coverage: dict[str, Any], source_routing: dict[str, Any] | None) -> list[str]:
    recommendations: list[str] = []
    if not coverage.get("document_count"):
        recommendations.append("Add local materials, news data, or macro data before running analyst synthesis.")
        return recommendations
    if coverage.get("missing_requested_tickers"):
        recommendations.append("Some requested tickers have no matched evidence; add ticker-specific documents or news rows.")
    if len(coverage.get("by_source_type", {})) < 2:
        recommendations.append("Add at least two source types before treating analyst conclusions as strong.")
    if source_routing is None:
        recommendations.append("Run SourceRoutingAgent or pass --source-routing-json to explain where each source belongs.")
    recommendations.append("Run Agent Lab with --evidence-pack-json, then review citations before creating paper decisions.")
    return recommendations


def _load_source_routing(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("source_routing"), dict):
        return payload["source_routing"]
    return payload if isinstance(payload, dict) else None


def _source_routing_summary(source_routing: dict[str, Any] | None) -> dict[str, Any]:
    if not source_routing:
        return {"available": False}
    return {
        "available": True,
        "summary": source_routing.get("summary", {}),
        "analyst_inputs": source_routing.get("analyst_inputs", {}),
        "warnings": source_routing.get("warnings", []),
    }


def _expand_material_paths(
    materials_paths: list[str | Path],
    source_routing: dict[str, Any] | None,
    include_routed_materials: bool,
) -> list[Path]:
    paths = [Path(path) for path in materials_paths]
    if include_routed_materials and source_routing:
        for item in source_routing.get("materials", {}).get("records", []):
            if isinstance(item, dict) and item.get("path"):
                paths.append(Path(item["path"]))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _dedupe_documents(documents: list[ResearchDocument]) -> tuple[list[ResearchDocument], int]:
    seen: dict[str, ResearchDocument] = {}
    duplicate_count = 0
    for document in documents:
        key = _document_id(document.uri or document.title, document.title, document.text)
        if key in seen:
            duplicate_count += 1
            existing = seen[key]
            existing.tickers = sorted(set(existing.tickers).union(document.tickers))
            existing.sectors = sorted(set(existing.sectors).union(document.sectors))
            existing.tags = normalize_context_tags([*existing.tags, *document.tags])
            continue
        document.document_id = key
        seen[key] = document
    return list(seen.values()), duplicate_count


def _first_datetime(row: dict[str, Any]) -> datetime | None:
    for key in DATE_COLUMNS:
        parsed = _parse_datetime(row.get(key))
        if parsed:
            return parsed
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text or text.lower() in {"nat", "nan", "none"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            import pandas as pd

            timestamp = pd.to_datetime(text, utc=True, errors="coerce")
            if timestamp is None or pd.isna(timestamp):
                return None
            return timestamp.to_pydatetime()
        except Exception:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _string(row.get(key))
        if value:
            return value
    return None


def _first_list(row: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [str(item) for item in value]
        text = _string(value)
        if text:
            return re.split(r"[,;|]\s*", text)
    return []


def _normalize_tickers(values: list[str]) -> list[str]:
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _normalize_strings(values: list[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _infer_tickers(text: str, requested_tickers: list[str]) -> list[str]:
    lowered = f" {text.upper()} "
    found = []
    for ticker in requested_tickers:
        if re.search(rf"(?<![A-Z0-9]){re.escape(ticker.upper())}(?![A-Z0-9])", lowered):
            found.append(ticker.upper())
    return sorted(set(found))


def _string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _document_id(uri: str, title: str, text: str) -> str:
    digest = sha256(f"{uri}|{title}|{text[:500]}".encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"ev_{digest}"


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
