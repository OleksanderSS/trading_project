from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import duckdb

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso


SEARCH_TERMS = (
    "capacity utilization",
    "utilization",
    "equipment orders",
    "book-to-bill",
    "lead time",
    "wafer starts",
)


class IndustryOperationalSourceCoverageBuilder:
    """Audit local stores for operational sources without promoting prose to metrics."""

    contract = "dean_industry_operational_source_coverage_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/industry_operational_source_coverage_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        duckdb_path: str | Path,
        research_sqlite_path: str | Path,
        knowledge_pack_path: str | Path,
        save: bool = True,
    ) -> dict[str, Any]:
        duck_path = Path(duckdb_path)
        research_path = Path(research_sqlite_path)
        pack_path = Path(knowledge_pack_path)
        structured = _duckdb_structured_coverage(duck_path)
        research = _research_coverage(research_path)
        pack = _knowledge_pack_coverage(pack_path)
        structured_count = len(structured["matching_columns"])
        numeric_pack_count = len(pack["structured_numeric_candidates"])
        eligible = structured_count + numeric_pack_count
        created_at = utc_now_iso()
        run_id = "industry_operational_source_coverage_" + created_at.replace(":", "").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "industry_operational_source_coverage",
            "contract": self.contract,
            "inputs": {
                "duckdb": _ref(duck_path),
                "research_sqlite": _ref(research_path),
                "knowledge_pack": _ref(pack_path),
            },
            "summary": {
                "structured_operational_candidate_count": eligible,
                "duckdb_matching_column_count": structured_count,
                "knowledge_pack_numeric_candidate_count": numeric_pack_count,
                "research_narrative_match_count": research["narrative_match_count"],
                "knowledge_pack_narrative_match_count": pack["narrative_match_count"],
                "gate_status": (
                    "local_structured_source_candidates_require_review"
                    if eligible else "structured_adapter_ready_source_feed_missing"
                ),
                "metric_extraction_performed": False,
                "gap_closure_allowed": False,
                "can_trade": False,
            },
            "duckdb_coverage": structured,
            "research_corpus_coverage": research,
            "knowledge_pack_coverage": pack,
            "required_source_feeds": [
                {
                    "metric_family": "semiconductor_equipment_orders",
                    "minimum_contract": [
                        "supplier_or_industry_body", "metric_value", "unit", "period",
                        "available_at", "methodology", "source_locator", "source_sha256",
                    ],
                },
                {
                    "metric_family": "foundry_capacity_utilization",
                    "minimum_contract": [
                        "foundry_or_methodology_backed_survey", "capacity_definition",
                        "utilization_value", "percent", "period", "available_at",
                        "geography_or_node", "source_locator", "source_sha256",
                    ],
                },
            ],
            "semantic_boundary": {
                "narrative_match_is_structured_metric": False,
                "topic_relevance_is_claim_support": False,
                "keyword_index_is_observation_series": False,
                "manual_source_review_required": True,
            },
            "safety": {
                "review_only": True,
                "external_fetch_performed": False,
                "metric_extraction_performed": False,
                "collector_execution_performed": False,
                "learning_write_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload, markdown=_markdown(payload), run_id=run_id
            )
        return payload


def _duckdb_structured_coverage(path: Path) -> dict[str, Any]:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        columns = connection.execute(
            "select table_name, column_name, data_type from information_schema.columns"
        ).fetchall()
        matching = [
            {"table": table, "column": column, "data_type": data_type}
            for table, column, data_type in columns
            if any(_token(term) in _token(column) for term in SEARCH_TERMS)
        ]
        keyword_matches = []
        tables = {row[0] for row in connection.execute("show tables").fetchall()}
        if "keyword_index" in tables:
            for term in SEARCH_TERMS:
                rows = connection.execute(
                    "select source_table, source_column, keyword, row_count from keyword_index where lower(keyword) like ?",
                    [f"%{term}%"],
                ).fetchall()
                keyword_matches.extend(
                    {"term": term, "source_table": row[0], "source_column": row[1], "keyword": row[2], "row_count": row[3]}
                    for row in rows
                )
        return {"matching_columns": matching, "keyword_index_matches": keyword_matches}
    finally:
        connection.close()


def _research_coverage(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        matches: dict[str, dict[str, Any]] = {}
        for term in SEARCH_TERMS:
            rows = connection.execute(
                "select document_id, title, source_type, published_at from documents where lower(title || ' ' || text) like ? limit 20",
                (f"%{term}%",),
            ).fetchall()
            for row in rows:
                record = matches.setdefault(
                    str(row[0]),
                    {
                        "document_id": row[0],
                        "title": row[1],
                        "source_type": row[2],
                        "published_at": row[3],
                        "terms": [],
                    },
                )
                record["terms"].append(term)
        normalized = [
            {**record, "terms": sorted(set(record["terms"]))}
            for record in matches.values()
        ]
        return {"narrative_match_count": len(normalized), "narrative_matches": normalized}
    finally:
        connection.close()


def _knowledge_pack_coverage(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    numeric = []
    for item in payload.get("items") or []:
        text = f"{item.get('title', '')} {item.get('body', '')}".lower()
        terms = [term for term in SEARCH_TERMS if term in text]
        if not terms:
            continue
        record = {"item_id": item.get("item_id"), "item_type": item.get("item_type"), "terms": terms, "title": item.get("title")}
        matches.append(record)
        if item.get("item_type") == "metric" and (item.get("metadata") or {}).get("required_lane_eligible") is True:
            numeric.append(record)
    return {
        "narrative_match_count": len(matches),
        "narrative_matches": matches,
        "structured_numeric_candidates": numeric,
    }


def _token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return (
        "# Industry Operational Source Coverage\n\n"
        f"- Gate: `{summary['gate_status']}`\n"
        f"- Structured candidates: `{summary['structured_operational_candidate_count']}`\n"
        f"- Narrative matches: `{summary['research_narrative_match_count'] + summary['knowledge_pack_narrative_match_count']}`\n"
        "- Metric extraction performed: `false`\n"
        "- Can trade: `false`\n"
    )


__all__ = ["IndustryOperationalSourceCoverageBuilder", "SEARCH_TERMS"]
