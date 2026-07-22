from __future__ import annotations

__all__ = [
    'DEFAULT_DATABASE_PATH',
    'DEFAULT_FORMS',
    'FILING_CANONICAL_FIELDS',
    'REQUIRED_COLUMNS',
    'SAVED_SEC_INDEX_CONTRACT',
    'SavedSECFilingIndexProducer',
    'render_saved_sec_filing_index_markdown',
    'verify_saved_sec_filing_index',
    'verify_sec_filing_index',
]

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso

SAVED_SEC_INDEX_CONTRACT = "dean_saved_sec_filing_index_v1"
DEFAULT_DATABASE_PATH = "data/trading_data.duckdb"
DEFAULT_FORMS = ("10-K", "10-Q", "20-F", "40-F")
REQUIRED_COLUMNS = {
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "isXBRL",
    "isInlineXBRL",
    "primaryDocument",
    "ticker",
    "cik",
    "hash",
}
FILING_CANONICAL_FIELDS = (
    "ticker",
    "cik",
    "form",
    "filing_date",
    "report_date",
    "accepted_at",
    "accession_number",
    "primary_document",
    "source_locator",
    "stored_hash",
    "expected_collector_hash",
    "is_xbrl",
    "is_inline_xbrl",
    "content_status",
)


class SavedSECFilingIndexProducer:
    """Build a point-in-time filing index from the saved DuckDB table.

    This producer intentionally stops at metadata. The current table does not
    store filing HTML or XBRL facts, so it cannot create fundamental metrics.
    """

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_sec_filing_index"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        tickers: list[str],
        as_of: str,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        forms: list[str] | tuple[str, ...] = DEFAULT_FORMS,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError(
                "SEC filing index as_of must be a timezone-aware "
                "ISO-8601 timestamp"
            )
        normalized_tickers = sorted(
            {
                str(ticker).upper().strip()
                for ticker in tickers
                if str(ticker).strip()
            }
        )
        normalized_forms = sorted(
            {
                str(form).upper().strip()
                for form in forms
                if str(form).strip()
            }
        )
        if not normalized_tickers:
            raise ValueError("at least one SEC filing ticker is required")
        if not normalized_forms:
            raise ValueError("at least one SEC filing form is required")

        database = Path(database_path)
        run_id = _run_id()
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "saved_sec_filing_index_producer",
            "producer_contract": SAVED_SEC_INDEX_CONTRACT,
            "inputs": {
                "database_path": str(database),
                "table": "sec_filings",
                "tickers": normalized_tickers,
                "forms": normalized_forms,
                "as_of": as_of_dt.isoformat(),
            },
            "source_provenance": _database_provenance(database),
        }
        if not database.exists():
            payload.update(
                _blocked(
                    "blocked_database_missing",
                    ["sec_database_missing"],
                )
            )
            return self._finish(payload, save=save)

        try:
            with duckdb.connect(
                str(database),
                read_only=True,
            ) as connection:
                schema_rows = connection.execute(
                    "DESCRIBE sec_filings"
                ).fetchall()
                columns = {str(row[0]) for row in schema_rows}
                missing = sorted(REQUIRED_COLUMNS - columns)
                total_rows = connection.execute(
                    "SELECT count(*) FROM sec_filings"
                ).fetchone()[0]
                if missing:
                    payload["source_provenance"].update(
                        {
                            "table_row_count": total_rows,
                            "table_columns": sorted(columns),
                            "missing_required_columns": missing,
                        }
                    )
                    payload.update(
                        _blocked(
                            "blocked_sec_table_schema",
                            [
                                f"sec_column_missing_{column}"
                                for column in missing
                            ],
                        )
                    )
                    return self._finish(payload, save=save)
                rows = _query_rows(
                    connection,
                    tickers=normalized_tickers,
                    forms=normalized_forms,
                )
        except (duckdb.Error, OSError) as exc:
            payload["source_provenance"]["read_error"] = str(exc)
            payload.update(
                _blocked(
                    "blocked_sec_database_unreadable",
                    ["sec_database_unreadable"],
                )
            )
            return self._finish(payload, save=save)

        payload["source_provenance"].update(
            {
                "table_row_count": int(total_rows),
                "table_columns": sorted(columns),
                "database_sha256_computed": False,
                "database_sha256_reason": (
                    "large mutable container; selected immutable filing rows "
                    "are hash-verified instead"
                ),
            }
        )
        normalized = _normalize_rows(rows, as_of=as_of_dt)
        records = normalized["accepted"]
        represented_tickers = sorted(
            {record["ticker"] for record in records}
        )
        missing_tickers = sorted(
            set(normalized_tickers) - set(represented_tickers)
        )
        fingerprint = _canonical_sha256(
            [
                {
                    key: record[key]
                    for key in (
                        "ticker",
                        "cik",
                        "form",
                        "report_date",
                        "accepted_at",
                        "accession_number",
                        "primary_document",
                        "source_locator",
                        "stored_hash",
                        "record_sha256",
                    )
                }
                for record in records
            ]
        )
        status = (
            "filing_index_ready_with_exclusions"
            if records and normalized["exclusions"]
            else "filing_index_ready"
            if records
            else "blocked_no_admissible_filings"
        )
        payload.update(
            {
                "status": status,
                "summary": {
                    "database_row_count": int(total_rows),
                    "query_row_count": len(rows),
                    "accepted_filing_count": len(records),
                    "excluded_filing_count": len(
                        normalized["exclusions"]
                    ),
                    "ticker_count": len(
                        represented_tickers
                    ),
                    "requested_ticker_count": len(
                        normalized_tickers
                    ),
                    "represented_tickers": represented_tickers,
                    "missing_tickers": missing_tickers,
                    "ticker_coverage_ratio": round(
                        len(represented_tickers)
                        / len(normalized_tickers),
                        6,
                    ),
                    "ticker_coverage_status": (
                        "complete"
                        if not missing_tickers
                        else "partial"
                    ),
                    "xbrl_filing_count": sum(
                        record["is_xbrl"] for record in records
                    ),
                    "accepted_fingerprint": fingerprint,
                    "reason_counts": normalized["reason_counts"],
                    "can_request_filing_content": bool(records),
                    "can_create_fundamental_metrics": False,
                    "can_run_ratio_templates": False,
                    "can_feed_value_screening": False,
                    "can_trade": False,
                },
                "filings": records,
                "exclusions": normalized["exclusions"],
                "content_extraction_requests": [
                    {
                        "request_id": (
                            f"sec_content_{record['record_sha256'][:16]}"
                        ),
                        "ticker": record["ticker"],
                        "form": record["form"],
                        "accession_number": record[
                            "accession_number"
                        ],
                        "source_locator": record["source_locator"],
                        "requested_outputs": [
                            "immutable_primary_document",
                            "primary_document_sha256",
                            "xbrl_facts_with_units_periods_and_contexts",
                        ],
                        "network_request_executed": False,
                        "status": "pending_external_content_acquisition",
                    }
                    for record in records
                ],
                "fundamental_boundary": {
                    "filing_metadata_available": bool(records),
                    "primary_document_content_stored": False,
                    "xbrl_fact_values_stored": False,
                    "metric_extraction_allowed_now": False,
                    "ratio_computation_allowed_now": False,
                    "valuation_allowed_now": False,
                    "reason": (
                        "sec_filings stores submission metadata only; filing "
                        "HTML and XBRL facts are not present"
                    ),
                },
                "safety": _safety(),
            }
        )
        return self._finish(payload, save=save)

    def _finish(
        self,
        payload: dict[str, Any],
        *,
        save: bool,
    ) -> dict[str, Any]:
        payload.setdefault("safety", _safety())
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_saved_sec_filing_index_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
        return payload


def verify_saved_sec_filing_index(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
    verify_source_database: bool = True,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("producer_contract") != SAVED_SEC_INDEX_CONTRACT:
        raise ValueError("unsupported SEC filing index contract")
    if payload.get("status") not in {
        "filing_index_ready",
        "filing_index_ready_with_exclusions",
    }:
        raise ValueError("SEC filing index artifact is not ready")
    as_of = parse_timezone_aware(payload.get("inputs", {}).get("as_of"))
    if as_of is None:
        raise ValueError("SEC filing index as_of is invalid")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != as_of:
            raise ValueError("SEC filing index expected as_of mismatch")
    database = Path(payload.get("inputs", {}).get("database_path", ""))
    if verify_source_database and not database.exists():
        raise ValueError("SEC filing source database is missing")
    filings = payload.get("filings")
    if not isinstance(filings, list):
        raise ValueError("SEC filing records are missing")

    verified: list[dict[str, Any]] = []
    for filing in filings:
        artifact_core = {
            key: filing.get(key)
            for key in FILING_CANONICAL_FIELDS
        }
        if _canonical_sha256(
            artifact_core
        ) != filing.get("record_sha256"):
            raise ValueError("SEC filing artifact payload mismatch")
    if verify_source_database:
        with duckdb.connect(str(database), read_only=True) as connection:
            for filing in filings:
                row = connection.execute(
                    (
                        "SELECT accessionNumber, filingDate, reportDate, "
                        "acceptanceDateTime, form, isXBRL, isInlineXBRL, "
                        "primaryDocument, ticker, cik, hash "
                        "FROM sec_filings WHERE accessionNumber = ? AND cik = ?"
                    ),
                    [
                        filing.get("accession_number"),
                        filing.get("cik"),
                    ],
                ).fetchone()
                if row is None:
                    raise ValueError("SEC filing source row is missing")
                normalized = _normalize_rows([row], as_of=as_of)
                if (
                    len(normalized["accepted"]) != 1
                    or normalized["accepted"][0]["record_sha256"]
                    != filing.get("record_sha256")
                ):
                    raise ValueError("SEC filing source row hash mismatch")
                verified.append(normalized["accepted"][0])
    else:
        verified = [dict(filing) for filing in filings]
    fingerprint = _canonical_sha256(
        [
            {
                key: filing[key]
                for key in (
                    "ticker",
                    "cik",
                    "form",
                    "report_date",
                    "accepted_at",
                    "accession_number",
                    "primary_document",
                    "source_locator",
                    "stored_hash",
                    "record_sha256",
                )
            }
            for filing in verified
        ]
    )
    if fingerprint != payload.get("summary", {}).get(
        "accepted_fingerprint"
    ):
        raise ValueError("SEC filing index fingerprint mismatch")
    return {
        "as_of": as_of.isoformat(),
        "filings": verified,
        "fingerprint": fingerprint,
        "artifact_path": str(path),
        "artifact_sha256": _sha256_file(path),
        "fundamental_metrics_available": False,
        "verified": True,
        "verification_mode": (
            "artifact_and_source_database"
            if verify_source_database
            else "hash_bound_artifact_only"
        ),
    }


def verify_sec_filing_index(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
    verify_source_database: bool = True,
) -> dict[str, Any]:
    """Verify either a DuckDB-backed or submissions-backed filing index."""

    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("producer_contract")
    if contract == SAVED_SEC_INDEX_CONTRACT:
        return verify_saved_sec_filing_index(
            path,
            expected_as_of=expected_as_of,
            verify_source_database=verify_source_database,
        )
    if contract == "dean_saved_sec_submissions_filing_index_v1":
        from dean_os.analysts._producers.sec.submissions_index import (
            verify_saved_sec_submissions_filing_index,
        )

        return verify_saved_sec_submissions_filing_index(
            path,
            expected_as_of=expected_as_of,
        )
    raise ValueError("unsupported SEC filing index contract")


def _query_rows(
    connection: Any,
    *,
    tickers: list[str],
    forms: list[str],
) -> list[tuple[Any, ...]]:
    ticker_marks = ",".join("?" for _ in tickers)
    form_marks = ",".join("?" for _ in forms)
    query = (
        "SELECT accessionNumber, filingDate, reportDate, "
        "acceptanceDateTime, form, isXBRL, isInlineXBRL, "
        "primaryDocument, ticker, cik, hash "
        "FROM sec_filings "
        f"WHERE upper(ticker) IN ({ticker_marks}) "
        f"AND upper(form) IN ({form_marks}) "
        "ORDER BY acceptanceDateTime, ticker, form, accessionNumber"
    )
    return connection.execute(
        query,
        [*tickers, *forms],
    ).fetchall()


def _normalize_rows(
    rows: list[tuple[Any, ...]],
    *,
    as_of: Any,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for index, row in enumerate(rows):
        (
            accession,
            filing_date,
            report_date,
            accepted_at_raw,
            form,
            is_xbrl,
            is_inline_xbrl,
            primary_document,
            ticker,
            cik,
            stored_hash,
        ) = row
        accession = str(accession or "").strip()
        cik = str(cik or "").strip().zfill(10)
        ticker = str(ticker or "").upper().strip()
        form = str(form or "").upper().strip()
        primary_document = str(primary_document or "").strip()
        accepted_at = parse_timezone_aware(accepted_at_raw)
        reasons: list[str] = []
        if not accession:
            reasons.append("sec_accession_missing")
        if not cik.strip("0"):
            reasons.append("sec_cik_missing")
        if not ticker:
            reasons.append("sec_ticker_missing")
        if not form:
            reasons.append("sec_form_missing")
        if accepted_at is None:
            reasons.append("sec_acceptance_time_missing_or_invalid")
        elif accepted_at > as_of:
            reasons.append("sec_filing_accepted_after_as_of")
        if not primary_document:
            reasons.append("sec_primary_document_missing")
        expected_hash = hashlib.sha256(
            f"{accession}{cik}".encode()
        ).hexdigest()
        if str(stored_hash or "") != expected_hash:
            reasons.append("sec_collector_hash_mismatch")
        if expected_hash in seen_hashes:
            reasons.append("duplicate_sec_filing")
        locator = (
            _sec_archive_url(cik, accession, primary_document)
            if accession and cik and primary_document
            else None
        )
        canonical = {
            "ticker": ticker,
            "cik": cik,
            "form": form,
            "filing_date": str(filing_date or "") or None,
            "report_date": str(report_date or "") or None,
            "accepted_at": (
                accepted_at.isoformat() if accepted_at else None
            ),
            "accession_number": accession,
            "primary_document": primary_document,
            "source_locator": locator,
            "stored_hash": str(stored_hash or ""),
            "expected_collector_hash": expected_hash,
            "is_xbrl": bool(is_xbrl),
            "is_inline_xbrl": bool(is_inline_xbrl),
            "content_status": "metadata_only_primary_document_not_stored",
        }
        canonical["record_sha256"] = _canonical_sha256(canonical)
        if reasons:
            exclusions.append(
                {
                    "index": index,
                    "ticker": ticker or None,
                    "accession_number": accession or None,
                    "status": "excluded",
                    "reasons": sorted(set(reasons)),
                    "record_sha256": canonical["record_sha256"],
                }
            )
            continue
        seen_hashes.add(expected_hash)
        accepted.append(canonical)
    reason_counts = Counter(
        reason
        for exclusion in exclusions
        for reason in exclusion["reasons"]
    )
    return {
        "accepted": accepted,
        "exclusions": exclusions,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def render_saved_sec_filing_index_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    boundary = payload.get("fundamental_boundary", {})
    lines = [
        "# DEAN-OS Saved SEC Filing Index",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Database: `{payload.get('inputs', {}).get('database_path')}`",
        f"- As-of: `{payload.get('inputs', {}).get('as_of')}`",
        f"- Tickers: {', '.join(payload.get('inputs', {}).get('tickers', []))}",
        f"- Forms: {', '.join(payload.get('inputs', {}).get('forms', []))}",
        f"- Query rows: {summary.get('query_row_count', 0)}",
        f"- Accepted filings: {summary.get('accepted_filing_count', 0)}",
        f"- Ticker coverage: `{summary.get('ticker_coverage_status')}` ({summary.get('ticker_coverage_ratio', 0)})",
        f"- Missing tickers: {', '.join(summary.get('missing_tickers', [])) or 'none'}",
        f"- XBRL filings: {summary.get('xbrl_filing_count', 0)}",
        f"- Fingerprint: `{summary.get('accepted_fingerprint')}`",
        f"- Fundamental metrics available: {boundary.get('xbrl_fact_values_stored', False)}",
        f"- Can feed value screening: {summary.get('can_feed_value_screening', False)}",
        f"- Can trade: {summary.get('can_trade', False)}",
        "",
        "## Filings",
        "",
    ]
    filings = payload.get("filings", [])
    if filings:
        lines.extend(
            (
                f"- `{item['ticker']}` `{item['form']}` "
                f"report=`{item['report_date']}` "
                f"accepted=`{item['accepted_at']}` "
                f"accession=`{item['accession_number']}` "
                f"content=`{item['content_status']}`"
            )
            for item in filings
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"- {boundary.get('reason', 'Filing content is unavailable.')}",
            "- The index may request immutable filing content and XBRL facts; it does not fetch them.",
            "- Ratio computation, valuation, recommendation, learning, paper execution, and trading remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def _database_provenance(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "size_bytes": None,
            "last_write_time_utc": None,
        }
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat.st_size,
        "last_write_time_utc": utc_now_iso()
        if stat.st_mtime is None
        else datetime.fromtimestamp(
            stat.st_mtime,
            tz=UTC,
        ).isoformat(),
        "read_only_connection": True,
    }


def _sec_archive_url(
    cik: str,
    accession: str,
    primary_document: str,
) -> str:
    cik_path = str(int(cik))
    accession_path = accession.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_path}/{accession_path}/{primary_document}"
    )


def _blocked(status: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "summary": {
            "accepted_filing_count": 0,
            "excluded_filing_count": 0,
            "reason_counts": dict.fromkeys(reasons, 1),
            "can_request_filing_content": False,
            "can_create_fundamental_metrics": False,
            "can_run_ratio_templates": False,
            "can_feed_value_screening": False,
            "can_trade": False,
        },
        "filings": [],
        "exclusions": [
            {"status": "excluded", "reasons": reasons}
        ],
    }


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "database_opened_read_only": True,
        "network_access_performed": False,
        "filing_content_fetch_performed": False,
        "xbrl_fact_fetch_performed": False,
        "ratio_computation_performed": False,
        "valuation_performed": False,
        "learning_write_performed": False,
        "paper_execution_performed": False,
        "live_execution_performed": False,
        "can_trade": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _run_id() -> str:
    return (
        "saved_sec_filing_index_"
        f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )
