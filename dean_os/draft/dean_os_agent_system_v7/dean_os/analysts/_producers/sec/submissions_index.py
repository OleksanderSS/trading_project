from __future__ import annotations

__all__ = [
    'DEFAULT_FORMS',
    'RECENT_REQUIRED_FIELDS',
    'SAVED_SEC_SUBMISSIONS_INDEX_CONTRACT',
    'SEC_SUBMISSIONS_SNAPSHOT_CONTRACT',
    'SavedSECSubmissionsFilingIndexProducer',
    'render_submissions_index_markdown',
    'verify_saved_sec_submissions_filing_index',
]

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso

SAVED_SEC_SUBMISSIONS_INDEX_CONTRACT = (
    "dean_saved_sec_submissions_filing_index_v1"
)
SEC_SUBMISSIONS_SNAPSHOT_CONTRACT = (
    "dean_sec_submissions_snapshot_v1"
)
DEFAULT_FORMS = ("10-K", "10-Q", "20-F", "40-F")
RECENT_REQUIRED_FIELDS = {
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "isXBRL",
    "isInlineXBRL",
    "primaryDocument",
}


class SavedSECSubmissionsFilingIndexProducer:
    """Create a latest-periodic filing index from immutable submissions."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_sec_submissions_filing_index"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        submissions_snapshot_path: str | Path,
        tickers: list[str],
        as_of: str,
        forms: list[str] | tuple[str, ...] = DEFAULT_FORMS,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError(
                "SEC submissions index as_of must be timezone-aware"
            )
        requested_tickers = sorted(
            {
                str(value).upper().strip()
                for value in tickers
                if str(value).strip()
            }
        )
        normalized_forms = sorted(
            {
                str(value).upper().strip()
                for value in forms
                if str(value).strip()
            }
        )
        if not requested_tickers:
            raise ValueError("at least one submissions ticker is required")
        snapshot_path = Path(submissions_snapshot_path)
        snapshot = json.loads(
            snapshot_path.read_text(encoding="utf-8")
        )
        _validate_snapshot(snapshot)
        normalized = _normalize_snapshot(
            snapshot=snapshot,
            tickers=requested_tickers,
            forms=normalized_forms,
            as_of=as_of_dt,
        )
        filings = normalized["selected"]
        represented = sorted({item["ticker"] for item in filings})
        missing = sorted(set(requested_tickers) - set(represented))
        fingerprint = _filing_fingerprint(filings)
        status = (
            "submissions_filing_index_ready_with_exclusions"
            if filings and normalized["exclusions"]
            else "submissions_filing_index_ready"
            if filings
            else "blocked_no_admissible_submissions_filings"
        )
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "saved_sec_submissions_filing_index_producer",
            "producer_contract": SAVED_SEC_SUBMISSIONS_INDEX_CONTRACT,
            "inputs": {
                "submissions_snapshot_path": str(snapshot_path),
                "submissions_snapshot_sha256": _sha256_file(
                    snapshot_path
                ),
                "tickers": requested_tickers,
                "forms": normalized_forms,
                "as_of": as_of_dt.isoformat(),
                "selection_rule": "latest_accepted_periodic_filing_per_ticker",
            },
            "status": status,
            "source_provenance": normalized["source_provenance"],
            "summary": {
                "snapshot_count": len(snapshot.get("snapshots", [])),
                "periodic_candidate_count": normalized[
                    "periodic_candidate_count"
                ],
                "not_selected_older_periodic_count": normalized[
                    "not_selected_older_periodic_count"
                ],
                "accepted_filing_count": len(filings),
                "excluded_filing_count": len(
                    normalized["exclusions"]
                ),
                "requested_ticker_count": len(requested_tickers),
                "ticker_count": len(represented),
                "represented_tickers": represented,
                "missing_tickers": missing,
                "ticker_coverage_ratio": _ratio(
                    len(represented),
                    len(requested_tickers),
                ),
                "ticker_coverage_status": (
                    "complete" if not missing else "partial"
                ),
                "xbrl_filing_count": sum(
                    bool(item["is_xbrl"]) for item in filings
                ),
                "accepted_fingerprint": fingerprint,
                "reason_counts": normalized["reason_counts"],
                "can_request_filing_content": bool(filings),
                "can_create_fundamental_metrics": False,
                "can_run_ratio_templates": False,
                "can_feed_value_screening": False,
                "can_trade": False,
            },
            "filings": filings,
            "exclusions": normalized["exclusions"],
            "content_extraction_requests": [
                {
                    "request_id": (
                        f"sec_content_{item['record_sha256'][:16]}"
                    ),
                    "ticker": item["ticker"],
                    "form": item["form"],
                    "accession_number": item["accession_number"],
                    "source_locator": item["source_locator"],
                    "requested_outputs": [
                        "immutable_primary_document",
                        "primary_document_sha256",
                        "xbrl_facts_with_units_periods_and_contexts",
                    ],
                    "network_request_executed": False,
                    "status": "pending_external_content_acquisition",
                }
                for item in filings
            ],
            "fundamental_boundary": {
                "filing_metadata_available": bool(filings),
                "primary_document_content_stored": False,
                "xbrl_fact_values_stored": False,
                "metric_extraction_allowed_now": False,
                "ratio_computation_allowed_now": False,
                "valuation_allowed_now": False,
                "reason": (
                    "This artifact verifies official submissions metadata; "
                    "content and facts remain separate source artifacts."
                ),
            },
            "safety": _safety(),
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_submissions_index_markdown(payload),
                run_id=payload["run_id"],
            )
        return payload


def verify_saved_sec_submissions_filing_index(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("producer_contract")
        != SAVED_SEC_SUBMISSIONS_INDEX_CONTRACT
    ):
        raise ValueError("unsupported submissions filing index contract")
    if payload.get("status") not in {
        "submissions_filing_index_ready",
        "submissions_filing_index_ready_with_exclusions",
    }:
        raise ValueError("submissions filing index is not ready")
    as_of = parse_timezone_aware(payload.get("inputs", {}).get("as_of"))
    if as_of is None:
        raise ValueError("submissions filing index as_of invalid")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != as_of:
            raise ValueError("submissions filing index expected as_of mismatch")
    snapshot_path = Path(
        payload.get("inputs", {}).get("submissions_snapshot_path", "")
    )
    if (
        not snapshot_path.exists()
        or _sha256_file(snapshot_path)
        != payload.get("inputs", {}).get(
            "submissions_snapshot_sha256"
        )
    ):
        raise ValueError("submissions snapshot artifact hash mismatch")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    _validate_snapshot(snapshot)
    for item in snapshot.get("snapshots", []):
        raw_path = Path(str(item.get("immutable_path") or ""))
        expected_sha = str(item.get("sha256") or "")
        if (
            not raw_path.exists()
            or not expected_sha
            or _sha256_file(raw_path) != expected_sha
        ):
            raise ValueError("submissions source hash mismatch")
    normalized = _normalize_snapshot(
        snapshot=snapshot,
        tickers=list(payload.get("inputs", {}).get("tickers", [])),
        forms=list(payload.get("inputs", {}).get("forms", [])),
        as_of=as_of,
    )
    filings = normalized["selected"]
    artifact_filings = payload.get("filings")
    if not isinstance(artifact_filings, list):
        raise ValueError("submissions filing records missing")
    if [
        item.get("record_sha256") for item in filings
    ] != [
        item.get("record_sha256") for item in artifact_filings
    ]:
        raise ValueError("submissions filing artifact payload mismatch")
    fingerprint = _filing_fingerprint(filings)
    if fingerprint != payload.get("summary", {}).get(
        "accepted_fingerprint"
    ):
        raise ValueError("submissions filing index fingerprint mismatch")
    return {
        "as_of": as_of.isoformat(),
        "filings": filings,
        "fingerprint": fingerprint,
        "artifact_path": str(path),
        "artifact_sha256": _sha256_file(path),
        "fundamental_metrics_available": False,
        "verified": True,
    }


def _normalize_snapshot(
    *,
    snapshot: dict[str, Any],
    tickers: list[str],
    forms: list[str],
    as_of: Any,
) -> dict[str, Any]:
    ticker_set = {str(value).upper() for value in tickers}
    form_set = {str(value).upper() for value in forms}
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_provenance: list[dict[str, Any]] = []
    for snapshot_item in snapshot.get("snapshots", []):
        ticker = str(snapshot_item.get("ticker") or "").upper()
        if ticker not in ticker_set:
            continue
        cik = str(snapshot_item.get("cik") or "").zfill(10)
        raw_path = Path(str(snapshot_item.get("immutable_path") or ""))
        expected_sha = str(snapshot_item.get("sha256") or "")
        if (
            not raw_path.exists()
            or not expected_sha
            or _sha256_file(raw_path) != expected_sha
        ):
            exclusions.append(
                {
                    "ticker": ticker,
                    "status": "excluded",
                    "reasons": ["sec_submissions_source_hash_mismatch"],
                }
            )
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        payload_cik = str(raw.get("cik") or "").zfill(10)
        payload_tickers = {
            str(value).upper() for value in raw.get("tickers", [])
        }
        if payload_cik != cik or ticker not in payload_tickers:
            exclusions.append(
                {
                    "ticker": ticker,
                    "status": "excluded",
                    "reasons": ["sec_submissions_identity_mismatch"],
                }
            )
            continue
        source_provenance.append(
            {
                "ticker": ticker,
                "cik": cik,
                "path": str(raw_path),
                "sha256": expected_sha,
                "size_bytes": raw_path.stat().st_size,
                "source_url": snapshot_item.get("source_url"),
                "status": "submissions_source_verified",
            }
        )
        recent = raw.get("filings", {}).get("recent", {})
        missing_fields = sorted(
            RECENT_REQUIRED_FIELDS - set(recent)
        )
        if missing_fields:
            exclusions.append(
                {
                    "ticker": ticker,
                    "status": "excluded",
                    "reasons": [
                        f"sec_submissions_field_missing_{field}"
                        for field in missing_fields
                    ],
                }
            )
            continue
        count = len(recent["accessionNumber"])
        for index in range(count):
            try:
                row = {
                    field: recent[field][index]
                    for field in RECENT_REQUIRED_FIELDS
                }
            except (IndexError, TypeError):
                exclusions.append(
                    {
                        "ticker": ticker,
                        "index": index,
                        "status": "excluded",
                        "reasons": [
                            "sec_submissions_parallel_array_mismatch"
                        ],
                    }
                )
                continue
            form = str(row["form"] or "").upper()
            if form not in form_set:
                continue
            record, reasons = _normalize_recent_row(
                row=row,
                ticker=ticker,
                cik=cik,
                as_of=as_of,
            )
            if reasons:
                exclusions.append(
                    {
                        "ticker": ticker,
                        "index": index,
                        "accession_number": row.get(
                            "accessionNumber"
                        ),
                        "status": "excluded",
                        "reasons": reasons,
                    }
                )
            else:
                candidates.append(record)
    selected_by_ticker: dict[str, dict[str, Any]] = {}
    for item in sorted(
        candidates,
        key=lambda value: (
            value["ticker"],
            value["accepted_at"],
            value["accession_number"],
        ),
    ):
        selected_by_ticker[item["ticker"]] = item
    selected = [
        selected_by_ticker[key] for key in sorted(selected_by_ticker)
    ]
    reason_counts = Counter(
        reason
        for exclusion in exclusions
        for reason in exclusion.get("reasons", [])
    )
    return {
        "selected": selected,
        "periodic_candidate_count": len(candidates),
        "not_selected_older_periodic_count": (
            len(candidates) - len(selected)
        ),
        "exclusions": exclusions,
        "reason_counts": dict(sorted(reason_counts.items())),
        "source_provenance": source_provenance,
    }


def _normalize_recent_row(
    *,
    row: dict[str, Any],
    ticker: str,
    cik: str,
    as_of: Any,
) -> tuple[dict[str, Any], list[str]]:
    accession = str(row.get("accessionNumber") or "").strip()
    primary_document = str(
        row.get("primaryDocument") or ""
    ).strip()
    accepted_at = parse_timezone_aware(
        row.get("acceptanceDateTime")
    )
    reasons: list[str] = []
    if not accession:
        reasons.append("sec_accession_missing")
    if not primary_document:
        reasons.append("sec_primary_document_missing")
    if accepted_at is None:
        reasons.append("sec_acceptance_time_missing_or_invalid")
    elif accepted_at > as_of:
        reasons.append("sec_filing_accepted_after_as_of")
    expected_hash = hashlib.sha256(
        f"{accession}{cik}".encode()
    ).hexdigest()
    source_locator = (
        _sec_archive_url(cik, accession, primary_document)
        if accession and primary_document
        else None
    )
    canonical = {
        "ticker": ticker,
        "cik": cik,
        "form": str(row.get("form") or "").upper(),
        "filing_date": str(row.get("filingDate") or "") or None,
        "report_date": str(row.get("reportDate") or "") or None,
        "accepted_at": (
            accepted_at.isoformat() if accepted_at else None
        ),
        "accession_number": accession,
        "primary_document": primary_document,
        "source_locator": source_locator,
        "stored_hash": expected_hash,
        "expected_collector_hash": expected_hash,
        "is_xbrl": bool(row.get("isXBRL")),
        "is_inline_xbrl": bool(row.get("isInlineXBRL")),
        "content_status": "metadata_only_primary_document_not_stored",
    }
    canonical["record_sha256"] = _canonical_sha256(canonical)
    return canonical, reasons


def _filing_fingerprint(filings: list[dict[str, Any]]) -> str:
    return _canonical_sha256(
        [
            {
                key: item[key]
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
            for item in filings
        ]
    )


def _validate_snapshot(payload: dict[str, Any]) -> None:
    if (
        payload.get("snapshot_contract")
        != SEC_SUBMISSIONS_SNAPSHOT_CONTRACT
    ):
        raise ValueError("unsupported SEC submissions snapshot contract")
    if payload.get("status") not in {
        "sec_submissions_snapshots_ready",
        "sec_submissions_snapshots_partial",
    }:
        raise ValueError("SEC submissions snapshot is not ready")
    safety = payload.get("safety", {})
    if (
        safety.get("official_sec_get_requests_only") is not True
        or safety.get("pipeline_run_performed") is not False
        or safety.get("live_execution_performed") is not False
        or safety.get("can_trade") is not False
    ):
        raise ValueError("SEC submissions snapshot safety invalid")


def render_submissions_index_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Saved SEC Submissions Filing Index",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- As-of: `{payload.get('inputs', {}).get('as_of')}`",
        (
            "- Tickers: "
            + (
                ", ".join(payload.get("inputs", {}).get("tickers", []))
                or "none"
            )
        ),
        f"- Periodic candidates: {summary.get('periodic_candidate_count', 0)}",
        (
            "- Older periodic candidates retained only as inventory: "
            f"{summary.get('not_selected_older_periodic_count', 0)}"
        ),
        f"- Accepted filings: {summary.get('accepted_filing_count', 0)}",
        f"- Fingerprint: `{summary.get('accepted_fingerprint')}`",
        f"- Can trade: {summary.get('can_trade', False)}",
        "",
        "## Selected Filings",
        "",
    ]
    filings = payload.get("filings", [])
    if filings:
        lines.extend(
            (
                f"- `{item['ticker']}` `{item['form']}` "
                f"report=`{item['report_date']}` "
                f"accepted=`{item['accepted_at']}` "
                f"accession=`{item['accession_number']}`"
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
            "- The latest admissible periodic filing is selected per ticker.",
            "- Older periodic filings remain source inventory and do not overwrite the selected context.",
            "- Filing content, facts, ratios, valuation, learning, and trading remain separate.",
            "",
        ]
    )
    return "\n".join(lines)


def _sec_archive_url(
    cik: str,
    accession: str,
    primary_document: str,
) -> str:
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{primary_document}"
    )


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "saved_submissions_read_performed": True,
        "filing_content_fetch_performed": False,
        "xbrl_fact_fetch_performed": False,
        "ratio_computation_performed": False,
        "valuation_performed": False,
        "pipeline_run_performed": False,
        "training_run_performed": False,
        "learning_write_performed": False,
        "paper_execution_performed": False,
        "live_execution_performed": False,
        "can_trade": False,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return (
        round(numerator / denominator, 6)
        if denominator > 0
        else 0.0
    )


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
        "saved_sec_submissions_filing_index_"
        f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )
