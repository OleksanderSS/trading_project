from __future__ import annotations

__all__ = [
    'ANNUAL_FORMS',
    'COMPANYFACTS_URL',
    'DEFAULT_METRIC_REGISTRY',
    'DEFAULT_RAW_DIR',
    'QUARTERLY_FORMS',
    'SAVED_SEC_COMPANYFACTS_CONTRACT',
    'SEC_COMPANYFACTS_SNAPSHOT_CONTRACT',
    'SUBMISSIONS_URL',
    'SavedSECCompanyFactsProducer',
    'fetch_companyfacts_snapshots',
    'fetch_primary_filing_snapshots',
    'fetch_sec_submissions_snapshots',
    'load_verified_fundamental_context_fragment',
    'render_saved_sec_companyfacts_markdown',
]

import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml

from dean_os.analysts._producers.sec.filing_index import (
    verify_sec_filing_index,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.structured_context_provenance import audit_structured_context

SAVED_SEC_COMPANYFACTS_CONTRACT = (
    "dean_saved_sec_companyfacts_evidence_v1"
)
SEC_COMPANYFACTS_SNAPSHOT_CONTRACT = (
    "dean_sec_companyfacts_snapshot_v1"
)
DEFAULT_METRIC_REGISTRY = (
    Path(__file__).parents[3]
    / "config"
    / "fundamental_metric_registry.yaml"
)
DEFAULT_RAW_DIR = Path("data/dean_os/sec_companyfacts_raw")
COMPANYFACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
)
SUBMISSIONS_URL = (
    "https://data.sec.gov/submissions/CIK{cik}.json"
)
ANNUAL_FORMS = {"10-K", "20-F", "40-F"}
QUARTERLY_FORMS = {"10-Q"}


class SavedSECCompanyFactsProducer:
    """Create point-in-time fundamental facts from saved SEC company facts.

    Only facts tied to an accession in a verified SavedSECFilingIndex artifact
    are eligible. The producer does not calculate ratios, translate currencies,
    infer missing periods, or treat a company-wide latest value as evidence for
    a specific filing.
    """

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_sec_companyfacts_producer"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        filing_index_path: str | Path,
        source_dir: str | Path = DEFAULT_RAW_DIR,
        registry_path: str | Path = DEFAULT_METRIC_REGISTRY,
        as_of: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        verified_index = verify_sec_filing_index(
            filing_index_path,
            expected_as_of=as_of,
        )
        resolved_as_of = verified_index["as_of"]
        as_of_dt = parse_timezone_aware(resolved_as_of)
        if as_of_dt is None:
            raise ValueError("verified SEC filing index as_of is invalid")

        registry_source = Path(registry_path)
        registry = _load_registry(registry_source)
        run_id = _run_id("saved_sec_companyfacts")
        source_root = Path(source_dir)
        filings = list(verified_index["filings"])
        requested_tickers = _requested_tickers(filing_index_path)
        filing_tickers = sorted(
            {str(item["ticker"]).upper() for item in filings}
        )
        missing_filing_tickers = sorted(
            set(requested_tickers) - set(filing_tickers)
        )
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "saved_sec_companyfacts_producer",
            "producer_contract": SAVED_SEC_COMPANYFACTS_CONTRACT,
            "inputs": {
                "filing_index_path": str(filing_index_path),
                "filing_index_fingerprint": verified_index[
                    "fingerprint"
                ],
                "filing_index_artifact_sha256": verified_index[
                    "artifact_sha256"
                ],
                "source_dir": str(source_root),
                "registry_path": str(registry_source),
                "as_of": resolved_as_of,
                "requested_tickers": requested_tickers,
            },
            "registry": {
                "registry_version": registry.get("registry_version"),
                "review_status": registry.get("review_status"),
                "metric_count": len(registry.get("metrics", {})),
                "path": str(registry_source),
                "sha256": (
                    _sha256_file(registry_source)
                    if registry_source.exists()
                    else None
                ),
            },
            "filing_coverage": {
                "requested_tickers": requested_tickers,
                "filing_tickers": filing_tickers,
                "missing_filing_tickers": missing_filing_tickers,
                "coverage_ratio": _ratio(
                    len(filing_tickers),
                    len(requested_tickers),
                ),
                "status": (
                    "complete"
                    if requested_tickers and not missing_filing_tickers
                    else "partial"
                ),
            },
        }
        if not registry.get("metrics"):
            payload.update(
                _blocked(
                    "blocked_metric_registry_missing_or_empty",
                    ["fundamental_metric_registry_missing_or_empty"],
                )
            )
            return self._finish(payload, save=save)

        normalized = _normalize_companyfacts_sources(
            filings=filings,
            source_root=source_root,
            registry=registry["metrics"],
            as_of=as_of_dt,
        )
        facts = normalized["facts"]
        fundamentals = _fundamentals_from_facts(facts)
        structured_audit = audit_structured_context(
            fundamentals=fundamentals,
            macro={},
            sector_data={},
            as_of=resolved_as_of,
        )
        accepted_fundamentals = structured_audit[
            "accepted_context"
        ]["fundamentals"]
        accepted_tickers = sorted(accepted_fundamentals)
        comparability = _comparability(
            facts=facts,
            requested_tickers=requested_tickers,
        )
        facts_fingerprint = _canonical_sha256(
            [_fact_fingerprint_row(item) for item in facts]
        )
        status = (
            "fundamental_facts_ready_with_gaps"
            if facts
            and (
                normalized["exclusions"]
                or missing_filing_tickers
                or set(accepted_tickers) != set(requested_tickers)
            )
            else "fundamental_facts_ready"
            if facts
            else "blocked_no_admissible_fundamental_facts"
        )
        payload.update(
            {
                "status": status,
                "source_provenance": normalized[
                    "source_provenance"
                ],
                "summary": {
                    "requested_ticker_count": len(requested_tickers),
                    "filing_ticker_count": len(filing_tickers),
                    "source_ticker_count": len(
                        normalized["source_tickers"]
                    ),
                    "accepted_fact_ticker_count": len(
                        accepted_tickers
                    ),
                    "accepted_fact_tickers": accepted_tickers,
                    "missing_filing_tickers": missing_filing_tickers,
                    "missing_source_tickers": normalized[
                        "missing_source_tickers"
                    ],
                    "accepted_fact_count": structured_audit[
                        "accepted_count"
                    ],
                    "excluded_fact_count": len(
                        normalized["exclusions"]
                    ),
                    "accepted_fingerprint": structured_audit[
                        "accepted_fingerprint"
                    ],
                    "facts_fingerprint": facts_fingerprint,
                    "reason_counts": normalized["reason_counts"],
                    "filing_coverage_status": payload[
                        "filing_coverage"
                    ]["status"],
                    "fact_ticker_coverage_ratio": _ratio(
                        len(accepted_tickers),
                        len(requested_tickers),
                    ),
                    "cross_ticker_comparability_status": comparability[
                        "status"
                    ],
                    "can_enter_fundamental_input_gate": bool(
                        accepted_fundamentals
                    ),
                    "requested_scope_complete": (
                        bool(requested_tickers)
                        and set(accepted_tickers)
                        == set(requested_tickers)
                    ),
                    "can_claim_complete_sector_fundamentals": (
                        len(requested_tickers) > 1
                        and set(accepted_tickers)
                        == set(requested_tickers)
                        and comparability["status"] == "comparable"
                    ),
                    "can_compute_ratios": False,
                    "can_translate_currencies": False,
                    "can_feed_ticker_prediction_directly": False,
                    "can_trade": False,
                },
                "fundamental_metric_rows": [
                    _gate_metric_row(item) for item in facts
                ],
                "facts": facts,
                "exclusions": normalized["exclusions"],
                "comparability": comparability,
                "structured_context_audit": {
                    key: value
                    for key, value in structured_audit.items()
                    if key
                    not in {
                        "accepted_context",
                        "accepted_observations",
                    }
                },
                "market_context_fragment": {
                    "as_of": resolved_as_of,
                    "fundamentals": accepted_fundamentals,
                    "metadata": {
                        "saved_sec_companyfacts_run_id": run_id,
                        "saved_sec_companyfacts_facts_fingerprint": (
                            facts_fingerprint
                        ),
                        "saved_sec_companyfacts_accepted_fingerprint": (
                            structured_audit["accepted_fingerprint"]
                        ),
                        "filing_index_fingerprint": verified_index[
                            "fingerprint"
                        ],
                        "requested_tickers": requested_tickers,
                        "accepted_fact_tickers": accepted_tickers,
                        "missing_filing_tickers": (
                            missing_filing_tickers
                        ),
                        "complete_sector_fundamentals": payload.get(
                            "summary", {}
                        ).get(
                            "can_claim_complete_sector_fundamentals",
                            False,
                        ),
                    },
                },
                "integration_boundary": {
                    "review_only": True,
                    "facts_are_bound_to_verified_accessions": True,
                    "company_wide_latest_value_fallback_allowed": False,
                    "missing_period_inference_allowed": False,
                    "currency_translation_allowed": False,
                    "ratio_computation_allowed": False,
                    "sector_generalization_requires_full_coverage": True,
                    "ticker_prediction_feature_promotion_allowed": False,
                },
                "safety": _producer_safety(),
            }
        )
        payload["market_context_fragment"]["metadata"][
            "complete_sector_fundamentals"
        ] = payload["summary"][
            "can_claim_complete_sector_fundamentals"
        ]
        return self._finish(payload, save=save)

    def _finish(
        self,
        payload: dict[str, Any],
        *,
        save: bool,
    ) -> dict[str, Any]:
        payload.setdefault("safety", _producer_safety())
        payload.setdefault("integration_boundary", {})
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_saved_sec_companyfacts_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
        return payload


def fetch_companyfacts_snapshots(
    *,
    filing_index_path: str | Path,
    output_dir: str | Path = DEFAULT_RAW_DIR,
    user_agent: str,
    expected_as_of: str | None = None,
    request_delay_seconds: float = 0.25,
    timeout_seconds: float = 30.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Fetch one immutable official companyfacts snapshot per filing CIK."""

    declared_user_agent = str(user_agent or "").strip()
    if len(declared_user_agent) < 8:
        raise ValueError(
            "SEC user agent must identify the research tool or operator"
        )
    verified = verify_sec_filing_index(
        filing_index_path,
        expected_as_of=expected_as_of,
    )
    root = Path(output_dir)
    unique = {
        filing["cik"]: filing["ticker"]
        for filing in verified["filings"]
    }
    snapshots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, (cik, ticker) in enumerate(sorted(unique.items())):
        if position and request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        url = COMPANYFACTS_URL.format(cik=cik)
        request = Request(
            url,
            headers={
                "User-Agent": declared_user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
            method="GET",
        )
        try:
            response = opener(request, timeout=timeout_seconds)
            with response:
                raw = response.read()
                encoding = str(
                    response.headers.get("Content-Encoding", "")
                ).lower()
            if "gzip" in encoding:
                raw = gzip.decompress(raw)
            parsed = json.loads(raw.decode("utf-8"))
            payload_cik = str(parsed.get("cik") or "").zfill(10)
            if payload_cik != cik:
                raise ValueError(
                    f"companyfacts CIK mismatch: expected={cik}, "
                    f"actual={payload_cik}"
                )
            sha256 = hashlib.sha256(raw).hexdigest()
            cik_dir = root / f"CIK{cik}"
            immutable_path = cik_dir / f"{sha256}.json"
            latest_path = cik_dir / "latest.json"
            _atomic_write_bytes(immutable_path, raw)
            _atomic_write_bytes(latest_path, raw)
            snapshots.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "source_url": url,
                    "sha256": sha256,
                    "size_bytes": len(raw),
                    "immutable_path": str(immutable_path),
                    "latest_path": str(latest_path),
                    "downloaded_at": utc_now_iso(),
                    "status": "snapshot_saved",
                }
            )
        except Exception as exc:  # network boundary is reported per CIK
            failures.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "source_url": url,
                    "status": "snapshot_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "run_id": _run_id("sec_companyfacts_snapshot"),
        "created_at": utc_now_iso(),
        "mode": "sec_companyfacts_snapshot",
        "snapshot_contract": SEC_COMPANYFACTS_SNAPSHOT_CONTRACT,
        "inputs": {
            "filing_index_path": str(filing_index_path),
            "filing_index_fingerprint": verified["fingerprint"],
            "as_of": verified["as_of"],
            "output_dir": str(root),
            "request_delay_seconds": request_delay_seconds,
            "timeout_seconds": timeout_seconds,
            "declared_user_agent_present": True,
        },
        "status": (
            "companyfacts_snapshots_ready"
            if snapshots and not failures
            else "companyfacts_snapshots_partial"
            if snapshots
            else "companyfacts_snapshots_failed"
        ),
        "summary": {
            "requested_cik_count": len(unique),
            "saved_snapshot_count": len(snapshots),
            "failed_snapshot_count": len(failures),
            "can_build_saved_fundamental_evidence": bool(snapshots),
            "can_trade": False,
        },
        "snapshots": snapshots,
        "failures": failures,
        "safety": {
            "official_sec_get_requests_only": True,
            "network_access_performed": True,
            "pipeline_run_performed": False,
            "training_run_performed": False,
            "learning_write_performed": False,
            "production_config_write_performed": False,
            "paper_execution_performed": False,
            "live_execution_performed": False,
            "can_trade": False,
        },
    }


def fetch_primary_filing_snapshots(
    *,
    filing_index_path: str | Path,
    output_dir: str | Path = "data/dean_os/sec_filings_raw",
    user_agent: str,
    tickers: list[str] | None = None,
    expected_as_of: str | None = None,
    request_delay_seconds: float = 0.25,
    timeout_seconds: float = 30.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Fetch immutable primary documents from a verified filing index."""

    declared_user_agent = str(user_agent or "").strip()
    if len(declared_user_agent) < 8:
        raise ValueError(
            "SEC user agent must identify the research tool or operator"
        )
    verified = verify_sec_filing_index(
        filing_index_path,
        expected_as_of=expected_as_of,
    )
    requested = {
        str(ticker).upper().strip()
        for ticker in tickers or []
        if str(ticker).strip()
    }
    filings = [
        filing
        for filing in verified["filings"]
        if not requested or filing["ticker"] in requested
    ]
    root = Path(output_dir)
    snapshots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, filing in enumerate(filings):
        if position and request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        url = str(filing["source_locator"])
        request = Request(
            url,
            headers={
                "User-Agent": declared_user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "gzip, deflate",
            },
            method="GET",
        )
        try:
            response = opener(request, timeout=timeout_seconds)
            with response:
                raw = response.read()
                encoding = str(
                    response.headers.get("Content-Encoding", "")
                ).lower()
                content_type = str(
                    response.headers.get("Content-Type", "")
                )
            if "gzip" in encoding:
                raw = gzip.decompress(raw)
            if not raw:
                raise ValueError("SEC primary document response is empty")
            sha256 = hashlib.sha256(raw).hexdigest()
            accession_path = _normalized_accession(
                filing["accession_number"]
            )
            filing_dir = (
                root
                / f"CIK{filing['cik']}"
                / accession_path
            )
            suffix = (
                Path(str(filing["primary_document"])).suffix
                or ".htm"
            )
            immutable_path = filing_dir / f"{sha256}{suffix}"
            latest_path = filing_dir / (
                f"latest{suffix}"
            )
            _atomic_write_bytes(immutable_path, raw)
            _atomic_write_bytes(latest_path, raw)
            snapshots.append(
                {
                    "ticker": filing["ticker"],
                    "cik": filing["cik"],
                    "form": filing["form"],
                    "report_date": filing["report_date"],
                    "accepted_at": filing["accepted_at"],
                    "accession_number": filing["accession_number"],
                    "primary_document": filing["primary_document"],
                    "source_url": url,
                    "content_type": content_type,
                    "sha256": sha256,
                    "size_bytes": len(raw),
                    "immutable_path": str(immutable_path),
                    "latest_path": str(latest_path),
                    "downloaded_at": utc_now_iso(),
                    "status": "primary_document_snapshot_saved",
                }
            )
        except Exception as exc:  # network boundary reported per filing
            failures.append(
                {
                    "ticker": filing["ticker"],
                    "cik": filing["cik"],
                    "form": filing["form"],
                    "accession_number": filing["accession_number"],
                    "source_url": url,
                    "status": "primary_document_snapshot_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "run_id": _run_id("sec_primary_document_snapshot"),
        "created_at": utc_now_iso(),
        "mode": "sec_primary_document_snapshot",
        "snapshot_contract": "dean_sec_primary_document_snapshot_v1",
        "inputs": {
            "filing_index_path": str(filing_index_path),
            "filing_index_fingerprint": verified["fingerprint"],
            "as_of": verified["as_of"],
            "output_dir": str(root),
            "tickers": sorted(requested),
            "request_delay_seconds": request_delay_seconds,
            "timeout_seconds": timeout_seconds,
            "declared_user_agent_present": True,
        },
        "status": (
            "primary_document_snapshots_ready"
            if snapshots and not failures
            else "primary_document_snapshots_partial"
            if snapshots
            else "primary_document_snapshots_failed"
        ),
        "summary": {
            "requested_filing_count": len(filings),
            "saved_snapshot_count": len(snapshots),
            "failed_snapshot_count": len(failures),
            "can_build_inline_xbrl_evidence": bool(snapshots),
            "can_trade": False,
        },
        "snapshots": snapshots,
        "failures": failures,
        "safety": {
            "official_sec_get_requests_only": True,
            "network_access_performed": True,
            "pipeline_run_performed": False,
            "training_run_performed": False,
            "learning_write_performed": False,
            "production_config_write_performed": False,
            "paper_execution_performed": False,
            "live_execution_performed": False,
            "can_trade": False,
        },
    }


def fetch_sec_submissions_snapshots(
    *,
    tickers: list[str],
    output_dir: str | Path = "data/dean_os/sec_submissions_raw",
    user_agent: str,
    assets_config_path: str | Path = "src/config/assets.yaml",
    request_delay_seconds: float = 0.25,
    timeout_seconds: float = 30.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Fetch immutable official submissions snapshots for configured CIKs."""

    declared_user_agent = str(user_agent or "").strip()
    if len(declared_user_agent) < 8:
        raise ValueError(
            "SEC user agent must identify the research tool or operator"
        )
    normalized_tickers = sorted(
        {
            str(ticker).upper().strip()
            for ticker in tickers
            if str(ticker).strip()
        }
    )
    if not normalized_tickers:
        raise ValueError("at least one submissions ticker is required")
    config_path = Path(assets_config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    details = (
        config.get("assets", {}).get("details", {})
        if isinstance(config, dict)
        else {}
    )
    ticker_ciks: dict[str, str] = {}
    missing_config: list[str] = []
    for ticker in normalized_tickers:
        raw_cik = (
            details.get(ticker, {}).get("cik")
            if isinstance(details.get(ticker), dict)
            else None
        )
        cik = str(raw_cik or "").zfill(10)
        if not cik.strip("0"):
            missing_config.append(ticker)
        else:
            ticker_ciks[ticker] = cik
    root = Path(output_dir)
    snapshots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = [
        {
            "ticker": ticker,
            "status": "submissions_snapshot_failed",
            "error_type": "CIKConfigError",
            "error": "ticker CIK missing from assets config",
        }
        for ticker in missing_config
    ]
    for position, (ticker, cik) in enumerate(
        sorted(ticker_ciks.items())
    ):
        if position and request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        url = SUBMISSIONS_URL.format(cik=cik)
        request = Request(
            url,
            headers={
                "User-Agent": declared_user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
            method="GET",
        )
        try:
            response = opener(request, timeout=timeout_seconds)
            with response:
                raw = response.read()
                encoding = str(
                    response.headers.get("Content-Encoding", "")
                ).lower()
            if "gzip" in encoding:
                raw = gzip.decompress(raw)
            parsed = json.loads(raw.decode("utf-8"))
            payload_cik = str(parsed.get("cik") or "").zfill(10)
            if payload_cik != cik:
                raise ValueError(
                    f"submissions CIK mismatch: expected={cik}, "
                    f"actual={payload_cik}"
                )
            tickers_in_payload = {
                str(value).upper()
                for value in parsed.get("tickers", [])
            }
            if ticker not in tickers_in_payload:
                raise ValueError(
                    f"submissions ticker mismatch: expected={ticker}"
                )
            sha256 = hashlib.sha256(raw).hexdigest()
            cik_dir = root / f"CIK{cik}"
            immutable_path = cik_dir / f"{sha256}.json"
            latest_path = cik_dir / "latest.json"
            _atomic_write_bytes(immutable_path, raw)
            _atomic_write_bytes(latest_path, raw)
            snapshots.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "source_url": url,
                    "sha256": sha256,
                    "size_bytes": len(raw),
                    "immutable_path": str(immutable_path),
                    "latest_path": str(latest_path),
                    "downloaded_at": utc_now_iso(),
                    "status": "submissions_snapshot_saved",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "source_url": url,
                    "status": "submissions_snapshot_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "run_id": _run_id("sec_submissions_snapshot"),
        "created_at": utc_now_iso(),
        "mode": "sec_submissions_snapshot",
        "snapshot_contract": "dean_sec_submissions_snapshot_v1",
        "inputs": {
            "tickers": normalized_tickers,
            "assets_config_path": str(config_path),
            "assets_config_sha256": _sha256_file(config_path),
            "output_dir": str(root),
            "request_delay_seconds": request_delay_seconds,
            "timeout_seconds": timeout_seconds,
            "declared_user_agent_present": True,
        },
        "status": (
            "sec_submissions_snapshots_ready"
            if snapshots and not failures
            else "sec_submissions_snapshots_partial"
            if snapshots
            else "sec_submissions_snapshots_failed"
        ),
        "summary": {
            "requested_cik_count": len(normalized_tickers),
            "saved_snapshot_count": len(snapshots),
            "failed_snapshot_count": len(failures),
            "can_build_saved_filing_index": bool(snapshots),
            "can_trade": False,
        },
        "snapshots": snapshots,
        "failures": failures,
        "safety": {
            "official_sec_get_requests_only": True,
            "network_access_performed": True,
            "pipeline_run_performed": False,
            "training_run_performed": False,
            "learning_write_performed": False,
            "production_config_write_performed": False,
            "paper_execution_performed": False,
            "live_execution_performed": False,
            "can_trade": False,
        },
    }


def load_verified_fundamental_context_fragment(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("producer_contract")
        != SAVED_SEC_COMPANYFACTS_CONTRACT
    ):
        raise ValueError("unsupported saved SEC companyfacts contract")
    if payload.get("status") not in {
        "fundamental_facts_ready",
        "fundamental_facts_ready_with_gaps",
    }:
        raise ValueError("saved SEC companyfacts artifact is not ready")
    safety = payload.get("safety", {})
    summary = payload.get("summary", {})
    if (
        safety.get("review_only") is not True
        or safety.get("network_access_performed") is not False
        or safety.get("live_execution_performed") is not False
        or summary.get("can_trade") is not False
    ):
        raise ValueError(
            "saved SEC companyfacts safety boundary is invalid"
        )
    fragment = payload.get("market_context_fragment")
    if not isinstance(fragment, dict):
        raise ValueError("saved SEC companyfacts fragment is missing")
    fragment_as_of = parse_timezone_aware(fragment.get("as_of"))
    if fragment_as_of is None:
        raise ValueError("saved SEC companyfacts fragment as_of invalid")
    input_as_of = parse_timezone_aware(
        payload.get("inputs", {}).get("as_of")
    )
    if input_as_of != fragment_as_of:
        raise ValueError(
            "saved SEC companyfacts fragment as_of mismatch"
        )
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != fragment_as_of:
            raise ValueError(
                "saved SEC companyfacts expected as_of mismatch"
            )

    filing_index = Path(
        payload.get("inputs", {}).get("filing_index_path", "")
    )
    verified_index = verify_sec_filing_index(
        filing_index,
        expected_as_of=fragment_as_of.isoformat(),
        verify_source_database=False,
    )
    if (
        verified_index["fingerprint"]
        != payload.get("inputs", {}).get(
            "filing_index_fingerprint"
        )
        or verified_index["artifact_sha256"]
        != payload.get("inputs", {}).get(
            "filing_index_artifact_sha256"
        )
    ):
        raise ValueError("SEC companyfacts filing index mismatch")
    registry = Path(payload.get("registry", {}).get("path", ""))
    if (
        not registry.exists()
        or _sha256_file(registry)
        != payload.get("registry", {}).get("sha256")
    ):
        raise ValueError("SEC companyfacts registry hash mismatch")
    for item in payload.get("source_provenance", []):
        source = Path(str(item.get("path") or ""))
        if (
            not source.exists()
            or _sha256_file(source) != item.get("sha256")
        ):
            raise ValueError(
                "SEC companyfacts source artifact hash mismatch"
            )

    fundamentals = fragment.get("fundamentals")
    if not isinstance(fundamentals, dict):
        raise ValueError(
            "saved SEC companyfacts fundamentals payload invalid"
        )
    audit = audit_structured_context(
        fundamentals=fundamentals,
        macro={},
        sector_data={},
        as_of=fragment_as_of.isoformat(),
    )
    if (
        audit["excluded_count"] != 0
        or audit["accepted_count"]
        != summary.get("accepted_fact_count")
        or audit["accepted_fingerprint"]
        != summary.get("accepted_fingerprint")
    ):
        raise ValueError(
            "saved SEC companyfacts fingerprint or count mismatch"
        )
    facts_fingerprint = _canonical_sha256(
        [
            _fact_fingerprint_row(item)
            for item in payload.get("facts", [])
        ]
    )
    if facts_fingerprint != summary.get("facts_fingerprint"):
        raise ValueError("saved SEC companyfacts fact payload mismatch")
    return {
        "as_of": fragment_as_of.isoformat(),
        "fundamentals": audit["accepted_context"]["fundamentals"],
        "metadata": {
            **dict(fragment.get("metadata", {})),
            "saved_sec_companyfacts_artifact_path": str(path),
            "saved_sec_companyfacts_artifact_sha256": _sha256_file(
                path
            ),
            "saved_sec_companyfacts_verified": True,
            "filing_index_verification_mode": verified_index.get(
                "verification_mode"
            ),
            "complete_sector_fundamentals": summary.get(
                "can_claim_complete_sector_fundamentals",
                False,
            ),
        },
    }


def _normalize_companyfacts_sources(
    *,
    filings: list[dict[str, Any]],
    source_root: Path,
    registry: dict[str, Any],
    as_of: datetime,
) -> dict[str, Any]:
    source_provenance: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_tickers: set[str] = set()
    missing_source_tickers: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for filing in filings:
        grouped[str(filing["cik"])].append(filing)

    for cik, cik_filings in sorted(grouped.items()):
        ticker = str(cik_filings[0]["ticker"]).upper()
        source_path = _companyfacts_source_path(source_root, cik)
        if source_path is None:
            missing_source_tickers.add(ticker)
            exclusions.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "metric_name": None,
                    "status": "excluded",
                    "reasons": ["sec_companyfacts_source_missing"],
                }
            )
            continue
        try:
            raw = source_path.read_bytes()
            companyfacts = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            missing_source_tickers.add(ticker)
            exclusions.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "metric_name": None,
                    "status": "excluded",
                    "reasons": [
                        "sec_companyfacts_source_unreadable"
                    ],
                    "error_type": type(exc).__name__,
                }
            )
            continue
        payload_cik = str(companyfacts.get("cik") or "").zfill(10)
        source_sha = hashlib.sha256(raw).hexdigest()
        provenance = {
            "ticker": ticker,
            "cik": cik,
            "path": str(source_path),
            "sha256": source_sha,
            "size_bytes": len(raw),
            "source_url": COMPANYFACTS_URL.format(cik=cik),
            "mtime_not_used_as_availability": True,
        }
        if payload_cik != cik:
            provenance["status"] = "blocked_cik_mismatch"
            source_provenance.append(provenance)
            missing_source_tickers.add(ticker)
            exclusions.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "metric_name": None,
                    "status": "excluded",
                    "reasons": ["sec_companyfacts_cik_mismatch"],
                }
            )
            continue
        provenance["status"] = "source_verified"
        source_provenance.append(provenance)
        source_tickers.add(ticker)
        for filing in cik_filings:
            accepted_at = parse_timezone_aware(
                filing.get("accepted_at")
            )
            if accepted_at is None or accepted_at > as_of:
                exclusions.append(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "accession_number": filing.get(
                            "accession_number"
                        ),
                        "metric_name": None,
                        "status": "excluded",
                        "reasons": [
                            "sec_filing_acceptance_not_as_of_compatible"
                        ],
                    }
                )
                continue
            for metric_name, metric_config in sorted(
                registry.items()
            ):
                fact, reasons = _select_metric_fact(
                    companyfacts=companyfacts,
                    filing=filing,
                    metric_name=str(metric_name),
                    metric_config=metric_config,
                    source_sha256=source_sha,
                )
                if fact is not None:
                    facts.append(fact)
                else:
                    exclusions.append(
                        {
                            "ticker": ticker,
                            "cik": cik,
                            "form": filing.get("form"),
                            "accession_number": filing.get(
                                "accession_number"
                            ),
                            "report_date": filing.get(
                                "report_date"
                            ),
                            "metric_name": metric_name,
                            "status": "excluded",
                            "reasons": reasons
                            or [
                                "sec_companyfacts_metric_not_found"
                            ],
                        }
                    )
    facts.sort(
        key=lambda item: (
            item["ticker"],
            item["accession_number"],
            item["metric_name"],
        )
    )
    reason_counts = Counter(
        reason
        for exclusion in exclusions
        for reason in exclusion.get("reasons", [])
    )
    return {
        "source_provenance": source_provenance,
        "source_tickers": sorted(source_tickers),
        "missing_source_tickers": sorted(missing_source_tickers),
        "facts": facts,
        "exclusions": exclusions,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _select_metric_fact(
    *,
    companyfacts: dict[str, Any],
    filing: dict[str, Any],
    metric_name: str,
    metric_config: Any,
    source_sha256: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(metric_config, dict):
        return None, ["fundamental_metric_registry_entry_invalid"]
    concepts = metric_config.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        return None, ["fundamental_metric_concepts_missing"]
    accepted_units = {
        str(value) for value in metric_config.get("accepted_units", [])
    }
    period_type = str(metric_config.get("period_type") or "")
    all_reasons: list[str] = []
    facts_root = companyfacts.get("facts", {})

    for mapping in concepts:
        if not isinstance(mapping, dict):
            continue
        taxonomy = str(mapping.get("taxonomy") or "")
        concept = str(mapping.get("concept") or "")
        concept_payload = (
            facts_root.get(taxonomy, {}).get(concept)
            if isinstance(facts_root, dict)
            else None
        )
        if not isinstance(concept_payload, dict):
            continue
        units = concept_payload.get("units", {})
        if not isinstance(units, dict):
            continue
        candidates: list[dict[str, Any]] = []
        saw_matching_accession = False
        for source_unit, values in units.items():
            if accepted_units and str(source_unit) not in accepted_units:
                continue
            if not isinstance(values, list):
                continue
            for raw in values:
                if not isinstance(raw, dict):
                    continue
                if _normalized_accession(raw.get("accn")) != (
                    _normalized_accession(
                        filing.get("accession_number")
                    )
                ):
                    continue
                saw_matching_accession = True
                candidate, reasons = _normalize_fact_candidate(
                    raw=raw,
                    source_unit=str(source_unit),
                    filing=filing,
                    period_type=period_type,
                )
                if candidate is not None:
                    candidates.append(candidate)
                else:
                    all_reasons.extend(reasons)
        if not candidates:
            if saw_matching_accession and not all_reasons:
                all_reasons.append(
                    "sec_companyfacts_context_not_admissible"
                )
            continue
        unique = _deduplicate_fact_candidates(candidates)
        if len(unique) != 1:
            return None, ["sec_companyfacts_metric_ambiguous"]
        selected = unique[0]
        canonical = {
            "ticker": str(filing["ticker"]).upper(),
            "cik": str(filing["cik"]),
            "metric_name": metric_name,
            "statement_role": metric_config.get(
                "statement_role"
            ),
            "taxonomy": taxonomy,
            "concept": concept,
            "concept_label": concept_payload.get("label"),
            "value": selected["value"],
            "unit": selected["unit"],
            "period_type": period_type,
            "period": selected["period"],
            "period_start": selected.get("period_start"),
            "period_end": selected["period_end"],
            "fiscal_year": selected.get("fiscal_year"),
            "fiscal_period": selected.get("fiscal_period"),
            "frame": selected.get("frame"),
            "form": str(filing["form"]).upper(),
            "filed_date": selected.get("filed_date"),
            "accepted_at": filing["accepted_at"],
            "available_at": filing["accepted_at"],
            "accession_number": filing["accession_number"],
            "filing_source_locator": filing["source_locator"],
            "source_locator": COMPANYFACTS_URL.format(
                cik=filing["cik"]
            ),
            "source_artifact_sha256": source_sha256,
            "source_fact_sha256": _canonical_sha256(
                {
                    "taxonomy": taxonomy,
                    "concept": concept,
                    "unit": selected["unit"],
                    "raw": selected["raw"],
                    "source_artifact_sha256": source_sha256,
                }
            ),
        }
        canonical["fact_sha256"] = _canonical_sha256(canonical)
        return canonical, []
    return None, sorted(
        set(
            all_reasons
            or ["sec_companyfacts_metric_not_found_for_accession"]
        )
    )


def _normalize_fact_candidate(
    *,
    raw: dict[str, Any],
    source_unit: str,
    filing: dict[str, Any],
    period_type: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    value = _finite_float(raw.get("val"))
    if value is None:
        reasons.append("sec_companyfacts_value_invalid")
    form = str(raw.get("form") or "").upper()
    if form != str(filing.get("form") or "").upper():
        reasons.append("sec_companyfacts_form_mismatch")
    report_end = _parse_date(filing.get("report_date"))
    end = _parse_date(raw.get("end"))
    start = _parse_date(raw.get("start"))
    if report_end is None or end != report_end:
        reasons.append("sec_companyfacts_report_end_mismatch")
    if period_type == "instant":
        if start is not None:
            reasons.append("sec_companyfacts_instant_has_start")
        period = end.isoformat() if end else None
    elif period_type == "duration":
        if start is None or end is None or start > end:
            reasons.append("sec_companyfacts_duration_invalid")
        elif not _admissible_duration(
            form=form,
            days=(end - start).days + 1,
        ):
            reasons.append(
                "sec_companyfacts_duration_not_filing_period"
            )
        period = (
            f"{start.isoformat()}/{end.isoformat()}"
            if start and end
            else None
        )
    else:
        reasons.append("fundamental_metric_period_type_invalid")
        period = None
    if reasons:
        return None, reasons
    return {
        "value": value,
        "unit": source_unit,
        "period": period,
        "period_start": start.isoformat() if start else None,
        "period_end": end.isoformat() if end else None,
        "fiscal_year": raw.get("fy"),
        "fiscal_period": raw.get("fp"),
        "frame": raw.get("frame"),
        "filed_date": raw.get("filed"),
        "raw": raw,
    }, []


def _deduplicate_fact_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in candidates:
        identity = _canonical_sha256(
            {
                "value": item["value"],
                "unit": item["unit"],
                "period_start": item.get("period_start"),
                "period_end": item.get("period_end"),
            }
        )
        if identity not in unique:
            unique[identity] = item
        elif not unique[identity].get("frame") and item.get("frame"):
            unique[identity] = item
    return [unique[key] for key in sorted(unique)]


def _admissible_duration(*, form: str, days: int) -> bool:
    if form in QUARTERLY_FORMS:
        return 60 <= days <= 120
    if form in ANNUAL_FORMS:
        return 300 <= days <= 400
    return False


def _fundamentals_from_facts(
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    fundamentals: dict[str, Any] = {}
    for fact in facts:
        ticker = fact["ticker"]
        ticker_payload = fundamentals.setdefault(
            ticker,
            {"metrics": {}},
        )
        ticker_payload["metrics"][fact["metric_name"]] = {
            "value": fact["value"],
            "unit": fact["unit"],
            "period": fact["period"],
            "available_at": fact["available_at"],
            "source_url": fact["source_locator"],
            "metadata": {
                "taxonomy": fact["taxonomy"],
                "concept": fact["concept"],
                "accession_number": fact["accession_number"],
                "filing_source_locator": fact[
                    "filing_source_locator"
                ],
                "source_artifact_sha256": fact[
                    "source_artifact_sha256"
                ],
                "source_fact_sha256": fact[
                    "source_fact_sha256"
                ],
            },
        }
    return fundamentals


def _gate_metric_row(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": fact["ticker"],
        "metric_name": fact["metric_name"],
        "value": fact["value"],
        "unit": fact["unit"],
        "period": fact["period"],
        "available_at": fact["available_at"],
        "source_citation": fact["source_locator"],
        "accession_number": fact["accession_number"],
        "fact_sha256": fact["fact_sha256"],
    }


def _comparability(
    *,
    facts: list[dict[str, Any]],
    requested_tickers: list[str],
) -> dict[str, Any]:
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        by_metric[fact["metric_name"]].append(fact)
    metrics: dict[str, Any] = {}
    comparable_count = 0
    for metric_name, values in sorted(by_metric.items()):
        tickers = sorted({item["ticker"] for item in values})
        units = sorted({item["unit"] for item in values})
        complete = set(tickers) == set(requested_tickers)
        comparable = complete and len(units) == 1
        if comparable:
            comparable_count += 1
        metrics[metric_name] = {
            "tickers": tickers,
            "missing_tickers": sorted(
                set(requested_tickers) - set(tickers)
            ),
            "units": units,
            "ticker_coverage_ratio": _ratio(
                len(tickers),
                len(requested_tickers),
            ),
            "comparable_without_transformation": comparable,
        }
    return {
        "status": (
            "comparable"
            if metrics and comparable_count == len(metrics)
            else "partial_or_unit_mismatch"
            if metrics
            else "unavailable"
        ),
        "requested_tickers": requested_tickers,
        "comparable_metric_count": comparable_count,
        "metric_count": len(metrics),
        "metrics": metrics,
        "rule": (
            "A metric is directly comparable only when every requested "
            "ticker is represented and the reported unit is identical. "
            "No currency translation or period inference is performed."
        ),
    }


def render_saved_sec_companyfacts_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    coverage = payload.get("filing_coverage", {})
    lines = [
        "# DEAN-OS Saved SEC Company Facts",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- As-of: `{payload.get('inputs', {}).get('as_of')}`",
        (
            "- Requested tickers: "
            + (", ".join(coverage.get("requested_tickers", [])) or "none")
        ),
        (
            "- Filing tickers: "
            + (", ".join(coverage.get("filing_tickers", [])) or "none")
        ),
        (
            "- Missing filing tickers: "
            + (
                ", ".join(
                    coverage.get("missing_filing_tickers", [])
                )
                or "none"
            )
        ),
        (
            "- Accepted fact tickers: "
            + (
                ", ".join(
                    summary.get("accepted_fact_tickers", [])
                )
                or "none"
            )
        ),
        f"- Accepted facts: {summary.get('accepted_fact_count', 0)}",
        (
            "- Complete sector fundamentals: "
            f"{summary.get('can_claim_complete_sector_fundamentals', False)}"
        ),
        (
            "- Cross-ticker comparability: "
            f"`{summary.get('cross_ticker_comparability_status')}`"
        ),
        f"- Can compute ratios: {summary.get('can_compute_ratios', False)}",
        f"- Can trade: {summary.get('can_trade', False)}",
        "",
        "## Facts",
        "",
    ]
    facts = payload.get("facts", [])
    if facts:
        lines.extend(
            (
                f"- `{item['ticker']}` `{item['metric_name']}` "
                f"value=`{item['value']}` unit=`{item['unit']}` "
                f"period=`{item['period']}` "
                f"accession=`{item['accession_number']}`"
            )
            for item in facts
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Every accepted fact is tied to a verified filing accession and acceptance time.",
            "- Missing metrics, periods, tickers, or sources remain explicit gaps.",
            "- Currency translation, ratios, valuation, prediction features, learning, and trading are not performed.",
            "",
        ]
    )
    return "\n".join(lines)


def _requested_tickers(
    filing_index_path: str | Path,
) -> list[str]:
    payload = json.loads(
        Path(filing_index_path).read_text(encoding="utf-8")
    )
    values = payload.get("inputs", {}).get("tickers", [])
    return sorted(
        {
            str(value).upper().strip()
            for value in values
            if str(value).strip()
        }
    )


def _companyfacts_source_path(
    source_root: Path,
    cik: str,
) -> Path | None:
    candidates = (
        source_root / f"CIK{cik}" / "latest.json",
        source_root / f"CIK{cik}.json",
        source_root / f"{cik}.json",
    )
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.exists() and candidate.is_file()
        ),
        None,
    )


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _normalized_accession(value: Any) -> str:
    return str(value or "").strip().replace("-", "")


def _fact_fingerprint_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "ticker",
            "cik",
            "metric_name",
            "taxonomy",
            "concept",
            "value",
            "unit",
            "period",
            "accepted_at",
            "accession_number",
            "source_artifact_sha256",
            "source_fact_sha256",
            "fact_sha256",
        )
    }


def _blocked(status: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "summary": {
            "accepted_fact_count": 0,
            "excluded_fact_count": 0,
            "reason_counts": dict.fromkeys(reasons, 1),
            "can_enter_fundamental_input_gate": False,
            "can_claim_complete_sector_fundamentals": False,
            "can_compute_ratios": False,
            "can_translate_currencies": False,
            "can_feed_ticker_prediction_directly": False,
            "can_trade": False,
        },
        "source_provenance": [],
        "fundamental_metric_rows": [],
        "facts": [],
        "exclusions": [
            {"status": "excluded", "reasons": reasons}
        ],
    }


def _producer_safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "filing_content_fetch_performed": False,
        "xbrl_fact_fetch_performed": False,
        "saved_companyfacts_read_performed": True,
        "ratio_computation_performed": False,
        "valuation_performed": False,
        "pipeline_run_performed": False,
        "training_run_performed": False,
        "learning_write_performed": False,
        "production_config_write_performed": False,
        "paper_execution_performed": False,
        "live_execution_performed": False,
        "can_trade": False,
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


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


def _run_id(prefix: str) -> str:
    return (
        f"{prefix}_"
        f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()
