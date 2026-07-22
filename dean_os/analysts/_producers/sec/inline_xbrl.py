from __future__ import annotations

__all__ = [
    'ANNUAL_FORMS',
    'PRIMARY_SNAPSHOT_CONTRACT',
    'QUARTERLY_FORMS',
    'SAVED_SEC_INLINE_XBRL_CONTRACT',
    'SavedSECInlineXBRLProducer',
    'load_verified_inline_xbrl_context_fragment',
    'render_saved_sec_inline_xbrl_markdown',
]

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml
from lxml import etree

from dean_os.analysts._producers.sec.companyfacts import (
    DEFAULT_METRIC_REGISTRY,
)
from dean_os.analysts._producers.sec.filing_index import (
    verify_sec_filing_index,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.structured_context_provenance import audit_structured_context

SAVED_SEC_INLINE_XBRL_CONTRACT = (
    "dean_saved_sec_inline_xbrl_evidence_v1"
)
PRIMARY_SNAPSHOT_CONTRACT = "dean_sec_primary_document_snapshot_v1"
ANNUAL_FORMS = {"10-K", "20-F", "40-F"}
QUARTERLY_FORMS = {"10-Q"}


class SavedSECInlineXBRLProducer:
    """Extract accession-bound consolidated facts from saved inline XBRL."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_sec_inline_xbrl_producer"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        primary_snapshot_path: str | Path,
        filing_index_path: str | Path,
        registry_path: str | Path = DEFAULT_METRIC_REGISTRY,
        as_of: str | None = None,
        tickers: list[str] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        verified_index = verify_sec_filing_index(
            filing_index_path,
            expected_as_of=as_of,
        )
        resolved_as_of = verified_index["as_of"]
        registry_source = Path(registry_path)
        registry = _load_registry(registry_source)
        snapshot_source = Path(primary_snapshot_path)
        snapshot = json.loads(
            snapshot_source.read_text(encoding="utf-8")
        )
        _validate_snapshot_artifact(snapshot)
        requested = {
            str(ticker).upper().strip()
            for ticker in tickers or []
            if str(ticker).strip()
        }
        filings = {
            _normal_accession(item["accession_number"]): item
            for item in verified_index["filings"]
            if not requested or item["ticker"] in requested
        }
        payload: dict[str, Any] = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "saved_sec_inline_xbrl_producer",
            "producer_contract": SAVED_SEC_INLINE_XBRL_CONTRACT,
            "inputs": {
                "primary_snapshot_path": str(snapshot_source),
                "primary_snapshot_sha256": _sha256_file(
                    snapshot_source
                ),
                "filing_index_path": str(filing_index_path),
                "filing_index_fingerprint": verified_index[
                    "fingerprint"
                ],
                "registry_path": str(registry_source),
                "as_of": resolved_as_of,
                "tickers": sorted(requested),
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
        }
        if not registry.get("metrics"):
            payload.update(
                _blocked(
                    "blocked_metric_registry_missing_or_empty",
                    ["fundamental_metric_registry_missing_or_empty"],
                )
            )
            return self._finish(payload, save=save)

        normalized = _extract_snapshot_facts(
            snapshot=snapshot,
            filings=filings,
            registry=registry["metrics"],
            issuer_reporting_units=registry.get(
                "issuer_reporting_units",
                {},
            ),
        )
        fundamentals = _fundamentals_from_facts(
            normalized["facts"]
        )
        audit = audit_structured_context(
            fundamentals=fundamentals,
            macro={},
            sector_data={},
            as_of=resolved_as_of,
        )
        accepted = audit["accepted_context"]["fundamentals"]
        accepted_tickers = sorted(accepted)
        fact_fingerprint = _canonical_sha256(
            [_fact_fingerprint_row(item) for item in normalized["facts"]]
        )
        payload.update(
            {
                "status": (
                    "inline_xbrl_facts_ready_with_gaps"
                    if normalized["facts"] and normalized["exclusions"]
                    else "inline_xbrl_facts_ready"
                    if normalized["facts"]
                    else "blocked_no_admissible_inline_xbrl_facts"
                ),
                "source_provenance": normalized[
                    "source_provenance"
                ],
                "summary": {
                    "snapshot_filing_count": len(
                        snapshot.get("snapshots", [])
                    ),
                    "matched_filing_count": normalized[
                        "matched_filing_count"
                    ],
                    "parsed_context_count": normalized[
                        "parsed_context_count"
                    ],
                    "parsed_unit_count": normalized[
                        "parsed_unit_count"
                    ],
                    "parsed_numeric_fact_count": normalized[
                        "parsed_numeric_fact_count"
                    ],
                    "accepted_fact_count": audit["accepted_count"],
                    "accepted_fact_tickers": accepted_tickers,
                    "excluded_fact_count": len(
                        normalized["exclusions"]
                    ),
                    "accepted_fingerprint": audit[
                        "accepted_fingerprint"
                    ],
                    "facts_fingerprint": fact_fingerprint,
                    "reason_counts": normalized["reason_counts"],
                    "can_enter_fundamental_input_gate": bool(accepted),
                    "can_compute_ratios": False,
                    "can_translate_currencies": False,
                    "can_feed_ticker_prediction_directly": False,
                    "can_trade": False,
                },
                "fundamental_metric_rows": [
                    _gate_metric_row(item)
                    for item in normalized["facts"]
                ],
                "facts": normalized["facts"],
                "exclusions": normalized["exclusions"],
                "structured_context_audit": {
                    key: value
                    for key, value in audit.items()
                    if key
                    not in {
                        "accepted_context",
                        "accepted_observations",
                    }
                },
                "market_context_fragment": {
                    "as_of": resolved_as_of,
                    "fundamentals": accepted,
                    "metadata": {
                        "saved_sec_inline_xbrl_run_id": payload[
                            "run_id"
                        ],
                        "saved_sec_inline_xbrl_facts_fingerprint": (
                            fact_fingerprint
                        ),
                        "saved_sec_inline_xbrl_accepted_fingerprint": (
                            audit["accepted_fingerprint"]
                        ),
                        "filing_index_fingerprint": verified_index[
                            "fingerprint"
                        ],
                        "accepted_fact_tickers": accepted_tickers,
                    },
                },
                "integration_boundary": {
                    "review_only": True,
                    "facts_bound_to_primary_document_sha": True,
                    "facts_bound_to_verified_accession": True,
                    "dimensional_contexts_allowed": False,
                    "reporting_currency_preferred": True,
                    "currency_translation_allowed": False,
                    "ratio_computation_allowed": False,
                    "ticker_prediction_feature_promotion_allowed": False,
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
                markdown=render_saved_sec_inline_xbrl_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
        return payload


def load_verified_inline_xbrl_context_fragment(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("producer_contract") != SAVED_SEC_INLINE_XBRL_CONTRACT:
        raise ValueError("unsupported saved SEC inline XBRL contract")
    if payload.get("status") not in {
        "inline_xbrl_facts_ready",
        "inline_xbrl_facts_ready_with_gaps",
    }:
        raise ValueError("saved SEC inline XBRL artifact is not ready")
    summary = payload.get("summary", {})
    safety = payload.get("safety", {})
    if (
        safety.get("review_only") is not True
        or safety.get("network_access_performed") is not False
        or safety.get("live_execution_performed") is not False
        or summary.get("can_trade") is not False
    ):
        raise ValueError("saved inline XBRL safety boundary invalid")
    fragment = payload.get("market_context_fragment")
    if not isinstance(fragment, dict):
        raise ValueError("saved inline XBRL fragment missing")
    fragment_as_of = parse_timezone_aware(fragment.get("as_of"))
    if fragment_as_of is None:
        raise ValueError("saved inline XBRL as_of invalid")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != fragment_as_of:
            raise ValueError("saved inline XBRL expected as_of mismatch")

    snapshot_path = Path(
        payload.get("inputs", {}).get("primary_snapshot_path", "")
    )
    if (
        not snapshot_path.exists()
        or _sha256_file(snapshot_path)
        != payload.get("inputs", {}).get("primary_snapshot_sha256")
    ):
        raise ValueError("inline XBRL snapshot artifact hash mismatch")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    _validate_snapshot_artifact(snapshot)
    expected_sources = {
        item.get("sha256"): item
        for item in snapshot.get("snapshots", [])
    }
    for source in payload.get("source_provenance", []):
        raw_path = Path(str(source.get("path") or ""))
        if (
            not raw_path.exists()
            or _sha256_file(raw_path) != source.get("sha256")
            or source.get("sha256") not in expected_sources
        ):
            raise ValueError("inline XBRL primary source hash mismatch")
    registry = Path(payload.get("registry", {}).get("path", ""))
    if (
        not registry.exists()
        or _sha256_file(registry)
        != payload.get("registry", {}).get("sha256")
    ):
        raise ValueError("inline XBRL registry hash mismatch")
    verified_index = verify_sec_filing_index(
        payload.get("inputs", {}).get("filing_index_path", ""),
        expected_as_of=fragment_as_of.isoformat(),
        verify_source_database=False,
    )
    snapshot_index_fingerprint = snapshot.get("inputs", {}).get(
        "filing_index_fingerprint"
    )
    if (
        verified_index["fingerprint"]
        != payload.get("inputs", {}).get("filing_index_fingerprint")
        or snapshot_index_fingerprint is not None
        and verified_index["fingerprint"]
        != snapshot_index_fingerprint
    ):
        raise ValueError("inline XBRL filing index mismatch")

    fundamentals = fragment.get("fundamentals")
    if not isinstance(fundamentals, dict):
        raise ValueError("inline XBRL fundamentals payload invalid")
    audit = audit_structured_context(
        fundamentals=fundamentals,
        macro={},
        sector_data={},
        as_of=fragment_as_of.isoformat(),
    )
    if (
        audit["excluded_count"] != 0
        or audit["accepted_count"] != summary.get("accepted_fact_count")
        or audit["accepted_fingerprint"]
        != summary.get("accepted_fingerprint")
    ):
        raise ValueError("inline XBRL fragment fingerprint mismatch")
    fact_fingerprint = _canonical_sha256(
        [_fact_fingerprint_row(item) for item in payload.get("facts", [])]
    )
    if fact_fingerprint != summary.get("facts_fingerprint"):
        raise ValueError("inline XBRL fact payload mismatch")
    return {
        "as_of": fragment_as_of.isoformat(),
        "fundamentals": audit["accepted_context"]["fundamentals"],
        "facts": payload.get("facts", []),
        "metadata": {
            **dict(fragment.get("metadata", {})),
            "saved_sec_inline_xbrl_artifact_path": str(path),
            "saved_sec_inline_xbrl_artifact_sha256": _sha256_file(path),
            "saved_sec_inline_xbrl_verified": True,
            "filing_index_verification_mode": verified_index.get(
                "verification_mode"
            ),
        },
    }


def _extract_snapshot_facts(
    *,
    snapshot: dict[str, Any],
    filings: dict[str, dict[str, Any]],
    registry: dict[str, Any],
    issuer_reporting_units: dict[str, Any],
) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_provenance: list[dict[str, Any]] = []
    matched_filing_count = 0
    parsed_context_count = 0
    parsed_unit_count = 0
    parsed_numeric_fact_count = 0
    for item in snapshot.get("snapshots", []):
        accession = _normal_accession(item.get("accession_number"))
        filing = filings.get(accession)
        if filing is None:
            exclusions.append(
                {
                    "ticker": item.get("ticker"),
                    "accession_number": item.get("accession_number"),
                    "metric_name": None,
                    "status": "excluded",
                    "reasons": [
                        "inline_xbrl_snapshot_not_in_verified_index"
                    ],
                }
            )
            continue
        mismatch = _snapshot_filing_mismatch(item, filing)
        if mismatch:
            exclusions.append(
                {
                    "ticker": filing["ticker"],
                    "accession_number": filing["accession_number"],
                    "metric_name": None,
                    "status": "excluded",
                    "reasons": mismatch,
                }
            )
            continue
        source = Path(str(item.get("immutable_path") or ""))
        expected_sha = str(item.get("sha256") or "")
        if (
            not source.exists()
            or not expected_sha
            or _sha256_file(source) != expected_sha
        ):
            exclusions.append(
                {
                    "ticker": filing["ticker"],
                    "accession_number": filing["accession_number"],
                    "metric_name": None,
                    "status": "excluded",
                    "reasons": ["inline_xbrl_primary_source_hash_mismatch"],
                }
            )
            continue
        source_provenance.append(
            {
                "ticker": filing["ticker"],
                "cik": filing["cik"],
                "accession_number": filing["accession_number"],
                "path": str(source),
                "sha256": expected_sha,
                "size_bytes": source.stat().st_size,
                "source_url": filing["source_locator"],
                "status": "primary_document_verified",
            }
        )
        try:
            parsed = _parse_inline_document(source)
        except (OSError, etree.Error, ValueError) as exc:
            exclusions.append(
                {
                    "ticker": filing["ticker"],
                    "accession_number": filing["accession_number"],
                    "metric_name": None,
                    "status": "excluded",
                    "reasons": ["inline_xbrl_document_unreadable"],
                    "error_type": type(exc).__name__,
                }
            )
            continue
        matched_filing_count += 1
        parsed_context_count += len(parsed["contexts"])
        parsed_unit_count += len(parsed["units"])
        parsed_numeric_fact_count += len(parsed["numeric_facts"])
        for metric_name, metric_config in sorted(registry.items()):
            fact, reasons = _select_inline_fact(
                parsed=parsed,
                filing=filing,
                metric_name=str(metric_name),
                metric_config=metric_config,
                source_sha256=expected_sha,
                registry_reporting_unit=(
                    issuer_reporting_units.get(filing["cik"])
                    if isinstance(issuer_reporting_units, dict)
                    else None
                ),
            )
            if fact is not None:
                facts.append(fact)
            else:
                exclusions.append(
                    {
                        "ticker": filing["ticker"],
                        "accession_number": filing["accession_number"],
                        "metric_name": metric_name,
                        "status": "excluded",
                        "reasons": reasons
                        or ["inline_xbrl_metric_not_found"],
                    }
                )
    facts.sort(
        key=lambda value: (
            value["ticker"],
            value["accession_number"],
            value["metric_name"],
        )
    )
    reason_counts = Counter(
        reason
        for exclusion in exclusions
        for reason in exclusion.get("reasons", [])
    )
    return {
        "facts": facts,
        "exclusions": exclusions,
        "source_provenance": source_provenance,
        "matched_filing_count": matched_filing_count,
        "parsed_context_count": parsed_context_count,
        "parsed_unit_count": parsed_unit_count,
        "parsed_numeric_fact_count": parsed_numeric_fact_count,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _parse_inline_document(path: Path) -> dict[str, Any]:
    root = etree.parse(
        str(path),
        etree.HTMLParser(huge_tree=True),
    ).getroot()
    contexts: dict[str, dict[str, Any]] = {}
    units: dict[str, str] = {}
    document_currency: str | None = None
    numeric_facts: list[dict[str, Any]] = []

    for element in root.iter():
        local = _local_name(element.tag)
        if local == "context":
            context_id = element.get("id")
            if context_id:
                contexts[context_id] = _parse_context(element)
        elif local == "unit":
            unit_id = element.get("id")
            if unit_id:
                units[unit_id] = _parse_unit(element)
        elif local == "nonnumeric":
            name = str(element.get("name") or "")
            if name.lower() == (
                "dei:documentreportingcurrencyisocode"
            ).lower():
                value = _element_text(element).strip().upper()
                if value:
                    document_currency = value

    for element in root.iter():
        if _local_name(element.tag) != "nonfraction":
            continue
        value = _parse_inline_number(element)
        numeric_facts.append(
            {
                "name": str(element.get("name") or ""),
                "context_id": str(element.get("contextref") or ""),
                "unit_id": str(element.get("unitref") or ""),
                "unit": units.get(
                    str(element.get("unitref") or ""),
                    "",
                ),
                "value": value,
                "scale": element.get("scale"),
                "sign": element.get("sign"),
                "format": element.get("format"),
                "decimals": element.get("decimals"),
                "fact_id": element.get("id"),
                "raw_text": _element_text(element).strip(),
            }
        )
    return {
        "contexts": contexts,
        "units": units,
        "document_currency": document_currency,
        "numeric_facts": numeric_facts,
    }


def _parse_context(element: Any) -> dict[str, Any]:
    identifier = None
    start = None
    end = None
    instant = None
    has_dimensions = False
    for child in element.iter():
        local = _local_name(child.tag)
        text = (child.text or "").strip()
        if local == "identifier":
            identifier = text
        elif local == "startdate":
            start = text
        elif local == "enddate":
            end = text
        elif local == "instant":
            instant = text
        elif local in {"explicitmember", "typedmember"}:
            has_dimensions = True
    return {
        "entity_identifier": str(identifier or "").zfill(10),
        "start": start,
        "end": end,
        "instant": instant,
        "has_dimensions": has_dimensions,
    }


def _parse_unit(element: Any) -> str:
    measures = [
        (child.text or "").strip()
        for child in element.iter()
        if _local_name(child.tag) == "measure"
        and (child.text or "").strip()
    ]
    normalized = [
        value.split(":")[-1]
        for value in measures
    ]
    if len(normalized) == 1:
        value = normalized[0]
        return "ratio" if value.lower() == "pure" else value
    return "-per-".join(normalized) if normalized else ""


def _select_inline_fact(
    *,
    parsed: dict[str, Any],
    filing: dict[str, Any],
    metric_name: str,
    metric_config: Any,
    source_sha256: str,
    registry_reporting_unit: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(metric_config, dict):
        return None, ["fundamental_metric_registry_entry_invalid"]
    concepts = metric_config.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        return None, ["fundamental_metric_concepts_missing"]
    accepted_units = [
        str(value)
        for value in metric_config.get("accepted_units", [])
    ]
    period_type = str(metric_config.get("period_type") or "")
    reasons: list[str] = []
    facts_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in parsed["numeric_facts"]:
        facts_by_name[fact["name"].lower()].append(fact)

    for mapping in concepts:
        if not isinstance(mapping, dict):
            continue
        taxonomy = str(mapping.get("taxonomy") or "")
        concept = str(mapping.get("concept") or "")
        qualified = f"{taxonomy}:{concept}"
        candidates: list[dict[str, Any]] = []
        for raw in facts_by_name.get(qualified.lower(), []):
            context = parsed["contexts"].get(raw["context_id"])
            candidate, candidate_reasons = _normalize_inline_candidate(
                raw=raw,
                context=context,
                filing=filing,
                period_type=period_type,
                accepted_units=accepted_units,
            )
            if candidate is not None:
                candidates.append(candidate)
            else:
                reasons.extend(candidate_reasons)
        if not candidates:
            continue
        document_reporting_currency = str(
            parsed.get("document_currency") or ""
        ).upper()
        registered_reporting_currency = str(
            registry_reporting_unit or ""
        ).upper()
        reporting_currency = (
            document_reporting_currency
            or registered_reporting_currency
        )
        if reporting_currency:
            native = [
                item
                for item in candidates
                if item["unit"].upper() == reporting_currency
            ]
            if native:
                candidates = native
        unique = _deduplicate_candidates(candidates)
        if len(unique) != 1:
            return None, ["inline_xbrl_metric_ambiguous"]
        selected = unique[0]
        canonical = {
            "ticker": filing["ticker"],
            "cik": filing["cik"],
            "metric_name": metric_name,
            "statement_role": metric_config.get("statement_role"),
            "taxonomy": taxonomy,
            "concept": concept,
            "concept_label": qualified,
            "value": selected["value"],
            "unit": selected["unit"],
            "period_type": period_type,
            "period": selected["period"],
            "period_start": selected.get("period_start"),
            "period_end": selected["period_end"],
            "fiscal_year": None,
            "fiscal_period": None,
            "frame": None,
            "form": filing["form"],
            "filed_date": filing["filing_date"],
            "accepted_at": filing["accepted_at"],
            "available_at": filing["accepted_at"],
            "accession_number": filing["accession_number"],
            "filing_source_locator": filing["source_locator"],
            "source_locator": filing["source_locator"],
            "source_artifact_sha256": source_sha256,
            "source_fact_sha256": _canonical_sha256(
                {
                    "qualified_name": qualified,
                    "context": selected["context"],
                    "unit": selected["unit"],
                    "value": selected["value"],
                    "raw_text": selected["raw_text"],
                    "fact_id": selected.get("fact_id"),
                    "source_artifact_sha256": source_sha256,
                }
            ),
            "source_kind": "sec_inline_xbrl_primary_document",
            "reporting_currency": reporting_currency or None,
            "reporting_currency_basis": (
                "inline_xbrl_document_reporting_currency"
                if document_reporting_currency
                else "hashed_issuer_reporting_unit_registry"
                if registered_reporting_currency
                else None
            ),
        }
        canonical["fact_sha256"] = _canonical_sha256(canonical)
        return canonical, []
    return None, sorted(
        set(reasons or ["inline_xbrl_metric_not_found_for_filing"])
    )


def _normalize_inline_candidate(
    *,
    raw: dict[str, Any],
    context: dict[str, Any] | None,
    filing: dict[str, Any],
    period_type: str,
    accepted_units: list[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if raw.get("value") is None:
        reasons.append("inline_xbrl_value_invalid")
    unit = str(raw.get("unit") or "")
    if accepted_units and unit not in accepted_units:
        reasons.append("inline_xbrl_unit_not_registered")
    if not context:
        reasons.append("inline_xbrl_context_missing")
        return None, reasons
    if context.get("entity_identifier") != str(filing["cik"]):
        reasons.append("inline_xbrl_entity_mismatch")
    if context.get("has_dimensions"):
        reasons.append("inline_xbrl_dimensional_context_blocked")
    report_end = _parse_date(filing.get("report_date"))
    if period_type == "instant":
        end = _parse_date(context.get("instant"))
        start = None
        if end != report_end:
            reasons.append("inline_xbrl_report_end_mismatch")
        period = end.isoformat() if end else None
    elif period_type == "duration":
        start = _parse_date(context.get("start"))
        end = _parse_date(context.get("end"))
        if end != report_end:
            reasons.append("inline_xbrl_report_end_mismatch")
        if start is None or end is None or start > end:
            reasons.append("inline_xbrl_duration_invalid")
        elif not _admissible_duration(
            form=filing["form"],
            days=(end - start).days + 1,
        ):
            reasons.append("inline_xbrl_duration_not_filing_period")
        period = (
            f"{start.isoformat()}/{end.isoformat()}"
            if start and end
            else None
        )
    else:
        start = None
        end = None
        period = None
        reasons.append("fundamental_metric_period_type_invalid")
    if reasons:
        return None, reasons
    return {
        "value": raw["value"],
        "unit": unit,
        "period": period,
        "period_start": start.isoformat() if start else None,
        "period_end": end.isoformat() if end else None,
        "context": context,
        "raw_text": raw.get("raw_text"),
        "fact_id": raw.get("fact_id"),
    }, []


def _parse_inline_number(element: Any) -> float | None:
    nil_value = (
        element.get("xsi:nil")
        or element.get("nil")
        or ""
    )
    if str(nil_value).lower() in {"true", "1"}:
        return None
    text = _element_text(element).strip()
    if not text or text in {"-", "—", "–"}:
        return None
    negative_parentheses = text.startswith("(") and text.endswith(")")
    cleaned = (
        text.replace("\u00a0", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("−", "-")
    )
    cleaned = re.sub(r"[^0-9eE+\-.]", "", cleaned)
    try:
        value = Decimal(cleaned)
        scale = int(element.get("scale") or 0)
        value *= Decimal(10) ** scale
    except (TypeError, ValueError, OverflowError, InvalidOperation):
        return None
    if negative_parentheses or element.get("sign") == "-":
        value = -abs(value)
    if not value.is_finite():
        return None
    integral = value.to_integral_value()
    return int(integral) if value == integral else float(value)


def _element_text(element: Any) -> str:
    return "".join(element.itertext())


def _local_name(tag: Any) -> str:
    value = str(tag).lower()
    if "}" in value:
        value = value.rsplit("}", 1)[-1]
    if ":" in value:
        value = value.rsplit(":", 1)[-1]
    return value


def _snapshot_filing_mismatch(
    item: dict[str, Any],
    filing: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    checks = {
        "ticker": (item.get("ticker"), filing.get("ticker")),
        "cik": (str(item.get("cik")), str(filing.get("cik"))),
        "form": (item.get("form"), filing.get("form")),
        "report_date": (
            item.get("report_date"),
            filing.get("report_date"),
        ),
        "accepted_at": (
            item.get("accepted_at"),
            filing.get("accepted_at"),
        ),
        "source_url": (
            item.get("source_url"),
            filing.get("source_locator"),
        ),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            reasons.append(f"inline_xbrl_snapshot_{field}_mismatch")
    return reasons


def _validate_snapshot_artifact(payload: dict[str, Any]) -> None:
    if payload.get("snapshot_contract") != PRIMARY_SNAPSHOT_CONTRACT:
        raise ValueError("unsupported SEC primary snapshot contract")
    if payload.get("status") not in {
        "primary_document_snapshots_ready",
        "primary_document_snapshots_partial",
    }:
        raise ValueError("SEC primary snapshot artifact is not ready")
    safety = payload.get("safety", {})
    if (
        safety.get("official_sec_get_requests_only") is not True
        or safety.get("pipeline_run_performed") is not False
        or safety.get("live_execution_performed") is not False
        or safety.get("can_trade") is not False
    ):
        raise ValueError("SEC primary snapshot safety boundary invalid")


def _fundamentals_from_facts(facts: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for fact in facts:
        ticker_payload = output.setdefault(
            fact["ticker"],
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
                "source_artifact_sha256": fact[
                    "source_artifact_sha256"
                ],
                "source_fact_sha256": fact["source_fact_sha256"],
                "source_kind": fact["source_kind"],
            },
        }
    return output


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


def _deduplicate_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in candidates:
        key = _canonical_sha256(
            {
                "value": item["value"],
                "unit": item["unit"],
                "period_start": item.get("period_start"),
                "period_end": item.get("period_end"),
            }
        )
        unique.setdefault(key, item)
    return [unique[key] for key in sorted(unique)]


def _admissible_duration(*, form: str, days: int) -> bool:
    if str(form).upper() in QUARTERLY_FORMS:
        return 60 <= days <= 120
    if str(form).upper() in ANNUAL_FORMS:
        return 300 <= days <= 400
    return False


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normal_accession(value: Any) -> str:
    return str(value or "").replace("-", "").strip()


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


def render_saved_sec_inline_xbrl_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Saved SEC Inline XBRL",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- As-of: `{payload.get('inputs', {}).get('as_of')}`",
        f"- Matched filings: {summary.get('matched_filing_count', 0)}",
        f"- Parsed contexts: {summary.get('parsed_context_count', 0)}",
        f"- Parsed numeric facts: {summary.get('parsed_numeric_fact_count', 0)}",
        f"- Accepted facts: {summary.get('accepted_fact_count', 0)}",
        (
            "- Accepted tickers: "
            + (
                ", ".join(summary.get("accepted_fact_tickers", []))
                or "none"
            )
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
                f"concept=`{item['taxonomy']}:{item['concept']}`"
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
            "- Facts are bound to the immutable primary document and verified filing accession.",
            "- Dimensional contexts are excluded from consolidated metrics.",
            "- Reporting currency is preferred; no currency translation is performed.",
            "- Ratios, valuation, pipeline features, learning, and trading remain disabled.",
            "",
        ]
    )
    return "\n".join(lines)


def _blocked(status: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "summary": {
            "accepted_fact_count": 0,
            "excluded_fact_count": 0,
            "reason_counts": dict.fromkeys(reasons, 1),
            "can_enter_fundamental_input_gate": False,
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


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "saved_primary_document_read_performed": True,
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
        "saved_sec_inline_xbrl_"
        f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )
