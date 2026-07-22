from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

CONTEXT_EVIDENCE_CONTRACT = "dean_context_evidence_point_in_time_v1"

NEWS_TIMESTAMP_FIELDS = (
    "published_at",
    "published_date",
    "publication_date",
    "pub_date",
    "publishedAt",
    "time_published",
    "timestamp",
    "datetime",
    "date",
)
NEWS_LOCATOR_FIELDS = (
    "url",
    "uri",
    "link",
    "reference",
    "source_id",
    "document_id",
    "id",
    "hash",
    "source",
)
NEWS_TICKER_FIELDS = ("tickers", "symbols", "ticker", "symbol")


def audit_news_records(
    records: list[Any],
    *,
    as_of: str,
    requested_tickers: list[str] | None = None,
) -> dict[str, Any]:
    as_of_dt = parse_timezone_aware(as_of)
    if as_of_dt is None:
        raise ValueError(
            "context evidence as_of must be a timezone-aware ISO-8601 timestamp"
        )
    requested = {
        str(ticker).upper().strip()
        for ticker in requested_tickers or []
        if str(ticker).strip()
    }
    accepted: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for index, raw in enumerate(records):
        reasons: list[str] = []
        if not isinstance(raw, dict):
            exclusions.append(
                {
                    "index": index,
                    "status": "excluded",
                    "reasons": ["news_record_not_structured"],
                }
            )
            continue
        record = dict(raw)
        timestamp_field, published_at = first_timestamp(record)
        locator_field, locator = first_nonempty(record, NEWS_LOCATOR_FIELDS)
        if published_at is None:
            reasons.append("publication_timestamp_missing_or_invalid")
        elif published_at > as_of_dt:
            reasons.append("publication_after_as_of")
        if not locator:
            reasons.append("stable_source_locator_missing")

        record_hash = canonical_record_sha256(record)
        if record_hash in seen_hashes:
            reasons.append("duplicate_news_record")
        explicit_tickers = explicit_record_tickers(record)
        direct_tickers = sorted(requested.intersection(explicit_tickers))
        text = news_text(record)
        cashtag_tickers = {
            match.upper()
            for match in re.findall(r"\$([A-Za-z][A-Za-z0-9.-]{0,9})\b", text)
        }
        direct_tickers = sorted(
            set(direct_tickers).union(requested.intersection(cashtag_tickers))
        )
        provenance = {
            "contract": CONTEXT_EVIDENCE_CONTRACT,
            "record_sha256": record_hash,
            "publication_timestamp_field": timestamp_field,
            "published_at": (
                published_at.isoformat() if published_at else None
            ),
            "source_locator_field": locator_field,
            "source_locator": str(locator) if locator else None,
            "explicit_tickers": sorted(explicit_tickers),
            "direct_requested_tickers": direct_tickers,
            "as_of": as_of_dt.isoformat(),
        }
        if reasons:
            exclusions.append(
                {
                    "index": index,
                    "status": "excluded",
                    "reasons": reasons,
                    "provenance": provenance,
                }
            )
            continue
        seen_hashes.add(record_hash)
        record["published_at"] = published_at.isoformat()
        record["_dean_context_provenance"] = {
            **provenance,
            "status": "point_in_time_compatible",
        }
        accepted.append(record)

    reason_counts: dict[str, int] = {}
    for item in exclusions:
        for reason in item["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "contract": CONTEXT_EVIDENCE_CONTRACT,
        "status": (
            "point_in_time_ready_with_exclusions"
            if accepted and exclusions
            else "point_in_time_ready"
            if accepted
            else "blocked_no_point_in_time_news"
        ),
        "as_of": as_of_dt.isoformat(),
        "input_count": len(records),
        "accepted_count": len(accepted),
        "excluded_count": len(exclusions),
        "accepted": accepted,
        "exclusions": exclusions,
        "reason_counts": dict(sorted(reason_counts.items())),
        "direct_ticker_rule": (
            "explicit ticker metadata or cashtag only; plain text substring "
            "matches cannot promote sector news to ticker evidence"
        ),
    }


def audit_market_context_news(context: Any) -> dict[str, Any]:
    records = list(getattr(context, "news", []) or [])
    as_of = getattr(context, "as_of", None)
    if not as_of:
        return {
            "contract": CONTEXT_EVIDENCE_CONTRACT,
            "status": "blocked_context_as_of_missing",
            "as_of": None,
            "input_count": len(records),
            "accepted_count": 0,
            "excluded_count": len(records),
            "accepted": [],
            "exclusions": [
                {
                    "index": index,
                    "status": "excluded",
                    "reasons": ["context_as_of_missing"],
                }
                for index in range(len(records))
            ],
            "reason_counts": {
                "context_as_of_missing": len(records)
            }
            if records
            else {},
        }
    try:
        return audit_news_records(
            records,
            as_of=str(as_of),
            requested_tickers=list(
                getattr(context, "tickers", []) or []
            ),
        )
    except ValueError:
        return {
            "contract": CONTEXT_EVIDENCE_CONTRACT,
            "status": "blocked_context_as_of_invalid",
            "as_of": str(as_of),
            "input_count": len(records),
            "accepted_count": 0,
            "excluded_count": len(records),
            "accepted": [],
            "exclusions": [
                {
                    "index": index,
                    "status": "excluded",
                    "reasons": ["context_as_of_invalid"],
                }
                for index in range(len(records))
            ],
            "reason_counts": {
                "context_as_of_invalid": len(records)
            }
            if records
            else {},
        }


def audit_research_documents(
    documents: list[Any],
    *,
    as_of: str,
) -> dict[str, Any]:
    as_of_dt = parse_timezone_aware(as_of)
    if as_of_dt is None:
        raise ValueError(
            "research document as_of must be a timezone-aware ISO-8601 timestamp"
        )
    accepted: list[Any] = []
    exclusions: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    time_sensitive_types = {
        "news",
        "article",
        "filing",
        "transcript",
    }
    for index, document in enumerate(documents):
        source_type = str(
            getattr(document, "source_type", "") or ""
        ).lower()
        published_at = parse_timezone_aware(
            getattr(document, "published_at", None)
        )
        ingested_at = parse_timezone_aware(
            getattr(document, "ingested_at", None)
        )
        uri = str(getattr(document, "uri", "") or "").strip()
        metadata = dict(getattr(document, "metadata", {}) or {})
        existing_document_provenance = metadata.get(
            "_dean_document_provenance"
        )
        if not isinstance(existing_document_provenance, dict):
            existing_document_provenance = {}
        replay = metadata.get("point_in_time_replay")
        replay_as_of = (
            str(replay.get("as_of"))
            if isinstance(replay, dict) and replay.get("as_of")
            else None
        )
        replay_reconstruction = (
            replay_as_of is not None
            and parse_timezone_aware(replay_as_of) == as_of_dt
        )
        text = str(getattr(document, "text", "") or "")
        content_sha256 = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        reasons: list[str] = []
        limitations: list[str] = []
        availability_at = published_at
        availability_basis = "published_at"
        if published_at is None:
            if source_type in time_sensitive_types:
                reasons.append(
                    "document_publication_timestamp_missing_or_invalid"
                )
            elif ingested_at is not None:
                availability_at = ingested_at
                availability_basis = "ingested_at_publication_unknown"
                limitations.append("publication_timestamp_unknown")
            else:
                reasons.append(
                    "document_availability_timestamp_missing_or_invalid"
                )
        elif published_at > as_of_dt:
            reasons.append("document_published_after_as_of")
        if not uri:
            reasons.append("document_source_locator_missing")
        if ingested_at is None:
            reasons.append("document_ingested_at_missing_or_invalid")
        elif ingested_at > as_of_dt and not replay_reconstruction:
            reasons.append("document_ingested_after_as_of")
        elif ingested_at > as_of_dt and replay_reconstruction:
            limitations.append(
                "historical_reconstruction_ingested_after_as_of"
            )
        if content_sha256 in seen_hashes:
            reasons.append("duplicate_research_document")

        provenance = {
            "contract": CONTEXT_EVIDENCE_CONTRACT,
            "document_id": getattr(document, "document_id", None),
            "source_type": source_type,
            "source_locator": uri or None,
            "content_sha256": content_sha256,
            "published_at": (
                published_at.isoformat() if published_at else None
            ),
            "ingested_at": (
                ingested_at.isoformat() if ingested_at else None
            ),
            "availability_at": (
                availability_at.isoformat()
                if availability_at
                else None
            ),
            "availability_basis": availability_basis,
            "replay_reconstruction": replay_reconstruction,
            "as_of": as_of_dt.isoformat(),
            "limitations": _merge_limitations(
                existing_document_provenance.get("limitations"),
                limitations,
            ),
        }
        for key in (
            "evidence_type",
            "domain_id",
            "source_path",
            "loader",
        ):
            value = existing_document_provenance.get(key)
            if value:
                provenance[key] = value
        if existing_document_provenance.get("availability_at"):
            provenance["source_declared_availability_at"] = (
                existing_document_provenance.get("availability_at")
            )
        if existing_document_provenance.get("availability_basis"):
            provenance["source_declared_availability_basis"] = (
                existing_document_provenance.get("availability_basis")
            )
        if reasons:
            exclusions.append(
                {
                    "index": index,
                    "document_id": getattr(
                        document,
                        "document_id",
                        None,
                    ),
                    "status": "excluded",
                    "reasons": reasons,
                    "provenance": provenance,
                }
            )
            continue
        seen_hashes.add(content_sha256)
        metadata["_dean_document_provenance"] = {
            **provenance,
            "status": "point_in_time_compatible",
        }
        document.metadata = metadata
        accepted.append(document)

    reason_counts: dict[str, int] = {}
    for item in exclusions:
        for reason in item["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "contract": CONTEXT_EVIDENCE_CONTRACT,
        "status": (
            "point_in_time_ready_with_exclusions"
            if accepted and exclusions
            else "point_in_time_ready"
            if accepted
            else "blocked_no_point_in_time_documents"
        ),
        "as_of": as_of_dt.isoformat(),
        "input_count": len(documents),
        "accepted_count": len(accepted),
        "excluded_count": len(exclusions),
        "accepted": accepted,
        "exclusions": exclusions,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def first_timestamp(
    record: dict[str, Any],
) -> tuple[str | None, datetime | None]:
    for field in NEWS_TIMESTAMP_FIELDS:
        if field not in record:
            continue
        parsed = parse_timezone_aware(record.get(field))
        if parsed is not None:
            return field, parsed
    return None, None


def _merge_limitations(
    existing: Any,
    generated: list[str],
) -> list[str]:
    merged: list[str] = []
    if isinstance(existing, str):
        candidates = [existing]
    elif isinstance(existing, list):
        candidates = [str(item) for item in existing if str(item).strip()]
    else:
        candidates = []
    candidates.extend(generated)
    for item in candidates:
        if item and item not in merged:
            merged.append(item)
    return merged


def first_nonempty(
    record: dict[str, Any],
    fields: tuple[str, ...],
) -> tuple[str | None, Any | None]:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return field, value
    return None, None


def explicit_record_tickers(record: dict[str, Any]) -> set[str]:
    tickers: set[str] = set()
    for field in NEWS_TICKER_FIELDS:
        value = record.get(field)
        if isinstance(value, (list, tuple, set)):
            values = value
        elif value is None:
            values = []
        else:
            values = re.split(r"[,;|\s]+", str(value))
        tickers.update(
            str(item).upper().strip()
            for item in values
            if str(item).strip()
        )
    return tickers


def news_text(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(key) or "")
        for key in (
            "title",
            "headline",
            "summary",
            "description",
            "content",
            "text",
        )
    ).strip()


def parse_timezone_aware(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_pydatetime"):
        try:
            parsed = value.to_pydatetime()
        except (TypeError, ValueError, AttributeError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def canonical_record_sha256(record: dict[str, Any]) -> str:
    normalized = {
        str(key): _json_safe(value)
        for key, value in record.items()
        if key != "_dean_context_provenance"
    }
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError, AttributeError):
            pass
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
