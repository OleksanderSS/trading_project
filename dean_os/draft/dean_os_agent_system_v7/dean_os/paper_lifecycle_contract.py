from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.utils import sha256_json

PAPER_LIFECYCLE_SCHEMA_VERSION = "dean_isolated_paper_lifecycle_v1"


def file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_fingerprint(payload: dict[str, Any]) -> str:
    return sha256_json(payload)


def fingerprint_matches(
    payload: dict[str, Any],
    *,
    object_key: str,
    fingerprint_key: str,
) -> bool:
    value = payload.get(object_key)
    fingerprint = payload.get(fingerprint_key)
    return (
        isinstance(value, dict)
        and valid_sha256(fingerprint)
        and object_fingerprint(value) == str(fingerprint).lower()
    )


def load_json_object(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def receipt_lineage_issues(
    payload: dict[str, Any],
    *,
    receipt_path: str | Path | None,
    now: datetime | None = None,
) -> list[str]:
    issues: list[str] = []
    receipt = payload.get("receipt")
    if payload.get("mode") != "review_decision_receipt":
        issues.append("receipt_mode_invalid")
    if payload.get("schema_version") != PAPER_LIFECYCLE_SCHEMA_VERSION:
        issues.append("receipt_schema_version_invalid")
    if not isinstance(receipt, dict):
        return [*issues, "receipt_object_missing"]
    if not fingerprint_matches(
        payload,
        object_key="receipt",
        fingerprint_key="receipt_fingerprint",
    ):
        issues.append("receipt_fingerprint_invalid")
    if not valid_sha256(file_sha256(receipt_path)):
        issues.append("receipt_file_sha256_unavailable")

    source_path = receipt.get("source_artifact_path")
    expected_source_sha = receipt.get("source_artifact_sha256")
    current_source_sha = file_sha256(source_path)
    if not valid_sha256(expected_source_sha):
        issues.append("receipt_source_artifact_sha256_missing")
    elif current_source_sha != expected_source_sha:
        issues.append("receipt_source_artifact_sha256_mismatch")

    source = load_json_object(source_path)
    if source is None:
        issues.append("receipt_source_artifact_unavailable")
    else:
        if source.get("mode") != "post_dry_run_review":
            issues.append("receipt_source_mode_not_post_dry_run_review")
        if receipt.get("source_artifact_mode") != source.get("mode"):
            issues.append("receipt_source_mode_mismatch")
        if receipt.get("source_artifact_run_id") != source.get("run_id"):
            issues.append("receipt_source_run_id_mismatch")
        source_review = source.get("post_dry_run_review")
        if not isinstance(source_review, dict):
            issues.append("receipt_source_review_object_missing")
        else:
            if receipt.get("source_decision") != source_review.get("decision"):
                issues.append("receipt_source_decision_mismatch")
            if receipt.get("source_verdict") != source_review.get("verdict"):
                issues.append("receipt_source_verdict_mismatch")
            if source_review.get("decision") != "ready_for_human_review":
                issues.append("receipt_source_not_ready_for_human_review")
            if source_review.get("verdict") not in {"clear", "caution"}:
                issues.append("receipt_source_verdict_not_reviewable")

    expires_at = parse_timestamp(receipt.get("expires_at"))
    if expires_at is None:
        issues.append("receipt_expiry_missing_or_invalid")
    else:
        reference = now or datetime.now(UTC)
        if expires_at <= reference.astimezone(UTC):
            issues.append("receipt_expired")
    return sorted(set(issues))


def valid_sha256(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return len(normalized) == 64 and all(
        char in "0123456789abcdef" for char in normalized
    )
