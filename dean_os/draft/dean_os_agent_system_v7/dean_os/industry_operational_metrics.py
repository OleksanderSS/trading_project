from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


CONTRACT = "dean_industry_operational_metrics_v1"
ALLOWED_VALUE_KINDS = {"actual", "guidance", "estimate", "target"}
ALLOWED_REVISION_STATUSES = {"original", "revised", "restated"}
PERCENT_METRICS = {"utilization", "capacity_utilization", "yield"}


class IndustryOperationalMetricsBuilder:
    """Validate operator-supplied industry metrics without inferring numbers from prose."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/industry_operational_metrics_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build_from_path(
        self,
        source_path: str | Path,
        *,
        as_of: str,
        domain_id: str,
        save: bool = True,
    ) -> dict[str, Any]:
        path = Path(source_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw.get("records") if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            raise ValueError("operational metric input must be a list or contain records[]")
        return self.build(
            records,
            as_of=as_of,
            domain_id=domain_id,
            input_reference=str(path),
            input_sha256=_file_sha256(path),
            save=save,
        )

    def build(
        self,
        records: list[Any],
        *,
        as_of: str,
        domain_id: str,
        input_reference: str = "in_memory_operator_packet",
        input_sha256: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = parse_timezone_aware(as_of)
        if cutoff is None:
            raise ValueError("as_of must be a timezone-aware ISO-8601 timestamp")
        accepted: list[dict[str, Any]] = []
        quarantined: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for index, raw in enumerate(records):
            normalized, reasons = _normalize_record(raw, cutoff=cutoff, domain_id=domain_id)
            record_id = normalized.get("record_id")
            if record_id and record_id in seen_ids:
                reasons.append("duplicate_record_id")
            if reasons:
                quarantined.append(
                    {
                        "index": index,
                        "status": "quarantined",
                        "reasons": sorted(set(reasons)),
                        "record": normalized,
                    }
                )
                continue
            seen_ids.add(record_id)
            normalized["status"] = "point_in_time_compatible_review_only"
            accepted.append(normalized)

        _apply_revision_lineage(accepted, quarantined)
        reason_counts = Counter(
            reason for item in quarantined for reason in item.get("reasons", [])
        )
        created_at = utc_now_iso()
        run_id = "industry_operational_metrics_" + created_at.replace(":", "").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "industry_operational_metrics_review_only",
            "contract": CONTRACT,
            "inputs": {
                "source_reference": input_reference,
                "source_sha256": input_sha256,
                "as_of": cutoff.isoformat(),
                "domain_id": domain_id,
            },
            "summary": {
                "input_count": len(records),
                "accepted_count": len(accepted),
                "quarantined_count": len(quarantined),
                "reason_counts": dict(sorted(reason_counts.items())),
                "actual_count": sum(row["value_kind"] == "actual" for row in accepted),
                "non_actual_count": sum(row["value_kind"] != "actual" for row in accepted),
                "active_count": sum(row["lifecycle_status"] == "active" for row in accepted),
                "superseded_count": sum(row["lifecycle_status"] == "superseded" for row in accepted),
                "can_support_gap_review": bool(accepted),
                "can_close_gap_automatically": False,
                "can_trade": False,
            },
            "accepted_records": accepted,
            "quarantined_records": quarantined,
            "integration_boundary": {
                "role": "structured_industry_evidence_candidate",
                "actual_guidance_estimate_kept_separate": True,
                "prose_to_number_inference_allowed": False,
                "point_in_time_filter_applied": True,
                "revisions_preserve_history": True,
                "manual_review_required": True,
                "stage5_feature_write_allowed": False,
                "learning_write_allowed": False,
                "replay_registration_allowed": False,
            },
            "safety": {
                "review_only": True,
                "automatic_gap_closure_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_render_markdown(payload),
                run_id=run_id,
            )
        return payload


def _normalize_record(raw: Any, *, cutoff: datetime, domain_id: str) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(raw, dict):
        return {"raw_type": type(raw).__name__}, ["record_not_structured"]
    row = dict(raw)
    reasons: list[str] = []
    required = (
        "record_id", "entity", "metric_name", "value", "unit", "period",
        "available_at", "source_locator", "source_sha256", "value_kind",
    )
    for field in required:
        if row.get(field) in (None, ""):
            reasons.append(f"{field}_missing")

    value = row.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        reasons.append("value_not_finite_numeric")
    available_at = parse_timezone_aware(str(row.get("available_at") or ""))
    if available_at is None:
        reasons.append("available_at_invalid_or_not_timezone_aware")
    elif available_at > cutoff:
        reasons.append("available_at_after_as_of")
    value_kind = str(row.get("value_kind") or "").lower()
    if value_kind not in ALLOWED_VALUE_KINDS:
        reasons.append("value_kind_invalid")
    revision_status = str(row.get("revision_status") or "original").lower()
    if revision_status not in ALLOWED_REVISION_STATUSES:
        reasons.append("revision_status_invalid")
    supersedes = row.get("supersedes_record_id")
    if revision_status in {"revised", "restated"} and not supersedes:
        reasons.append("revision_missing_supersedes_record_id")
    if revision_status == "original" and supersedes:
        reasons.append("original_cannot_supersede_record")
    source_hash = str(row.get("source_sha256") or "").lower()
    if source_hash and (len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash)):
        reasons.append("source_sha256_invalid")

    metric_name = str(row.get("metric_name") or "").lower()
    unit = str(row.get("unit") or "").lower()
    normalized_value = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    if metric_name in PERCENT_METRICS:
        if unit not in {"percent", "percentage_point"}:
            reasons.append("percent_metric_requires_explicit_percent_unit")
        elif unit == "percent" and isinstance(normalized_value, float) and not 0 <= normalized_value <= 100:
            reasons.append("percent_value_out_of_range")

    normalized = {
        "record_id": str(row.get("record_id") or ""),
        "domain_id": str(row.get("domain_id") or domain_id),
        "entity": str(row.get("entity") or ""),
        "ticker": str(row.get("ticker") or "").upper() or None,
        "geography": row.get("geography"),
        "product_segment": row.get("product_segment"),
        "metric_name": metric_name,
        "value": normalized_value,
        "unit": unit,
        "period": str(row.get("period") or ""),
        "available_at": available_at.isoformat() if available_at else None,
        "source_locator": str(row.get("source_locator") or ""),
        "source_sha256": source_hash,
        "source_tier": row.get("source_tier"),
        "methodology": row.get("methodology"),
        "value_kind": value_kind,
        "revision_status": revision_status,
        "supersedes_record_id": supersedes,
        "lifecycle_status": "active",
        "as_of": cutoff.isoformat(),
    }
    normalized["observation_sha256"] = _canonical_sha256(normalized)
    return normalized, reasons


def _apply_revision_lineage(accepted: list[dict[str, Any]], quarantined: list[dict[str, Any]]) -> None:
    by_id = {row["record_id"]: row for row in accepted}
    invalid_ids: set[str] = set()
    for row in accepted:
        parent_id = row.get("supersedes_record_id")
        if not parent_id:
            continue
        parent = by_id.get(str(parent_id))
        if parent is None:
            invalid_ids.add(row["record_id"])
            quarantined.append({
                "status": "quarantined",
                "reasons": ["superseded_record_not_present_in_packet"],
                "record": row,
            })
            continue
        identity = ("domain_id", "entity", "metric_name", "unit")
        if any(parent.get(key) != row.get(key) for key in identity):
            invalid_ids.add(row["record_id"])
            quarantined.append({
                "status": "quarantined",
                "reasons": ["revision_metric_identity_mismatch"],
                "record": row,
            })
            continue
        parent["lifecycle_status"] = "superseded"
        parent["superseded_by_record_id"] = row["record_id"]
    if invalid_ids:
        accepted[:] = [row for row in accepted if row["record_id"] not in invalid_ids]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DEAN-OS Industry Operational Metrics",
        "",
        f"- Status: review-only structured evidence candidate",
        f"- Accepted: {summary['accepted_count']}",
        f"- Quarantined: {summary['quarantined_count']}",
        f"- Active: {summary['active_count']}",
        f"- Superseded: {summary['superseded_count']}",
        f"- Can close gaps automatically: {summary['can_close_gap_automatically']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "Numbers are accepted only from structured records with explicit units, periods, point-in-time availability, and source hashes. Guidance and estimates remain separate from actuals.",
    ]
    return "\n".join(lines) + "\n"
