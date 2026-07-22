"""Read-only hypothesis state projection from the canonical SystemJournal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.analyst_core.schemas import HypothesisLedgerEntry
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.system_journal import SystemJournal


HYPOTHESIS_FIELDS = {
    "hypothesis_id",
    "as_of",
    "hypothesis",
    "confidence",
    "trigger_evidence_ids",
    "supporting_evidence_ids",
    "contradicting_evidence_ids",
    "expected_observations",
    "invalidation_signals",
    "horizons_to_check",
    "status",
    "calibration_note",
    "safety",
}
INACTIVE_DISPOSITIONS = {"reject", "reformulate"}


def project_active_hypotheses(
    journal_path: str | Path,
    *,
    domain_id: str,
    as_of: str,
) -> dict[str, Any]:
    """Project review-eligible hypotheses without writing a second store."""
    cutoff = parse_timezone_aware(as_of)
    if cutoff is None:
        raise ValueError("hypothesis projection as_of must be timezone-aware")
    records = SystemJournal(journal_path).read_verified()
    rows: dict[str, dict[str, Any]] = {}
    dispositions: dict[str, str] = {}
    exclusions: list[dict[str, str]] = []
    considered = 0
    for record in records:
        if record.get("domain_id") != domain_id:
            continue
        effective = parse_timezone_aware(record.get("effective_at"))
        if effective is None or effective > cutoff:
            continue
        entity = record.get("entity") or {}
        hypothesis_id = str(entity.get("entity_id") or "").strip()
        event_type = record.get("event_type")
        if event_type == "hypothesis_created" and hypothesis_id:
            considered += 1
            payload = record.get("payload") or {}
            existing = rows.get(hypothesis_id, {})
            merged = {
                **existing,
                **{
                    key: value
                    for key, value in payload.items()
                    if key in HYPOTHESIS_FIELDS and value is not None
                },
                "hypothesis_id": hypothesis_id,
            }
            merged.setdefault("as_of", record.get("effective_at"))
            rows[hypothesis_id] = merged
        elif event_type == "hypothesis_reviewed" and hypothesis_id:
            disposition = str(
                (record.get("payload") or {}).get("disposition") or ""
            ).strip()
            if disposition:
                dispositions[hypothesis_id] = disposition

    active: list[HypothesisLedgerEntry] = []
    for hypothesis_id, row in sorted(rows.items()):
        disposition = dispositions.get(hypothesis_id)
        if disposition in INACTIVE_DISPOSITIONS:
            exclusions.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "reason": f"inactive_manual_disposition:{disposition}",
                }
            )
            continue
        try:
            active.append(HypothesisLedgerEntry(**row))
        except ValueError as exc:
            exclusions.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "reason": f"projection_schema_invalid:{exc}",
                }
            )
    status = SystemJournal(journal_path).status()
    return {
        "contract": "dean_hypothesis_journal_projection_v1",
        "domain_id": domain_id,
        "as_of": cutoff.isoformat(),
        "journal_path": str(Path(journal_path)),
        "journal_tip_sha256": status.get("tip_sha256"),
        "journal_record_count": status.get("record_count"),
        "created_event_count_considered": considered,
        "active_hypotheses": active,
        "active_hypothesis_count": len(active),
        "exclusions": exclusions,
        "read_only": True,
        "status_change_performed": False,
    }


__all__ = ["project_active_hypotheses"]
