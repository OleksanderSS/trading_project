from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

SYSTEM_JOURNAL_EVENT_CONTRACT = "dean_system_journal_event_v1"

SYSTEM_JOURNAL_EVENT_TYPES = {
    "source_snapshot_recorded",
    "news_observed",
    "evidence_observed",
    "analysis_cycle_recorded",
    "hypothesis_created",
    "hypothesis_reviewed",
    "replay_checkpoint_matured",
    "outcome_recorded",
    "hypothesis_assessed",
    "action_proposed",
    "action_reviewed",
    "action_executed",
    "learning_proposal_created",
    "learning_proposal_reviewed",
    "incident_recorded",
    "governance_closure_recorded",
    "report_generated",
}


class SystemJournal:
    """Append-only, hash-chained journal for evidence, reasoning and actions.

    The journal is an audit/event log, not production learning memory. Entries
    can propose a lesson or action, but they cannot change a prompt, template,
    model, configuration, trading state or broker state.
    """

    def __init__(
        self,
        journal_path: str | Path = "data/dean_os/system_journal.jsonl",
    ) -> None:
        self.journal_path = Path(journal_path)

    def append(
        self,
        *,
        event_type: str,
        effective_at: str,
        actor: str,
        domain_id: str,
        entity_type: str,
        entity_id: str,
        source_artifact: dict[str, Any] | None = None,
        parent_event_ids: list[str] | None = None,
        context: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        recorded_at: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        normalized = _normalize_event(
            event_type=event_type,
            effective_at=effective_at,
            actor=actor,
            domain_id=domain_id,
            entity_type=entity_type,
            entity_id=entity_id,
            source_artifact=source_artifact,
            parent_event_ids=parent_event_ids,
            context=context,
            payload=payload,
            recorded_at=recorded_at,
        )
        event_key = normalized["event_key"]
        existing = self.read_verified()
        duplicate = next(
            (record for record in existing if record.get("event_key") == event_key),
            None,
        )
        if duplicate is not None:
            return duplicate, False

        previous_hash = existing[-1]["record_sha256"] if existing else None
        record = _record_from_normalized(normalized, previous_hash)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
        return record, True

    def append_many(
        self,
        events: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        verified = self.read_verified()
        by_key = {str(record["event_key"]): record for record in verified}
        previous_hash = verified[-1]["record_sha256"] if verified else None
        appended: list[str] = []
        existing: list[str] = []
        new_records: list[dict[str, Any]] = []
        for event in events:
            normalized = _normalize_event(**event)
            event_key = str(normalized["event_key"])
            duplicate = by_key.get(event_key)
            if duplicate is not None:
                existing.append(str(duplicate["journal_event_id"]))
                continue
            record = _record_from_normalized(normalized, previous_hash)
            previous_hash = record["record_sha256"]
            by_key[event_key] = record
            new_records.append(record)
            appended.append(str(record["journal_event_id"]))
        if new_records:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
                for record in new_records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
        return {
            "requested_count": len(appended) + len(existing),
            "appended_count": len(appended),
            "existing_count": len(existing),
            "appended_event_ids": appended,
            "existing_event_ids": existing,
        }

    def read_verified(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        records: list[dict[str, Any]] = []
        previous_hash = None
        for line_number, raw in enumerate(
            self.journal_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid system journal JSON at line {line_number}"
                ) from exc
            if (
                not isinstance(record, dict)
                or record.get("contract") != SYSTEM_JOURNAL_EVENT_CONTRACT
            ):
                raise ValueError(
                    f"invalid system journal contract at line {line_number}"
                )
            if record.get("previous_record_sha256") != previous_hash:
                raise ValueError(
                    f"system journal chain break at line {line_number}"
                )
            expected_hash = _record_sha256(record)
            if record.get("record_sha256") != expected_hash:
                raise ValueError(
                    f"system journal record hash mismatch at line {line_number}"
                )
            previous_hash = expected_hash
            records.append(record)
        return records

    def status(self) -> dict[str, Any]:
        records = self.read_verified()
        event_counts = Counter(str(record.get("event_type")) for record in records)
        domain_counts = Counter(str(record.get("domain_id")) for record in records)
        return {
            "contract": "dean_system_journal_status_v1",
            "journal_path": str(self.journal_path),
            "record_count": len(records),
            "chain_valid": True,
            "tip_sha256": records[-1]["record_sha256"] if records else None,
            "event_type_counts": dict(sorted(event_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "learning_memory_write_performed": False,
            "production_rule_update_performed": False,
            "action_execution_performed": any(
                record.get("event_type") == "action_executed" for record in records
            ),
            "can_trade": False,
        }


def artifact_binding(
    path: str | Path,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"journal source artifact missing: {artifact_path}")
    loaded = payload if payload is not None else _load_object(artifact_path)
    return {
        "path": str(artifact_path),
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "run_id": loaded.get("run_id"),
        "contract": loaded.get("contract") or loaded.get("producer_contract"),
    }


def _normalize_event(
    *,
    event_type: str,
    effective_at: str,
    actor: str,
    domain_id: str,
    entity_type: str,
    entity_id: str,
    source_artifact: dict[str, Any] | None = None,
    parent_event_ids: list[str] | None = None,
    context: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    if event_type not in SYSTEM_JOURNAL_EVENT_TYPES:
        raise ValueError(f"unsupported system journal event type: {event_type}")
    if not actor.strip():
        raise ValueError("journal actor is required")
    if not domain_id.strip():
        raise ValueError("journal domain_id is required")
    if not entity_type.strip() or not entity_id.strip():
        raise ValueError("journal entity_type and entity_id are required")
    effective_dt = parse_timezone_aware(effective_at)
    recorded_dt = parse_timezone_aware(recorded_at or utc_now_iso())
    if effective_dt is None or recorded_dt is None:
        raise ValueError("journal timestamps must be timezone-aware")
    normalized_source = _normalized_source_artifact(source_artifact or {})
    normalized_context = json_ready(context or {})
    normalized_payload = json_ready(payload or {})
    normalized_parents = sorted(
        {str(item).strip() for item in parent_event_ids or [] if str(item).strip()}
    )
    identity = {
        "event_type": event_type,
        "effective_at": effective_dt.isoformat(),
        "domain_id": domain_id.strip(),
        "entity_type": entity_type.strip(),
        "entity_id": entity_id.strip(),
        "source_artifact": normalized_source,
        "parent_event_ids": normalized_parents,
        "context": normalized_context,
        "payload": normalized_payload,
    }
    return {
        **identity,
        "event_key": _sha256_json(identity),
        "recorded_at": recorded_dt.isoformat(),
        "actor": actor.strip(),
    }


def _record_from_normalized(
    normalized: dict[str, Any], previous_hash: str | None
) -> dict[str, Any]:
    event_key = str(normalized["event_key"])
    record: dict[str, Any] = {
        "contract": SYSTEM_JOURNAL_EVENT_CONTRACT,
        "journal_event_id": "journal_event_" + event_key[:24],
        "event_key": event_key,
        "event_type": normalized["event_type"],
        "effective_at": normalized["effective_at"],
        "recorded_at": normalized["recorded_at"],
        "actor": normalized["actor"],
        "domain_id": normalized["domain_id"],
        "entity": {
            "entity_type": normalized["entity_type"],
            "entity_id": normalized["entity_id"],
        },
        "source_artifact": normalized["source_artifact"],
        "parent_event_ids": normalized["parent_event_ids"],
        "context": normalized["context"],
        "payload": normalized["payload"],
        "previous_record_sha256": previous_hash,
        "safety": _safety(),
    }
    record["record_sha256"] = _record_sha256(record)
    return record


def _normalized_source_artifact(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    normalized = {
        "path": str(value.get("path") or "").strip() or None,
        "sha256": str(value.get("sha256") or "").strip().lower() or None,
        "run_id": value.get("run_id"),
        "contract": value.get("contract"),
    }
    sha = normalized["sha256"]
    if sha is not None and (len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha)):
        raise ValueError("source artifact sha256 must be a 64-character hex digest")
    return normalized


def _record_sha256(record: dict[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return _sha256_json(body)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"journal source artifact must be an object: {path}")
    return payload


def _safety() -> dict[str, bool]:
    return {
        "append_only": True,
        "review_only": True,
        "learning_memory_write_performed": False,
        "production_rule_update_performed": False,
        "model_promotion_performed": False,
        "broker_access_performed": False,
        "can_trade": False,
    }


__all__ = [
    "SYSTEM_JOURNAL_EVENT_CONTRACT",
    "SYSTEM_JOURNAL_EVENT_TYPES",
    "SystemJournal",
    "artifact_binding",
]
