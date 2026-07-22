from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


class AccumulationAuthorizationLedger:
    """Append-only, hash-chained approvals for accumulation commands."""

    contract = "dean_accumulation_authorization_v1"

    def __init__(
        self,
        ledger_path: str | Path = "data/dean_os/accumulation_authorization_ledger.jsonl",
    ) -> None:
        self.ledger_path = Path(ledger_path)

    def approve(
        self,
        schedule_path: str | Path,
        *,
        lane_id: str,
        approved_by: str,
        confirm_command_sha256: str,
        expires_at: str,
        approved_at: str | None = None,
    ) -> dict[str, Any]:
        schedule_file = Path(schedule_path)
        schedule = _load_object(schedule_file)
        if schedule.get("contract") != "dean_prospective_accumulation_schedule_v1":
            raise ValueError("unsupported accumulation schedule contract")
        request = next(
            (item for item in schedule.get("authorization_requests") or [] if item.get("lane_id") == lane_id),
            None,
        )
        if request is None:
            raise ValueError(f"lane has no current authorization request: {lane_id}")
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        command = str(request.get("command") or "").strip()
        if not command:
            raise ValueError("authorization request has no command")
        command_sha = command_sha256(command)
        if confirm_command_sha256.lower() != command_sha:
            raise ValueError("confirmed command SHA-256 does not match the scheduled command")

        approved_dt = parse_timezone_aware(approved_at or utc_now_iso())
        expires_dt = parse_timezone_aware(expires_at)
        if approved_dt is None or expires_dt is None:
            raise ValueError("approval timestamps must be timezone-aware")
        if expires_dt <= approved_dt:
            raise ValueError("authorization expiry must be after approval time")

        records = self.read_verified()
        schedule_sha = _sha256_file(schedule_file)
        if any(
            record.get("schedule", {}).get("sha256") == schedule_sha
            and record.get("lane_id") == lane_id
            and record.get("event_type") == "authorization_granted"
            for record in records
        ):
            raise ValueError("this schedule lane already has an authorization record")

        previous_hash = records[-1]["record_sha256"] if records else None
        record: dict[str, Any] = {
            "contract": self.contract,
            "event_type": "authorization_granted",
            "authorization_id": _authorization_id(schedule_sha, lane_id, command_sha, approved_dt.isoformat()),
            "schedule": {"path": str(schedule_file), "sha256": schedule_sha},
            "runbook": dict((schedule.get("inputs") or {}).get("runbook") or {}),
            "lane_id": lane_id,
            "command": command,
            "command_sha256": command_sha,
            "approved_by": approved_by.strip(),
            "approved_at": approved_dt.isoformat(),
            "expires_at": expires_dt.isoformat(),
            "network_or_external_access_may_occur": bool(
                request.get("network_or_external_access_may_occur")
            ),
            "previous_record_sha256": previous_hash,
            "execution_performed": False,
        }
        record["record_sha256"] = _record_sha256(record)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def read_verified(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        records: list[dict[str, Any]] = []
        previous_hash = None
        for line_number, raw in enumerate(self.ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid authorization ledger JSON at line {line_number}") from exc
            if not isinstance(record, dict) or record.get("contract") != self.contract:
                raise ValueError(f"invalid authorization ledger contract at line {line_number}")
            if record.get("previous_record_sha256") != previous_hash:
                raise ValueError(f"authorization ledger chain break at line {line_number}")
            expected = _record_sha256(record)
            if record.get("record_sha256") != expected:
                raise ValueError(f"authorization ledger record hash mismatch at line {line_number}")
            previous_hash = expected
            records.append(record)
        return records

    def status(self) -> dict[str, Any]:
        records = self.read_verified()
        return {
            "contract": "dean_accumulation_authorization_ledger_status_v1",
            "ledger_path": str(self.ledger_path),
            "record_count": len(records),
            "chain_valid": True,
            "tip_sha256": records[-1]["record_sha256"] if records else None,
            "command_execution_performed": False,
        }


def command_sha256(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def _record_sha256(record: dict[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _authorization_id(schedule_sha: str, lane_id: str, command_sha: str, approved_at: str) -> str:
    seed = f"{schedule_sha}|{lane_id}|{command_sha}|{approved_at}".encode("utf-8")
    return "accum_auth_" + hashlib.sha256(seed).hexdigest()[:24]


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["AccumulationAuthorizationLedger", "command_sha256"]
