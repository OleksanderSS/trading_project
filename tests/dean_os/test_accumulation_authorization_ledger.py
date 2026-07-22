from __future__ import annotations

import json
from pathlib import Path

import pytest

from dean_os.accumulation_authorization_ledger import (
    AccumulationAuthorizationLedger,
    command_sha256,
)


def _schedule(path: Path, command: str = "python safe_runner.py --review") -> Path:
    path.write_text(
        json.dumps(
            {
                "contract": "dean_prospective_accumulation_schedule_v1",
                "inputs": {"runbook": {"path": "runbook.json", "sha256": "a" * 64}},
                "authorization_requests": [
                    {
                        "lane_id": "macro_context",
                        "command": command,
                        "approved": False,
                        "network_or_external_access_may_occur": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_appends_hash_chained_explicit_authorization(tmp_path: Path) -> None:
    command = "python safe_runner.py --review"
    schedule = _schedule(tmp_path / "schedule.json", command)
    ledger = AccumulationAuthorizationLedger(tmp_path / "ledger.jsonl")
    record = ledger.approve(
        schedule,
        lane_id="macro_context",
        approved_by="human-reviewer",
        confirm_command_sha256=command_sha256(command),
        approved_at="2026-07-12T12:00:00+00:00",
        expires_at="2026-07-13T12:00:00+00:00",
    )
    assert record["execution_performed"] is False
    assert record["previous_record_sha256"] is None
    assert ledger.status()["record_count"] == 1
    assert ledger.status()["chain_valid"] is True


def test_rejects_command_hash_mismatch(tmp_path: Path) -> None:
    schedule = _schedule(tmp_path / "schedule.json")
    with pytest.raises(ValueError, match="does not match"):
        AccumulationAuthorizationLedger(tmp_path / "ledger.jsonl").approve(
            schedule,
            lane_id="macro_context",
            approved_by="reviewer",
            confirm_command_sha256="0" * 64,
            approved_at="2026-07-12T12:00:00+00:00",
            expires_at="2026-07-13T12:00:00+00:00",
        )


def test_detects_tampered_record(tmp_path: Path) -> None:
    command = "python safe_runner.py --review"
    schedule = _schedule(tmp_path / "schedule.json", command)
    path = tmp_path / "ledger.jsonl"
    ledger = AccumulationAuthorizationLedger(path)
    ledger.approve(
        schedule,
        lane_id="macro_context",
        approved_by="reviewer",
        confirm_command_sha256=command_sha256(command),
        approved_at="2026-07-12T12:00:00+00:00",
        expires_at="2026-07-13T12:00:00+00:00",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    record["approved_by"] = "attacker"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ledger.read_verified()


def test_rejects_duplicate_schedule_lane_authorization(tmp_path: Path) -> None:
    command = "python safe_runner.py --review"
    schedule = _schedule(tmp_path / "schedule.json", command)
    ledger = AccumulationAuthorizationLedger(tmp_path / "ledger.jsonl")
    kwargs = dict(
        lane_id="macro_context",
        approved_by="reviewer",
        confirm_command_sha256=command_sha256(command),
        approved_at="2026-07-12T12:00:00+00:00",
        expires_at="2026-07-13T12:00:00+00:00",
    )
    ledger.approve(schedule, **kwargs)
    with pytest.raises(ValueError, match="already has"):
        ledger.approve(schedule, **kwargs)
