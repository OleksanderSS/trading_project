from __future__ import annotations

import json

import pytest

from dean_os.system_journal import SystemJournal


def _event(entity_id: str) -> dict:
    return {
        "event_type": "hypothesis_created",
        "effective_at": "2026-07-13T10:00:00+00:00",
        "actor": "test_analyst",
        "domain_id": "test_domain",
        "entity_type": "hypothesis",
        "entity_id": entity_id,
        "context": {"cycle_run_id": "cycle_1"},
        "payload": {"hypothesis": f"claim {entity_id}"},
    }


def test_system_journal_is_hash_chained_and_idempotent(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = SystemJournal(path)

    first = journal.append_many([_event("h1"), _event("h2")])
    second = journal.append_many([_event("h1"), _event("h2")])

    assert first["appended_count"] == 2
    assert second["appended_count"] == 0
    assert second["existing_count"] == 2
    status = journal.status()
    assert status["record_count"] == 2
    assert status["chain_valid"] is True
    records = journal.read_verified()
    assert records[1]["previous_record_sha256"] == records[0]["record_sha256"]


def test_system_journal_rejects_tampering(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = SystemJournal(path)
    journal.append_many([_event("h1"), _event("h2")])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["payload"]["hypothesis"] = "tampered"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        journal.read_verified()
