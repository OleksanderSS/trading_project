from __future__ import annotations

import json
from pathlib import Path

import pytest

from dean_os.domain_macro_collection_execution_gate import (
    DomainMacroCollectionExecutionGate,
)
from dean_os.domain_macro_collection_executor import (
    DomainMacroCollectionExecutor,
    TicketConsumptionLedger,
)
from dean_os.domain_macro_collection_request import CONTRACT as REQUEST_CONTRACT

AS_OF = "2026-07-14T09:50:00+00:00"
SCOPE = [
    "CPIAUCSL",
    "DCOILWTICO",
    "DGS10",
    "FEDFUNDS",
    "INDPRO",
    "PPIACO",
    "VIXCLS",
]


def _request(tmp_path: Path) -> Path:
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "macro_request_fixture",
                "mode": "domain_macro_collection_request",
                "contract": REQUEST_CONTRACT,
                "domain_id": "energy",
                "inputs": {"request_as_of": AS_OF},
                "summary": {
                    "status": "macro_collection_request_ready",
                    "request_required": True,
                    "execution_authorized": False,
                    "collector_run_performed": False,
                },
                "collection_request": {
                    "replacement_series_scope": SCOPE,
                    "runtime_parameters": {
                        "series_ids": SCOPE,
                        "maximum_collection_runs": 1,
                        "automatic_retry_allowed": False,
                    },
                },
                "point_in_time_contract": {
                    "availability_field": "realtime_start",
                    "missing_availability_action": "reject_snapshot",
                    "canonical_pipeline_target": "data/processed/features/macro_data.parquet",
                },
                "safety": {"proposal_only": True, "automatic_retry_allowed": False},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _gate(tmp_path: Path, request: Path) -> Path:
    payload = DomainMacroCollectionExecutionGate(tmp_path / "gate").build(
        request_path=request,
        evaluated_at=AS_OF,
        credential_present_override=True,
        journal_path=tmp_path / "gate_journal.jsonl",
        save=True,
    )
    return Path(payload["saved_paths"]["latest_json"])


def _rows(scope: list[str] = SCOPE) -> list[dict]:
    return [
        {
            "series_id": series_id,
            "date": "2026-07-13",
            "realtime_start": "2026-07-14",
            "realtime_end": "2026-07-14",
            "value": str(index + 1.0),
            "source_locator": f"https://fred.stlouisfed.org/series/{series_id}",
        }
        for index, series_id in enumerate(scope)
    ]


def _executor_args(tmp_path: Path) -> dict:
    request = _request(tmp_path)
    gate = _gate(tmp_path, request)
    return {
        "gate_path": gate,
        "request_path": request,
        "ledger_path": tmp_path / "ticket_ledger.jsonl",
        "journal_path": tmp_path / "journal.jsonl",
        "workspace_root": tmp_path,
        "credential_present_override": True,
        "save": False,
    }


def test_without_execute_flag_does_not_claim_ticket(tmp_path: Path) -> None:
    payload = DomainMacroCollectionExecutor(tmp_path / "report").build(
        **_executor_args(tmp_path),
        execute_network=False,
    )

    assert payload["summary"]["status"] == "macro_collection_execution_awaiting_explicit_execute_flag"
    assert payload["summary"]["ticket_claimed"] is False
    assert payload["ticket_ledger"]["record_count"] == 0


def test_success_consumes_ticket_writes_snapshot_and_runs_envelope_once(tmp_path: Path) -> None:
    calls = {"fetch": 0, "envelope": 0}

    def fetch(scope: list[str], as_of: str) -> list[dict]:
        calls["fetch"] += 1
        assert scope == SCOPE
        assert as_of == AS_OF
        return _rows()

    def envelope(path: Path, as_of: str) -> dict:
        calls["envelope"] += 1
        assert path.is_file()
        assert as_of == AS_OF
        return {
            "summary": {
                "status": "domain_macro_binding_candidate_ready",
                "candidate_ready_for_binding_review": True,
            }
        }

    args = _executor_args(tmp_path)
    first = DomainMacroCollectionExecutor(tmp_path / "report").build(
        **args,
        execute_network=True,
        apply_journal=True,
        fetch_rows=fetch,
        envelope_runner=envelope,
    )
    second = DomainMacroCollectionExecutor(tmp_path / "report").build(
        **args,
        execute_network=True,
        fetch_rows=fetch,
        envelope_runner=envelope,
    )

    assert first["summary"]["status"] == "macro_collection_execution_completed_candidate_ready"
    assert first["summary"]["ticket_consumed"] is True
    assert first["summary"]["snapshot_written"] is True
    assert first["collection_result"]["present_series_scope"] == SCOPE
    assert first["ticket_ledger"]["record_count"] == 2
    assert second["summary"]["status"] == "macro_collection_execution_blocked"
    assert "execution_ticket_already_consumed" in second["summary"]["structural_blockers"]
    assert calls == {"fetch": 1, "envelope": 1}


def test_partial_scope_fails_and_ticket_cannot_retry(tmp_path: Path) -> None:
    args = _executor_args(tmp_path)
    payload = DomainMacroCollectionExecutor(tmp_path / "report").build(
        **args,
        execute_network=True,
        fetch_rows=lambda _scope, _as_of: _rows(SCOPE[:-1]),
        envelope_runner=lambda _path, _as_of: {},
    )

    assert payload["summary"]["status"] == "macro_collection_execution_failed_no_retry"
    assert payload["collection_result"]["failure_code"] == "ValueError"
    assert payload["summary"]["snapshot_written"] is False
    assert payload["summary"]["ticket_consumed"] is True
    assert TicketConsumptionLedger(args["ledger_path"]).has_claim(
        payload["inputs"]["ticket_id"]
    )


def test_future_availability_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["realtime_start"] = "2026-07-15"
    payload = DomainMacroCollectionExecutor(tmp_path / "report").build(
        **_executor_args(tmp_path),
        execute_network=True,
        fetch_rows=lambda _scope, _as_of: rows,
        envelope_runner=lambda _path, _as_of: {},
    )

    assert payload["summary"]["status"] == "macro_collection_execution_failed_no_retry"
    assert payload["summary"]["snapshot_written"] is False


def test_tampered_ticket_is_blocked_before_fetch(tmp_path: Path) -> None:
    request = _request(tmp_path)
    gate = _gate(tmp_path, request)
    payload = json.loads(gate.read_text(encoding="utf-8"))
    payload["execution_ticket"]["series_scope"] = ["DGS10"]
    gate.write_text(json.dumps(payload), encoding="utf-8")
    called = {"fetch": 0}

    def fetch(_scope: list[str], _as_of: str) -> list[dict]:
        called["fetch"] += 1
        return _rows()

    result = DomainMacroCollectionExecutor(tmp_path / "report").build(
        gate_path=gate,
        request_path=request,
        ledger_path=tmp_path / "ticket_ledger.jsonl",
        journal_path=tmp_path / "journal.jsonl",
        workspace_root=tmp_path,
        credential_present_override=True,
        execute_network=True,
        fetch_rows=fetch,
        save=False,
    )

    assert "ticket_hash_invalid" in result["summary"]["structural_blockers"]
    assert called["fetch"] == 0


def test_ticket_ledger_detects_tampering(tmp_path: Path) -> None:
    args = _executor_args(tmp_path)
    result = DomainMacroCollectionExecutor(tmp_path / "report").build(
        **args,
        execute_network=True,
        fetch_rows=lambda _scope, _as_of: _rows(),
        envelope_runner=lambda _path, _as_of: {
            "summary": {"candidate_ready_for_binding_review": True}
        },
    )
    ledger_path = Path(args["ledger_path"])
    records = ledger_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(records[0])
    first["ticket_id"] = "tampered"
    records[0] = json.dumps(first)
    ledger_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    assert result["summary"]["ticket_consumed"] is True
    with pytest.raises(ValueError, match="hash mismatch"):
        TicketConsumptionLedger(ledger_path).read_verified()
