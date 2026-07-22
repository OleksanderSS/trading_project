from __future__ import annotations

import json
from pathlib import Path

from dean_os.domain_macro_collection_execution_gate import (
    DomainMacroCollectionExecutionGate,
)
from dean_os.domain_macro_collection_request import CONTRACT as REQUEST_CONTRACT
from dean_os.system_journal import SystemJournal

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
                "safety": {
                    "proposal_only": True,
                    "automatic_retry_allowed": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _build(tmp_path: Path, credential_present: bool, **kwargs) -> dict:
    return DomainMacroCollectionExecutionGate(tmp_path / "report").build(
        request_path=_request(tmp_path),
        evaluated_at=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        credential_present_override=credential_present,
        save=False,
        **kwargs,
    )


def test_missing_credential_blocks_without_exposing_secret(tmp_path: Path) -> None:
    payload = _build(tmp_path, False)

    assert payload["summary"]["status"] == "macro_collection_execution_blocked_missing_fred_api_key"
    assert payload["summary"]["readiness_blockers"] == ["fred_api_key_missing"]
    assert payload["summary"]["credential_value_read_into_report"] is False
    assert payload["summary"]["single_run_authorized"] is False
    assert payload["execution_ticket"] is None


def test_ready_gate_issues_exact_single_run_ticket_only(tmp_path: Path) -> None:
    payload = _build(tmp_path, True)

    assert payload["summary"]["status"] == "macro_collection_execution_ready_single_run"
    assert payload["summary"]["single_run_authorized"] is True
    assert payload["execution_ticket"]["maximum_collection_runs"] == 1
    assert payload["execution_ticket"]["automatic_retry_allowed"] is False
    assert payload["execution_ticket"]["series_scope"] == SCOPE
    assert payload["summary"]["collector_run_performed"] is False


def test_non_allowlisted_series_fails_closed(tmp_path: Path) -> None:
    path = _request(tmp_path)
    request = json.loads(path.read_text(encoding="utf-8"))
    request["collection_request"]["replacement_series_scope"].append("NOT_A_FRED_SERIES")
    request["collection_request"]["runtime_parameters"]["series_ids"].append("NOT_A_FRED_SERIES")
    path.write_text(json.dumps(request), encoding="utf-8")
    payload = DomainMacroCollectionExecutionGate(tmp_path / "report").build(
        request_path=path,
        evaluated_at=AS_OF,
        credential_present_override=True,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["summary"]["single_run_authorized"] is False
    assert "series_not_allowlisted:NOT_A_FRED_SERIES" in payload["summary"]["structural_blockers"]


def test_retry_or_second_run_contract_is_rejected(tmp_path: Path) -> None:
    path = _request(tmp_path)
    request = json.loads(path.read_text(encoding="utf-8"))
    request["collection_request"]["runtime_parameters"]["maximum_collection_runs"] = 2
    path.write_text(json.dumps(request), encoding="utf-8")
    payload = DomainMacroCollectionExecutionGate(tmp_path / "report").build(
        request_path=path,
        evaluated_at=AS_OF,
        credential_present_override=True,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert "runtime_not_single_pass" in payload["summary"]["structural_blockers"]
    assert payload["summary"]["execution_ticket_issued"] is False


def test_wrong_output_target_is_rejected(tmp_path: Path) -> None:
    path = _request(tmp_path)
    request = json.loads(path.read_text(encoding="utf-8"))
    request["point_in_time_contract"]["canonical_pipeline_target"] = "somewhere/else.parquet"
    path.write_text(json.dumps(request), encoding="utf-8")
    payload = DomainMacroCollectionExecutionGate(tmp_path / "report").build(
        request_path=path,
        evaluated_at=AS_OF,
        credential_present_override=True,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert "canonical_output_target_mismatch" in payload["summary"]["structural_blockers"]


def test_blocked_preflight_journal_is_idempotent_and_report_is_safe(tmp_path: Path) -> None:
    path = _request(tmp_path)
    builder = DomainMacroCollectionExecutionGate(tmp_path / "report")
    args = {
        "request_path": path,
        "evaluated_at": AS_OF,
        "credential_present_override": False,
        "journal_path": tmp_path / "journal.jsonl",
        "apply_journal": True,
        "save": True,
    }
    first = builder.build(**args)
    second = builder.build(**args)

    assert first["journal"]["appended_count"] == 1
    assert second["journal"]["appended_count"] == 0
    assert SystemJournal(tmp_path / "journal.jsonl").status()["chain_valid"] is True
    report = Path(second["saved_paths"]["latest_markdown"]).read_text(encoding="utf-8")
    assert "Credential value exposed: False" in report
    assert "does not instantiate or run the collector" in report
