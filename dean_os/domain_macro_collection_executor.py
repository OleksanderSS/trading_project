from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.domain_macro_collection_execution_gate import CONTRACT as GATE_CONTRACT
from dean_os.domain_macro_collection_request import CONTRACT as REQUEST_CONTRACT
from dean_os.domain_scoped_macro_envelope import DomainScopedMacroEnvelopeCeremony
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready
from src.core.file_management.file_manager import FileManager
from src.data.collectors.fred_collector import FredCollector
from src.pipeline.stages.processing.data_handler import ProcessingDataHandler
from src.pipeline.stages.processing.storage import ProcessingStorage

CONTRACT = "dean_domain_macro_collection_executor_v1"
LEDGER_CONTRACT = "dean_domain_macro_ticket_consumption_v1"
DEFAULT_GATE_PATH = "reports/dean_os/domain_macro_collection_execution_gate_current/latest.json"
DEFAULT_REQUEST_PATH = "reports/dean_os/domain_macro_collection_request_current/latest.json"
DEFAULT_REGISTRY_PATH = "dean_os/config/macro_series_registry.yaml"
DEFAULT_DISPATCH_PATH = "reports/dean_os/domain_binding_task_dispatch_current/latest.json"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_macro_collection_executor_current"
DEFAULT_ENVELOPE_OUTPUT_DIR = "reports/dean_os/domain_scoped_macro_envelope_current"
DEFAULT_LEDGER_PATH = "data/dean_os/domain_macro_ticket_consumption.jsonl"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"

FetchRows = Callable[[list[str], str], list[dict[str, Any]]]
EnvelopeRunner = Callable[[Path, str], dict[str, Any]]


class DomainMacroCollectionExecutor:
    """Consume one SHA-bound ticket and execute exactly one macro vertical run."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str = "energy",
        gate_path: str | Path = DEFAULT_GATE_PATH,
        request_path: str | Path = DEFAULT_REQUEST_PATH,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        dispatch_path: str | Path = DEFAULT_DISPATCH_PATH,
        envelope_output_dir: str | Path = DEFAULT_ENVELOPE_OUTPUT_DIR,
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        workspace_root: str | Path = ".",
        execute_network: bool = False,
        apply_journal: bool = False,
        credential_present_override: bool | None = None,
        fetch_rows: FetchRows | None = None,
        envelope_runner: EnvelopeRunner | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        gate_file = Path(gate_path)
        request_file = Path(request_path)
        gate = _load_json(gate_file)
        request = _load_json(request_file)
        request_sha = _sha256_file(request_file)
        gate_sha = _sha256_file(gate_file)
        ticket = dict(gate.get("execution_ticket") or {})
        credential_present = (
            credential_present_override
            if credential_present_override is not None
            else bool(str(os.getenv("FRED_API_KEY") or "").strip())
        )
        blockers = _validate(
            gate, request, request_sha, domain_id, credential_present
        )
        ticket_id = str(ticket.get("ticket_id") or "missing_ticket")
        ledger = TicketConsumptionLedger(ledger_path)
        already_consumed = ledger.has_claim(ticket_id)
        if already_consumed:
            blockers.append("execution_ticket_already_consumed")

        scope = list((request.get("collection_request") or {}).get("replacement_series_scope") or [])
        as_of = str((request.get("inputs") or {}).get("request_as_of") or "")
        result: dict[str, Any] = {
            "collector_run_performed": False,
            "network_access_performed": False,
            "source_row_count": 0,
            "normalized_row_count": 0,
            "invalid_value_row_count": 0,
            "present_series_scope": [],
            "missing_series_scope": scope,
            "raw_snapshot_path": None,
            "canonical_snapshot_path": None,
            "canonical_snapshot_sha256": None,
            "envelope_status": None,
            "envelope_candidate_ready": False,
            "failure_code": None,
        }
        claim_record: dict[str, Any] | None = None
        completion_record: dict[str, Any] | None = None

        if blockers:
            status = "macro_collection_execution_blocked"
        elif not execute_network:
            status = "macro_collection_execution_awaiting_explicit_execute_flag"
        else:
            claim_record = ledger.claim(
                ticket=ticket,
                gate_path=gate_file,
                gate_sha256=gate_sha,
                request_path=request_file,
                request_sha256=request_sha,
                claimed_at=utc_now_iso(),
            )
            try:
                result["collector_run_performed"] = True
                result["network_access_performed"] = fetch_rows is None
                rows = (fetch_rows or _fetch_fred_rows)(scope, as_of)
                frame, validation = _validated_frame(rows, scope, as_of)
                result.update(validation)
                workspace = Path(workspace_root).resolve()
                raw_path = workspace / "data" / "dean_os" / "macro_collection_runs" / ticket_id / "fred_validated.parquet"
                manager = FileManager(base_dir=workspace)
                manager.save_dataframe(frame, raw_path, format="parquet", remove_tz=False, index=False)
                result["raw_snapshot_path"] = str(raw_path)
                normalized = ProcessingDataHandler(None, None).clean_and_normalize_macro_data(frame)
                canonical_relative = ProcessingStorage(manager)._save_persistent_macro_snapshot(normalized)
                canonical_path = workspace / canonical_relative
                manager._executor.shutdown(wait=True)
                result["normalized_row_count"] = len(normalized)
                result["canonical_snapshot_path"] = str(canonical_path)
                result["canonical_snapshot_sha256"] = _sha256_file(canonical_path)
                if envelope_runner is None:
                    envelope = DomainScopedMacroEnvelopeCeremony(envelope_output_dir).build(
                        domain_id=domain_id,
                        source_path=canonical_path,
                        as_of=as_of,
                        registry_path=registry_path,
                        dispatch_path=dispatch_path,
                        execution_gate_path=gate_path,
                        journal_path=journal_path,
                        apply_journal=apply_journal,
                    )
                else:
                    envelope = envelope_runner(canonical_path, as_of)
                result["envelope_status"] = (envelope.get("summary") or {}).get("status")
                result["envelope_candidate_ready"] = bool(
                    (envelope.get("summary") or {}).get("candidate_ready_for_binding_review")
                )
                status = (
                    "macro_collection_execution_completed_candidate_ready"
                    if result["envelope_candidate_ready"]
                    else "macro_collection_execution_completed_envelope_not_ready"
                )
            except Exception as exc:
                status = "macro_collection_execution_failed_no_retry"
                result["failure_code"] = f"{type(exc).__name__}"
            completion_record = ledger.complete(
                ticket_id=ticket_id,
                status=status,
                completed_at=utc_now_iso(),
                result={
                    "source_row_count": result["source_row_count"],
                    "normalized_row_count": result["normalized_row_count"],
                    "present_series_scope": result["present_series_scope"],
                    "missing_series_scope": result["missing_series_scope"],
                    "canonical_snapshot_sha256": result["canonical_snapshot_sha256"],
                    "envelope_status": result["envelope_status"],
                    "failure_code": result["failure_code"],
                    "retry_allowed": False,
                },
            )

        payload = {
            "run_id": _run_id("domain_macro_collection_executor"),
            "created_at": utc_now_iso(),
            "mode": "domain_macro_collection_executor",
            "contract": CONTRACT,
            "domain_id": domain_id,
            "inputs": {
                "gate_path": str(gate_path),
                "gate_sha256": gate_sha,
                "request_path": str(request_path),
                "request_sha256": request_sha,
                "registry_path": str(registry_path),
                "dispatch_path": str(dispatch_path),
                "ticket_id": ticket_id,
                "ticket_sha256": ticket.get("ticket_sha256"),
                "request_as_of": as_of,
                "execute_network_requested": execute_network,
            },
            "summary": {
                "status": status,
                "structural_blockers": sorted(set(blockers)),
                "ticket_claimed": claim_record is not None,
                "ticket_consumed": completion_record is not None,
                "second_run_allowed": False,
                "automatic_retry_allowed": False,
                "collector_run_performed": result["collector_run_performed"],
                "network_access_performed": result["network_access_performed"],
                "snapshot_written": result["canonical_snapshot_path"] is not None,
                "stage2_validation_performed": result["normalized_row_count"] > 0,
                "macro_envelope_run_performed": result["envelope_status"] is not None,
                "binding_accepted": False,
                "can_invoke_domain_analysis": False,
                "can_trade": False,
            },
            "collection_result": result,
            "ticket_ledger": {
                "path": str(ledger_path),
                "claim_record_sha256": (claim_record or {}).get("record_sha256"),
                "completion_record_sha256": (completion_record or {}).get("record_sha256"),
                **ledger.status(),
            },
            "safety": {
                "exact_ticket_sha_enforced": True,
                "maximum_collection_runs": 1,
                "automatic_retry_allowed": False,
                "credential_value_logged": False,
                "exception_message_logged": False,
                "binding_write_performed": False,
                "learning_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        payload["journal"] = _journal(
            payload=payload,
            gate_path=gate_file,
            journal_path=Path(journal_path),
            apply=apply_journal,
        )
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


class TicketConsumptionLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read_verified(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        previous = None
        for line_number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("contract") != LEDGER_CONTRACT:
                raise ValueError(f"invalid ticket ledger contract at line {line_number}")
            if record.get("previous_record_sha256") != previous:
                raise ValueError(f"ticket ledger chain break at line {line_number}")
            expected = _record_sha(record)
            if record.get("record_sha256") != expected:
                raise ValueError(f"ticket ledger hash mismatch at line {line_number}")
            records.append(record)
            previous = expected
        return records

    def has_claim(self, ticket_id: str) -> bool:
        return any(
            record.get("ticket_id") == ticket_id and record.get("event_type") == "ticket_claimed"
            for record in self.read_verified()
        )

    def claim(
        self,
        *,
        ticket: dict[str, Any],
        gate_path: Path,
        gate_sha256: str,
        request_path: Path,
        request_sha256: str,
        claimed_at: str,
    ) -> dict[str, Any]:
        ticket_id = str(ticket["ticket_id"])
        if self.has_claim(ticket_id):
            raise ValueError("execution ticket already consumed")
        return self._append(
            {
                "event_type": "ticket_claimed",
                "ticket_id": ticket_id,
                "ticket_sha256": ticket["ticket_sha256"],
                "gate": {"path": str(gate_path), "sha256": gate_sha256},
                "request": {"path": str(request_path), "sha256": request_sha256},
                "claimed_at": claimed_at,
                "maximum_collection_runs": 1,
                "automatic_retry_allowed": False,
            }
        )

    def complete(
        self, *, ticket_id: str, status: str, completed_at: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.has_claim(ticket_id):
            raise ValueError("cannot complete unclaimed ticket")
        return self._append(
            {
                "event_type": "ticket_completed",
                "ticket_id": ticket_id,
                "completed_at": completed_at,
                "status": status,
                "result": result,
            }
        )

    def _append(self, body: dict[str, Any]) -> dict[str, Any]:
        records = self.read_verified()
        record = {
            "contract": LEDGER_CONTRACT,
            **body,
            "previous_record_sha256": records[-1]["record_sha256"] if records else None,
        }
        record["record_sha256"] = _record_sha(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def status(self) -> dict[str, Any]:
        records = self.read_verified()
        return {
            "contract": "dean_domain_macro_ticket_consumption_status_v1",
            "record_count": len(records),
            "chain_valid": True,
            "tip_sha256": records[-1]["record_sha256"] if records else None,
        }


def _validate(
    gate: dict[str, Any],
    request: dict[str, Any],
    request_sha: str,
    domain_id: str,
    credential_present: bool,
) -> list[str]:
    blockers: list[str] = []
    summary = gate.get("summary") or {}
    ticket = gate.get("execution_ticket") or {}
    if gate.get("contract") != GATE_CONTRACT or gate.get("mode") != "domain_macro_collection_execution_gate":
        blockers.append("unsupported_gate_contract")
    if request.get("contract") != REQUEST_CONTRACT:
        blockers.append("unsupported_request_contract")
    if gate.get("domain_id") != domain_id or request.get("domain_id") != domain_id:
        blockers.append("domain_mismatch")
    if summary.get("status") != "macro_collection_execution_ready_single_run" or summary.get("single_run_authorized") is not True:
        blockers.append("gate_not_authorized")
    if (gate.get("inputs") or {}).get("request_sha256") != request_sha or ticket.get("request_sha256") != request_sha:
        blockers.append("request_sha_mismatch")
    if ticket.get("maximum_collection_runs") != 1 or ticket.get("automatic_retry_allowed") is not False:
        blockers.append("ticket_not_single_pass")
    if ticket and not _ticket_hash_valid(ticket):
        blockers.append("ticket_hash_invalid")
    if not credential_present:
        blockers.append("fred_api_key_missing")
    return sorted(set(blockers))


def _ticket_hash_valid(ticket: dict[str, Any]) -> bool:
    body = {key: value for key, value in ticket.items() if key not in {"ticket_id", "ticket_sha256"}}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest() == ticket.get("ticket_sha256")


def _fetch_fred_rows(scope: list[str], as_of: str) -> list[dict[str, Any]]:
    return asyncio.run(_fetch_fred_rows_async(scope, as_of))


async def _fetch_fred_rows_async(scope: list[str], as_of: str) -> list[dict[str, Any]]:
    api_key = str(os.getenv("FRED_API_KEY") or "").strip()
    # httpx otherwise logs full query URLs, including the FRED API key.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    as_of_dt = _aware(as_of)
    as_of_date = as_of_dt.date().isoformat()
    observation_start = (as_of_dt.date() - timedelta(days=730)).isoformat()
    collector = object.__new__(FredCollector)
    collector.start_date = observation_start
    collector.timeout = 30
    collector.logger = logging.getLogger("bounded_fred_macro_executor")
    transport = httpx.AsyncHTTPTransport(retries=0)
    async with httpx.AsyncClient(
        transport=transport,
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": "DEAN-OS/1.0 bounded-macro-collection"},
    ) as client:
        results = await asyncio.gather(
            *[
                collector._fetch_series(
                    series_id,
                    client,
                    api_key,
                    observation_start=observation_start,
                    observation_end=as_of_date,
                    vintage_date=as_of_date,
                )
                for series_id in scope
            ],
            return_exceptions=True,
        )
    if any(isinstance(item, BaseException) for item in results):
        raise RuntimeError("one_or_more_fred_series_failed")
    return [row for rows in results for row in rows]


def _validated_frame(
    rows: list[dict[str, Any]], scope: list[str], as_of: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame(rows)
    required = {"series_id", "date", "realtime_start", "value", "source_locator"}
    missing_columns = sorted(required - set(frame.columns))
    if frame.empty or missing_columns:
        raise ValueError("macro_collection_schema_invalid")
    unexpected = sorted(set(frame["series_id"].astype(str)) - set(scope))
    if unexpected:
        raise ValueError("unexpected_macro_series")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    invalid_value_count = int(frame["value"].isna().sum())
    frame = frame.dropna(subset=["value"]).copy()
    observation = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    availability = pd.to_datetime(frame["realtime_start"], errors="coerce", utc=True)
    cutoff = pd.Timestamp(_aware(as_of))
    if observation.isna().any() or availability.isna().any():
        raise ValueError("macro_collection_timestamp_invalid")
    if (observation > cutoff).any() or (availability > cutoff).any():
        raise ValueError("macro_collection_after_as_of")
    present = sorted(set(frame["series_id"].astype(str)))
    missing = sorted(set(scope) - set(present))
    if missing:
        raise ValueError("macro_collection_incomplete_series_scope")
    frame["hash"] = frame.apply(
        lambda row: hashlib.sha256(
            "|".join(
                str(row.get(key, ""))
                for key in ("series_id", "date", "realtime_start", "value")
            ).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    return frame, {
        "source_row_count": len(rows),
        "invalid_value_row_count": invalid_value_count,
        "present_series_scope": present,
        "missing_series_scope": missing,
    }


def _journal(
    *, payload: dict[str, Any], gate_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    if not payload["summary"]["ticket_claimed"]:
        return {
            "apply_requested": apply,
            "events_proposed": 0,
            "appended_count": 0,
            "existing_count": 0,
            "chain_valid": SystemJournal(journal_path).status()["chain_valid"],
        }
    succeeded = payload["summary"]["status"] == "macro_collection_execution_completed_candidate_ready"
    event = {
        "event_type": "action_executed" if succeeded else "incident_recorded",
        "effective_at": payload["created_at"],
        "actor": "domain_macro_collection_executor",
        "domain_id": payload["domain_id"],
        "entity_type": "bounded_macro_collection_execution",
        "entity_id": payload["inputs"]["ticket_id"],
        "source_artifact": artifact_binding(gate_path),
        "context": {"context_family": "macro", "single_pass": True},
        "payload": {
            "status": payload["summary"]["status"],
            "request_sha256": payload["inputs"]["request_sha256"],
            "ticket_sha256": payload["inputs"]["ticket_sha256"],
            "collector_run_performed": payload["summary"]["collector_run_performed"],
            "network_access_performed": payload["summary"]["network_access_performed"],
            "canonical_snapshot_sha256": payload["collection_result"]["canonical_snapshot_sha256"],
            "envelope_status": payload["collection_result"]["envelope_status"],
            "failure_code": payload["collection_result"]["failure_code"],
            "retry_allowed": False,
            "binding_accepted": False,
        },
    }
    journal = SystemJournal(journal_path)
    if not apply:
        return {
            "apply_requested": False,
            "events_proposed": 1,
            "appended_count": 0,
            "existing_count": 0,
            "chain_valid": journal.status()["chain_valid"],
        }
    result = journal.append_many([event])
    status = journal.status()
    return {"apply_requested": True, **result, "record_count": status["record_count"], "chain_valid": status["chain_valid"], "tip_sha256": status["tip_sha256"]}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    result = payload["collection_result"]
    lines = [
        "# DEAN-OS Domain Macro Collection Executor",
        "",
        f"- Status: `{summary['status']}`",
        f"- Ticket: `{payload['inputs']['ticket_id']}`",
        f"- Ticket claimed: {summary['ticket_claimed']}",
        f"- Ticket consumed: {summary['ticket_consumed']}",
        f"- Network access: {summary['network_access_performed']}",
        f"- Collector run: {summary['collector_run_performed']}",
        f"- Source rows: {result['source_row_count']}",
        f"- Normalized rows: {result['normalized_row_count']}",
        f"- Snapshot written: {summary['snapshot_written']}",
        f"- Envelope status: `{result['envelope_status']}`",
        f"- Binding accepted: {summary['binding_accepted']}",
        "",
        "## Series",
        "",
        "- Present: " + (", ".join(result["present_series_scope"]) or "none"),
        "- Missing: " + (", ".join(result["missing_series_scope"]) or "none"),
        "",
        "## Boundary",
        "",
        "- The ticket is single-use even if the external call fails.",
        "- Automatic retry and a second collection run are forbidden.",
        "- Binding acceptance, analyst invocation, learning writes and trading remain disabled.",
    ]
    if summary["structural_blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in summary["structural_blockers"])
    if result["failure_code"]:
        lines.extend(["", "## Failure", "", f"- Code: `{result['failure_code']}`", "- Exception text was intentionally not recorded to prevent credential leakage."])
    return "\n".join(lines).strip() + "\n"


def _record_sha(record: dict[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _aware(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = ["CONTRACT", "DomainMacroCollectionExecutor", "TicketConsumptionLedger"]
