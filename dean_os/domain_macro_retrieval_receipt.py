from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.domain_macro_collection_executor import (
    CONTRACT as EXECUTOR_CONTRACT,
    TicketConsumptionLedger,
)
from dean_os.domain_scoped_macro_envelope import DomainScopedMacroEnvelopeCeremony
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready
from src.core.file_management.file_manager import FileManager
from src.pipeline.stages.processing.data_handler import ProcessingDataHandler
from src.pipeline.stages.processing.storage import ProcessingStorage

CONTRACT = "dean_domain_macro_retrieval_receipt_v1"
DEFAULT_EXECUTOR_PATH = "reports/dean_os/domain_macro_collection_executor_current/latest.json"
DEFAULT_GATE_PATH = "reports/dean_os/domain_macro_collection_execution_gate_current/latest.json"
DEFAULT_LEDGER_PATH = "data/dean_os/domain_macro_ticket_consumption.jsonl"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_macro_retrieval_receipt_current"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"


class DomainMacroRetrievalReceipt:
    """Stamp actual system availability from a verified completed collection."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        executor_path: str | Path = DEFAULT_EXECUTOR_PATH,
        gate_path: str | Path = DEFAULT_GATE_PATH,
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        workspace_root: str | Path = ".",
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        executor_file = Path(executor_path)
        executor = _load_json(executor_file)
        ticket_id = str((executor.get("inputs") or {}).get("ticket_id") or "")
        ledger = TicketConsumptionLedger(ledger_path)
        records = ledger.read_verified()
        completion = next(
            (
                item
                for item in records
                if item.get("ticket_id") == ticket_id
                and item.get("event_type") == "ticket_completed"
            ),
            None,
        )
        result = executor.get("collection_result") or {}
        source_path = Path(str(result.get("canonical_snapshot_path") or ""))
        source_sha = _sha256_file(source_path) if source_path.is_file() else None
        expected_sha = result.get("canonical_snapshot_sha256")
        blockers: list[str] = []
        if executor.get("contract") != EXECUTOR_CONTRACT:
            blockers.append("unsupported_executor_contract")
        summary = executor.get("summary") or {}
        if summary.get("ticket_consumed") is not True or summary.get("network_access_performed") is not True:
            blockers.append("verified_collection_not_completed")
        if not completion:
            blockers.append("ticket_completion_receipt_missing")
        if not source_path.is_file() or source_sha != expected_sha:
            blockers.append("canonical_snapshot_sha_mismatch")
        available_at = str((completion or {}).get("completed_at") or "")
        envelope: dict[str, Any] = {}
        stamped_path: Path | None = None
        stamped_sha: str | None = None
        row_count = 0

        if not blockers:
            frame = pd.read_parquet(source_path)
            if frame.empty or "realtime_start" not in frame.columns:
                blockers.append("canonical_snapshot_point_in_time_contract_invalid")
            else:
                frame["available_at"] = available_at
                normalized = ProcessingDataHandler(None, None).clean_and_normalize_macro_data(frame)
                workspace = Path(workspace_root).resolve()
                stamped_path = (
                    workspace
                    / "data"
                    / "dean_os"
                    / "macro_collection_runs"
                    / ticket_id
                    / "fred_retrieval_stamped.parquet"
                )
                manager = FileManager(base_dir=workspace)
                manager.save_dataframe(
                    normalized,
                    stamped_path,
                    format="parquet",
                    remove_tz=False,
                    index=False,
                )
                ProcessingStorage(manager)._save_persistent_macro_snapshot(normalized)
                manager._executor.shutdown(wait=True)
                stamped_sha = _sha256_file(stamped_path)
                row_count = len(normalized)
                envelope = DomainScopedMacroEnvelopeCeremony().build(
                    domain_id=str(executor.get("domain_id") or "energy"),
                    source_path=stamped_path,
                    as_of=available_at,
                    execution_gate_path=gate_path,
                    journal_path=journal_path,
                    apply_journal=apply_journal,
                )

        candidate_ready = bool(
            (envelope.get("summary") or {}).get("candidate_ready_for_binding_review")
        )
        status = (
            "macro_retrieval_receipt_blocked"
            if blockers
            else "macro_retrieval_receipt_completed_candidate_ready"
            if candidate_ready
            else "macro_retrieval_receipt_completed_envelope_not_ready"
        )
        payload = {
            "run_id": _run_id("domain_macro_retrieval_receipt"),
            "created_at": utc_now_iso(),
            "mode": "domain_macro_retrieval_receipt",
            "contract": CONTRACT,
            "domain_id": executor.get("domain_id"),
            "inputs": {
                "executor_path": str(executor_path),
                "executor_sha256": _sha256_file(executor_file),
                "ticket_id": ticket_id,
                "source_snapshot_path": str(source_path),
                "source_snapshot_sha256": source_sha,
                "retrieved_available_at": available_at,
                "availability_basis": "ticket_completion_receipt",
            },
            "summary": {
                "status": status,
                "structural_blockers": blockers,
                "row_count": row_count,
                "retrieval_timestamp_applied": not blockers,
                "network_access_performed": False,
                "second_collection_performed": False,
                "snapshot_written": stamped_path is not None,
                "envelope_run_performed": bool(envelope),
                "candidate_ready_for_binding_review": candidate_ready,
                "binding_accepted": False,
                "can_invoke_domain_analysis": False,
                "can_trade": False,
            },
            "stamped_snapshot": {
                "path": str(stamped_path) if stamped_path else None,
                "sha256": stamped_sha,
                "available_at": available_at,
                "original_realtime_start_preserved": True,
                "file_mtime_used": False,
            },
            "envelope": {
                "status": (envelope.get("summary") or {}).get("status"),
                "candidate_path": (envelope.get("saved_paths") or {}).get("latest_json"),
                "candidate_ready": candidate_ready,
            },
            "safety": {
                "offline_recovery_only": True,
                "network_access_performed": False,
                "ticket_reused": False,
                "release_time_fabricated": False,
                "retrieval_time_used_as_system_availability": True,
                "binding_write_performed": False,
                "learning_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        payload["journal"] = _journal(
            payload=payload,
            executor_path=executor_file,
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


def _journal(
    *, payload: dict[str, Any], executor_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    if payload["summary"]["structural_blockers"]:
        return {"apply_requested": apply, "events_proposed": 0, "appended_count": 0, "existing_count": 0, "chain_valid": SystemJournal(journal_path).status()["chain_valid"]}
    event = {
        "event_type": "action_executed",
        "effective_at": payload["inputs"]["retrieved_available_at"],
        "actor": "domain_macro_retrieval_receipt",
        "domain_id": str(payload["domain_id"]),
        "entity_type": "macro_retrieval_availability_receipt",
        "entity_id": payload["inputs"]["ticket_id"],
        "source_artifact": artifact_binding(executor_path),
        "context": {"context_family": "macro", "offline_recovery_only": True},
        "payload": {
            "availability_basis": "ticket_completion_receipt",
            "available_at": payload["inputs"]["retrieved_available_at"],
            "stamped_snapshot_sha256": payload["stamped_snapshot"]["sha256"],
            "candidate_ready": payload["summary"]["candidate_ready_for_binding_review"],
            "network_access_performed": False,
            "ticket_reused": False,
            "binding_accepted": False,
        },
    }
    journal = SystemJournal(journal_path)
    if not apply:
        return {"apply_requested": False, "events_proposed": 1, "appended_count": 0, "existing_count": 0, "chain_valid": journal.status()["chain_valid"]}
    result = journal.append_many([event])
    status = journal.status()
    return {"apply_requested": True, **result, "record_count": status["record_count"], "chain_valid": status["chain_valid"], "tip_sha256": status["tip_sha256"]}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DEAN-OS Macro Retrieval Receipt",
        "",
        f"- Status: `{summary['status']}`",
        f"- Available at: `{payload['inputs']['retrieved_available_at']}`",
        f"- Availability basis: `ticket_completion_receipt`",
        f"- Rows: {summary['row_count']}",
        f"- Network access: {summary['network_access_performed']}",
        f"- Second collection: {summary['second_collection_performed']}",
        f"- Candidate ready: {summary['candidate_ready_for_binding_review']}",
        f"- Binding accepted: {summary['binding_accepted']}",
        "",
        "## Boundary",
        "",
        "- Original FRED realtime_start is preserved.",
        "- available_at is the hash-chained ticket completion time when the system had the snapshot.",
        "- File mtime was not used and original release time was not fabricated.",
    ]
    if summary["structural_blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in summary["structural_blockers"])
    return "\n".join(lines).strip() + "\n"


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


__all__ = ["CONTRACT", "DomainMacroRetrievalReceipt"]
