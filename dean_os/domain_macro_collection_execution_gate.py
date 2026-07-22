from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.domain_macro_collection_request import CONTRACT as REQUEST_CONTRACT
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready
from src.data.collectors.fred_collector import FredCollector

CONTRACT = "dean_domain_macro_collection_execution_gate_v1"
DEFAULT_REQUEST_PATH = "reports/dean_os/domain_macro_collection_request_current/latest.json"
DEFAULT_REGISTRY_PATH = "dean_os/config/macro_series_registry.yaml"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_macro_collection_execution_gate_current"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"
EXPECTED_TARGET = Path("data/processed/features/macro_data.parquet")


class DomainMacroCollectionExecutionGate:
    """Authorize at most one exact FRED request; never execute it."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str = "energy",
        request_path: str | Path = DEFAULT_REQUEST_PATH,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        evaluated_at: str | None = None,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        apply_journal: bool = False,
        credential_present_override: bool | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        request_file = Path(request_path)
        registry_file = Path(registry_path)
        request = _load_json(request_file)
        registry = _load_yaml(registry_file)
        cutoff = evaluated_at or utc_now_iso()
        _aware(cutoff)
        request_sha = _sha256_file(request_file)
        runtime_contract = dict(FredCollector.runtime_request_contract)
        credential_present = (
            credential_present_override
            if credential_present_override is not None
            else bool(str(os.getenv("FRED_API_KEY") or "").strip())
        )
        blockers = _blockers(
            request=request,
            request_sha=request_sha,
            domain_id=domain_id,
            registry=registry,
            runtime_contract=runtime_contract,
            cutoff=cutoff,
        )
        readiness_blockers = list(blockers)
        if not credential_present:
            readiness_blockers.append("fred_api_key_missing")
        authorized = not readiness_blockers
        if blockers:
            status = "macro_collection_execution_blocked_contract"
        elif not credential_present:
            status = "macro_collection_execution_blocked_missing_fred_api_key"
        else:
            status = "macro_collection_execution_ready_single_run"

        scope = list((request.get("collection_request") or {}).get("replacement_series_scope") or [])
        request_as_of = str((request.get("inputs") or {}).get("request_as_of") or "")
        ticket = _ticket(
            domain_id=domain_id,
            request_sha=request_sha,
            request_as_of=request_as_of,
            scope=scope,
            evaluated_at=cutoff,
            runtime_contract=runtime_contract,
        ) if authorized else None
        payload = {
            "run_id": _run_id("domain_macro_collection_execution_gate"),
            "created_at": utc_now_iso(),
            "mode": "domain_macro_collection_execution_gate",
            "contract": CONTRACT,
            "domain_id": domain_id,
            "inputs": {
                "request_path": str(request_path),
                "request_sha256": request_sha,
                "registry_path": str(registry_path),
                "registry_sha256": _sha256_file(registry_file),
                "evaluated_at": cutoff,
            },
            "summary": {
                "status": status,
                "structural_blockers": blockers,
                "readiness_blockers": sorted(set(readiness_blockers)),
                "credential_name": "FRED_API_KEY",
                "credential_present": credential_present,
                "credential_value_read_into_report": False,
                "single_run_authorized": authorized,
                "execution_ticket_issued": ticket is not None,
                "maximum_collection_runs": 1,
                "automatic_retry_allowed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "snapshot_written": False,
                "stage2_validation_performed": False,
                "macro_envelope_run_performed": False,
                "binding_accepted": False,
                "can_invoke_domain_analysis": False,
                "can_trade": False,
            },
            "preflight_checks": {
                "request_contract_valid": not blockers,
                "request_sha256": request_sha,
                "series_scope": scope,
                "series_count": len(scope),
                "series_allowlisted_by_registry": not any(
                    item.startswith("series_not_allowlisted:") for item in blockers
                ),
                "runtime_request_contract": runtime_contract,
                "point_in_time_cutoff": request_as_of,
                "canonical_output_target": str(EXPECTED_TARGET).replace("\\", "/"),
                "output_target_contract_valid": "canonical_output_target_mismatch" not in blockers,
                "credential_presence_checked": True,
                "credential_secret_exposed": False,
            },
            "execution_ticket": ticket,
            "executor_contract": {
                "consume_exact_request_sha256": request_sha,
                "consume_exact_ticket_sha256": (ticket or {}).get("ticket_sha256"),
                "allowed_steps": [
                    "one_fred_collection",
                    "stage2_point_in_time_validation",
                    "atomic_macro_snapshot_write",
                    "one_domain_macro_envelope_run",
                ],
                "forbidden_steps": [
                    "automatic_retry",
                    "second_collection_run",
                    "binding_acceptance",
                    "analyst_invocation",
                    "hypothesis_decision",
                    "learning_write",
                    "trade",
                ],
            },
            "safety": {
                "preflight_only": True,
                "network_access_performed": False,
                "collector_instantiated": False,
                "collector_run_performed": False,
                "credential_value_logged": False,
                "filesystem_write_outside_reports_and_journal": False,
                "snapshot_write_performed": False,
                "binding_write_performed": False,
                "learning_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        payload["journal"] = _journal(
            payload=payload,
            source_path=request_file,
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


def _blockers(
    *,
    request: dict[str, Any],
    request_sha: str,
    domain_id: str,
    registry: dict[str, Any],
    runtime_contract: dict[str, Any],
    cutoff: str,
) -> list[str]:
    del request_sha
    blockers: list[str] = []
    summary = request.get("summary") or {}
    collection = request.get("collection_request") or {}
    runtime = collection.get("runtime_parameters") or {}
    pit = request.get("point_in_time_contract") or {}
    safety = request.get("safety") or {}
    if request.get("contract") != REQUEST_CONTRACT or request.get("mode") != "domain_macro_collection_request":
        blockers.append("unsupported_collection_request_contract")
    if request.get("domain_id") != domain_id:
        blockers.append("request_domain_mismatch")
    if summary.get("status") != "macro_collection_request_ready" or summary.get("request_required") is not True:
        blockers.append("collection_request_not_ready")
    if summary.get("execution_authorized") is not False or summary.get("collector_run_performed") is not False:
        blockers.append("request_execution_boundary_invalid")
    if safety.get("proposal_only") is not True or safety.get("automatic_retry_allowed") is not False:
        blockers.append("request_safety_boundary_invalid")
    scope = list(collection.get("replacement_series_scope") or [])
    if not scope or len(scope) != len(set(scope)) or sorted(scope) != sorted(runtime.get("series_ids") or []):
        blockers.append("runtime_series_scope_mismatch")
    catalog = dict(registry.get("series") or {})
    for series_id in scope:
        if series_id not in catalog:
            blockers.append(f"series_not_allowlisted:{series_id}")
    if runtime.get("maximum_collection_runs") != 1 or runtime.get("automatic_retry_allowed") is not False:
        blockers.append("runtime_not_single_pass")
    expected_capabilities = {
        "contract": "fred_bounded_runtime_request_v1",
        "runtime_series_ids_supported": True,
        "timezone_aware_as_of_required": True,
        "fred_vintage_dates_supported": True,
        "observation_end_cutoff_supported": True,
        "point_in_time_availability_field": "realtime_start",
        "maximum_runs_enforced_by_external_gate": True,
    }
    if runtime_contract != expected_capabilities:
        blockers.append("fred_runtime_capability_contract_mismatch")
    if pit.get("availability_field") != "realtime_start" or pit.get("missing_availability_action") != "reject_snapshot":
        blockers.append("point_in_time_contract_invalid")
    if Path(str(pit.get("canonical_pipeline_target") or "")) != EXPECTED_TARGET:
        blockers.append("canonical_output_target_mismatch")
    request_as_of = str((request.get("inputs") or {}).get("request_as_of") or "")
    try:
        if not request_as_of or _aware(request_as_of) > _aware(cutoff):
            blockers.append("request_after_gate_cutoff")
    except ValueError:
        blockers.append("request_as_of_invalid")
    return sorted(set(blockers))


def _ticket(
    *,
    domain_id: str,
    request_sha: str,
    request_as_of: str,
    scope: list[str],
    evaluated_at: str,
    runtime_contract: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "contract": "dean_domain_macro_single_run_ticket_v1",
        "domain_id": domain_id,
        "request_sha256": request_sha,
        "request_as_of": request_as_of,
        "series_scope": scope,
        "authorized_at": evaluated_at,
        "maximum_collection_runs": 1,
        "automatic_retry_allowed": False,
        "runtime_contract": runtime_contract["contract"],
        "consumed": False,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["ticket_sha256"] = hashlib.sha256(encoded).hexdigest()
    body["ticket_id"] = "macro_run_" + body["ticket_sha256"][:24]
    return body


def _journal(
    *, payload: dict[str, Any], source_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    event = {
        "event_type": "action_reviewed",
        "effective_at": payload["inputs"]["evaluated_at"],
        "actor": "domain_macro_collection_execution_gate",
        "domain_id": payload["domain_id"],
        "entity_type": "bounded_macro_collection_preflight",
        "entity_id": "macro_gate_" + payload["inputs"]["request_sha256"][:16],
        "source_artifact": artifact_binding(source_path),
        "context": {"context_family": "macro", "preflight_only": True},
        "payload": {
            "status": payload["summary"]["status"],
            "readiness_blockers": payload["summary"]["readiness_blockers"],
            "credential_present": payload["summary"]["credential_present"],
            "credential_value_logged": False,
            "single_run_authorized": payload["summary"]["single_run_authorized"],
            "collector_run_performed": False,
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
    return {
        "apply_requested": True,
        **result,
        "record_count": status["record_count"],
        "chain_valid": status["chain_valid"],
        "tip_sha256": status["tip_sha256"],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    checks = payload["preflight_checks"]
    lines = [
        "# DEAN-OS Domain Macro Collection Execution Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Request SHA-256: `{checks['request_sha256']}`",
        f"- Series: {checks['series_count']}",
        f"- FRED credential present: {summary['credential_present']}",
        f"- Credential value exposed: {summary['credential_value_read_into_report']}",
        f"- Single run authorized: {summary['single_run_authorized']}",
        f"- Ticket issued: {summary['execution_ticket_issued']}",
        f"- Collector run performed: {summary['collector_run_performed']}",
        f"- Network access performed: {summary['network_access_performed']}",
        "",
        "## Readiness blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["readiness_blockers"] or ["none"])
    lines.extend(
        [
            "",
            "## Allowed future execution",
            "",
            "- One FRED collection for the exact SHA-bound request.",
            "- Stage 2 point-in-time validation and atomic macro snapshot write.",
            "- One domain macro envelope run.",
            "",
            "## Boundary",
            "",
            "- This gate does not instantiate or run the collector.",
            "- No retry, binding acceptance, analyst invocation, learning write or trade is authorized.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


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


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = ["CONTRACT", "DomainMacroCollectionExecutionGate", "render_markdown"]
