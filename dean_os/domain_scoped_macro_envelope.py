from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.analysts._producers.macro import (
    SAVED_MACRO_PRODUCER_CONTRACT,
    SavedMacroEvidenceProducer,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.analyst_core.domain_analyst_lifecycle_profile import DomainAnalystLifecycleProfileCompiler
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready

DOMAIN_MACRO_ENVELOPE_CONTRACT = "dean_domain_scoped_macro_evidence_envelope_v1"
DEFAULT_DISPATCH_PATH = "reports/dean_os/domain_binding_task_dispatch_current/latest.json"
DEFAULT_EXECUTION_GATE_PATH = "reports/dean_os/domain_macro_collection_execution_gate_current/latest.json"
DEFAULT_REGISTRY_PATH = "dean_os/config/macro_series_registry.yaml"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_scoped_macro_envelope_current"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"


class DomainScopedMacroEnvelopeCeremony:
    """Run one offline macro preview and emit a domain-bound candidate only."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str = "energy",
        source_path: str | Path | None = None,
        as_of: str,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        dispatch_path: str | Path = DEFAULT_DISPATCH_PATH,
        execution_gate_path: str | Path | None = None,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        dispatch_file = Path(dispatch_path)
        dispatch = _load_json(dispatch_file)
        execution_gate_file = Path(execution_gate_path) if execution_gate_path else None
        execution_gate = (
            _load_json(execution_gate_file)
            if execution_gate_file and execution_gate_file.is_file()
            else None
        )
        lifecycle = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
        overlay = lifecycle.get("domain_overlay") or {}
        requested_scope = sorted(
            {str(item).strip() for item in overlay.get("macro_series_scope") or [] if str(item).strip()}
        )
        blockers = _ceremony_blockers(
            dispatch=dispatch,
            domain_id=domain_id,
            requested_scope=requested_scope,
            lifecycle=lifecycle,
            execution_gate=execution_gate,
            as_of=as_of,
        )
        source = Path(source_path) if source_path else None
        source_sha = _sha256_file(source) if source and source.is_file() else None
        core: dict[str, Any] | None = None
        relevant: list[dict[str, Any]] = []
        missing_scope = list(requested_scope)
        adapter_run = False

        if blockers:
            status = "blocked_macro_envelope_contract"
        elif source is None:
            status = "awaiting_explicit_local_macro_source"
        elif not source.is_file():
            status = "blocked_explicit_macro_source_missing"
        else:
            adapter_run = True
            core = SavedMacroEvidenceProducer().build(
                source_path=source,
                as_of=as_of,
                registry_path=registry_path,
                save=False,
            )
            core_selected = list(core.get("selected_observations") or [])
            relevant = [item for item in core_selected if item.get("series_id") in requested_scope]
            present = {str(item.get("series_id")) for item in relevant}
            missing_scope = sorted(set(requested_scope) - present)
            if core.get("status") not in {"macro_evidence_ready", "macro_evidence_ready_with_exclusions"}:
                status = "blocked_macro_core_not_ready"
            elif not relevant:
                status = "blocked_no_domain_relevant_macro_observations"
            elif missing_scope:
                status = "domain_macro_binding_candidate_ready_with_scope_gaps"
            else:
                status = "domain_macro_binding_candidate_ready"

        relevant_context_keys = {str(item.get("context_key")) for item in relevant}
        core_fragment = (core or {}).get("market_context_fragment") or {}
        core_macro = core_fragment.get("macro") or {}
        filtered_macro = {
            key: value for key, value in core_macro.items() if key in relevant_context_keys
        }
        candidate_ready = status in {
            "domain_macro_binding_candidate_ready",
            "domain_macro_binding_candidate_ready_with_scope_gaps",
        }
        payload = {
            "run_id": _run_id("domain_scoped_macro_envelope"),
            "created_at": utc_now_iso(),
            "mode": "domain_scoped_macro_evidence_envelope",
            "contract": DOMAIN_MACRO_ENVELOPE_CONTRACT,
            "source_producer_contract": SAVED_MACRO_PRODUCER_CONTRACT,
            "domain_envelope_contract": DOMAIN_MACRO_ENVELOPE_CONTRACT,
            "domain_id": domain_id,
            "status": status,
            "inputs": {
                "domain_id": domain_id,
                "source_path": str(source) if source else None,
                "source_sha256": source_sha,
                "registry_path": str(registry_path),
                "registry_sha256": _sha256_file(Path(registry_path)) if Path(registry_path).is_file() else None,
                "dispatch_path": str(dispatch_path),
                "dispatch_sha256": _sha256_file(dispatch_file) if dispatch_file.is_file() else None,
                "execution_gate_path": str(execution_gate_path) if execution_gate_path else None,
                "execution_gate_sha256": (
                    _sha256_file(execution_gate_file)
                    if execution_gate_file and execution_gate_file.is_file()
                    else None
                ),
                "as_of": as_of,
            },
            "domain_binding": {
                "profile_contract": lifecycle.get("contract"),
                "fixed_contract_sha256": lifecycle.get("fixed_contract_sha256"),
                "domain_overlay_sha256": lifecycle.get("domain_overlay_sha256"),
                "macro_context_role": overlay.get("macro_context_role"),
                "requested_series_scope": requested_scope,
                "present_series_scope": sorted({str(item.get("series_id")) for item in relevant}),
                "missing_series_scope": missing_scope,
            },
            "core_preview": {
                "adapter": "SavedMacroEvidenceProducer",
                "adapter_run_performed": adapter_run,
                "core_contract": (core or {}).get("producer_contract"),
                "core_status": (core or {}).get("status"),
                "core_payload_sha256": _stable_payload_sha256(core) if core else None,
                "source_row_count": ((core or {}).get("summary") or {}).get("source_row_count"),
                "selected_series_count": ((core or {}).get("summary") or {}).get("selected_series_count"),
                "domain_relevant_series_count": len(relevant),
                "exclusion_count": len((core or {}).get("exclusions") or []),
                "schema_mapping": (core or {}).get("schema_mapping") or {},
                "reason_counts": ((core or {}).get("summary") or {}).get("reason_counts") or {},
                "exclusion_reasons": sorted(
                    {
                        str(reason)
                        for item in (core or {}).get("exclusions") or []
                        for reason in item.get("reasons") or []
                    }
                ),
            },
            "selected_observations": relevant,
            "market_context_fragment": {
                "as_of": as_of,
                "domain_id": domain_id,
                "macro": filtered_macro,
                "metadata": {
                    "domain_macro_envelope_contract": DOMAIN_MACRO_ENVELOPE_CONTRACT,
                    "source_sha256": source_sha,
                    "profile_domain_overlay_sha256": lifecycle.get("domain_overlay_sha256"),
                    "supporting_context_only": True,
                },
            },
            "summary": {
                "domain_id": domain_id,
                "status": status,
                "structural_blockers": blockers,
                "adapter_run_performed": adapter_run,
                "source_lineage_verified": candidate_ready,
                "candidate_ready_for_binding_review": candidate_ready,
                "binding_accepted": False,
                "can_update_profile_binding": False,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "binding_gate": {
                "status": "candidate_ready_pending_explicit_binding_decision" if candidate_ready else "not_open",
                "allowed_decisions": ["accept_binding", "replace_candidate", "defer"],
                "candidate_sha256_binding_required": True,
                "decision_recorded": False,
            },
            "safety": {
                "review_only": True,
                "single_adapter_run_limit": 1,
                "automatic_retry_allowed": False,
                "network_access_performed": False,
                "binding_write_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        journal_summary = _journal(
            payload=payload,
            dispatch_path=dispatch_file,
            journal_path=Path(journal_path),
            apply=apply_journal,
        )
        payload["journal"] = journal_summary
        if save:
            saved = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_macro_envelope_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved
        return json_ready(payload)


def load_verified_domain_macro_context_fragment(
    artifact_path: str | Path,
    *,
    expected_domain_id: str | None = None,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    """Rebuild the offline macro core and verify a saved domain envelope."""

    path = Path(artifact_path).resolve()
    payload = _load_json(path)
    if payload.get("contract") != DOMAIN_MACRO_ENVELOPE_CONTRACT:
        raise ValueError("unsupported domain macro envelope contract")
    if payload.get("mode") != "domain_scoped_macro_evidence_envelope":
        raise ValueError("unsupported domain macro envelope mode")
    if payload.get("status") not in {
        "domain_macro_binding_candidate_ready",
        "domain_macro_binding_candidate_ready_with_scope_gaps",
    }:
        raise ValueError("domain macro envelope is not ready")
    inputs = payload.get("inputs") or {}
    domain_id = str(payload.get("domain_id") or "")
    as_of_text = str(inputs.get("as_of") or "")
    as_of = parse_timezone_aware(as_of_text)
    if as_of is None:
        raise ValueError("domain macro envelope as_of invalid")
    if not domain_id or inputs.get("domain_id") != domain_id:
        raise ValueError("domain macro envelope identity invalid")
    if expected_domain_id is not None and domain_id != expected_domain_id:
        raise ValueError("domain macro envelope expected domain mismatch")
    expected_dt = parse_timezone_aware(expected_as_of) if expected_as_of else None
    if expected_as_of is not None and (expected_dt is None or as_of != expected_dt):
        raise ValueError("domain macro envelope expected as_of mismatch")
    summary = payload.get("summary") or {}
    safety = payload.get("safety") or {}
    if summary.get("structural_blockers"):
        raise ValueError("domain macro envelope has structural blockers")
    for key in (
        "adapter_run_performed",
        "source_lineage_verified",
        "candidate_ready_for_binding_review",
    ):
        if summary.get(key) is not True:
            raise ValueError(f"domain macro envelope required flag invalid: {key}")
    for key in (
        "binding_accepted",
        "can_invoke_domain_analysis",
        "can_approve_hypothesis",
        "can_write_learning_memory",
        "can_trade",
    ):
        if summary.get(key) is not False:
            raise ValueError(f"domain macro envelope forbidden flag invalid: {key}")
    if (
        safety.get("review_only") is not True
        or safety.get("network_access_performed") is not False
        or safety.get("binding_write_performed") is not False
        or safety.get("learning_write_performed") is not False
        or safety.get("live_execution_performed") is not False
    ):
        raise ValueError("domain macro envelope safety boundary invalid")

    source_path = Path(str(inputs.get("source_path") or "")).resolve()
    registry_path = Path(str(inputs.get("registry_path") or "")).resolve()
    dispatch_path = Path(str(inputs.get("dispatch_path") or "")).resolve()
    for bound, expected_sha, label in (
        (source_path, inputs.get("source_sha256"), "source"),
        (registry_path, inputs.get("registry_sha256"), "registry"),
        (dispatch_path, inputs.get("dispatch_sha256"), "dispatch"),
    ):
        if not bound.is_file() or _sha256_file(bound) != expected_sha:
            raise ValueError(f"domain macro envelope {label} hash mismatch")
    gate_path_value = inputs.get("execution_gate_path")
    gate_path = Path(str(gate_path_value)).resolve() if gate_path_value else None
    gate = _load_json(gate_path) if gate_path and gate_path.is_file() else None
    if gate_path_value and (
        gate_path is None
        or not gate_path.is_file()
        or _sha256_file(gate_path) != inputs.get("execution_gate_sha256")
    ):
        raise ValueError("domain macro envelope execution gate hash mismatch")

    lifecycle = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
    overlay = lifecycle.get("domain_overlay") or {}
    requested_scope = sorted(
        {
            str(item).strip()
            for item in overlay.get("macro_series_scope") or []
            if str(item).strip()
        }
    )
    blockers = _ceremony_blockers(
        dispatch=_load_json(dispatch_path),
        domain_id=domain_id,
        requested_scope=requested_scope,
        lifecycle=lifecycle,
        execution_gate=gate,
        as_of=as_of_text,
    )
    if blockers:
        raise ValueError(
            "domain macro recursive verification failed: " + ",".join(blockers)
        )
    core = SavedMacroEvidenceProducer().build(
        source_path=str(inputs.get("source_path") or ""),
        as_of=as_of_text,
        registry_path=str(inputs.get("registry_path") or ""),
        save=False,
    )
    relevant = [
        item
        for item in core.get("selected_observations") or []
        if item.get("series_id") in requested_scope
    ]
    context_keys = {str(item.get("context_key")) for item in relevant}
    core_macro = ((core.get("market_context_fragment") or {}).get("macro") or {})
    filtered_macro = {
        key: value for key, value in core_macro.items() if key in context_keys
    }
    fragment = payload.get("market_context_fragment") or {}
    if (
        payload.get("core_preview", {}).get("core_payload_sha256")
        != _stable_payload_sha256(core)
        or _sha256_json(payload.get("selected_observations") or [])
        != _sha256_json(relevant)
        or fragment.get("domain_id") != domain_id
        or parse_timezone_aware(str(fragment.get("as_of") or "")) != as_of
        or _sha256_json(fragment.get("macro") or {})
        != _sha256_json(filtered_macro)
    ):
        raise ValueError("domain macro envelope fragment mismatch")
    return {
        "as_of": as_of.isoformat(),
        "domain_id": domain_id,
        "macro": filtered_macro,
        "metadata": {
            **dict(fragment.get("metadata") or {}),
            "domain_macro_envelope_verified": True,
            "domain_macro_envelope_path": str(path),
            "domain_macro_envelope_sha256": _sha256_file(path),
        },
    }


def _ceremony_blockers(
    *,
    dispatch: dict[str, Any],
    domain_id: str,
    requested_scope: list[str],
    lifecycle: dict[str, Any],
    execution_gate: dict[str, Any] | None,
    as_of: str,
) -> list[str]:
    blockers: list[str] = []
    if dispatch.get("mode") != "domain_binding_task_dispatch":
        blockers.append("unsupported_dispatch_artifact")
    if dispatch.get("summary", {}).get("domain_id") != domain_id:
        blockers.append("dispatch_domain_mismatch")
    macro = next(
        (item for item in dispatch.get("task_dispatches", []) if item.get("context_family") == "macro"),
        None,
    )
    dispatch_authorized = bool(
        macro
        and macro.get("recommended_action")
        in {
            "domain_scoped_macro_evidence_envelope",
            "prepare_one_allowlisted_offline_adapter_run",
        }
    )
    gate_authorized = _execution_gate_allows_offline_envelope(
        execution_gate,
        domain_id=domain_id,
        requested_scope=requested_scope,
        as_of=as_of,
    )
    if not dispatch_authorized and not gate_authorized:
        blockers.append("macro_envelope_not_dispatched")
    if not requested_scope:
        blockers.append("domain_macro_series_scope_missing")
    if lifecycle.get("readiness", {}).get("schema_valid") is not True:
        blockers.append("domain_lifecycle_profile_invalid")
    return sorted(set(blockers))


def _execution_gate_allows_offline_envelope(
    gate: dict[str, Any] | None,
    *,
    domain_id: str,
    requested_scope: list[str],
    as_of: str,
) -> bool:
    if not gate:
        return False
    summary = gate.get("summary") or {}
    ticket = gate.get("execution_ticket") or {}
    ticket_as_of = parse_timezone_aware(ticket.get("request_as_of"))
    envelope_as_of = parse_timezone_aware(as_of)
    return bool(
        gate.get("contract") == "dean_domain_macro_collection_execution_gate_v1"
        and gate.get("mode") == "domain_macro_collection_execution_gate"
        and gate.get("domain_id") == domain_id
        and summary.get("status") == "macro_collection_execution_ready_single_run"
        and summary.get("single_run_authorized") is True
        and ticket.get("domain_id") == domain_id
        and ticket_as_of is not None
        and envelope_as_of is not None
        and ticket_as_of <= envelope_as_of
        and sorted(ticket.get("series_scope") or []) == sorted(requested_scope)
        and ticket.get("maximum_collection_runs") == 1
        and ticket.get("automatic_retry_allowed") is False
    )


def _journal(
    *, payload: dict[str, Any], dispatch_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    events = _journal_events(payload, dispatch_path)
    if not apply:
        return {
            "apply_requested": False,
            "events_proposed": len(events),
            "appended_count": 0,
            "existing_count": 0,
            "chain_valid": SystemJournal(journal_path).status()["chain_valid"],
        }
    result = SystemJournal(journal_path).append_many(events)
    status = SystemJournal(journal_path).status()
    return {
        "apply_requested": True,
        **result,
        "record_count": status["record_count"],
        "chain_valid": status["chain_valid"],
        "tip_sha256": status["tip_sha256"],
    }


def _journal_events(payload: dict[str, Any], dispatch_path: Path) -> list[dict[str, Any]]:
    source = artifact_binding(dispatch_path)
    task_id = "bind_{}_macro".format(payload["domain_id"])
    as_of = payload["inputs"]["as_of"]
    proposal = {
        "event_type": "action_proposed",
        "effective_at": as_of,
        "actor": "domain_binding_task_dispatcher",
        "domain_id": payload["domain_id"],
        "entity_type": "domain_context_binding_task",
        "entity_id": task_id,
        "source_artifact": source,
        "context": {"context_family": "macro", "review_only": True},
        "payload": {
            "recommended_action": "domain_scoped_macro_evidence_envelope",
            "execution_authorized": False,
            "binding_acceptance_authorized": False,
        },
    }
    result_type = "action_executed" if payload["summary"]["adapter_run_performed"] else "action_reviewed"
    result = {
        "event_type": result_type,
        "effective_at": as_of,
        "actor": "domain_scoped_macro_envelope",
        "domain_id": payload["domain_id"],
        "entity_type": "domain_context_binding_preview",
        "entity_id": task_id + ":" + (payload["inputs"].get("source_sha256") or "no_source"),
        "source_artifact": source,
        "context": {"context_family": "macro", "review_only": True},
        "payload": {
            "status": payload["status"],
            "adapter_run_performed": payload["summary"]["adapter_run_performed"],
            "candidate_ready_for_binding_review": payload["summary"]["candidate_ready_for_binding_review"],
            "binding_accepted": False,
            "source_sha256": payload["inputs"].get("source_sha256"),
            "core_payload_sha256": payload["core_preview"].get("core_payload_sha256"),
        },
    }
    return [proposal, result]


def render_macro_envelope_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    binding = payload["domain_binding"]
    journal = payload["journal"]
    lines = [
        "# DEAN-OS Domain-Scoped Macro Envelope",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Adapter run performed: {summary['adapter_run_performed']}",
        f"- Candidate ready for binding review: {summary['candidate_ready_for_binding_review']}",
        f"- Binding accepted: {summary['binding_accepted']}",
        f"- Requested series: {len(binding['requested_series_scope'])}",
        f"- Present series: {len(binding['present_series_scope'])}",
        f"- Missing series: {len(binding['missing_series_scope'])}",
        f"- Journal appended: {journal.get('appended_count', 0)}",
        f"- Journal chain valid: {journal.get('chain_valid')}",
        f"- Can invoke analyst: {summary['can_invoke_domain_analysis']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Series scope",
        "",
        "- Requested: " + (", ".join(binding["requested_series_scope"]) or "none"),
        "- Present: " + (", ".join(binding["present_series_scope"]) or "none"),
        "- Missing: " + (", ".join(binding["missing_series_scope"]) or "none"),
        "- Core blockers: " + (", ".join(payload["core_preview"].get("exclusion_reasons") or []) or "none"),
        "",
        "## Boundary",
        "",
        "- This artifact is a binding candidate only; it cannot accept the binding.",
        "- Macro remains supporting context and cannot approve a hypothesis.",
        "- One explicit local source, one offline adapter pass, no retry and no network.",
    ]
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


def _sha256_json(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    raw = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_payload_sha256(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    stable = _without_runtime_metadata(payload)
    return _sha256_json(stable)


def _without_runtime_metadata(value: Any) -> Any:
    runtime_keys = {
        "run_id",
        "created_at",
        "saved_paths",
        "artifact_safety",
        "saved_macro_producer_run_id",
    }
    if isinstance(value, dict):
        return {
            key: _without_runtime_metadata(item)
            for key, item in value.items()
            if key not in runtime_keys
        }
    if isinstance(value, list):
        return [_without_runtime_metadata(item) for item in value]
    return value


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "DOMAIN_MACRO_ENVELOPE_CONTRACT",
    "DomainScopedMacroEnvelopeCeremony",
    "load_verified_domain_macro_context_fragment",
]
