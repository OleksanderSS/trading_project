"""Universal, declarative context-acquisition state machine for DEAN-OS.

Family-specific collectors remain outside this module.  The machine consumes
their JSON artifacts through a registry of declarative stage contracts and
enforces ordering, SHA lineage, single-stage execution, journaling and the
system authority boundary.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready


REGISTRY_CONTRACT = "dean_context_acquisition_family_registry_v1"
RECEIPT_CONTRACT = "dean_context_acquisition_transition_receipt_v1"
LEDGER_CONTRACT = "dean_context_acquisition_transition_ledger_v1"
DEFAULT_REGISTRY_PATH = "dean_os/config/context_acquisition_family_registry.json"
DEFAULT_LEDGER_PATH = "data/dean_os/context_acquisition_transition_ledger.jsonl"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"
DEFAULT_OUTPUT_DIR = "reports/dean_os/context_acquisition_state_machine_current"

STATE_ORDER = [
    "idle",
    "gap_identified",
    "request_prepared",
    "execution_authorized",
    "execution_completed",
    "retrieval_verified",
    "awaiting_binding_decision",
]
CONTEXT_FAMILIES = {
    "news",
    "official_policy",
    "macro",
    "fundamentals",
    "sector_market",
    "pipeline_context",
}
FORBIDDEN_AUTHORITIES = {
    "automatic_multi_stage_loop_allowed",
    "automatic_retry_allowed",
    "binding_acceptance_allowed",
    "analyst_invocation_allowed",
    "hypothesis_approval_allowed",
    "learning_write_allowed",
    "trading_allowed",
}


class ContextAcquisitionTransitionLedger:
    """Append-only global ledger containing per-acquisition receipt chains."""

    def __init__(self, path: str | Path = DEFAULT_LEDGER_PATH):
        self.path = Path(path)

    def read_verified(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        previous_record_sha: str | None = None
        per_acquisition_tip: dict[str, dict[str, Any]] = {}
        for line_number, raw in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("contract") != LEDGER_CONTRACT:
                raise ValueError(f"invalid context acquisition ledger contract at line {line_number}")
            if record.get("previous_record_sha256") != previous_record_sha:
                raise ValueError(f"context acquisition ledger chain break at line {line_number}")
            if record.get("record_sha256") != _record_sha(record):
                raise ValueError(f"context acquisition ledger hash mismatch at line {line_number}")
            receipt = record.get("transition_receipt") or {}
            if receipt.get("contract") != RECEIPT_CONTRACT:
                raise ValueError(f"invalid transition receipt contract at line {line_number}")
            if receipt.get("transition_receipt_sha256") != _receipt_sha(receipt):
                raise ValueError(f"transition receipt hash mismatch at line {line_number}")
            if receipt.get("decision") != "approved":
                raise ValueError(f"non-approved transition persisted at line {line_number}")
            acquisition_id = str(receipt.get("acquisition_id") or "")
            prior = per_acquisition_tip.get(acquisition_id)
            expected_state = (prior or {}).get("to_state") or "idle"
            expected_receipt_sha = (prior or {}).get("transition_receipt_sha256")
            if receipt.get("from_state") != expected_state:
                raise ValueError(f"acquisition state chain break at line {line_number}")
            if receipt.get("previous_transition_receipt_sha256") != expected_receipt_sha:
                raise ValueError(f"acquisition receipt chain break at line {line_number}")
            records.append(record)
            previous_record_sha = str(record["record_sha256"])
            per_acquisition_tip[acquisition_id] = receipt
        return records

    def receipts_for(self, acquisition_id: str) -> list[dict[str, Any]]:
        return [
            dict(record["transition_receipt"])
            for record in self.read_verified()
            if (record.get("transition_receipt") or {}).get("acquisition_id") == acquisition_id
        ]

    def append(self, receipt: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        records = self.read_verified()
        receipt_dict = json_ready(dict(receipt))
        if receipt_dict.get("decision") != "approved":
            raise ValueError("only approved context transitions may be persisted")
        if receipt_dict.get("transition_receipt_sha256") != _receipt_sha(receipt_dict):
            raise ValueError("invalid transition receipt hash")
        for record in records:
            existing = record.get("transition_receipt") or {}
            if (
                existing.get("acquisition_id") == receipt_dict.get("acquisition_id")
                and existing.get("stage_id") == receipt_dict.get("stage_id")
                and (existing.get("artifact") or {}).get("sha256")
                == (receipt_dict.get("artifact") or {}).get("sha256")
            ):
                return record, False
        record = {
            "contract": LEDGER_CONTRACT,
            "transition_receipt": receipt_dict,
            "previous_record_sha256": records[-1]["record_sha256"] if records else None,
        }
        record["record_sha256"] = _record_sha(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record, True

    def status(self) -> dict[str, Any]:
        records = self.read_verified()
        return {
            "path": str(self.path),
            "record_count": len(records),
            "chain_valid": True,
            "tip_sha256": records[-1]["record_sha256"] if records else None,
        }


class ContextAcquisitionStateMachine:
    """Advance one context family by exactly one verified stage per call."""

    def __init__(
        self,
        *,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.ledger = ContextAcquisitionTransitionLedger(ledger_path)
        self.journal_path = Path(journal_path)
        self.output_dir = Path(output_dir)

    def advance(
        self,
        *,
        acquisition_id: str,
        domain_id: str,
        context_family: str,
        stage_id: str,
        artifact_path: str | Path,
        evaluated_at: str | None = None,
        apply_transition: bool = False,
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        registry = _load_json(self.registry_path)
        registry_blockers = _registry_blockers(registry)
        artifact_file = Path(artifact_path).resolve()
        artifact, artifact_sha, artifact_blockers = _load_bound_artifact(artifact_file)
        receipts = self.ledger.receipts_for(acquisition_id)
        previous = receipts[-1] if receipts else None
        current_state = str((previous or {}).get("to_state") or "idle")
        timestamp = evaluated_at or str(artifact.get("created_at") or utc_now_iso())
        timestamp_blocker = _timestamp_blocker(timestamp)

        family = (registry.get("families") or {}).get(context_family) or {}
        stages = list(family.get("stages") or [])
        stage = next((item for item in stages if item.get("stage_id") == stage_id), None)
        blockers = [*registry_blockers, *artifact_blockers]
        if timestamp_blocker:
            blockers.append(timestamp_blocker)
        if not acquisition_id.strip():
            blockers.append("acquisition_id_missing")
        if not domain_id.strip():
            blockers.append("domain_id_missing")
        if context_family not in CONTEXT_FAMILIES:
            blockers.append("unknown_context_family")
        elif family.get("status") != "implemented":
            blockers.append(f"context_family_adapter_not_implemented:{context_family}")
        if stage is None:
            blockers.append(f"unknown_stage_for_context_family:{stage_id}")

        # Exact repeat of the persisted last transition is idempotent.
        if (
            previous
            and previous.get("stage_id") == stage_id
            and (previous.get("artifact") or {}).get("sha256") == artifact_sha
            and not blockers
        ):
            payload = self._payload(
                acquisition_id=acquisition_id,
                domain_id=domain_id,
                context_family=context_family,
                stage_id=stage_id,
                state_before=str(previous.get("from_state")),
                proposed_state=str(previous.get("to_state")),
                persisted_state=str(previous.get("to_state")),
                status="transition_already_recorded",
                blockers=[],
                receipt=previous,
                ledger_appended=False,
                journal_result=_journal_noop(self.journal_path, apply_journal),
                apply_transition=apply_transition,
            )
            return self._save(payload, save)

        previous_artifact: dict[str, Any] = {}
        if previous:
            previous_path = Path(str((previous.get("artifact") or {}).get("path") or ""))
            previous_artifact, current_previous_sha, prior_load_blockers = _load_bound_artifact(previous_path)
            blockers.extend(f"previous_{item}" for item in prior_load_blockers)
            if current_previous_sha != (previous.get("artifact") or {}).get("sha256"):
                blockers.append("previous_artifact_sha256_changed")

        proposed_state = str((stage or {}).get("to_state") or current_state)
        if stage:
            if stage.get("from_state") != current_state:
                blockers.append(f"non_sequential_transition:{current_state}->{stage_id}")
            blockers.extend(
                _artifact_contract_blockers(
                    artifact=artifact,
                    artifact_file=artifact_file,
                    artifact_sha=artifact_sha,
                    previous_receipt=previous,
                    previous_artifact=previous_artifact,
                    domain_id=domain_id,
                    stage=stage,
                )
            )

        blockers = sorted(set(blockers))
        decision = "approved" if not blockers else "blocked"
        receipt = {
            "contract": RECEIPT_CONTRACT,
            "acquisition_id": acquisition_id,
            "domain_id": domain_id,
            "context_family": context_family,
            "stage_id": stage_id,
            "from_state": current_state,
            "to_state": proposed_state,
            "decision": decision,
            "evaluated_at": timestamp,
            "artifact": {
                "path": str(artifact_file),
                "sha256": artifact_sha,
                "contract": artifact.get("contract"),
                "mode": artifact.get("mode"),
                "run_id": artifact.get("run_id"),
            },
            "previous_artifact_sha256": (
                (previous.get("artifact") or {}).get("sha256") if previous else None
            ),
            "previous_transition_receipt_sha256": (
                previous.get("transition_receipt_sha256") if previous else None
            ),
            "checks": {
                "single_stage_only": True,
                "registry_driven": True,
                "artifact_sha_bound": artifact_sha is not None,
                "previous_artifact_unchanged": "previous_artifact_sha256_changed" not in blockers,
                "authority_boundary_fail_closed": not registry_blockers,
            },
            "blockers": blockers,
            "authority": {
                "binding_accepted": False,
                "analyst_invoked": False,
                "hypothesis_approved": False,
                "learning_written": False,
                "trade_executed": False,
            },
        }
        receipt["transition_receipt_sha256"] = _receipt_sha(receipt)

        ledger_appended = False
        if decision == "approved" and apply_transition:
            _, ledger_appended = self.ledger.append(receipt)
        persisted_state = proposed_state if ledger_appended else current_state
        status = (
            "transition_blocked"
            if blockers
            else "transition_recorded"
            if ledger_appended
            else "transition_ready_not_recorded"
        )
        journal_result = _journal_transition(
            receipt=receipt,
            artifact_path=artifact_file,
            journal_path=self.journal_path,
            apply=apply_journal,
        )
        payload = self._payload(
            acquisition_id=acquisition_id,
            domain_id=domain_id,
            context_family=context_family,
            stage_id=stage_id,
            state_before=current_state,
            proposed_state=proposed_state,
            persisted_state=persisted_state,
            status=status,
            blockers=blockers,
            receipt=receipt,
            ledger_appended=ledger_appended,
            journal_result=journal_result,
            apply_transition=apply_transition,
        )
        return self._save(payload, save)

    def reconcile(self, acquisition_id: str) -> dict[str, Any]:
        registry = _load_json(self.registry_path)
        blockers = _registry_blockers(registry)
        receipts = self.ledger.receipts_for(acquisition_id)
        for receipt in receipts:
            artifact = receipt.get("artifact") or {}
            path = Path(str(artifact.get("path") or ""))
            _, current_sha, load_blockers = _load_bound_artifact(path)
            blockers.extend(
                f"artifact_{receipt.get('stage_id')}:{item}" for item in load_blockers
            )
            if current_sha != artifact.get("sha256"):
                blockers.append(f"artifact_{receipt.get('stage_id')}:sha256_changed")
        current_state = str((receipts[-1] if receipts else {}).get("to_state") or "idle")
        context_family = str((receipts[-1] if receipts else {}).get("context_family") or "")
        family = (registry.get("families") or {}).get(context_family) or {}
        next_stage = next(
            (item.get("stage_id") for item in family.get("stages") or [] if item.get("from_state") == current_state),
            None,
        )
        return json_ready(
            {
                "contract": "dean_context_acquisition_reconciliation_v1",
                "acquisition_id": acquisition_id,
                "status": "reconciliation_valid" if not blockers else "reconciliation_blocked",
                "current_state": current_state,
                "next_stage_id": next_stage,
                "transition_count": len(receipts),
                "blockers": sorted(set(blockers)),
                "ledger": self.ledger.status(),
                "authority": {
                    "binding_accepted": False,
                    "analyst_invoked": False,
                    "learning_written": False,
                    "trade_executed": False,
                },
            }
        )

    def _payload(
        self,
        *,
        acquisition_id: str,
        domain_id: str,
        context_family: str,
        stage_id: str,
        state_before: str,
        proposed_state: str,
        persisted_state: str,
        status: str,
        blockers: list[str],
        receipt: Mapping[str, Any],
        ledger_appended: bool,
        journal_result: dict[str, Any],
        apply_transition: bool,
    ) -> dict[str, Any]:
        return {
            "run_id": _run_id("context_acquisition_state_machine"),
            "created_at": utc_now_iso(),
            "mode": "context_acquisition_state_machine",
            "contract": "dean_context_acquisition_state_machine_run_v1",
            "acquisition_id": acquisition_id,
            "domain_id": domain_id,
            "context_family": context_family,
            "summary": {
                "status": status,
                "stage_id": stage_id,
                "state_before": state_before,
                "proposed_state": proposed_state,
                "persisted_state": persisted_state,
                "structural_blockers": blockers,
                "apply_transition_requested": apply_transition,
                "ledger_appended": ledger_appended,
                "automatic_next_stage_run": False,
                "binding_accepted": False,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "transition_receipt": json_ready(dict(receipt)),
            "ledger": self.ledger.status(),
            "journal": journal_result,
            "safety": {
                "one_transition_per_call": True,
                "automatic_retry_allowed": False,
                "family_collector_called": False,
                "network_access_performed": False,
                "binding_write_performed": False,
                "learning_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }

    def _save(self, payload: dict[str, Any], save: bool) -> dict[str, Any]:
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def _registry_blockers(registry: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if registry.get("contract") != REGISTRY_CONTRACT:
        blockers.append("unsupported_context_acquisition_registry_contract")
    if registry.get("state_order") != STATE_ORDER:
        blockers.append("context_acquisition_state_order_invalid")
    authority = registry.get("authority_boundary") or {}
    for key in sorted(FORBIDDEN_AUTHORITIES):
        if authority.get(key) is not False:
            blockers.append(f"authority_boundary_not_fail_closed:{key}")
    families = registry.get("families") or {}
    if set(families) != CONTEXT_FAMILIES:
        blockers.append("context_family_registry_set_mismatch")
    for family_id, family in families.items():
        stages = list((family or {}).get("stages") or [])
        if (family or {}).get("status") == "implemented":
            if not stages:
                blockers.append(f"implemented_family_has_no_stages:{family_id}")
                continue
            seen: set[str] = set()
            incoming_states = {
                str(stage.get("to_state") or "") for stage in stages
            }
            has_idle_entry = False
            for stage in stages:
                stage_id = str(stage.get("stage_id") or "")
                if not stage_id or stage_id in seen:
                    blockers.append(f"family_stage_id_invalid:{family_id}")
                seen.add(stage_id)
                from_state = str(stage.get("from_state") or "")
                to_state = str(stage.get("to_state") or "")
                if from_state == "idle":
                    has_idle_entry = True
                if from_state not in STATE_ORDER:
                    blockers.append(
                        f"family_stage_source_invalid:{family_id}:{stage_id}"
                    )
                if to_state not in STATE_ORDER:
                    blockers.append(f"family_stage_target_invalid:{family_id}:{stage_id}")
                elif (
                    from_state in STATE_ORDER
                    and STATE_ORDER.index(to_state) <= STATE_ORDER.index(from_state)
                ):
                    blockers.append(
                        f"family_stage_sequence_invalid:{family_id}:{stage_id}"
                    )
                if from_state != "idle" and from_state not in incoming_states:
                    blockers.append(
                        f"family_stage_unreachable:{family_id}:{stage_id}"
                    )
                if not stage.get("artifact_contract") or not stage.get("artifact_mode"):
                    blockers.append(f"family_stage_artifact_contract_missing:{family_id}:{stage_id}")
            if not has_idle_entry:
                blockers.append(f"implemented_family_has_no_idle_entry:{family_id}")
    return sorted(set(blockers))


def _artifact_contract_blockers(
    *,
    artifact: Mapping[str, Any],
    artifact_file: Path,
    artifact_sha: str | None,
    previous_receipt: Mapping[str, Any] | None,
    previous_artifact: Mapping[str, Any],
    domain_id: str,
    stage: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if artifact.get("contract") != stage.get("artifact_contract"):
        blockers.append("artifact_contract_mismatch")
    if artifact.get("mode") != stage.get("artifact_mode"):
        blockers.append("artifact_mode_mismatch")
    if artifact.get("domain_id") != domain_id:
        blockers.append("artifact_domain_mismatch")
    status = _path_value(artifact, str(stage.get("status_path") or "summary.status"))
    if status not in list(stage.get("allowed_statuses") or []):
        blockers.append(f"artifact_status_not_ready:{status}")
    structural = _path_value(artifact, "summary.structural_blockers")
    if structural:
        blockers.append("artifact_has_structural_blockers")
    for path in stage.get("required_true") or []:
        if _path_value(artifact, path) is not True:
            blockers.append(f"required_true_failed:{path}")
    for path in stage.get("required_false") or []:
        if _path_value(artifact, path) is not False:
            blockers.append(f"required_false_failed:{path}")
    for path, expected in (stage.get("required_equals") or {}).items():
        if _path_value(artifact, path) != expected:
            blockers.append(f"required_equals_failed:{path}")
    for path, allowed in (stage.get("required_in") or {}).items():
        if _path_value(artifact, path) not in allowed:
            blockers.append(f"required_in_failed:{path}")

    prior_sha_path = stage.get("previous_artifact_sha_path")
    if prior_sha_path:
        expected_sha = (previous_receipt or {}).get("artifact", {}).get("sha256")
        if not expected_sha or _path_value(artifact, prior_sha_path) != expected_sha:
            blockers.append(f"previous_artifact_sha_binding_failed:{prior_sha_path}")

    reference = stage.get("referenced_artifact") or {}
    if reference:
        current_path_value = _path_value(artifact, str(reference.get("current_path") or ""))
        current_sha = _path_value(artifact, str(reference.get("current_sha256") or ""))
        previous_path_value = _path_value(
            previous_artifact, str(reference.get("previous_declared_path") or "")
        )
        if not current_path_value or not previous_path_value:
            blockers.append("referenced_artifact_path_missing")
        else:
            current_path = _resolve_reference_path(current_path_value, artifact_file)
            previous_path = _resolve_reference_path(
                previous_path_value,
                Path(str((previous_receipt or {}).get("artifact", {}).get("path") or artifact_file)),
            )
            if current_path != previous_path:
                blockers.append("referenced_artifact_path_binding_failed")
            if not current_path.is_file():
                blockers.append("referenced_artifact_unreadable")
            elif _sha256_file(current_path) != current_sha:
                blockers.append("referenced_artifact_sha256_invalid")
    if not artifact_sha:
        blockers.append("artifact_sha256_missing")
    return blockers


def _journal_transition(
    *,
    receipt: Mapping[str, Any],
    artifact_path: Path,
    journal_path: Path,
    apply: bool,
) -> dict[str, Any]:
    journal = SystemJournal(journal_path)
    event = {
        "event_type": "action_reviewed",
        "effective_at": receipt["evaluated_at"],
        "actor": "context_acquisition_state_machine",
        "domain_id": receipt["domain_id"],
        "entity_type": "context_acquisition_transition",
        "entity_id": f"{receipt['acquisition_id']}:{receipt['stage_id']}",
        "source_artifact": artifact_binding(artifact_path) if artifact_path.is_file() else {},
        "context": {
            "context_family": receipt["context_family"],
            "from_state": receipt["from_state"],
            "to_state": receipt["to_state"],
            "single_stage_only": True,
        },
        "payload": {
            "decision": receipt["decision"],
            "transition_receipt_sha256": receipt["transition_receipt_sha256"],
            "blockers": receipt["blockers"],
            "binding_accepted": False,
            "analyst_invoked": False,
            "learning_written": False,
            "trade_executed": False,
        },
    }
    if not apply:
        return {
            "apply_requested": False,
            "events_proposed": 1,
            "appended_count": 0,
            "existing_count": 0,
            "chain_valid": journal.status()["chain_valid"],
        }
    result = journal.append_many([event])
    return {"apply_requested": True, **result, **journal.status()}


def _journal_noop(journal_path: Path, apply: bool) -> dict[str, Any]:
    return {
        "apply_requested": apply,
        "events_proposed": 0,
        "appended_count": 0,
        "existing_count": 1,
        "chain_valid": SystemJournal(journal_path).status()["chain_valid"],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DEAN-OS Context Acquisition State Machine",
        "",
        f"- Acquisition: `{payload['acquisition_id']}`",
        f"- Domain: `{payload['domain_id']}`",
        f"- Context family: `{payload['context_family']}`",
        f"- Status: `{summary['status']}`",
        f"- Transition: `{summary['state_before']}` -> `{summary['proposed_state']}`",
        f"- Persisted state: `{summary['persisted_state']}`",
        f"- Ledger appended: {summary['ledger_appended']}",
        f"- Automatic next-stage run: {summary['automatic_next_stage_run']}",
        f"- Binding accepted: {summary['binding_accepted']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["structural_blockers"] or ["none"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This call evaluates one transition only and never invokes a family collector.",
            "- It cannot accept a binding, invoke the analyst, approve a hypothesis, write learning memory, or trade.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _load_bound_artifact(path: Path) -> tuple[dict[str, Any], str | None, list[str]]:
    if not path.is_file():
        return {}, None, ["artifact_unreadable"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, None, ["artifact_json_invalid"]
    return payload, _sha256_file(path), []


def _path_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split(".") if dotted_path else []:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_reference_path(value: Any, declaring_artifact: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    workspace_candidate = Path.cwd() / path
    if workspace_candidate.exists():
        return workspace_candidate.resolve()
    return (declaring_artifact.parent / path).resolve()


def _timestamp_blocker(value: str) -> str | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return "evaluated_at_invalid"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return "evaluated_at_not_timezone_aware"
    return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_sha(receipt: Mapping[str, Any]) -> str:
    body = {key: value for key, value in receipt.items() if key != "transition_receipt_sha256"}
    return _sha256_json(body)


def _record_sha(record: Mapping[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return _sha256_json(body)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "ContextAcquisitionStateMachine",
    "ContextAcquisitionTransitionLedger",
    "DEFAULT_REGISTRY_PATH",
    "REGISTRY_CONTRACT",
    "RECEIPT_CONTRACT",
    "STATE_ORDER",
    "render_markdown",
]
