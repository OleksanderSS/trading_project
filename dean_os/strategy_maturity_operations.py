from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.execution.maturity_gates import (
    REPLAY_GATE_CHECKS,
    run_promotion_pipeline,
    verify_gate_receipt,
)
from dean_os.schemas import utc_now_iso
from dean_os.strategies.strategy_playbook import (
    MaturityLevel,
    PromotionPolicy,
    StrategyDescription,
    StrategyPlaybook,
    StrategyStatus,
)
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready


LEDGER_CONTRACT = "dean_strategy_maturity_decision_ledger_v1"
ASSESSMENT_CONTRACT = "dean_strategy_replay_candidate_assessment_v1"
RECONCILIATION_CONTRACT = "dean_strategy_maturity_daily_reconciliation_v1"
DEFAULT_LEDGER_PATH = "data/dean_os/strategy_maturity_decisions.jsonl"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"
DEFAULT_ASSESSMENT_OUTPUT = "reports/dean_os/strategy_replay_candidate_assessment_current"
DEFAULT_RECONCILIATION_OUTPUT = "reports/dean_os/strategy_maturity_reconciliation_current"


class StrategyMaturityDecisionLedger:
    """Append-only audit ledger for approved, blocked and review-required gates."""

    def __init__(self, path: str | Path = DEFAULT_LEDGER_PATH):
        self.path = Path(path)

    def read_verified(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        previous: str | None = None
        for line_number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("contract") != LEDGER_CONTRACT:
                raise ValueError(f"invalid maturity ledger contract at line {line_number}")
            if record.get("previous_record_sha256") != previous:
                raise ValueError(f"maturity ledger chain break at line {line_number}")
            if record.get("record_sha256") != _record_sha(record):
                raise ValueError(f"maturity ledger hash mismatch at line {line_number}")
            receipt = record.get("gate_receipt") or {}
            valid, failures = verify_gate_receipt(
                receipt,
                require_approved=False,
                verify_evidence=False,
            )
            if not valid:
                raise ValueError(
                    f"invalid maturity receipt at line {line_number}: {','.join(failures)}"
                )
            records.append(record)
            previous = str(record["record_sha256"])
        return records

    def decisions_for(self, strategy_id: str) -> list[dict[str, Any]]:
        return [
            record
            for record in self.read_verified()
            if (record.get("gate_receipt") or {}).get("strategy_id") == strategy_id
        ]

    def append(
        self,
        *,
        receipt: Mapping[str, Any],
        source_artifact_path: str | Path,
    ) -> tuple[dict[str, Any], bool]:
        valid, failures = verify_gate_receipt(receipt, require_approved=False)
        if not valid:
            raise ValueError("invalid maturity receipt: " + ",".join(failures))
        records = self.read_verified()
        receipt_sha = receipt.get("receipt_sha256")
        duplicate = next(
            (
                record
                for record in records
                if (record.get("gate_receipt") or {}).get("receipt_sha256") == receipt_sha
            ),
            None,
        )
        if duplicate:
            return duplicate, False
        source_path = Path(source_artifact_path).resolve()
        record = {
            "contract": LEDGER_CONTRACT,
            "recorded_at": utc_now_iso(),
            "gate_receipt": json_ready(dict(receipt)),
            "source_artifact": artifact_binding(source_path),
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


class StrategyReplayCandidateAssessment:
    """Evaluate one real reviewed hypothesis as a research-only strategy candidate."""

    def __init__(self, output_dir: str | Path = DEFAULT_ASSESSMENT_OUTPUT):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        review_gate_path: str | Path,
        hypothesis_id: str | None = None,
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        apply_ledger: bool = False,
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        source_file = Path(review_gate_path).resolve()
        source = _load_json(source_file)
        source_sha = _sha256_file(source_file)
        blockers = _source_blockers(source)
        reviews = list(source.get("hypothesis_review") or [])
        eligible = [
            item
            for item in reviews
            if item.get("disposition") == "accept_for_replay"
            and not item.get("registration_blockers")
        ]
        selected = next(
            (item for item in eligible if item.get("hypothesis_id") == hypothesis_id),
            eligible[0] if eligible and hypothesis_id is None else None,
        )
        if selected is None:
            blockers.append("eligible_hypothesis_not_found")
        selected_id = str((selected or {}).get("hypothesis_id") or "missing")
        tasks = [
            item
            for item in (source.get("registration_bundle") or {}).get("tasks") or []
            if item.get("hypothesis_id") == selected_id
        ]
        primary_task = next(
            (item for item in tasks if item.get("horizon_days") == 20),
            tasks[0] if tasks else None,
        )
        if primary_task is None:
            blockers.append("replay_task_lineage_missing")
        strategy_id = f"research_strategy_{selected_id}"
        playbook = _playbook(strategy_id, selected or {}, primary_task or {})
        checks = {
            "as_of_data_only": not blockers and _has_point_in_time_lineage(source, selected or {}, primary_task or {}),
            "source_manifest_present": bool(
                (source.get("source_packet") or {}).get("sha256")
                or (source.get("source_packet") or {}).get("source_packet_sha256")
            ),
            "no_future_leakage": False,
            "model_state_manifest_present": False,
            "decision_lineage_complete": _decision_lineage_complete(selected or {}, primary_task or {}),
            "risk_limits_simulated": False,
            "outcome_review_generated": False,
        }
        gate = run_promotion_pipeline(
            strategy_id,
            "replay",
            checks,
            current_level="research",
            evidence_artifacts={"reviewed_replay_bundle": source_file},
        )
        receipt = gate["receipt"]
        ledger = StrategyMaturityDecisionLedger(ledger_path)
        ledger_appended = False
        if apply_ledger:
            _, ledger_appended = ledger.append(
                receipt=receipt,
                source_artifact_path=source_file,
            )
        journal = _journal_gate(
            receipt=receipt,
            source_path=source_file,
            journal_path=Path(journal_path),
            apply=apply_journal,
        )
        payload = {
            "run_id": _run_id("strategy_replay_candidate_assessment"),
            "created_at": utc_now_iso(),
            "mode": "strategy_replay_candidate_assessment",
            "contract": ASSESSMENT_CONTRACT,
            "strategy_id": strategy_id,
            "inputs": {
                "review_gate_path": str(source_file),
                "review_gate_sha256": source_sha,
                "selected_hypothesis_id": selected_id,
                "selected_task_id": (primary_task or {}).get("task_id"),
            },
            "summary": {
                "status": "replay_gate_blocked_missing_strategy_evidence" if receipt["decision"] == "blocked" else "replay_gate_review_required",
                "source_blockers": sorted(set(blockers)),
                "gate_decision": receipt["decision"],
                "failed_check_count": len(receipt["checks_failed"]),
                "failed_checks": receipt["checks_failed"],
                "ledger_appended": ledger_appended,
                "strategy_registry_mutated": False,
                "replay_task_registered": False,
                "strategy_promoted": False,
                "paper_execution_performed": False,
                "can_trade": False,
            },
            "strategy_playbook": playbook.model_dump(mode="json"),
            "replay_gate": gate,
            "ledger": ledger.status(),
            "journal": journal,
            "safety": {
                "assessment_only": True,
                "automatic_promotion_allowed": False,
                "registry_write_performed": False,
                "replay_registration_performed": False,
                "learning_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_assessment_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


class StrategyMaturityDailyReconciler:
    """Reconcile a candidate playbook with the verified maturity-decision ledger."""

    def __init__(self, output_dir: str | Path = DEFAULT_RECONCILIATION_OUTPUT):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        candidate_assessment_path: str | Path,
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
        risk_snapshot_path: str | Path | None = None,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        assessment_file = Path(candidate_assessment_path).resolve()
        assessment = _load_json(assessment_file)
        playbook = StrategyPlaybook.model_validate(assessment.get("strategy_playbook") or {})
        ledger = StrategyMaturityDecisionLedger(ledger_path)
        decisions = ledger.decisions_for(playbook.strategy_id)
        approved = [
            record for record in decisions if (record.get("gate_receipt") or {}).get("decision") == "approved"
        ]
        latest = decisions[-1].get("gate_receipt") if decisions else None
        latest_approved = approved[-1].get("gate_receipt") if approved else None
        derived_level = str((latest_approved or {}).get("target_gate") or "research")
        registry_level = playbook.promotion_policy.current_maturity_level.value
        blockers: list[str] = []
        level_order = ["research", "replay", "paper", "shadow", "supervised_live"]
        risk_required = level_order.index(derived_level) >= level_order.index("replay")
        risk_snapshot: dict[str, Any] | None = None
        risk_valid = False
        if risk_snapshot_path:
            risk_snapshot = _load_json(Path(risk_snapshot_path))
            risk_valid = bool(
                risk_snapshot.get("contract") == "dean_strategy_risk_snapshot_v1"
                and risk_snapshot.get("strategy_id") == playbook.strategy_id
                and (risk_snapshot.get("summary") or {}).get("risk_check_passed") is True
                and (risk_snapshot.get("summary") or {}).get("kill_switch_active") is False
            )
            if not risk_valid:
                blockers.append("strategy_risk_snapshot_invalid")
        elif risk_required:
            blockers.append("strategy_risk_snapshot_missing")
        if assessment.get("contract") != ASSESSMENT_CONTRACT:
            blockers.append("unsupported_candidate_assessment_contract")
        if latest:
            latest_valid, latest_failures = verify_gate_receipt(
                latest,
                expected_strategy_id=playbook.strategy_id,
                require_approved=False,
            )
            if not latest_valid:
                blockers.extend(
                    f"latest_receipt_invalid:{item}" for item in latest_failures
                )
        if derived_level != registry_level:
            blockers.append("registry_maturity_does_not_match_approved_receipt")
        if playbook.status.value != registry_level:
            blockers.append("playbook_status_does_not_match_registry_maturity")
        if latest_approved:
            valid, failures = verify_gate_receipt(
                latest_approved,
                expected_strategy_id=playbook.strategy_id,
                expected_target_gate=derived_level,
            )
            if not valid:
                blockers.extend(f"approved_receipt_invalid:{item}" for item in failures)
        if derived_level in {"shadow", "supervised_live"} and not playbook.promotion_policy.rollback_strategy_id:
            blockers.append("rollback_strategy_missing")
        if derived_level == "supervised_live":
            blockers.append("supervised_live_disabled_by_system_policy")
        status = "maturity_reconciliation_valid" if not blockers else "maturity_reconciliation_blocked"
        journal = _journal_reconciliation(
            strategy_id=playbook.strategy_id,
            status=status,
            blockers=blockers,
            assessment_path=assessment_file,
            journal_path=Path(journal_path),
            apply=apply_journal,
        )
        payload = {
            "run_id": _run_id("strategy_maturity_reconciliation"),
            "created_at": utc_now_iso(),
            "mode": "strategy_maturity_daily_reconciliation",
            "contract": RECONCILIATION_CONTRACT,
            "strategy_id": playbook.strategy_id,
            "inputs": {
                "candidate_assessment_path": str(assessment_file),
                "candidate_assessment_sha256": _sha256_file(assessment_file),
                "ledger_path": str(ledger_path),
                "risk_snapshot_path": str(risk_snapshot_path) if risk_snapshot_path else None,
            },
            "summary": {
                "status": status,
                "structural_blockers": sorted(set(blockers)),
                "registry_maturity_level": registry_level,
                "derived_approved_maturity_level": derived_level,
                "decision_count": len(decisions),
                "approved_decision_count": len(approved),
                "latest_gate_decision": (latest or {}).get("decision"),
                "latest_gate_failed_checks": (latest or {}).get("checks_failed") or [],
                "rollback_ready": bool(playbook.promotion_policy.rollback_strategy_id),
                "risk_snapshot_required": risk_required,
                "risk_snapshot_present": risk_snapshot is not None,
                "risk_snapshot_valid": risk_valid,
                "registry_write_performed": False,
                "strategy_promoted": False,
                "paper_execution_performed": False,
                "can_trade": False,
            },
            "latest_decision_receipt": latest,
            "latest_approved_receipt": latest_approved,
            "ledger": ledger.status(),
            "journal": journal,
            "safety": {
                "reconciliation_only": True,
                "registry_write_performed": False,
                "promotion_performed": False,
                "simulated_order_submitted": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_reconciliation_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def journal_simulated_order_decision(
    *,
    order: Mapping[str, Any],
    result: Mapping[str, Any],
    source_artifact_path: str | Path,
    journal_path: str | Path = DEFAULT_JOURNAL_PATH,
    apply: bool = False,
) -> dict[str, Any]:
    mode = str(order.get("mode") or "")
    decision = str(result.get("decision") or "")
    if mode not in {"paper", "shadow"}:
        raise ValueError("only paper/shadow simulated decisions may be journaled")
    if decision not in {"approved_simulated", "rejected", "blocked_hard"}:
        raise ValueError("unsupported simulated order decision")
    source_path = Path(source_artifact_path).resolve()
    event = {
        "event_type": "action_reviewed",
        "effective_at": str(order.get("created_at") or utc_now_iso()),
        "actor": "strategy_execution_gateway",
        "domain_id": str(order.get("domain_id") or "strategy_system"),
        "entity_type": "simulated_order_decision",
        "entity_id": str(order.get("order_id") or "missing_order_id"),
        "source_artifact": artifact_binding(source_path),
        "context": {"mode": mode, "simulation_only": True},
        "payload": {
            "strategy_id": order.get("strategy_id"),
            "decision": decision,
            "lineage_id": result.get("lineage_id"),
            "maturity_receipt_sha256": result.get("maturity_receipt_sha256"),
            "broker_send_performed": False,
            "live_execution_performed": False,
        },
    }
    journal = SystemJournal(journal_path)
    if not apply:
        return {"apply_requested": False, "events_proposed": 1, "appended_count": 0, "chain_valid": journal.status()["chain_valid"]}
    appended = journal.append_many([event])
    return {"apply_requested": True, **appended, **journal.status()}


def _source_blockers(source: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    summary = source.get("summary") or {}
    if source.get("contract") != "dean_world_model_replay_review_gate_v1":
        blockers.append("unsupported_replay_review_gate_contract")
    if source.get("mode") != "world_model_replay_review_gate":
        blockers.append("unsupported_replay_review_gate_mode")
    if summary.get("approved") is not True or summary.get("manual_hypothesis_review_complete") is not True:
        blockers.append("manual_replay_review_not_complete")
    if summary.get("can_register_replay_tasks") is not True:
        blockers.append("replay_bundle_not_registration_eligible")
    if summary.get("replay_task_registration_performed") is not False:
        blockers.append("replay_registration_boundary_invalid")
    if summary.get("can_write_learning_memory") is not False or summary.get("can_trade") is not False:
        blockers.append("replay_review_authority_boundary_invalid")
    return blockers


def _playbook(
    strategy_id: str,
    hypothesis: Mapping[str, Any],
    task: Mapping[str, Any],
) -> StrategyPlaybook:
    return StrategyPlaybook(
        strategy_id=strategy_id,
        version="0.1-research-candidate",
        status=StrategyStatus.RESEARCH,
        description=StrategyDescription(
            name=f"Observation candidate for {hypothesis.get('hypothesis_id')}",
            thesis=str(hypothesis.get("hypothesis") or "Research hypothesis observation candidate"),
            strategy_family=["hypothesis_replay_observation"],
            time_horizon=f"{task.get('horizon_days') or 'unknown'}d_event_response",
            asset_universe=[],
        ),
        promotion_policy=PromotionPolicy(
            current_maturity_level=MaturityLevel.RESEARCH,
            next_allowed_level=MaturityLevel.REPLAY,
            approval_required=True,
            rollback_strategy_id=None,
        ),
    )


def _has_point_in_time_lineage(
    source: Mapping[str, Any], hypothesis: Mapping[str, Any], task: Mapping[str, Any]
) -> bool:
    return bool(
        source.get("created_at")
        and task.get("packet_as_of")
        and task.get("trigger_event_at")
        and (hypothesis.get("trigger_event") or {}).get("record_sha256")
    )


def _decision_lineage_complete(
    hypothesis: Mapping[str, Any], task: Mapping[str, Any]
) -> bool:
    lineage = task.get("resolution_lineage") or {}
    return bool(
        hypothesis.get("disposition") == "accept_for_replay"
        and lineage.get("source_packet_sha256")
        and lineage.get("source_review_gate_sha256")
        and task.get("trigger_evidence_id")
    )


def _journal_gate(
    *, receipt: Mapping[str, Any], source_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    event = {
        "event_type": "action_reviewed",
        "effective_at": receipt["evaluated_at"],
        "actor": "strategy_maturity_gate",
        "domain_id": "strategy_system",
        "entity_type": "strategy_maturity_decision",
        "entity_id": f"{receipt['strategy_id']}:{receipt['target_gate']}:{receipt['receipt_sha256'][:16]}",
        "source_artifact": artifact_binding(source_path),
        "context": {"target_gate": receipt["target_gate"], "live_execution_allowed": False},
        "payload": {
            "decision": receipt["decision"],
            "failed_checks": receipt["checks_failed"],
            "receipt_sha256": receipt["receipt_sha256"],
            "strategy_promoted": False,
            "paper_execution_performed": False,
            "live_execution_performed": False,
        },
    }
    return _append_or_preview(event, journal_path, apply)


def _journal_reconciliation(
    *, strategy_id: str, status: str, blockers: list[str], assessment_path: Path,
    journal_path: Path, apply: bool
) -> dict[str, Any]:
    event = {
        "event_type": "governance_closure_recorded",
        "effective_at": utc_now_iso(),
        "actor": "strategy_maturity_daily_reconciler",
        "domain_id": "strategy_system",
        "entity_type": "strategy_maturity_reconciliation",
        "entity_id": strategy_id,
        "source_artifact": artifact_binding(assessment_path),
        "context": {"daily_reconciliation": True},
        "payload": {
            "status": status,
            "blockers": blockers,
            "strategy_promoted": False,
            "paper_execution_performed": False,
            "live_execution_performed": False,
        },
    }
    return _append_or_preview(event, journal_path, apply)


def _append_or_preview(event: dict[str, Any], journal_path: Path, apply: bool) -> dict[str, Any]:
    journal = SystemJournal(journal_path)
    if not apply:
        return {"apply_requested": False, "events_proposed": 1, "appended_count": 0, "chain_valid": journal.status()["chain_valid"]}
    result = journal.append_many([event])
    return {"apply_requested": True, **result, **journal.status()}


def render_assessment_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DEAN-OS Strategy Replay Candidate Assessment",
        "",
        f"- Strategy: `{payload['strategy_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Replay gate decision: `{summary['gate_decision']}`",
        f"- Failed checks: {summary['failed_check_count']}",
        f"- Strategy promoted: {summary['strategy_promoted']}",
        f"- Replay task registered: {summary['replay_task_registered']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Failed replay checks",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["failed_checks"] or ["none"])
    lines.extend(["", "## Boundary", "", "- This assessment creates a research-only playbook candidate; it does not promote or register it."])
    return "\n".join(lines).strip() + "\n"


def render_reconciliation_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DEAN-OS Strategy Maturity Reconciliation",
        "",
        f"- Strategy: `{payload['strategy_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Registry maturity: `{summary['registry_maturity_level']}`",
        f"- Approved maturity: `{summary['derived_approved_maturity_level']}`",
        f"- Latest decision: `{summary['latest_gate_decision']}`",
        f"- Approved decisions: {summary['approved_decision_count']}",
        f"- Strategy promoted: {summary['strategy_promoted']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Structural blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["structural_blockers"] or ["none"])
    lines.extend(["", "## Latest gate gaps", ""])
    lines.extend(f"- {item}" for item in summary["latest_gate_failed_checks"] or ["none"])
    return "\n".join(lines).strip() + "\n"


def _record_sha(record: Mapping[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return _sha256_json(body)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


__all__ = [
    "StrategyMaturityDecisionLedger",
    "StrategyMaturityDailyReconciler",
    "StrategyReplayCandidateAssessment",
    "journal_simulated_order_decision",
]
