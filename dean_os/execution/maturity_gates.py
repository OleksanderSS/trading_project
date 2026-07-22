"""Fail-closed strategy maturity gates for replay, paper and shadow modes.

Gate decisions are evidence-bound receipts.  A boolean such as
``paper_gate_passed=True`` is never accepted as proof of a previous gate: the
caller must provide the approved, untampered receipt from that gate.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


RECEIPT_CONTRACT = "dean_strategy_maturity_gate_receipt_v1"
MATURITY_ORDER = ("research", "replay", "paper", "shadow", "supervised_live")
LIVE_EXECUTION_ENABLED = False


class GateDecision(str, Enum):
    BLOCKED = "blocked"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"


@dataclass
class GateCheckResult:
    gate_name: str
    decision: GateDecision
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    approver: str | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "decision": self.decision.value,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "approver": self.approver,
            "notes": self.notes,
        }


REPLAY_GATE_CHECKS = [
    "as_of_data_only",
    "source_manifest_present",
    "no_future_leakage",
    "model_state_manifest_present",
    "decision_lineage_complete",
    "risk_limits_simulated",
    "outcome_review_generated",
]

PAPER_GATE_CHECKS = [
    "replay_gate_passed",
    "simulated_broker_only",
    "live_data_permitted",
    "real_orders_forbidden",
    "risk_engine_active",
    "kill_switch_tested",
    "order_lineage_logged",
    "decision_lineage_complete",
    "rollback_plan_exists",
]

SHADOW_GATE_CHECKS = [
    "paper_gate_passed",
    "broker_send_disabled_and_verified",
    "slippage_tracking_active",
    "risk_breach_simulation_active",
    "daily_shadow_report_generated",
    "kill_switch_tested",
    "out_of_sample_passed",
    "walk_forward_passed",
    "decision_lineage_complete",
    "rollback_plan_exists",
]

SUPERVISED_LIVE_GATE_CHECKS = [
    "shadow_gate_passed",
    "small_capital_allocation",
    "allowed_assets_only",
    "allowed_hours_only",
    "human_approval_or_emergency_stop_available",
    "max_position_limit_active",
    "max_daily_loss_limit_active",
    "max_drawdown_limit_active",
    "unsupported_assets_blocked",
    "execution_gateway_required",
    "kill_switch_required",
    "decision_lineage_complete",
    "operator_review_record_present",
]

GATE_CHECKS: dict[str, list[str]] = {
    "replay": REPLAY_GATE_CHECKS,
    "paper": PAPER_GATE_CHECKS,
    "shadow": SHADOW_GATE_CHECKS,
    "supervised_live": SUPERVISED_LIVE_GATE_CHECKS,
}

PREVIOUS_GATE_CHECK = {
    "paper": "replay_gate_passed",
    "shadow": "paper_gate_passed",
    "supervised_live": "shadow_gate_passed",
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_hash(receipt: Mapping[str, Any]) -> str:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return _sha256_bytes(_canonical_json(body).encode("utf-8"))


def _bind_evidence(
    evidence_artifacts: Mapping[str, str | Path] | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    bindings: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    if not evidence_artifacts:
        return bindings, ["evidence_artifact_missing"]

    for artifact_id, raw_path in sorted(evidence_artifacts.items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            failures.append(f"evidence_artifact_unreadable:{artifact_id}")
            continue
        bindings[artifact_id] = {
            "path": str(path),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    if not bindings and "evidence_artifact_missing" not in failures:
        failures.append("evidence_artifact_missing")
    return bindings, failures


def verify_gate_receipt(
    receipt: Mapping[str, Any] | None,
    *,
    expected_strategy_id: str | None = None,
    expected_target_gate: str | None = None,
    require_approved: bool = True,
    verify_evidence: bool = True,
) -> tuple[bool, list[str]]:
    """Verify receipt structure, self-hash and current evidence hashes."""
    failures: list[str] = []
    if not isinstance(receipt, Mapping):
        return False, ["maturity_receipt_missing"]
    if receipt.get("contract") != RECEIPT_CONTRACT:
        failures.append("maturity_receipt_contract_invalid")
    if receipt.get("receipt_sha256") != _receipt_hash(receipt):
        failures.append("maturity_receipt_hash_invalid")
    if expected_strategy_id and receipt.get("strategy_id") != expected_strategy_id:
        failures.append("maturity_receipt_strategy_mismatch")
    if expected_target_gate and receipt.get("target_gate") != expected_target_gate:
        failures.append("maturity_receipt_gate_mismatch")
    if require_approved and receipt.get("decision") != GateDecision.APPROVED.value:
        failures.append("maturity_receipt_not_approved")
    if receipt.get("live_execution_allowed") is not False:
        failures.append("maturity_receipt_live_flag_invalid")

    bindings = receipt.get("evidence_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        failures.append("maturity_receipt_evidence_missing")
    elif verify_evidence:
        for artifact_id, binding in bindings.items():
            if not isinstance(binding, Mapping):
                failures.append(f"maturity_receipt_evidence_invalid:{artifact_id}")
                continue
            path = Path(str(binding.get("path", "")))
            if not path.is_file():
                failures.append(f"maturity_receipt_evidence_unreadable:{artifact_id}")
            elif _file_sha256(path) != binding.get("sha256"):
                failures.append(f"maturity_receipt_evidence_hash_invalid:{artifact_id}")
    return not failures, failures


def _evaluate(
    gate_name: str,
    required_checks: list[str],
    provided: Mapping[str, bool],
    approver: str | None = None,
) -> GateCheckResult:
    passed = [check for check in required_checks if provided.get(check) is True]
    failed = [check for check in required_checks if provided.get(check) is not True]
    operator = approver.strip() if isinstance(approver, str) and approver.strip() else None
    if failed:
        decision = GateDecision.BLOCKED
    elif operator:
        decision = GateDecision.APPROVED
    else:
        decision = GateDecision.REVIEW_REQUIRED
    return GateCheckResult(gate_name, decision, passed, failed, operator)


def replay_gate(provided: dict[str, bool], approver: str | None = None) -> GateCheckResult:
    return _evaluate("replay_gate", REPLAY_GATE_CHECKS, provided, approver)


def paper_gate(provided: dict[str, bool], approver: str | None = None) -> GateCheckResult:
    return _evaluate("paper_gate", PAPER_GATE_CHECKS, provided, approver)


def shadow_gate(provided: dict[str, bool], approver: str | None = None) -> GateCheckResult:
    return _evaluate("shadow_gate", SHADOW_GATE_CHECKS, provided, approver)


def supervised_live_gate(
    provided: dict[str, bool], approver: str | None = None
) -> GateCheckResult:
    result = _evaluate(
        "supervised_live_gate", SUPERVISED_LIVE_GATE_CHECKS, provided, approver
    )
    result.decision = GateDecision.BLOCKED
    result.notes = "supervised_live_disabled_by_system_policy"
    if "supervised_live_disabled_by_system_policy" not in result.checks_failed:
        result.checks_failed.append("supervised_live_disabled_by_system_policy")
    return result


def run_promotion_pipeline(
    strategy_id: str,
    target_gate: str,
    provided_checks: dict[str, bool],
    approver: str | None = None,
    *,
    current_level: str | None = None,
    previous_receipt: Mapping[str, Any] | None = None,
    evidence_artifacts: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Evaluate exactly one sequential transition and return a SHA-bound receipt."""
    if target_gate not in GATE_CHECKS:
        return {
            "strategy_id": strategy_id,
            "target_gate": target_gate,
            "error": f"Unknown gate: {target_gate}",
        }

    target_index = MATURITY_ORDER.index(target_gate)
    inferred_current = MATURITY_ORDER[target_index - 1]
    current = current_level or inferred_current
    transition_failures: list[str] = []
    if current not in MATURITY_ORDER or MATURITY_ORDER.index(current) + 1 != target_index:
        transition_failures.append(f"non_sequential_transition:{current}->{target_gate}")

    checks = dict(provided_checks)
    prior_check = PREVIOUS_GATE_CHECK.get(target_gate)
    previous_sha: str | None = None
    if prior_check:
        prior_ok, prior_failures = verify_gate_receipt(
            previous_receipt,
            expected_strategy_id=strategy_id,
            expected_target_gate=current,
        )
        checks[prior_check] = prior_ok
        transition_failures.extend(prior_failures)
        if prior_ok and previous_receipt is not None:
            previous_sha = str(previous_receipt.get("receipt_sha256"))

    evidence_bindings, evidence_failures = _bind_evidence(evidence_artifacts)
    transition_failures.extend(evidence_failures)
    result = _evaluate(
        f"{target_gate}_gate", GATE_CHECKS[target_gate], checks, approver
    )
    for failure in transition_failures:
        if failure not in result.checks_failed:
            result.checks_failed.append(failure)

    if target_gate == "supervised_live":
        result.decision = GateDecision.BLOCKED
        result.notes = "supervised_live_disabled_by_system_policy"
        if result.notes not in result.checks_failed:
            result.checks_failed.append(result.notes)
    elif result.checks_failed:
        result.decision = GateDecision.BLOCKED
        result.notes = "fail_closed_gate_rejection"

    receipt: dict[str, Any] = {
        "contract": RECEIPT_CONTRACT,
        "strategy_id": strategy_id,
        "from_level": current,
        "target_gate": target_gate,
        "decision": result.decision.value,
        "checks_passed": result.checks_passed,
        "checks_failed": result.checks_failed,
        "evidence_bindings": evidence_bindings,
        "previous_receipt_sha256": previous_sha,
        "approver": result.approver,
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "live_execution_allowed": LIVE_EXECUTION_ENABLED,
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    return {
        "strategy_id": strategy_id,
        "target_gate": target_gate,
        "result": result.as_dict(),
        "receipt": receipt,
    }
