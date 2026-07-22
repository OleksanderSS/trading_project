from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.analyst_core.domain_analyst_lifecycle_profile import (
    DomainAnalystLifecycleProfileCompiler,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_POLICY_PATH = "dean_os/config/domain_context_binding_policy.template.json"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_analyst_binding_plan_current"


class DomainAnalystBindingPlanner:
    """Plan domain-context reuse/collection without executing either action."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str = "energy",
        candidate_artifacts: dict[str, list[str | Path]] | None = None,
        as_of: str | None = None,
        policy_path: str | Path = DEFAULT_POLICY_PATH,
        lifecycle_template_path: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = as_of or utc_now_iso()
        _aware_timestamp(cutoff, "as_of")
        compiler = (
            DomainAnalystLifecycleProfileCompiler(lifecycle_template_path)
            if lifecycle_template_path
            else DomainAnalystLifecycleProfileCompiler()
        )
        lifecycle = compiler.compile(domain_id)
        policy = _load_json(Path(policy_path))
        supplied = candidate_artifacts or {}
        policy_families = {
            item["family_id"]: item for item in policy.get("families", [])
        }
        expected_families = list(
            lifecycle.get("fixed_contract", {}).get("fixed_context_families") or []
        )
        structural_blockers = list(
            lifecycle.get("readiness", {}).get("structural_blockers") or []
        )
        if set(policy_families) != set(expected_families):
            structural_blockers.append("binding_policy_context_family_set_mismatch")
        unknown_supplied = sorted(set(supplied) - set(expected_families))
        if unknown_supplied:
            structural_blockers.append("unknown_candidate_context_family")

        plans = [
            _family_plan(
                domain_id=domain_id,
                family_id=family_id,
                policy=policy_families.get(family_id) or {},
                candidates=supplied.get(family_id) or [],
                as_of=cutoff,
            )
            for family_id in expected_families
        ]
        ready_candidates = [
            item for item in plans if item["status"] == "reuse_candidate_ready_for_review"
        ]
        unresolved = [
            item for item in plans if item["status"] != "binding_accepted"
        ]
        collection_tasks = [item["collection_task"] for item in plans if item.get("collection_task")]
        payload = {
            "run_id": _run_id("domain_analyst_binding_plan"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_binding_plan",
            "contract": policy.get("contract"),
            "inputs": {
                "domain_id": domain_id,
                "as_of": cutoff,
                "policy_path": str(policy_path),
                "candidate_artifact_count": sum(len(value) for value in supplied.values()),
            },
            "summary": {
                "domain_id": domain_id,
                "plan_status": (
                    "binding_plan_blocked_structurally"
                    if structural_blockers
                    else "binding_plan_ready_pending_artifacts_and_manual_acceptance"
                ),
                "context_family_count": len(plans),
                "reuse_candidate_ready_count": len(ready_candidates),
                "unresolved_binding_count": len(unresolved),
                "collection_task_proposal_count": len(collection_tasks),
                "structural_blockers": sorted(set(structural_blockers)),
                "can_propose_reuse": not structural_blockers,
                "can_propose_collection_tasks": not structural_blockers,
                "can_execute_collection_tasks": False,
                "can_accept_bindings": False,
                "can_build_vertical_slice_invocation": False,
                "can_invoke_domain_analysis_now": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "lifecycle_profile_reference": {
                "contract": lifecycle.get("contract"),
                "fixed_contract_sha256": lifecycle.get("fixed_contract_sha256"),
                "domain_overlay_sha256": lifecycle.get("domain_overlay_sha256"),
                "profile_status": lifecycle.get("profile_status"),
            },
            "family_plans": plans,
            "collection_task_proposals": collection_tasks,
            "manual_gate": {
                "status": "not_open_no_complete_binding_set",
                "required_decisions": ["accept_binding", "replace_candidate", "defer"],
                "rule": "A candidate is never a binding until its SHA-256 and validation result are accepted explicitly.",
            },
            "explicit_non_actions": [
                "No filesystem-wide artifact discovery was performed.",
                "No collection adapter was executed.",
                "No profile binding or production configuration was changed.",
                "No vertical slice, hypothesis approval, learning write or trading action was run.",
            ],
        }
        if save:
            saved = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_binding_plan_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved
        return json_ready(payload)


def _family_plan(
    *,
    domain_id: str,
    family_id: str,
    policy: dict[str, Any],
    candidates: list[str | Path],
    as_of: str,
) -> dict[str, Any]:
    reviews = [
        _candidate_review(
            path=Path(path),
            domain_id=domain_id,
            family_id=family_id,
            policy=policy,
            as_of=as_of,
        )
        for path in candidates
    ]
    valid = [item for item in reviews if item["validation_status"] == "valid"]
    if len(valid) == 1:
        status = "reuse_candidate_ready_for_review"
        selected = valid[0]
        reason = "One explicitly supplied candidate passed contract, domain, time, lineage and safety checks."
    elif len(valid) > 1:
        status = "manual_candidate_selection_required"
        selected = None
        reason = "Multiple valid candidates exist; the planner cannot select lineage automatically."
    elif reviews:
        status = "binding_blocked_invalid_candidates"
        selected = None
        reason = "All explicitly supplied candidates failed validation."
    else:
        status = "binding_blocked_no_candidate"
        selected = None
        reason = "No candidate was explicitly supplied; automatic filesystem discovery is forbidden."
    task = None
    if status != "reuse_candidate_ready_for_review":
        task = {
            "task_id": f"bind_{domain_id}_{family_id}",
            "domain_id": domain_id,
            "context_family": family_id,
            "task_type": "prepare_validated_context_artifact",
            "suggested_adapter": policy.get("suggested_adapter"),
            "required_contract": policy.get("required_contract"),
            "required_validation": list(policy.get("required_validation") or []),
            "execution_authorized": False,
            "synthetic_placeholder_allowed": False,
            "completion_condition": "One domain-matched point-in-time artifact passes validation and is accepted at the binding gate.",
        }
    return {
        "context_family": family_id,
        "status": status,
        "reason": reason,
        "required_contract": policy.get("required_contract"),
        "legacy_contracts": list(policy.get("legacy_contracts") or []),
        "required_validation": list(policy.get("required_validation") or []),
        "candidate_reviews": reviews,
        "proposed_candidate": selected,
        "binding_written": False,
        "collection_task": task,
    }


def _candidate_review(
    *,
    path: Path,
    domain_id: str,
    family_id: str,
    policy: dict[str, Any],
    as_of: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    payload: dict[str, Any] = {}
    if not path.is_file():
        reasons.append("candidate_file_missing")
    elif path.suffix.lower() != ".json":
        reasons.append("candidate_must_be_json_review_artifact")
    else:
        try:
            payload = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append("candidate_json_invalid")
    contract = _artifact_contract(payload)
    allowed_contracts = {
        str(policy.get("required_contract") or ""),
        *[str(value) for value in policy.get("legacy_contracts") or []],
    }
    if payload and contract not in allowed_contracts:
        reasons.append("artifact_contract_not_allowed_for_family")
    candidate_domain = _artifact_domain(payload)
    if payload and not candidate_domain:
        reasons.append("domain_identity_missing")
    elif payload and candidate_domain != domain_id:
        reasons.append("cross_domain_artifact_reuse_forbidden")
    artifact_as_of = _artifact_as_of(payload)
    if payload and not artifact_as_of:
        reasons.append("artifact_as_of_missing")
    elif payload:
        try:
            if _aware_timestamp(artifact_as_of, "artifact as_of") > _aware_timestamp(as_of, "analysis as_of"):
                reasons.append("future_artifact_forbidden")
        except ValueError:
            reasons.append("artifact_as_of_invalid")
    if payload and not _review_only(payload, family_id):
        reasons.append("review_only_safety_boundary_missing")
    status = str(payload.get("status") or "").lower()
    if payload and (
        "ready" not in status
        or "blocked" in status
        or "awaiting" in status
    ):
        reasons.append("artifact_not_ready_for_binding_review")
    return {
        "path": str(path),
        "sha256": _sha256_file(path) if path.is_file() else None,
        "contract": contract,
        "domain_id": candidate_domain,
        "artifact_as_of": artifact_as_of,
        "validation_status": "valid" if not reasons else "invalid",
        "reasons": reasons,
    }


def render_binding_plan_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DEAN-OS Domain Analyst Binding Plan",
        "",
        f"- Domain: `{summary['domain_id']}`",
        f"- Status: `{summary['plan_status']}`",
        f"- Context families: {summary['context_family_count']}",
        f"- Reuse candidates ready: {summary['reuse_candidate_ready_count']}",
        f"- Unresolved bindings: {summary['unresolved_binding_count']}",
        f"- Collection task proposals: {summary['collection_task_proposal_count']}",
        f"- Can execute collection: {summary['can_execute_collection_tasks']}",
        f"- Can invoke analysis: {summary['can_invoke_domain_analysis_now']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Context family plan",
        "",
    ]
    for item in payload["family_plans"]:
        lines.append(
            f"- `{item['context_family']}`: `{item['status']}` - {item['reason']}"
        )
    lines.extend(["", "## Proposed tasks", ""])
    for task in payload["collection_task_proposals"]:
        lines.append(
            f"- `{task['task_id']}` -> `{task['required_contract']}` via `{task['suggested_adapter']}`; execution authorized={task['execution_authorized']}"
        )
    lines.extend(["", "## Manual gate", "", f"- {payload['manual_gate']['rule']}"])
    lines.extend(["", "## Explicit non-actions", ""])
    lines.extend(f"- {item}" for item in payload["explicit_non_actions"])
    return "\n".join(lines).strip() + "\n"


def _artifact_contract(payload: dict[str, Any]) -> str | None:
    return str(
        payload.get("producer_contract")
        or payload.get("contract")
        or payload.get("schema_version")
        or ""
    ).strip() or None


def _artifact_domain(payload: dict[str, Any]) -> str | None:
    candidates = [
        payload.get("domain_id"),
        (payload.get("summary") or {}).get("domain_id"),
        (payload.get("inputs") or {}).get("domain_id"),
        (payload.get("metadata") or {}).get("domain_id"),
        (payload.get("market_context_fragment") or {}).get("domain_id"),
        ((payload.get("market_context_fragment") or {}).get("metadata") or {}).get("domain_id"),
    ]
    return next((str(value).strip() for value in candidates if str(value or "").strip()), None)


def _artifact_as_of(payload: dict[str, Any]) -> str | None:
    values = [
        (payload.get("inputs") or {}).get("as_of"),
        payload.get("as_of"),
        payload.get("created_at"),
    ]
    return next((str(value).strip() for value in values if str(value or "").strip()), None)


def _review_only(payload: dict[str, Any], family_id: str) -> bool:
    safety = payload.get("safety") or payload.get("artifact_safety") or {}
    prohibited = [
        "learning_write_performed",
        "production_config_write_performed",
        "broker_access_performed",
        "live_execution_performed",
    ]
    if any(safety.get(key) is True for key in prohibited):
        return False
    if safety.get("review_only") is True or safety.get("review_artifact") is True:
        return True
    return family_id == "pipeline_context" and payload.get("contract") == "dean_world_model_pipeline_context_v1"


def _aware_timestamp(value: str, label: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
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


__all__ = ["DomainAnalystBindingPlanner", "render_binding_plan_markdown"]
