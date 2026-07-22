from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_BINDING_PLAN_PATH = "reports/dean_os/domain_analyst_binding_plan_current/latest.json"
DEFAULT_POLICY_PATH = "dean_os/config/domain_binding_dispatch_policy.template.json"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_binding_task_dispatch_current"


class DomainBindingTaskDispatcher:
    """Classify binding tasks and fail closed before any adapter execution."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        binding_plan_path: str | Path = DEFAULT_BINDING_PLAN_PATH,
        policy_path: str | Path = DEFAULT_POLICY_PATH,
        save: bool = True,
    ) -> dict[str, Any]:
        binding_plan = _load_json(Path(binding_plan_path))
        policy = _load_json(Path(policy_path))
        blockers = _structural_blockers(binding_plan, policy)
        catalog = dict(policy.get("adapters") or {})
        domain_id = str(binding_plan.get("summary", {}).get("domain_id") or "")
        dispatches = [
            _dispatch_task(
                item,
                catalog.get(item.get("context_family")) or {},
                domain_id=domain_id,
            )
            for item in binding_plan.get("family_plans", [])
        ]
        dispatches.sort(key=lambda item: (item["priority"], item["context_family"]))
        reuse = [item for item in dispatches if item["dispatch_class"] == "local_reuse_validation"]
        existing = [item for item in dispatches if item["dispatch_class"] == "existing_adapter_run"]
        generalize = [item for item in dispatches if item["dispatch_class"] == "adapter_generalization_work"]
        executable = [item for item in dispatches if item["execution_eligible"]]
        summary_status = "dispatch_plan_blocked_structurally" if blockers else (
            "dispatch_plan_ready_one_or_more_tasks_eligible"
            if executable
            else "dispatch_plan_ready_no_executable_tasks"
        )
        payload = {
            "run_id": _run_id("domain_binding_task_dispatch"),
            "created_at": utc_now_iso(),
            "mode": "domain_binding_task_dispatch",
            "contract": policy.get("contract"),
            "inputs": {
                "binding_plan_path": str(binding_plan_path),
                "binding_plan_run_id": binding_plan.get("run_id"),
                "domain_id": binding_plan.get("summary", {}).get("domain_id"),
                "policy_path": str(policy_path),
            },
            "summary": {
                "domain_id": binding_plan.get("summary", {}).get("domain_id"),
                "status": summary_status,
                "task_count": len(dispatches),
                "local_reuse_validation_count": len(reuse),
                "existing_adapter_run_count": len(existing),
                "adapter_generalization_work_count": len(generalize),
                "execution_eligible_count": len(executable),
                "structural_blockers": blockers,
                "next_priority_task_id": dispatches[0]["task_id"] if dispatches else None,
                "next_priority_context_family": dispatches[0]["context_family"] if dispatches else None,
                "can_execute_dispatch_now": bool(executable) and not blockers,
                "adapter_run_performed": False,
                "binding_written": False,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "dispatch_policy": {
                "maximum_adapter_runs_per_dispatch": policy.get("maximum_adapter_runs_per_dispatch"),
                "automatic_multi_task_loop_allowed": policy.get("automatic_multi_task_loop_allowed"),
                "explicit_local_inputs_required": policy.get("explicit_local_inputs_required"),
                "execution_boundary": policy.get("execution_boundary"),
            },
            "task_dispatches": dispatches,
            "recommended_next_step": (
                _next_step(dispatches, blockers)
            ),
            "journal_events_proposed": [
                {
                    "event_type": "binding_task_dispatch_proposed",
                    "entity_id": item["task_id"],
                    "payload_sha_binding_required": True,
                    "append_performed": False,
                }
                for item in dispatches
            ],
            "explicit_non_actions": [
                "No adapter was imported or executed by this dispatcher.",
                "No network or filesystem discovery was performed.",
                "No candidate was accepted and no profile binding was written.",
                "No analyst run, hypothesis decision, learning write or trade occurred.",
            ],
        }
        if save:
            saved = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_dispatch_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved
        return json_ready(payload)


def _dispatch_task(
    family_plan: dict[str, Any], adapter: dict[str, Any], *, domain_id: str
) -> dict[str, Any]:
    family_id = str(family_plan.get("context_family") or "")
    proposed = family_plan.get("proposed_candidate")
    compatibility = str(adapter.get("compatibility") or "unknown")
    if proposed and proposed.get("validation_status") == "valid":
        dispatch_class = "local_reuse_validation"
        status = "ready_for_binding_gate_review"
        action = "review_sha_bound_candidate_without_rebuilding_source_artifact"
    elif compatibility == "generic_offline_ready":
        dispatch_class = "existing_adapter_run"
        status = "waiting_for_explicit_local_inputs"
        action = "prepare_one_allowlisted_offline_adapter_run"
    else:
        dispatch_class = "adapter_generalization_work"
        status = "implementation_required_before_dispatch"
        action = str(adapter.get("generalization_target") or "define_domain_scoped_adapter")
    return {
        "task_id": (family_plan.get("collection_task") or {}).get("task_id") or f"bind_{domain_id}_{family_id}",
        "domain_id": domain_id,
        "context_family": family_id,
        "priority": int(adapter.get("priority") or 999),
        "depends_on_context_families": list(adapter.get("depends_on") or []),
        "dispatch_class": dispatch_class,
        "dispatch_status": status,
        "recommended_action": action,
        "implementation": adapter.get("implementation"),
        "adapter_actual_contract": adapter.get("actual_contract"),
        "binding_required_contract": family_plan.get("required_contract"),
        "compatibility": compatibility,
        "compatibility_reasons": list(adapter.get("reasons") or []),
        "core_reuse_possible": "core" in compatibility or compatibility == "generic_offline_ready",
        "explicit_local_inputs_present": False,
        "execution_eligible": False,
        "execution_authorized": False,
        "adapter_run_performed": False,
        "journal_append_performed": False,
    }


def _structural_blockers(binding_plan: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if binding_plan.get("mode") != "domain_analyst_binding_plan":
        blockers.append("unsupported_binding_plan_mode")
    if binding_plan.get("contract") != "dean_domain_context_binding_policy_v1":
        blockers.append("unsupported_binding_plan_contract")
    families = [item.get("context_family") for item in binding_plan.get("family_plans", [])]
    catalog = dict(policy.get("adapters") or {})
    if set(families) != set(catalog):
        blockers.append("binding_plan_dispatch_catalog_family_mismatch")
    if len(families) != len(set(families)):
        blockers.append("duplicate_context_family_in_binding_plan")
    if policy.get("maximum_adapter_runs_per_dispatch") != 1:
        blockers.append("dispatch_run_limit_must_equal_one")
    if policy.get("automatic_multi_task_loop_allowed") is not False:
        blockers.append("automatic_multi_task_loop_must_be_false")
    boundary = dict(policy.get("execution_boundary") or {})
    forbidden = [
        "network_access_allowed",
        "production_config_write_allowed",
        "binding_acceptance_allowed",
        "hypothesis_approval_allowed",
        "learning_write_allowed",
        "trading_allowed",
    ]
    if any(boundary.get(key) is not False for key in forbidden):
        blockers.append("dispatch_authority_boundary_not_fail_closed")
    for family_id, item in catalog.items():
        if not item.get("actual_contract"):
            blockers.append(f"adapter_actual_contract_missing:{family_id}")
        if not item.get("compatibility"):
            blockers.append(f"adapter_compatibility_missing:{family_id}")
    return sorted(set(blockers))


def _next_step(dispatches: list[dict[str, Any]], blockers: list[str]) -> str:
    if blockers:
        return "Repair dispatch policy/plan structural blockers before any adapter work."
    if not dispatches:
        return "No binding tasks are available."
    item = dispatches[0]
    return (
        f"Implement and test `{item['recommended_action']}` for `{item['context_family']}` first; "
        "do not dispatch the remaining tasks in a loop."
    )


def render_dispatch_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DEAN-OS Domain Binding Task Dispatch",
        "",
        f"- Domain: `{summary['domain_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Tasks: {summary['task_count']}",
        f"- Local reuse validation: {summary['local_reuse_validation_count']}",
        f"- Existing adapter runs: {summary['existing_adapter_run_count']}",
        f"- Adapter generalization work: {summary['adapter_generalization_work_count']}",
        f"- Execution eligible: {summary['execution_eligible_count']}",
        f"- Can execute now: {summary['can_execute_dispatch_now']}",
        f"- Can invoke analyst: {summary['can_invoke_domain_analysis']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Priority dispatch",
        "",
    ]
    for item in payload["task_dispatches"]:
        lines.append(
            f"- P{item['priority']} `{item['task_id']}`: `{item['dispatch_class']}` / "
            f"`{item['dispatch_status']}` -> `{item['recommended_action']}`"
        )
        for reason in item["compatibility_reasons"]:
            lines.append(f"  - {reason}")
    lines.extend(["", "## Recommended next step", "", f"- {payload['recommended_next_step']}"])
    lines.extend(["", "## Explicit non-actions", ""])
    lines.extend(f"- {item}" for item in payload["explicit_non_actions"])
    return "\n".join(lines).strip() + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = ["DomainBindingTaskDispatcher", "render_dispatch_markdown"]
