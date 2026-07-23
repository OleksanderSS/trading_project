from __future__ import annotations

import json
from pathlib import Path

from dean_os.analyst_core.domain_analyst_binding_planner import DomainAnalystBindingPlanner
from dean_os.domain_binding_task_dispatcher import DomainBindingTaskDispatcher


def test_energy_dispatch_classifies_all_tasks_without_execution(tmp_path):
    plan_path = _save_plan(tmp_path)
    payload = DomainBindingTaskDispatcher().build(
        binding_plan_path=plan_path, save=False
    )

    summary = payload["summary"]
    assert summary["status"] == "dispatch_plan_ready_no_executable_tasks"
    assert summary["task_count"] == 6
    assert summary["adapter_generalization_work_count"] == 0
    assert summary["existing_adapter_run_count"] == 6
    assert summary["execution_eligible_count"] == 0
    assert summary["can_execute_dispatch_now"] is False
    assert summary["adapter_run_performed"] is False
    assert summary["can_trade"] is False
    assert all(item["execution_authorized"] is False for item in payload["task_dispatches"])


def test_macro_domain_envelope_is_first_priority(tmp_path):
    payload = DomainBindingTaskDispatcher().build(
        binding_plan_path=_save_plan(tmp_path), save=False
    )

    first = payload["task_dispatches"][0]
    assert first["context_family"] == "macro"
    assert first["priority"] == 1
    assert first["recommended_action"] == "prepare_one_allowlisted_offline_adapter_run"
    assert first["core_reuse_possible"] is True


def test_fundamentals_contract_matches_domain_envelope(tmp_path):
    payload = DomainBindingTaskDispatcher().build(
        binding_plan_path=_save_plan(tmp_path), save=False
    )
    item = _family(payload, "fundamentals")

    assert item["adapter_actual_contract"] == "dean_domain_scoped_fundamentals_envelope_v1"
    assert item["binding_required_contract"] == "dean_domain_scoped_fundamentals_envelope_v1"


def test_valid_reuse_candidate_routes_to_binding_review_not_adapter(tmp_path):
    candidate = tmp_path / "energy-news.json"
    candidate.write_text(
        json.dumps(
            {
                "contract": "dean_domain_scoped_news_envelope_v1",
                "mode": "domain_scoped_news_envelope",
                "domain_id": "energy",
                "created_at": "2026-07-13T00:00:00Z",
                "inputs": {"as_of": "2026-07-13T00:00:00Z", "domain_id": "energy"},
                "status": "domain_news_candidate_ready_with_gaps",
                "safety": {"review_only": True},
            }
        ),
        encoding="utf-8",
    )
    plan = DomainAnalystBindingPlanner(tmp_path / "binding").build(
        as_of="2026-07-14T12:00:00Z",
        candidate_artifacts={"news": [candidate]},
    )
    payload = DomainBindingTaskDispatcher().build(
        binding_plan_path=plan["saved_paths"]["latest_json"], save=False
    )
    item = _family(payload, "news")

    assert item["dispatch_class"] == "local_reuse_validation"
    assert item["dispatch_status"] == "ready_for_binding_gate_review"
    assert item["task_id"] == "bind_energy_news"
    assert item["execution_eligible"] is False


def test_policy_authority_widening_blocks_dispatch(tmp_path):
    policy = json.loads(
        Path("dean_os/config/domain_binding_dispatch_policy.template.json").read_text(encoding="utf-8")
    )
    policy["execution_boundary"]["trading_allowed"] = True
    policy_path = tmp_path / "unsafe-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    payload = DomainBindingTaskDispatcher().build(
        binding_plan_path=_save_plan(tmp_path),
        policy_path=policy_path,
        save=False,
    )

    assert payload["summary"]["status"] == "dispatch_plan_blocked_structurally"
    assert "dispatch_authority_boundary_not_fail_closed" in payload["summary"]["structural_blockers"]


def test_dispatch_report_saves_priority_and_non_actions(tmp_path):
    payload = DomainBindingTaskDispatcher(tmp_path / "dispatch").build(
        binding_plan_path=_save_plan(tmp_path)
    )

    markdown = (tmp_path / "dispatch" / "latest.md").read_text(encoding="utf-8")
    assert "P1 `bind_energy_macro`" in markdown
    assert "Can execute now: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")


def _save_plan(tmp_path):
    plan = DomainAnalystBindingPlanner(tmp_path / "plan").build(
        as_of="2026-07-14T12:00:00Z"
    )
    return plan["saved_paths"]["latest_json"]


def _family(payload, family_id):
    return next(item for item in payload["task_dispatches"] if item["context_family"] == family_id)
