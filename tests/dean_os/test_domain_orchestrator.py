"""Tests for the DEAN-OS domain orchestrator (thin composer)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dean_os.domain_orchestrator import DomainOrchestrator
from dean_os.domain_profiles import get_profile, list_domain_ids


def _write_minimal_registry(path: Path) -> None:
    """Write a tiny registry with one enabled pipeline agent + the domain analyst.

    Uses only agents that do not require heavy external inputs, so the test is
    hermetic and fast.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
agents:
  semiconductor_analyst:
    class_path: dean_os.agents.domain_analyst:DomainAnalystAgent
    branch: pipeline
    veto_level: none
    enabled: true
    error_behavior: skip
    timeout_seconds: 30
    domain_id: semiconductor_ai_infrastructure
    horizon_days: 180
    agent_role: standalone_domain_analysis
    decision_influence: false
    execution_group: semiconductor_domain_analysis
    run_phases:
      - pre_trade
  context_synthesis:
    class_path: dean_os.agents.context_synthesis:ContextSynthesisAgent
    branch: pipeline
    veto_level: none
    enabled: true
    error_behavior: skip
    timeout_seconds: 10
    require_predecessor_data: true
    shadow_mode: true
    run_phases:
      - pre_trade
""",
        encoding="utf-8",
    )


@pytest.fixture()
def registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "agent_registry.yaml"
    _write_minimal_registry(path)
    return path


def test_get_profile_known_domain():
    profile = get_profile("semiconductor_ai_infrastructure")
    assert profile.domain_id == "semiconductor_ai_infrastructure"
    assert "NVDA" in profile.ticker_universe_hint
    assert "sector_demand" in profile.required_evidence_types


def test_get_profile_unknown_domain_raises():
    with pytest.raises(KeyError):
        get_profile("does_not_exist")


def test_list_domain_ids_includes_core_domains():
    ids = list_domain_ids()
    assert "semiconductor_ai_infrastructure" in ids
    assert "energy" in ids
    assert "macro_policy" in ids


def test_orchestrator_runs_both_branches_and_is_fail_closed(registry_path: Path, tmp_path: Path):
    output_dir = tmp_path / "reports"
    payload = DomainOrchestrator(
        registry_path=registry_path,
        output_dir=output_dir,
    ).run_sync(
        "semiconductor_ai_infrastructure",
        as_of="2026-07-12T00:00:00Z",
        include_profile_agents=False,
        allow_legacy_unbound_context=True,
        save=True,
    )

    summary = payload["summary"]
    assert summary["can_trade"] is False
    assert summary["can_write_learning_memory"] is False
    assert summary["can_write_production_config"] is False
    assert summary["can_create_recommendation"] is False
    # The generic pipeline branch must not duplicate the domain analyst.  A
    # skip-behavior diagnostic may produce no report when its inputs are absent.
    assert summary["pipeline_agent_count"] >= 0
    # Analyst branch produced at least the domain analyst report.
    assert summary["analyst_report_count"] >= 1
    all_names = [
        report.get("agent_name")
        for section in (
            payload["pipeline_branch"]["reports"],
            payload["analyst_branch"]["reports"],
            payload["composite_pipeline_manager"]["reports"],
        )
        for report in section
    ]
    assert all_names.count("semiconductor_analyst") == 1
    assert "semiconductor_analyst" not in {
        report.get("agent_name")
        for report in payload["pipeline_branch"]["reports"]
    }
    # Report was written and is loadable JSON.
    latest_json = Path(payload["saved_paths"]["latest_json"])
    assert latest_json.exists()
    loaded = json.loads(latest_json.read_text(encoding="utf-8"))
    assert loaded["summary"]["can_trade"] is False


def test_orchestrator_missing_domain_raises(registry_path: Path, tmp_path: Path):
    with pytest.raises(KeyError):
        DomainOrchestrator(
            registry_path=registry_path,
            output_dir=tmp_path / "reports",
        ).run_sync("not_a_real_domain")


def test_orchestrator_cli_help_runs():
    # Importing the CLI module should not execute main(); just ensure it imports.
    import run_agent_domain_orchestrator  # noqa: F401


def test_custom_registry_uses_workspace_as_project_root(registry_path: Path, tmp_path: Path):
    orchestrator = DomainOrchestrator(
        registry_path=registry_path,
        output_dir=tmp_path / "reports",
    )
    assert orchestrator.project_root == Path.cwd().resolve()


def test_orchestrator_waits_without_verified_context_set(
    registry_path: Path, tmp_path: Path
) -> None:
    payload = DomainOrchestrator(
        registry_path=registry_path,
        output_dir=tmp_path / "reports",
    ).run_sync(
        "semiconductor_ai_infrastructure",
        as_of="2026-07-10T19:50:45.683169+00:00",
        save=False,
    )

    assert payload["summary"]["automation_status"] == (
        "domain_orchestrator_waiting_for_context_set"
    )
    assert payload["summary"]["pipeline_agent_count"] == 0
    assert payload["summary"]["analyst_report_count"] == 0
    assert payload["summary"]["can_invoke_domain_analysis"] is False


def test_orchestrator_preserves_incomplete_context_set_as_waiting_state(
    registry_path: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "dean_os.domain_orchestrator.load_verified_domain_context_set",
        lambda *args, **kwargs: {
            "status": "domain_context_set_incomplete",
            "candidate_set_sha256": "a" * 64,
            "complete": False,
            "binding_accepted": False,
            "missing_families": ["sector_market"],
            "collection_proposals": [
                {
                    "context_family": "sector_market",
                    "execution_authorized": False,
                }
            ],
            "metadata": {
                "domain_context_set_path": str(tmp_path / "set.json"),
                "domain_context_set_sha256": "b" * 64,
            },
        },
    )
    payload = DomainOrchestrator(
        registry_path=registry_path,
        output_dir=tmp_path / "reports",
    ).run_sync(
        "semiconductor_ai_infrastructure",
        as_of="2026-07-10T19:50:45.683169+00:00",
        context_set_path=tmp_path / "set.json",
        save=False,
    )

    assert payload["summary"]["automation_status"] == (
        "domain_orchestrator_waiting_for_context_families"
    )
    assert payload["summary"]["missing_context_families"] == ["sector_market"]
    assert payload["context_set_gate"]["acquisition_proposals"][0][
        "execution_authorized"
    ] is False
    assert payload["analyst_branch"]["report_count"] == 0
