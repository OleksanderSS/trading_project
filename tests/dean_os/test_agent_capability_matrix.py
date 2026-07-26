from __future__ import annotations

from dean_os.agent_capability_matrix import (
    AgentCapabilityMatrixBuilder,
)


def test_capability_matrix_matches_registry_and_preserves_boundaries():
    payload = AgentCapabilityMatrixBuilder().build(save=False)
    entries = {
        item["agent_name"]: item
        for item in payload["entries"]
    }

    # matrix_complete (not undeclared and not stale_contracts) is the real
    # invariant: every agent_registry.yaml entry has a declared, non-stale
    # CAPABILITY_CONTRACTS entry. A hardcoded agent_count assertion was
    # here before and had already gone stale once (registry grew 28->39
    # agents without the test being updated) -- removed rather than bumped
    # again, since matrix_complete already proves coverage without a magic
    # number that drifts every time the registry grows.
    assert payload["summary"]["matrix_complete"] is True
    assert payload["summary"]["agent_count"] == len(entries) > 0
    assert entries["pipeline_manager"]["decision_influence"] is False
    assert entries["pipeline_manager"]["run_phases"] == ["pre_trade"]
    assert entries["semiconductor_analyst"][
        "decision_influence"
    ] is False
    assert entries["regime"]["enabled"] is True
    assert entries["regime"]["activation_mode"] == (
        "active_shadow_review"
    )
    assert entries["regime"]["run_phases"] == ["pre_trade"]
    assert entries["regime"]["decision_influence"] is False
    assert entries["context_synthesis"]["enabled"] is True
    assert entries["context_synthesis"]["activation_mode"] == (
        "active_shadow_review"
    )
    assert entries["context_synthesis"]["decision_influence"] is False
    assert entries["model_performance"]["enabled"] is False
    assert "fixed AMD artifact paths" in (
        entries["model_performance"]["known_gap"]
    )
    assert payload["scope_boundaries"]["amd_current_case"] == (
        "ticker_model_evaluation_only"
    )
    assert payload["scope_boundaries"][
        "sector_evidence_can_be_ticker_evidence"
    ] is False
    assert payload["safety"]["is_activation_gate"] is False
    assert payload["safety"]["can_trade"] is False
