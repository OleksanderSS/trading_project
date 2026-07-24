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

    assert payload["summary"]["matrix_complete"] is True
    # Registry grew from 28 to 39 agents since this test was written
    # (coherence_scan, freshness_audit, agent_evaluation_controller,
    # pipeline_readiness, and 5 more standalone domain analysts) -- all
    # confirmed live, not scaffolding. See CAPABILITY_CONTRACTS in
    # agent_capability_matrix.py for their contract entries.
    assert payload["summary"]["agent_count"] == 39
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
