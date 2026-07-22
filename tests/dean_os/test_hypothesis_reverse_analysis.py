from __future__ import annotations

from dean_os.hypothesis_reverse_analysis import build_hypothesis_reverse_analysis


def test_reverse_analysis_separates_true_mechanism_from_wrong_market_reaction():
    card = build_hypothesis_reverse_analysis(
        hypothesis={"hypothesis_id": "h1", "hypothesis": "Demand and price rise"},
        review={
            "hypothesis_id": "h1",
            "disposition": "accept_for_replay",
            "expectation_context_available": False,
        },
        outcome={
            "outcome_id": "o1",
            "result_label": "falsified",
            "fundamental_result": "confirmed",
            "market_reaction_result": "miss",
        },
    )

    codes = set(card["proposal_eligible_error_codes"])
    assert "true_hypothesis_wrong_market_reaction" in codes
    assert "priced_in_blindness" in codes
    assert card["recommended_next_action"]["action"] == (
        "retain_the_fundamental_mechanism_and_reformulate_the_market_reaction_leg"
    )
    assert card["safety"]["machine_may_propose"] is True
    assert card["safety"]["machine_may_apply_rule"] is False
    assert card["safety"]["can_trade"] is False


def test_reverse_analysis_does_not_invent_root_cause_from_bare_falsification():
    card = build_hypothesis_reverse_analysis(
        hypothesis={"hypothesis_id": "h2", "hypothesis": "Demand rises"},
        review={"hypothesis_id": "h2", "disposition": "accept_for_replay"},
        outcome={"outcome_id": "o2", "result_label": "falsified"},
    )

    assert card["proposal_eligible_error_codes"] == []
    assert card["machine_diagnosis_candidates"][0]["error_code"] == (
        "unknown_falsification_cause"
    )
    assert card["machine_diagnosis_candidates"][0]["diagnostic_strength"] == (
        "candidate"
    )
    assert card["agent_change_proposal"]["automatic_application_allowed"] is False


def test_reverse_analysis_identifies_data_and_horizon_failures_from_structure():
    card = build_hypothesis_reverse_analysis(
        hypothesis={
            "hypothesis_id": "h3",
            "hypothesis": "Policy affects the sector",
            "horizon_family": "event_response_fixed_v1",
        },
        review={"hypothesis_id": "h3", "disposition": "accept_for_replay"},
        outcome={
            "outcome_id": "o3",
            "result_label": "unobservable",
            "horizon_family": "sector_thesis_fixed_v1",
            "data_quality_status": "incomplete",
            "observable": False,
        },
    )

    codes = set(card["proposal_eligible_error_codes"])
    assert {"data_quality_failure", "outcome_not_observable", "horizon_mismatch"} <= codes
    assert card["recommended_next_action"]["action"] == (
        "repair_or_reconstruct_outcome_evidence_before_judging_the_hypothesis"
    )
