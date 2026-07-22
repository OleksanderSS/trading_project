from __future__ import annotations

from dean_os.hypothesis_quality_assessment import (
    HYPOTHESIS_OUTCOME_ASSESSMENT_CONTRACT,
    HYPOTHESIS_QUALITY_ASSESSMENT_CONTRACT,
    assess_hypothesis_quality,
    assessment_policy,
)


def _trigger():
    return {
        "evidence_id": "e1",
        "source_id": "https://example.test/official",
        "provenance": {
            "source_tier": "tier_1_core_evidence",
            "source_identity": "official_source",
            "published_at": "2026-07-01T10:00:00+00:00",
        },
    }


def _hypothesis():
    return {
        "hypothesis_id": "h1",
        "as_of": "2026-07-01T10:00:00+00:00",
        "hypothesis": "A named basket will weaken relative to its benchmark.",
        "confidence": 0.8,
        "trigger_evidence_ids": ["e1"],
        "supporting_evidence_ids": [],
        "expected_observations": ["negative relative return"],
        "invalidation_signals": ["non-negative relative return"],
        "horizons_to_check": [1, 5, 20],
        "registration_blockers": [],
        "measurement_spec": {
            "primary_horizon_days": 20,
            "target_metrics": ["basket_relative_return"],
            "assessment_rule": "Compare the basket with its pre-event benchmark baseline.",
            "measurement_context": {
                "baseline": "pre_event_close",
                "basket": {"members": ["AAA", "BBB"]},
            },
        },
    }


def test_quality_score_is_not_a_truth_probability_or_trade_signal():
    hypothesis = _hypothesis()
    trigger = _trigger()

    assessment = assess_hypothesis_quality(
        hypothesis,
        trigger_event=trigger,
        evidence_events=[trigger],
        packet_summary={"expectation_context_available": False},
        alignment={"mechanism": "capex_cycle"},
        replay_tasks=[
            {
                "trigger_event_at": "2026-07-01T10:00:00+00:00",
                "horizon_days": 20,
            }
        ],
    )

    assert assessment["contract"] == HYPOTHESIS_QUALITY_ASSESSMENT_CONTRACT
    assert assessment["confidence_probability"] is None
    assert assessment["directional_trading_signal"] is None
    assert assessment["reported_generator_confidence"] == 0.8
    assert assessment["hypothesis_quality_score"] <= 69
    assert assessment["replay_eligible"] is True
    assert assessment["max_allowed_use"] == "replay_observation_only"
    assert assessment["safety"]["can_trade"] is False


def test_missing_falsification_fails_the_replay_quality_floor():
    hypothesis = _hypothesis()
    hypothesis["invalidation_signals"] = []
    hypothesis["measurement_spec"].pop("assessment_rule")
    trigger = _trigger()

    assessment = assess_hypothesis_quality(
        hypothesis,
        trigger_event=trigger,
        evidence_events=[trigger],
        packet_summary={},
        alignment={},
        replay_tasks=[{"trigger_event_at": "2026-07-01T10:00:00+00:00"}],
    )

    assert assessment["replay_eligible"] is False
    assert "falsifiability_or_measurement_floor_not_met" in assessment[
        "replay_eligibility_blockers"
    ]
    assert assessment["hypothesis_quality_score"] <= 39


def test_outcome_policy_separates_claim_result_from_market_reaction():
    policy = assessment_policy()

    assert policy["outcome_contract"] == HYPOTHESIS_OUTCOME_ASSESSMENT_CONTRACT
    assert "right_thesis_wrong_market_reaction" in policy["outcome_labels"]
    assert "right_market_reaction_wrong_causal_explanation" in policy[
        "outcome_labels"
    ]
    assert "confidence_calibration" in policy["post_outcome_dimensions"]
