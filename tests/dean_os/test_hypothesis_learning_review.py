from __future__ import annotations

import hashlib
import json

from dean_os.hypothesis_learning_review import HypothesisLearningReview


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_learning_review_diagnoses_error_but_does_not_promote_rule(tmp_path):
    packet_path = tmp_path / "packet.json"
    gate_path = tmp_path / "gate.json"
    packet = {
        "run_id": "world_1",
        "contract": "dean_world_model_event_learning_v1",
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "hypothesis": "Capex growth will persist",
            }
        ],
    }
    _write(packet_path, packet)
    gate = {
        "run_id": "gate_1",
        "created_at": "2026-07-13T10:05:00+00:00",
        "source_packet": {"run_id": "world_1", "sha256": _sha(packet_path)},
        "hypothesis_review": [
            {
                "hypothesis_id": "h1",
                "hypothesis": "Capex growth will persist",
                "disposition": "reformulate",
                "rationale": "The trigger questions capex sustainability.",
                "proposed_hypothesis": "Capex expectations may weaken.",
                "expectation_context_available": False,
                "source_assessment": "credible_context_source_but_trigger_polarity_conflicts_with_generated_claim",
                "trigger_event": {"title": "Questions on capex"},
            }
        ],
    }
    _write(gate_path, gate)

    result = HypothesisLearningReview(tmp_path / "reports").build(
        packet_path,
        gate_path,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert result["summary"]["learning_candidate_count"] == 1
    assert result["summary"]["learning_proposal_count"] == 1
    case = result["hypothesis_cases"][0]
    assert "trigger_polarity_mismatch" in case["error_codes"]
    proposal = result["learning_proposals"][0]
    assert proposal["current_independent_case_count"] == 1
    assert proposal["minimum_independent_case_count"] == 3
    assert proposal["promotion_status"] == "collect_more_independent_reviewed_cases"
    assert proposal["recommended_action"]
    assert proposal["fallback_action"]
    assert proposal["verification_requirements"]
    assert proposal["production_rule_update_performed"] is False


def test_falsified_outcome_without_root_cause_is_not_used_to_rewrite_rules(tmp_path):
    packet_path = tmp_path / "packet.json"
    gate_path = tmp_path / "gate.json"
    outcome_path = tmp_path / "outcomes.json"
    packet = {
        "run_id": "world_2",
        "hypotheses": [{"hypothesis_id": "h2", "hypothesis": "Demand rises"}],
    }
    _write(packet_path, packet)
    gate = {
        "run_id": "gate_2",
        "created_at": "2026-07-13T10:05:00+00:00",
        "source_packet": {"run_id": "world_2", "sha256": _sha(packet_path)},
        "hypothesis_review": [
            {
                "hypothesis_id": "h2",
                "hypothesis": "Demand rises",
                "disposition": "accept_for_replay",
                "source_assessment": "coherent_trigger",
            }
        ],
    }
    _write(gate_path, gate)
    _write(
        outcome_path,
        {
            "outcomes": [
                {"outcome_id": "o2", "hypothesis_id": "h2", "result_label": "falsified"}
            ]
        },
    )

    result = HypothesisLearningReview(tmp_path / "reports").build(
        packet_path,
        gate_path,
        outcome_json=outcome_path,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert result["summary"]["unresolved_root_cause_count"] == 1
    assert result["summary"]["learning_proposal_count"] == 0
    assert result["hypothesis_cases"][0]["error_codes"] == [
        "unknown_falsification_cause"
    ]


def test_structured_outcome_can_generate_machine_proposals_but_not_apply_them(tmp_path):
    packet_path = tmp_path / "packet.json"
    gate_path = tmp_path / "gate.json"
    outcome_path = tmp_path / "outcomes.json"
    packet = {
        "run_id": "world_3",
        "hypotheses": [
            {
                "hypothesis_id": "h3",
                "hypothesis": "Fundamentals and price rise",
            }
        ],
    }
    _write(packet_path, packet)
    _write(
        gate_path,
        {
            "run_id": "gate_3",
            "created_at": "2026-07-13T10:05:00+00:00",
            "source_packet": {"run_id": "world_3", "sha256": _sha(packet_path)},
            "hypothesis_review": [
                {
                    "hypothesis_id": "h3",
                    "hypothesis": "Fundamentals and price rise",
                    "disposition": "accept_for_replay",
                    "expectation_context_available": False,
                }
            ],
        },
    )
    _write(
        outcome_path,
        {
            "outcomes": [
                {
                    "outcome_id": "o3",
                    "hypothesis_id": "h3",
                    "result_label": "falsified",
                    "fundamental_result": "confirmed",
                    "market_reaction_result": "miss",
                }
            ]
        },
    )

    result = HypothesisLearningReview(tmp_path / "reports").build(
        packet_path,
        gate_path,
        outcome_json=outcome_path,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    codes = {proposal["error_code"] for proposal in result["learning_proposals"]}
    assert "true_hypothesis_wrong_market_reaction" in codes
    assert "priced_in_blindness" in codes
    assert result["summary"]["machine_root_cause_ready_count"] == 1
    assert result["summary"]["production_rule_update_performed"] is False


def test_unobservable_outcome_proposes_measurement_repairs_only(tmp_path):
    packet_path = tmp_path / "packet.json"
    gate_path = tmp_path / "gate.json"
    outcome_path = tmp_path / "outcomes.json"
    packet = {"run_id": "world_4", "hypotheses": [{"hypothesis_id": "h4"}]}
    _write(packet_path, packet)
    _write(
        gate_path,
        {
            "run_id": "gate_4",
            "created_at": "2026-07-13T10:05:00+00:00",
            "source_packet": {"run_id": "world_4", "sha256": _sha(packet_path)},
            "hypothesis_review": [
                {"hypothesis_id": "h4", "disposition": "accept_for_replay"}
            ],
        },
    )
    _write(
        outcome_path,
        {
            "outcomes": [
                {
                    "outcome_id": "o4",
                    "hypothesis_id": "h4",
                    "result_label": "unobservable",
                    "observable": False,
                    "coverage_status": "insufficient",
                    "data_quality_status": "incomplete",
                }
            ]
        },
    )

    result = HypothesisLearningReview(tmp_path / "reports").build(
        packet_path,
        gate_path,
        outcome_json=outcome_path,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    codes = {proposal["error_code"] for proposal in result["learning_proposals"]}
    assert codes == {"data_quality_failure", "outcome_not_observable"}
    assert result["summary"]["production_rule_update_performed"] is False
