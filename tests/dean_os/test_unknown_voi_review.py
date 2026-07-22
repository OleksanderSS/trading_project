import json

import pytest

from dean_os.unknown_voi_review import UnknownValueOfInformationReviewBuilder


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _plan(tmp_path):
    return _write(tmp_path / "plan.json", {
        "contract": "dean_replay_outcome_evidence_plan_v1",
        "task_plans": [{"evidence_lanes": [{
            "gap_id": "g1", "description": "cancelled orders", "resolution_status": "missing",
            "expected_source_type": "industry_data", "collection_route": {"status": "missing"},
            "value_of_information": {"linked_hypothesis_ids": ["h1"]},
        }]}],
    })


def test_unassessed_real_plan_remains_unranked(tmp_path):
    payload = UnknownValueOfInformationReviewBuilder(tmp_path / "out").build(
        _plan(tmp_path), save=False
    )
    assert payload["summary"]["unscored_count"] == 1
    assert payload["validated_collector_ranking"] == []
    assert payload["safety"]["collector_execution_performed"] is False


def test_validated_assessment_produces_review_ranking_only(tmp_path):
    assessments = _write(tmp_path / "assessments.json", {"assessments": [{
        "gap_id": "g1", "status": "validated", "uncertainty_type": "epistemic",
        "scenario_change_potential": 0.9, "confidence_change_potential": 0.7,
        "wrong_conclusion_blocking_value": 0.9, "decision_relevance": 0.8,
        "collection_feasibility": 0.6, "normalized_collection_cost": 0.4,
        "evidence_basis": ["h1 depends on confirmed cancellations"],
        "assessor": "reviewer", "assessed_at": "2026-07-12T00:00:00+00:00",
    }]})
    payload = UnknownValueOfInformationReviewBuilder(tmp_path / "out").build(
        _plan(tmp_path), assessments_path=assessments, save=False
    )
    assert payload["summary"]["validated_scored_count"] == 1
    assert payload["validated_collector_ranking"][0]["gap_id"] == "g1"
    assert payload["summary"]["collector_execution_allowed"] is False


def test_unknown_gap_assessment_is_rejected(tmp_path):
    assessments = _write(tmp_path / "assessments.json", {
        "assessments": [{"gap_id": "not_in_plan"}]
    })
    with pytest.raises(ValueError, match="unknown gap"):
        UnknownValueOfInformationReviewBuilder().build(
            _plan(tmp_path), assessments_path=assessments, save=False
        )
