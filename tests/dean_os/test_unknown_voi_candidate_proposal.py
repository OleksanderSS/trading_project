import hashlib
import json

from dean_os.unknown_voi_candidate_proposal import UnknownValueOfInformationCandidateProposalBuilder


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_selects_small_multi_hypothesis_review_set_without_scores(tmp_path):
    plan = _write(tmp_path / "plan.json", {
        "contract": "dean_replay_outcome_evidence_plan_v1",
        "task_plans": [
            {"hypothesis_id": "h1", "horizon_days": 30, "evidence_lanes": [
                {"gap_id": "multi", "description": "orders", "resolution_status": "missing", "expected_source_type": "industry_data", "collection_route": {"status": "dedicated_collector_missing"}},
                {"gap_id": "single", "description": "roi", "resolution_status": "missing", "expected_source_type": "industry_report", "collection_route": {"status": "intake_path_available_source_refresh_required"}},
            ]},
            {"hypothesis_id": "h2", "horizon_days": 90, "evidence_lanes": [
                {"gap_id": "multi", "description": "orders", "resolution_status": "missing", "expected_source_type": "industry_data", "collection_route": {"status": "dedicated_collector_missing"}},
            ]},
        ],
    })
    review = _write(tmp_path / "review.json", {
        "contract": "dean_unknown_voi_review_v1",
        "inputs": {"evidence_plan": {"sha256": _sha(plan)}},
        "gap_reviews": [
            {"gap_id": "multi", "assessment": {"triage_score": None}},
            {"gap_id": "single", "assessment": {"triage_score": None}},
        ],
    })
    payload = UnknownValueOfInformationCandidateProposalBuilder(tmp_path / "out").build(
        plan, review, max_candidates=1, save=False
    )
    assert payload["candidates"][0]["gap_id"] == "multi"
    assert payload["candidates"][0]["triage_score"] is None
    assert payload["safety"]["numeric_voi_values_inferred"] is False


def test_scored_gap_is_not_proposed_again(tmp_path):
    plan = _write(tmp_path / "plan.json", {
        "contract": "dean_replay_outcome_evidence_plan_v1",
        "task_plans": [{"hypothesis_id": "h1", "horizon_days": 30, "evidence_lanes": [
            {"gap_id": "g1", "description": "orders", "resolution_status": "missing", "collection_route": {"status": "route_available_metric_gap_open"}}
        ]}],
    })
    review = _write(tmp_path / "review.json", {
        "contract": "dean_unknown_voi_review_v1",
        "inputs": {"evidence_plan": {"sha256": _sha(plan)}},
        "gap_reviews": [{"gap_id": "g1", "assessment": {"triage_score": 0.8}}],
    })
    payload = UnknownValueOfInformationCandidateProposalBuilder().build(plan, review, save=False)
    assert payload["candidates"] == []
