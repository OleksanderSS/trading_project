from __future__ import annotations

import hashlib
import json

from dean_os.replay_outcome_evidence_plan import ReplayOutcomeEvidencePlanBuilder


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_preserves_gap_sources_and_keeps_price_secondary(tmp_path) -> None:
    gap = _write(
        tmp_path / "gap.json",
        {
            "gap_reviews": [
                {
                    "gap_id": "g1",
                    "resolution_status": "missing",
                    "expected_source_type": "company_filing",
                    "description": "capex guidance",
                    "supporting_evidence": [],
                    "limitations": ["not yet filed"],
                }
            ]
        },
    )
    packet = _write(
        tmp_path / "packet.json",
        {
            "run_id": "packet-1",
            "contract": "dean_world_model_event_learning_packet_v1",
            "source_lineage": {
                "gap_review": {"path": str(gap), "sha256": _sha(gap)}
            },
            "replay_tasks": [
                {
                    "task_id": "t1",
                    "hypothesis_id": "h1",
                    "horizon_days": 30,
                    "as_of": "2026-07-01T00:00:00+00:00",
                    "due_at": "2026-07-31T00:00:00+00:00",
                    "linked_gap_ids": ["g1"],
                    "expected_observations": ["guidance rises"],
                    "invalidation_signals": ["guidance cut"],
                }
            ],
        },
    )
    routing = _write(
        tmp_path / "routing.json",
        {
            "contract": "dean_replay_evaluation_routing_v1",
            "source_packet": {"run_id": "packet-1"},
            "routes": [
                {
                    "task_id": "t1",
                    "route": "hypothesis_outcome_replay",
                    "evaluation_status": "waiting",
                }
            ],
        },
    )

    payload = ReplayOutcomeEvidencePlanBuilder(tmp_path / "out").build(
        packet, routing, save=False
    )
    plan = payload["task_plans"][0]

    assert payload["summary"]["task_plan_count"] == 1
    assert payload["summary"]["unique_gap_count"] == 1
    assert payload["summary"]["unique_gap_status_counts"] == {"missing": 1}
    assert payload["summary"]["outcome_evaluation_can_run"] is False
    assert plan["evidence_lanes"][0]["expected_source_type"] == "company_filing"
    assert plan["evidence_lanes"][0]["collection_route"]["status"] == (
        "route_available_metric_gap_open"
    )
    assert payload["summary"]["collection_route_status_counts"] == {
        "route_available_metric_gap_open": 1
    }
    assert payload["summary"]["voi_status_counts"] == {"unassessed": 1}
    assert payload["summary"]["voi_scored_gap_count"] == 0
    assert plan["evidence_lanes"][0]["value_of_information"]["triage_score"] is None
    assert plan["secondary_market_context"]["role"] == "context_only"
    assert plan["secondary_market_context"]["event_study_allowed"] is False
    assert plan["checkpoints"]["pre_due_source_review"] == "2026-07-24T00:00:00+00:00"
