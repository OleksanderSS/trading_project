from __future__ import annotations

import json

from dean_os.domain_analyst_binding_planner import DomainAnalystBindingPlanner


def test_energy_plan_creates_six_proposal_only_binding_tasks():
    payload = DomainAnalystBindingPlanner().build(save=False, as_of="2026-07-14T12:00:00Z")

    summary = payload["summary"]
    assert summary["plan_status"] == "binding_plan_ready_pending_artifacts_and_manual_acceptance"
    assert summary["context_family_count"] == 6
    assert summary["unresolved_binding_count"] == 6
    assert summary["collection_task_proposal_count"] == 6
    assert summary["can_execute_collection_tasks"] is False
    assert summary["can_invoke_domain_analysis_now"] is False
    assert summary["can_trade"] is False
    assert all(task["execution_authorized"] is False for task in payload["collection_task_proposals"])
    assert all(task["synthetic_placeholder_allowed"] is False for task in payload["collection_task_proposals"])


def test_cross_domain_candidate_is_rejected(tmp_path):
    path = tmp_path / "wrong-domain.json"
    _write_candidate(path, domain_id="semiconductor_ai_infrastructure")

    payload = DomainAnalystBindingPlanner().build(
        save=False,
        as_of="2026-07-14T12:00:00Z",
        candidate_artifacts={"news": [path]},
    )

    news = _family(payload, "news")
    assert news["status"] == "binding_blocked_invalid_candidates"
    assert "cross_domain_artifact_reuse_forbidden" in news["candidate_reviews"][0]["reasons"]
    assert news["binding_written"] is False


def test_valid_domain_candidate_is_proposed_not_bound(tmp_path):
    path = tmp_path / "energy-news.json"
    _write_candidate(path, domain_id="energy")

    payload = DomainAnalystBindingPlanner().build(
        save=False,
        as_of="2026-07-14T12:00:00Z",
        candidate_artifacts={"news": [path]},
    )

    news = _family(payload, "news")
    assert news["status"] == "reuse_candidate_ready_for_review"
    assert news["proposed_candidate"]["validation_status"] == "valid"
    assert news["proposed_candidate"]["sha256"]
    assert news["binding_written"] is False
    assert payload["summary"]["reuse_candidate_ready_count"] == 1
    assert payload["summary"]["unresolved_binding_count"] == 6


def test_future_candidate_is_rejected(tmp_path):
    path = tmp_path / "future.json"
    _write_candidate(path, domain_id="energy", as_of="2026-07-15T00:00:00Z")

    payload = DomainAnalystBindingPlanner().build(
        save=False,
        as_of="2026-07-14T12:00:00Z",
        candidate_artifacts={"news": [path]},
    )

    news = _family(payload, "news")
    assert "future_artifact_forbidden" in news["candidate_reviews"][0]["reasons"]


def test_unknown_family_blocks_plan(tmp_path):
    path = tmp_path / "unknown.json"
    _write_candidate(path, domain_id="energy")

    payload = DomainAnalystBindingPlanner().build(
        save=False,
        as_of="2026-07-14T12:00:00Z",
        candidate_artifacts={"social_sentiment": [path]},
    )

    assert payload["summary"]["plan_status"] == "binding_plan_blocked_structurally"
    assert "unknown_candidate_context_family" in payload["summary"]["structural_blockers"]


def test_saved_report_is_operator_readable(tmp_path):
    payload = DomainAnalystBindingPlanner(tmp_path / "reports").build(
        as_of="2026-07-14T12:00:00Z"
    )

    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")
    assert "Unresolved bindings: 6" in markdown
    assert "Can execute collection: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")


def _family(payload, family_id):
    return next(item for item in payload["family_plans"] if item["context_family"] == family_id)


def _write_candidate(path, *, domain_id, as_of="2026-07-13T00:00:00Z"):
    path.write_text(
        json.dumps(
            {
                "contract": "dean_domain_scoped_news_envelope_v1",
                "mode": "domain_scoped_news_envelope",
                "domain_id": domain_id,
                "created_at": as_of,
                "inputs": {"as_of": as_of, "domain_id": domain_id},
                "status": "domain_news_candidate_ready_with_gaps",
                "safety": {
                    "review_only": True,
                    "learning_write_performed": False,
                    "production_config_write_performed": False,
                    "broker_access_performed": False,
                    "live_execution_performed": False,
                },
            }
        ),
        encoding="utf-8",
    )
