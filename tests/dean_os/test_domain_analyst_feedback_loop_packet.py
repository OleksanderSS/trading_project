from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.domain_analyst_feedback_loop_packet import DomainAnalystFeedbackLoopPacket


def test_feedback_loop_packet_waits_for_manual_feedback(tmp_path):
    paths = _fixture_paths(tmp_path)

    payload = DomainAnalystFeedbackLoopPacket(tmp_path / "reports").build(
        case_registry_json=paths["case_registry"],
        forecast_review_json=paths["forecast_review"],
        profile_policy_json=paths["profile_policy"],
        template_decision_json=paths["template_decision"],
        save=False,
    )

    assert payload["summary"]["packet_status"] == "domain_analyst_feedback_loop_ready_pending_manual_feedback"
    assert payload["summary"]["domain_id"] == "semiconductor_ai_infrastructure"
    assert payload["summary"]["feedback_target_count"] == 4
    assert payload["summary"]["manual_feedback_record_count"] == 0
    assert payload["summary"]["learning_candidate_proposal_count"] == 0
    assert payload["summary"]["can_capture_manual_feedback"] is True
    assert payload["summary"]["can_create_learning_candidate_proposals"] is True
    assert payload["summary"]["can_apply_learning"] is False
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_create_analyst_research_recommendation"] is True
    assert payload["summary"]["can_create_execution_recommendation"] is False
    assert payload["summary"]["can_trade"] is False
    assert "correct_but_lucky_or_wrong_reason" in payload["review_label_taxonomy"]["labels"]["causal_quality"]
    assert "approved" in payload["review_label_taxonomy"]["labels"]["process_review"]
    assert any(check["code"] == "manual_feedback_not_supplied" and check["status"] == "warn" for check in payload["review_checks"])
    assert any(item["source_file"] == "FEEDBACK_TO_LEARNING_PIPELINE_TEMPLATE.yaml" for item in payload["after_385_harvest_decisions"])


def test_feedback_loop_packet_creates_proposal_only_learning_candidate_from_valid_feedback(tmp_path):
    paths = _fixture_paths(tmp_path)
    feedback_path = _write_json(
        tmp_path / "manual_feedback.json",
        [
            {
                "feedback_id": "fb_lucky_hit_1",
                "target_id": "case:semiconductor_ai_infrastructure:ai_capex_cycle",
                "reviewer": "operator",
                "review_type": "outcome_reasoning",
                "severity": "high",
                "labels": ["correct_but_lucky_or_wrong_reason", "missed_counterforce"],
                "proposed_learning_actions": ["create_eval_case", "request_more_evidence"],
                "notes": "Direction may have been right, but causal chain needs stricter counterforce review.",
            }
        ],
    )

    payload = DomainAnalystFeedbackLoopPacket(tmp_path / "reports").build(
        case_registry_json=paths["case_registry"],
        forecast_review_json=paths["forecast_review"],
        profile_policy_json=paths["profile_policy"],
        template_decision_json=paths["template_decision"],
        manual_feedback_json=feedback_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "domain_analyst_feedback_loop_ready_with_feedback_candidates"
    assert payload["summary"]["manual_feedback_record_count"] == 1
    assert payload["summary"]["learning_candidate_proposal_count"] == 1
    assert payload["summary"]["can_apply_learning"] is False
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_trade"] is False

    candidate = payload["learning_candidate_proposals"][0]
    assert candidate["candidate_id"] == "learning_candidate:fb_lucky_hit_1"
    assert candidate["promotion_status"] == "proposal_only_pending_human_approval"
    assert candidate["can_apply_now"] is False
    assert set(candidate["proposed_actions"]) == {"create_eval_case", "request_more_evidence"}
    assert any(check["code"] == "manual_feedback_records_valid" and check["status"] == "pass" for check in payload["review_checks"])


def test_feedback_loop_packet_blocks_unsafe_or_unknown_feedback(tmp_path):
    paths = _fixture_paths(tmp_path)
    feedback_path = _write_json(
        tmp_path / "manual_feedback.json",
        {
            "feedback_records": [
                {
                    "feedback_id": "fb_bad_1",
                    "target_id": "unknown_target",
                    "labels": ["not_a_known_label"],
                    "proposed_learning_actions": ["create_eval_case"],
                    "requests_execution": True,
                }
            ]
        },
    )

    payload = DomainAnalystFeedbackLoopPacket(tmp_path / "reports").build(
        case_registry_json=paths["case_registry"],
        forecast_review_json=paths["forecast_review"],
        profile_policy_json=paths["profile_policy"],
        template_decision_json=paths["template_decision"],
        manual_feedback_json=feedback_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "domain_analyst_feedback_loop_blocked"
    assert payload["summary"]["learning_candidate_proposal_count"] == 0
    record = payload["manual_feedback_records"][0]
    assert record["can_be_learning_candidate"] is False
    assert "unknown_target_id" in record["blockers"]
    assert "unknown_labels:not_a_known_label" in record["blockers"]
    assert "requests_execution" in record["blockers"]
    assert any(check["code"] == "manual_feedback_records_valid" and check["status"] == "fail" for check in payload["review_checks"])


def test_feedback_loop_packet_saves_markdown_and_cli_runs(tmp_path):
    paths = _fixture_paths(tmp_path)
    payload = DomainAnalystFeedbackLoopPacket(tmp_path / "reports").build(
        case_registry_json=paths["case_registry"],
        forecast_review_json=paths["forecast_review"],
        profile_policy_json=paths["profile_policy"],
        template_decision_json=paths["template_decision"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Packet status: `domain_analyst_feedback_loop_ready_pending_manual_feedback`" in markdown
    assert "Can apply learning: False" in markdown
    assert "Can create execution recommendation: False" in markdown
    assert "FEEDBACK_TO_LEARNING_PIPELINE_TEMPLATE.yaml" in json.dumps(payload["after_385_harvest_decisions"])
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_feedback_loop_packet.py"),
            "--case-registry-json",
            str(paths["case_registry"]),
            "--forecast-review-json",
            str(paths["forecast_review"]),
            "--profile-policy-json",
            str(paths["profile_policy"]),
            "--template-decision-json",
            str(paths["template_decision"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Packet status: domain_analyst_feedback_loop_ready_pending_manual_feedback" in result.stdout
    assert "Can apply learning: False" in result.stdout
    assert "Can create execution recommendation: False" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "case_registry": _write_json(tmp_path / "case_registry.json", _case_registry_fixture()),
        "forecast_review": _write_json(tmp_path / "forecast_review.json", _forecast_review_fixture()),
        "profile_policy": _write_json(tmp_path / "profile_policy.json", _profile_policy_fixture()),
        "template_decision": _write_json(tmp_path / "template_decision.json", _template_decision_fixture()),
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _case_registry_fixture() -> dict:
    return {
        "mode": "domain_analyst_case_registry_packet",
        "summary": {
            "registry_status": "case_registry_ready_pending_outcomes",
            "domain_id": "semiconductor_ai_infrastructure",
            "expectation_case_count": 1,
        },
        "case_entries": [
            {
                "case_id": "case:semiconductor_ai_infrastructure:ai_capex_cycle",
                "case_type": "forecast_expectation_case",
                "domain_id": "semiconductor_ai_infrastructure",
                "outcome_bucket": "pending_expectation_outcome",
                "allowed_future_labels": ["correct_for_stated_reasons", "correct_but_lucky_or_wrong_reason"],
                "allowed_review_outputs": ["operator_rationale", "self_improvement_proposal"],
            }
        ],
    }


def _forecast_review_fixture() -> dict:
    return {
        "mode": "domain_analyst_forecast_review_packet",
        "summary": {
            "packet_status": "forecast_review_ready_with_cautions_pending_outcomes",
            "domain_id": "semiconductor_ai_infrastructure",
        },
        "outcome_taxonomy": [
            {"bucket_id": "hit_correct_reason"},
            {"bucket_id": "hit_lucky_or_wrong_reason"},
            {"bucket_id": "miss_wrong_reason"},
        ],
        "forecast_candidates": [
            {
                "expectation_id": "expectation:semiconductor_ai_infrastructure:ai_capex_cycle",
                "domain_id": "semiconductor_ai_infrastructure",
                "allowed_future_labels": ["hit_correct_reason", "hit_lucky_or_wrong_reason", "miss_wrong_reason"],
                "allowed_review_outputs": ["analyst_research_recommendation", "self_improvement_proposal"],
            }
        ],
    }


def _profile_policy_fixture() -> dict:
    return {
        "mode": "domain_analyst_profile_policy_packet",
        "summary": {"packet_status": "domain_profile_policy_packet_ready"},
        "profile_policy_reviews": [
            {
                "domain_id": "semiconductor_ai_infrastructure",
                "checks": [
                    {"status": "pass", "code": "feedback_issue_types_present", "message": "ok"},
                    {"status": "pass", "code": "feedback_severity_labels_present", "message": "ok"},
                ],
            }
        ],
    }


def _template_decision_fixture() -> dict:
    return {
        "mode": "domain_analyst_template_decision_packet",
        "summary": {
            "decision_status": "manual_template_decision_pending",
            "decision": "pending_review",
            "domain_id": "semiconductor_ai_infrastructure",
            "can_create_analyst_research_recommendation": True,
            "can_create_execution_recommendation": False,
            "can_trade": False,
        },
    }
