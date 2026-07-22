from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dean_os.analysts.schemas import DomainProfile
from dean_os.domain_analyst_profile_policy_packet import DomainAnalystProfilePolicyPacket


def test_profile_policy_packet_reviews_all_profiles_as_policy_ready(tmp_path):
    payload = DomainAnalystProfilePolicyPacket(tmp_path / "reports").build(save=False)

    assert payload["summary"]["packet_status"] == "domain_profile_policy_packet_ready"
    assert payload["summary"]["profile_count"] >= 5
    assert payload["summary"]["profiles_policy_ready_count"] == payload["summary"]["profile_count"]
    assert payload["summary"]["blocked_profile_ids"] == []
    assert payload["summary"]["can_support_one_next_domain_clone_candidate_after_manual_acceptance"] is True
    assert payload["summary"]["can_clone_domain_profiles_now"] is False
    assert payload["summary"]["can_create_analyst_research_recommendation"] is True
    assert payload["summary"]["can_create_execution_recommendation"] is False
    assert payload["summary"]["can_trade"] is False

    contract_slots = {item["slot_id"] for item in payload["policy_contract"]["required_policy_slots"]}
    assert {"source_registry_policy", "ingestion_filter_policy", "evidence_scoring_policy", "review_output_policy", "feedback_label_policy"}.issubset(contract_slots)
    assert all(item["policy_status"] == "profile_policy_ready" for item in payload["profile_policy_reviews"])
    assert all("research_recommendation" in item["allowed_review_outputs"] for item in payload["profile_policy_reviews"])
    assert all("buy_sell_hold" in item["blocked_outputs"] for item in payload["profile_policy_reviews"])
    assert any(item["source_file"] == "EVIDENCE_SCORING_TEMPLATE.yaml" and item["classification"] == "adapted_to_default_policy_slot" for item in payload["after_385_harvest_decisions"])


def test_profile_policy_packet_blocks_malformed_profile(monkeypatch, tmp_path):
    bad_profile = DomainProfile(
        domain_id="bad_domain",
        display_name="Bad Domain",
        description="Missing policy internals.",
        required_evidence_types=["market_confirmation"],
        useful_evidence_types=[],
        sector_keywords=["bad"],
        ticker_universe_hint=["BAD"],
        contradiction_rules=[],
        direct_ticker_evidence_rules=["ticker evidence required"],
        blocked_if_missing=["No market confirmation."],
        source_registry_policy={"policy_id": "bad_source_policy"},
        ingestion_filter_policy={"policy_id": "bad_ingestion_policy", "fail_closed_rules": []},
        evidence_scoring_policy={"policy_id": "bad_scoring_policy", "weights": {}},
        review_output_policy={"policy_id": "bad_review_policy", "allowed_review_outputs": [], "blocked_outputs": []},
        feedback_label_policy={"policy_id": "bad_feedback_policy", "issue_types": [], "severity_labels": []},
    )

    monkeypatch.setattr("dean_os.domain_analyst_profile_policy_packet.list_domain_profiles", lambda: ["bad_domain"])
    monkeypatch.setattr("dean_os.domain_analyst_profile_policy_packet.get_domain_profile", lambda domain_id: bad_profile)

    payload = DomainAnalystProfilePolicyPacket(tmp_path / "reports").build(save=False)

    assert payload["summary"]["packet_status"] == "domain_profile_policy_packet_blocked"
    assert payload["summary"]["blocked_profile_ids"] == ["bad_domain"]
    assert payload["summary"]["can_support_one_next_domain_clone_candidate_after_manual_acceptance"] is False
    assert any(check["code"] == "all_profiles_policy_ready" and check["status"] == "fail" for check in payload["review_checks"])
    profile_checks = payload["profile_policy_reviews"][0]["checks"]
    assert any(check["code"] == "source_registry_trust_tiers_present" and check["status"] == "fail" for check in profile_checks)
    assert any(check["code"] == "execution_outputs_blocked" and check["status"] == "fail" for check in profile_checks)


def test_profile_policy_packet_saves_markdown_and_cli_runs(tmp_path):
    payload = DomainAnalystProfilePolicyPacket(tmp_path / "reports").build()
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Packet status: `domain_profile_policy_packet_ready`" in markdown
    assert "Can create analyst research recommendation: True" in markdown
    assert "Can create execution recommendation: False" in markdown
    assert "EVIDENCE_SCORING_TEMPLATE.yaml" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_profile_policy_packet.py"),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Packet status: domain_profile_policy_packet_ready" in result.stdout
    assert "Can clone domain profiles now: False" in result.stdout
    assert "Can create analyst research recommendation: True" in result.stdout
    assert "Can create execution recommendation: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
