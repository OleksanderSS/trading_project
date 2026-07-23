from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.analyst_core.domain_analyst_portability_review import DomainAnalystPortabilityReview


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _vertical_slice(**summary_overrides) -> dict:
    summary = {
        "run_status": "domain_analyst_candidate_complete_pending_manual_acceptance",
        "domain_id": "semiconductor_ai_infrastructure",
        "template_candidate_status": "ready_for_manual_template_acceptance",
        "document_count": 260,
        "evidence_item_count": 200,
        "regime_scenario_status": "domain_analyst_regime_scenario_ready_with_review_items",
        "scenario_node_count": 29,
        "scenario_probability_mass_valid": True,
        "scenario_evidence_gap_count": 4,
        "manual_acceptance_required": True,
        "can_mark_template_accepted_now": False,
        "can_scale_to_other_domains_now": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "mode": "domain_analyst_vertical_slice_run",
        "summary": summary,
        "inputs": {
            "news_data_paths": ["data/processed/features/news_data.parquet"],
            "macro_data_paths": ["data/processed/features/macro_data.parquet"],
            "materials_paths": [],
        },
        "synthetic_fixture_audit": {
            "has_synthetic_marker": False,
            "has_fixture_marker": False,
            "has_smoke_label": False,
        },
        "artifact_paths": {
            "regime_scenario_json": "reports/dean_os/domain_analyst_vertical_slice_current/regime_scenario/latest.json",
            "forecast_review_json": "reports/dean_os/domain_analyst_vertical_slice_current/forecast_review/latest.json",
        },
    }


def _architecture() -> dict:
    return {
        "architecture_version": "test",
        "summary": {
            "can_clone_domain_profiles_now": False,
            "can_trade": False,
        },
    }


def test_portability_review_marks_profiles_portable_but_keeps_cloning_blocked(tmp_path):
    vertical = _write_json(tmp_path / "vertical" / "latest.json", _vertical_slice())
    architecture = _write_json(tmp_path / "architecture" / "latest.json", _architecture())

    payload = DomainAnalystPortabilityReview(tmp_path / "reports").build(
        vertical_slice_json=vertical,
        architecture_map_json=architecture,
        save=False,
    )

    assert payload["summary"]["review_status"] == "domain_analyst_portability_review_ready"
    assert payload["summary"]["profile_count"] >= 5
    assert payload["summary"]["blocked_profile_ids"] == []
    assert payload["summary"]["can_clone_domain_profiles_now"] is False
    assert payload["summary"]["can_wire_gpt_as_optional_adapter_later"] is True
    assert payload["summary"]["can_wire_local_finbert_as_optional_adapter_later"] is True
    assert payload["summary"]["can_trade"] is False
    assert all(not item["missing_evidence_aliases"] for item in payload["profile_reviews"])
    assert all(item["source_registry_policy_id"] == "default_domain_source_registry_policy_v1" for item in payload["profile_reviews"])
    assert all(item["evidence_scoring_policy_id"] == "default_domain_evidence_scoring_policy_v1" for item in payload["profile_reviews"])
    slot_ids = {item["slot_id"] for item in payload["reusable_template_slots"]["slots"]}
    assert {"source_registry_policy", "ingestion_filter_policy", "evidence_scoring_policy", "review_output_policy", "feedback_label_policy"}.issubset(slot_ids)
    assert {"regime_context_vector", "news_vs_regime_assessments", "scenario_outcome_graph", "evidence_gap_priorities", "self_check_horizons"}.issubset(slot_ids)
    assert payload["summary"]["context_analysis_slot_count"] == 5
    assert payload["summary"]["regime_scenario_context_portable"] is True
    assert payload["reusable_template_slots"]["context_analysis_source_status"]["scenario_probability_mass_valid"] is True
    assert any(item == "DomainAnalystForecastReviewPacket" for item in payload["reusable_template_slots"]["fixed_non_portable_contract"])
    assert any(item == "DomainAnalystRegimeScenarioPacket" for item in payload["reusable_template_slots"]["fixed_non_portable_contract"])
    assert any(check["code"] == "forecast_review_available_for_learning_trace" and check["status"] == "pass" for check in payload["review_checks"])
    assert any(check["code"] == "portable_context_analysis_slots_present" and check["status"] == "pass" for check in payload["review_checks"])
    assert any(check["code"] == "source_regime_scenario_context_reviewable" and check["status"] == "pass" for check in payload["review_checks"])
    assert any(check["code"] == "source_registry_policy_present" and check["status"] == "pass" for item in payload["profile_reviews"] for check in item["checks"])


def test_portability_review_keeps_gpt_and_finbert_optional(tmp_path):
    vertical = _write_json(tmp_path / "vertical" / "latest.json", _vertical_slice())
    architecture = _write_json(tmp_path / "architecture" / "latest.json", _architecture())

    payload = DomainAnalystPortabilityReview(tmp_path / "reports").build(
        vertical_slice_json=vertical,
        architecture_map_json=architecture,
        save=False,
    )

    adapter = payload["optional_enrichment_adapter_contract"]
    assert adapter["gpt_required_for_mvp"] is False
    assert adapter["finbert_required_for_mvp"] is False
    assert "local_files_only=True" in adapter["finbert_current_path"]
    assert any(check["code"] == "gpt_not_required_for_mvp" and check["status"] == "pass" for check in payload["review_checks"])
    assert any(check["code"] == "finbert_not_required_for_mvp" and check["status"] == "pass" for check in payload["review_checks"])


def test_portability_review_blocks_auto_clone_if_vertical_slice_allows_scaling(tmp_path):
    vertical = _write_json(tmp_path / "vertical" / "latest.json", _vertical_slice(can_scale_to_other_domains_now=True))
    architecture = _write_json(tmp_path / "architecture" / "latest.json", _architecture())

    payload = DomainAnalystPortabilityReview(tmp_path / "reports").build(
        vertical_slice_json=vertical,
        architecture_map_json=architecture,
        save=False,
    )

    assert payload["summary"]["review_status"] == "domain_analyst_portability_blocked"
    assert any(check["code"] == "domain_cloning_disabled_until_manual_acceptance" and check["status"] == "fail" for check in payload["review_checks"])
    assert payload["summary"]["can_clone_domain_profiles_now"] is False


def test_portability_review_saves_markdown_and_cli_runs(tmp_path):
    vertical = _write_json(tmp_path / "vertical" / "latest.json", _vertical_slice())
    architecture = _write_json(tmp_path / "architecture" / "latest.json", _architecture())

    payload = DomainAnalystPortabilityReview(tmp_path / "reports").build(
        vertical_slice_json=vertical,
        architecture_map_json=architecture,
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "GPT required now: False" in markdown
    assert "Can clone domains now: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_portability_review.py"),
            "--vertical-slice-json",
            str(vertical),
            "--architecture-map-json",
            str(architecture),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Review status: domain_analyst_portability_review_ready" in result.stdout
    assert "Can clone now: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
