from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.analyst_core.domain_analyst_template_standardization_packet import DomainAnalystTemplateStandardizationPacket


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _domain_instance(**summary_overrides) -> dict:
    summary = {
        "instance_status": "domain_analyst_instance_review_ready",
        "domain_id": "semiconductor_ai_infrastructure",
        "sectors": ["semiconductor"],
        "document_count": 144,
        "evidence_item_count": 144,
        "can_reuse_as_template_after_manual_review": True,
        "manual_acceptance_required": True,
        "can_scale_to_other_domains_now": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "domain_instance_fixture",
        "mode": "domain_analyst_instance_contract",
        "summary": summary,
        "portable_template_slots": {
            "domain_id": "semiconductor_ai_infrastructure",
            "sectors": ["semiconductor"],
            "sector_keywords": ["semiconductor", "AI accelerator", "advanced packaging"],
            "required_evidence_types": [
                "sector_demand",
                "capex_cycle",
                "supply_chain",
                "policy_or_geopolitical",
                "market_confirmation",
            ],
            "useful_evidence_types": ["hyperscaler_capex", "inventory_cycle"],
            "ticker_universe_hint": ["AMD", "NVDA", "TSM"],
            "source_registry_policy": {"policy_id": "default_domain_source_registry_policy_v1"},
            "ingestion_filter_policy": {"policy_id": "default_domain_ingestion_filter_policy_v1"},
            "evidence_scoring_policy": {"policy_id": "default_domain_evidence_scoring_policy_v1"},
            "review_output_policy": {
                "policy_id": "default_domain_review_output_policy_v1",
                "allowed_review_outputs": ["research_recommendation", "self_improvement_proposal"],
            },
            "feedback_label_policy": {"policy_id": "default_domain_feedback_label_policy_v1"},
            "portable_rule": "Replace domain-specific fields; keep gates and non-actions unchanged.",
        },
        "fixed_contract_sequence": ["source gate", "domain intake", "thesis review"],
    }


def _thesis_review(**summary_overrides) -> dict:
    summary = {
        "packet_status": "domain_thesis_review_ready",
        "domain_id": "semiconductor_ai_infrastructure",
        "sectors": ["semiconductor"],
        "thesis_stance": "mixed",
        "expected_direction": "mixed",
        "confidence": 0.68,
        "evidence_item_count": 144,
        "ticker_direct_count": 0,
        "manual_review_required": True,
        "can_enter_manual_thesis_review": True,
        "can_standardize_domain_template_after_manual_review": True,
        "can_prepare_separate_ticker_bridge_after_manual_review": True,
        "can_create_direct_ticker_thesis_without_bridge": False,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "domain_thesis_review_fixture",
        "mode": "domain_analyst_thesis_review_packet",
        "summary": summary,
        "regime_scenario_context": {
            "available": True,
            "packet_status": "domain_analyst_regime_scenario_ready_with_review_items",
            "source_run_id": "regime_scenario_fixture",
            "active_regime_fields": [
                {
                    "field": "ai_tech_cycle",
                    "state": "capex_boom",
                    "intensity": 0.8,
                    "trend": "rising",
                    "confidence": "high",
                    "evidence_ids": ["event_1"],
                }
            ],
            "scenario_probabilities": {
                "upside_acceleration": 0.3,
                "base_case_continuation": 0.45,
                "downside_reset": 0.25,
            },
            "probability_mass_valid": True,
            "top_evidence_gaps": [{"gap_id": "gap_1", "priority": "high", "description": "Validate capex persistence."}],
            "self_check_horizons": ["1d", "5d", "20d", "60d", "120d"],
        },
    }


def _regime_scenario() -> dict:
    return {
        "run_id": "regime_scenario_fixture",
        "mode": "domain_analyst_regime_scenario_packet",
        "summary": {
            "packet_status": "domain_analyst_regime_scenario_ready_with_review_items",
            "probability_mass_valid": True,
            "can_create_execution_recommendation": False,
            "can_trade": False,
        },
        "regime_context_vector": {
            "fields": {
                "ai_tech_cycle": {
                    "state": "capex_boom",
                    "intensity": 0.9,
                    "trend": "rising",
                    "confidence": "high",
                    "evidence_ids": ["event_1"],
                },
                "market_state": {
                    "state": "crowded_theme",
                    "intensity": 0.5,
                    "trend": "stable",
                    "confidence": "medium",
                    "evidence_ids": ["event_2"],
                },
            }
        },
        "scenario_outcome_graph": {
            "scenario_probabilities": {
                "upside_acceleration": 0.25,
                "base_case_continuation": 0.5,
                "downside_reset": 0.25,
            },
            "probability_mass_check": {"sum": 1.0, "valid": True},
            "horizons": ["1d", "5d", "20d", "60d", "120d"],
        },
        "evidence_gap_priorities": [{"gap_id": "gap_1", "priority": "high", "description": "Validate demand breadth."}],
    }


def _architecture() -> dict:
    return {
        "mode": "current_architecture_map",
        "summary": {
            "can_clone_domain_profiles_now": False,
            "can_write_production_config_now": False,
            "can_trade": False,
        },
    }


def _write_inputs(
    tmp_path: Path,
    *,
    instance: dict | None = None,
    thesis: dict | None = None,
    regime: dict | None = None,
) -> dict[str, Path]:
    return {
        "instance": _write_json(tmp_path / "instance" / "latest.json", instance or _domain_instance()),
        "thesis": _write_json(tmp_path / "thesis" / "latest.json", thesis or _thesis_review()),
        "regime": _write_json(tmp_path / "regime" / "latest.json", regime or _regime_scenario()),
        "architecture": _write_json(tmp_path / "architecture" / "latest.json", _architecture()),
    }


def test_template_standardization_packet_marks_candidate_ready_for_manual_acceptance(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystTemplateStandardizationPacket(tmp_path / "reports").build(
        domain_instance_contract_json=paths["instance"],
        domain_thesis_review_json=paths["thesis"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["candidate_status"] == "ready_for_manual_template_acceptance"
    assert payload["summary"]["manual_acceptance_required"] is True
    assert payload["summary"]["can_mark_template_accepted_now"] is False
    assert payload["summary"]["can_standardize_domain_template_after_manual_acceptance"] is True
    assert payload["summary"]["can_prepare_sector_to_ticker_bridge_after_manual_acceptance"] is True
    assert payload["summary"]["can_run_sector_to_ticker_bridge_now"] is False
    assert payload["summary"]["can_scale_to_other_domains_now"] is False
    assert payload["summary"]["regime_scenario_context_available"] is True
    assert payload["summary"]["scenario_probability_mass_valid"] is True
    assert payload["summary"]["self_check_horizon_count"] == 5
    assert payload["summary"]["can_trade"] is False
    assert payload["template_scope"]["required_evidence_types"]
    assert payload["template_scope"]["regime_scenario_context"]["available"] is True
    assert any(slot["slot_id"] == "scenario_outcome_graph" for slot in payload["template_scope"]["portable_context_analysis_slots"])
    assert payload["template_scope"]["source_registry_policy"]["policy_id"] == "default_domain_source_registry_policy_v1"
    assert payload["template_scope"]["review_output_policy"]["policy_id"] == "default_domain_review_output_policy_v1"
    assert any(check["code"] == "no_auto_template_acceptance" and check["status"] == "pass" for check in payload["review_checks"])
    assert any(check["code"] == "regime_scenario_context_available_for_template" and check["status"] == "pass" for check in payload["review_checks"])


def test_template_standardization_packet_accepts_explicit_regime_scenario_artifact(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystTemplateStandardizationPacket(tmp_path / "reports").build(
        domain_instance_contract_json=paths["instance"],
        domain_thesis_review_json=paths["thesis"],
        regime_scenario_json=paths["regime"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["regime_scenario_context_source"] == "regime_scenario_json"
    assert payload["summary"]["active_regime_field_count"] == 2
    assert payload["summary"]["self_check_horizon_count"] == 5
    assert any(check["code"] == "regime_scenario_artifact_type" and check["status"] == "pass" for check in payload["review_checks"])
    assert any(check["code"] == "regime_scenario_no_execution_recommendation" and check["status"] == "pass" for check in payload["review_checks"])


def test_template_standardization_packet_blocks_unsafe_thesis_review(tmp_path):
    paths = _write_inputs(tmp_path, thesis=_thesis_review(can_trade=True))

    payload = DomainAnalystTemplateStandardizationPacket(tmp_path / "reports").build(
        domain_instance_contract_json=paths["instance"],
        domain_thesis_review_json=paths["thesis"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["candidate_status"] == "blocked_template_standardization"
    assert payload["summary"]["can_standardize_domain_template_after_manual_acceptance"] is False
    assert any(check["code"] == "thesis_no_trading" and check["status"] == "fail" for check in payload["review_checks"])


def test_template_standardization_packet_needs_more_review_when_thesis_not_ready(tmp_path):
    thesis = _thesis_review(
        packet_status="domain_thesis_review_needs_more_evidence",
        can_standardize_domain_template_after_manual_review=False,
        can_prepare_separate_ticker_bridge_after_manual_review=False,
    )
    paths = _write_inputs(tmp_path, thesis=thesis)

    payload = DomainAnalystTemplateStandardizationPacket(tmp_path / "reports").build(
        domain_instance_contract_json=paths["instance"],
        domain_thesis_review_json=paths["thesis"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["candidate_status"] == "needs_more_template_review"
    assert payload["summary"]["can_standardize_domain_template_after_manual_acceptance"] is False
    assert any(check["code"] == "domain_thesis_review_needs_more_review" for check in payload["review_checks"])


def test_template_standardization_packet_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystTemplateStandardizationPacket(tmp_path / "reports").build(
        domain_instance_contract_json=paths["instance"],
        domain_thesis_review_json=paths["thesis"],
        architecture_map_json=paths["architecture"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can mark template accepted now: False" in markdown
    assert "Evidence scoring policy: `default_domain_evidence_scoring_policy_v1`" in markdown
    assert "Can trade: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_template_standardization_packet.py"),
            "--domain-instance-contract-json",
            str(paths["instance"]),
            "--domain-thesis-review-json",
            str(paths["thesis"]),
            "--regime-scenario-json",
            str(paths["regime"]),
            "--architecture-map-json",
            str(paths["architecture"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Candidate status: ready_for_manual_template_acceptance" in result.stdout
    assert "Regime/scenario context: True" in result.stdout
    assert "Can mark template accepted now: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
