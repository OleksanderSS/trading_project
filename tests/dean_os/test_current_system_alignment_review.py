from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.current_system_alignment_review import CurrentSystemAlignmentReview


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evidence_pack_payload() -> dict:
    return {
        "mode": "analyst_evidence_pack",
        "coverage": {
            "document_count": 158,
            "data_quality": "strong",
            "research_ready": True,
            "agent_lab_ready": True,
            "warning_count": 0,
            "dropped_count": 0,
            "by_source_type": {"news": 111, "report": 47},
            "tickers": ["AAPL", "AMD", "MSFT", "NVDA", "TSM"],
            "sectors": ["semiconductor"],
            "date_range": {"start": "2026-02-25T08:00:00+00:00", "end": "2026-05-05T10:15:00+00:00"},
        },
    }


def _source_gate_payload(**summary_overrides) -> dict:
    summary = {
        "gate_status": "source_evidence_ready_for_domain_research",
        "can_enter_domain_research": True,
        "can_promote_to_evidence": False,
        "can_extract_claims_events_entities": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "mode": "source_evidence_validation_gate",
        "summary": summary,
        "decision_guidance": {"pass_count": 321, "warning_count": 0, "fail_count": 0},
    }


def _agent_lab_payload(**summary_overrides) -> dict:
    summary = {
        "latest_thesis": "Cited research is mixed.",
        "learning_record_count": 0,
        "proposal_count": 0,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "agent_lab_fixture",
        "document_count": 158,
        "note_count": 4,
        "learning_records": [],
        "action_proposals": [],
        "summary": summary,
    }


def _dropzone_payload(supported_file_count: int = 0) -> dict:
    return {
        "mode": "real_source_dropzone_inventory",
        "summary": {
            "dropzone_status": "empty_dropzone" if supported_file_count == 0 else "ready",
            "supported_file_count": supported_file_count,
            "can_build_normalized_packet": supported_file_count > 0,
            "can_promote_to_evidence": False,
            "can_trade": False,
        },
    }


def _fundamental_gate_payload() -> dict:
    return {
        "mode": "fundamental_input_readiness_gate",
        "summary": {
            "readiness_status": "fundamental_input_ready_for_manual_review",
            "metric_count": 2,
            "can_feed_value_screening_after_manual_review": True,
            "can_compute_ratios_now": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _domain_analyst_intake_payload() -> dict:
    return {
        "mode": "domain_analyst_intake_packet",
        "summary": {
            "intake_status": "domain_analyst_intake_ready",
            "domain_id": "semiconductor_ai_infrastructure",
            "evidence_item_count": 12,
            "ticker_direct_count": 4,
            "analyst_report_created": True,
            "can_create_direct_ticker_thesis_without_bridge": False,
            "can_write_learning_memory": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _domain_analyst_instance_contract_payload() -> dict:
    return {
        "mode": "domain_analyst_instance_contract",
        "summary": {
            "instance_status": "domain_analyst_instance_review_ready",
            "domain_id": "semiconductor_ai_infrastructure",
            "can_reuse_as_template_after_manual_review": True,
            "manual_acceptance_required": True,
            "can_scale_to_other_domains_now": False,
            "can_write_learning_memory": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _domain_analyst_thesis_review_payload() -> dict:
    return {
        "mode": "domain_analyst_thesis_review_packet",
        "summary": {
            "packet_status": "domain_thesis_review_ready",
            "domain_id": "semiconductor_ai_infrastructure",
            "can_enter_manual_thesis_review": True,
            "can_standardize_domain_template_after_manual_review": True,
            "can_prepare_separate_ticker_bridge_after_manual_review": True,
            "manual_review_required": True,
            "can_create_direct_ticker_thesis_without_bridge": False,
            "can_write_learning_memory": False,
            "can_change_analyst_weights": False,
            "can_write_config": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _domain_analyst_template_standardization_payload() -> dict:
    return {
        "mode": "domain_analyst_template_standardization_packet",
        "summary": {
            "candidate_status": "ready_for_manual_template_acceptance",
            "domain_id": "semiconductor_ai_infrastructure",
            "manual_acceptance_required": True,
            "can_mark_template_accepted_now": False,
            "can_standardize_domain_template_after_manual_acceptance": True,
            "can_prepare_sector_to_ticker_bridge_after_manual_acceptance": True,
            "can_run_sector_to_ticker_bridge_now": False,
            "can_scale_to_other_domains_now": False,
            "can_create_direct_ticker_thesis_without_bridge": False,
            "can_write_learning_memory": False,
            "can_change_analyst_weights": False,
            "can_write_config": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _domain_analyst_case_registry_payload() -> dict:
    return {
        "mode": "domain_analyst_case_registry_packet",
        "summary": {
            "registry_status": "case_registry_ready_pending_outcomes",
            "domain_id": "semiconductor_ai_infrastructure",
            "case_count": 1,
            "source_observation_count": 3,
            "outcome_bucket_counts": {"pending_domain_outcome": 1},
            "can_write_case_registry_artifact": True,
            "can_promote_learning_now": False,
            "can_write_learning_memory": False,
            "can_change_analyst_weights": False,
            "can_write_config": False,
            "can_train_from_hits_only": False,
            "can_drop_miss_cases": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _pipeline_control_instance_contract_payload() -> dict:
    return {
        "mode": "pipeline_control_instance_contract",
        "summary": {
            "instance_status": "pipeline_control_instance_review_ready",
            "blocked_metric_planes": [],
            "caution_metric_planes": [],
            "can_propose_reviewed_experiments_after_manual_review": True,
            "can_run_autonomous_tuning_now": False,
            "can_write_production_config": False,
            "can_write_learning_memory": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _pipeline_control_caution_review_payload() -> dict:
    return {
        "mode": "pipeline_control_caution_review_packet",
        "summary": {
            "caution_review_status": "pipeline_cautions_need_reviewed_inputs",
            "blocked_metric_planes": [],
            "caution_metric_planes": ["risk", "validation", "feature_stability"],
            "missing_evidence_planes": ["risk", "validation", "feature_stability"],
            "can_propose_reviewed_experiments_after_manual_caution_acceptance": True,
            "can_run_autonomous_tuning_now": False,
            "can_write_production_config": False,
            "can_write_learning_memory": False,
            "can_create_recommendation": False,
            "can_trade": False,
        },
    }


def _pipeline_metric_input_readiness_payload(**summary_overrides) -> dict:
    summary = {
        "readiness_status": "metric_inputs_ready",
        "blocked_metric_planes": [],
        "caution_metric_planes": [],
        "can_refresh_pipeline_control_surface_now": True,
        "can_propose_reviewed_tuning_after_surface_and_manual_review": True,
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "mode": "pipeline_metric_input_readiness_gate",
        "summary": summary,
    }


def _architecture_map_payload() -> dict:
    return {
        "mode": "current_architecture_map",
        "summary": {
            "architecture_status": "current_architecture_map_ready",
            "active_design": "source_first_two_branch_review_system",
            "pipeline_metric_plane_count": 8,
            "domain_profile_count": 5,
            "can_clone_domain_profiles_now": False,
            "can_write_production_config_now": False,
            "can_trade": False,
        },
    }


def _write_alignment_inputs(tmp_path: Path, *, source_gate_payload: dict | None = None) -> dict[str, Path]:
    paths = {
        "evidence": _write_json(tmp_path / "analyst_evidence_pack" / "latest.json", _evidence_pack_payload()),
        "source_gate": _write_json(
            tmp_path / "source_gate" / "latest.json",
            source_gate_payload or _source_gate_payload(),
        ),
        "agent_lab_dir": tmp_path / "agent_lab",
        "dropzone": _write_json(tmp_path / "dropzone" / "latest.json", _dropzone_payload()),
        "fundamental": _write_json(tmp_path / "fundamental" / "latest.json", _fundamental_gate_payload()),
        "architecture": _write_json(tmp_path / "architecture" / "latest.json", _architecture_map_payload()),
        "domain_intake": _write_json(tmp_path / "domain_intake" / "latest.json", _domain_analyst_intake_payload()),
        "domain_instance": _write_json(tmp_path / "domain_instance" / "latest.json", _domain_analyst_instance_contract_payload()),
        "domain_thesis": _write_json(tmp_path / "domain_thesis" / "latest.json", _domain_analyst_thesis_review_payload()),
        "domain_template": _write_json(
            tmp_path / "domain_template" / "latest.json",
            _domain_analyst_template_standardization_payload(),
        ),
        "domain_case_registry": _write_json(
            tmp_path / "domain_case_registry" / "latest.json",
            _domain_analyst_case_registry_payload(),
        ),
        "metric_input": _write_json(tmp_path / "metric_input" / "latest.json", _pipeline_metric_input_readiness_payload()),
        "pipeline_instance": _write_json(tmp_path / "pipeline_instance" / "latest.json", _pipeline_control_instance_contract_payload()),
        "pipeline_caution_review": _write_json(
            tmp_path / "pipeline_caution_review" / "latest.json",
            _pipeline_control_caution_review_payload(),
        ),
    }
    _write_json(paths["agent_lab_dir"] / "agent_lab_fixture.json", _agent_lab_payload())
    return paths


def test_alignment_review_marks_current_source_first_path_aligned_with_cautions(tmp_path):
    paths = _write_alignment_inputs(tmp_path)

    payload = CurrentSystemAlignmentReview(tmp_path / "reports").build(
        evidence_pack_json=paths["evidence"],
        source_gate_json=paths["source_gate"],
        agent_lab_path=paths["agent_lab_dir"],
        dropzone_inventory_json=paths["dropzone"],
        fundamental_gate_json=paths["fundamental"],
        architecture_map_json=paths["architecture"],
        domain_analyst_intake_json=paths["domain_intake"],
        domain_analyst_instance_contract_json=paths["domain_instance"],
        domain_analyst_thesis_review_json=paths["domain_thesis"],
        domain_analyst_template_standardization_json=paths["domain_template"],
        domain_analyst_case_registry_json=paths["domain_case_registry"],
        pipeline_metric_input_readiness_json=paths["metric_input"],
        pipeline_control_instance_contract_json=paths["pipeline_instance"],
        pipeline_control_caution_review_json=paths["pipeline_caution_review"],
        save=False,
    )

    assert payload["summary"]["alignment_status"] == "aligned_with_cautions"
    assert payload["summary"]["recommended_action"] == "continue_cached_source_review_path"
    assert payload["summary"]["can_trade"] is False
    assert payload["artifact_statuses"]["cached_evidence_pack"]["status"] == "useful"
    assert payload["artifact_statuses"]["current_architecture_map"]["status"] == "useful"
    assert payload["artifact_statuses"]["domain_analyst_intake"]["status"] == "useful"
    assert payload["artifact_statuses"]["domain_analyst_instance_contract"]["status"] == "useful"
    assert payload["artifact_statuses"]["domain_analyst_thesis_review"]["status"] == "useful"
    assert payload["artifact_statuses"]["domain_analyst_template_standardization"]["status"] == "useful"
    assert payload["artifact_statuses"]["domain_analyst_case_registry"]["status"] == "useful"
    assert payload["artifact_statuses"]["pipeline_metric_input_readiness"]["status"] == "useful"
    assert payload["artifact_statuses"]["pipeline_control_instance_contract"]["status"] == "useful"
    assert payload["artifact_statuses"]["pipeline_control_caution_review"]["status"] == "useful"
    assert payload["artifact_statuses"]["source_evidence_gate"]["status"] == "useful"
    assert payload["artifact_statuses"]["isolated_agent_lab"]["status"] == "useful"
    assert any(check["code"] == "source_gate_no_trading" and check["status"] == "pass" for check in payload["boundary_checks"])


def test_alignment_review_blocks_if_source_gate_boundary_allows_trading(tmp_path):
    paths = _write_alignment_inputs(tmp_path, source_gate_payload=_source_gate_payload(can_trade=True))

    payload = CurrentSystemAlignmentReview(tmp_path / "reports").build(
        evidence_pack_json=paths["evidence"],
        source_gate_json=paths["source_gate"],
        agent_lab_path=paths["agent_lab_dir"],
        dropzone_inventory_json=paths["dropzone"],
        fundamental_gate_json=paths["fundamental"],
        architecture_map_json=paths["architecture"],
        domain_analyst_intake_json=paths["domain_intake"],
        domain_analyst_instance_contract_json=paths["domain_instance"],
        domain_analyst_thesis_review_json=paths["domain_thesis"],
        domain_analyst_template_standardization_json=paths["domain_template"],
        domain_analyst_case_registry_json=paths["domain_case_registry"],
        pipeline_metric_input_readiness_json=paths["metric_input"],
        pipeline_control_instance_contract_json=paths["pipeline_instance"],
        pipeline_control_caution_review_json=paths["pipeline_caution_review"],
        save=False,
    )

    assert payload["summary"]["alignment_status"] == "misaligned_blocked"
    assert payload["summary"]["recommended_action"] == "fix_boundary_violation_before_more_integration"
    assert any(check["code"] == "source_gate_no_trading" and check["status"] == "fail" for check in payload["boundary_checks"])


def test_alignment_review_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_alignment_inputs(tmp_path)

    payload = CurrentSystemAlignmentReview(tmp_path / "reports").build(
        evidence_pack_json=paths["evidence"],
        source_gate_json=paths["source_gate"],
        agent_lab_path=paths["agent_lab_dir"],
        dropzone_inventory_json=paths["dropzone"],
        fundamental_gate_json=paths["fundamental"],
        architecture_map_json=paths["architecture"],
        domain_analyst_intake_json=paths["domain_intake"],
        domain_analyst_instance_contract_json=paths["domain_instance"],
        domain_analyst_thesis_review_json=paths["domain_thesis"],
        domain_analyst_template_standardization_json=paths["domain_template"],
        domain_analyst_case_registry_json=paths["domain_case_registry"],
        pipeline_metric_input_readiness_json=paths["metric_input"],
        pipeline_control_instance_contract_json=paths["pipeline_instance"],
        pipeline_control_caution_review_json=paths["pipeline_caution_review"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can trade: False" in markdown
    assert "source-first" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_current_system_alignment_review.py"),
            "--evidence-pack-json",
            str(paths["evidence"]),
            "--source-gate-json",
            str(paths["source_gate"]),
            "--agent-lab-path",
            str(paths["agent_lab_dir"]),
            "--dropzone-inventory-json",
            str(paths["dropzone"]),
            "--fundamental-gate-json",
            str(paths["fundamental"]),
            "--architecture-map-json",
            str(paths["architecture"]),
            "--domain-analyst-intake-json",
            str(paths["domain_intake"]),
            "--domain-analyst-instance-contract-json",
            str(paths["domain_instance"]),
            "--domain-analyst-thesis-review-json",
            str(paths["domain_thesis"]),
            "--domain-analyst-template-standardization-json",
            str(paths["domain_template"]),
            "--domain-analyst-case-registry-json",
            str(paths["domain_case_registry"]),
            "--pipeline-metric-input-readiness-json",
            str(paths["metric_input"]),
            "--pipeline-control-instance-contract-json",
            str(paths["pipeline_instance"]),
            "--pipeline-control-caution-review-json",
            str(paths["pipeline_caution_review"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Alignment: aligned_with_cautions" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
