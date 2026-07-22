from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.domain_analyst_template_decision_packet import DomainAnalystTemplateDecisionPacket


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _vertical(**summary_overrides) -> dict:
    summary = {
        "run_status": "domain_analyst_candidate_complete_pending_manual_acceptance",
        "domain_id": "semiconductor_ai_infrastructure",
        "can_create_analyst_research_recommendation": True,
        "can_create_execution_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "vertical_fixture",
        "mode": "domain_analyst_vertical_slice_run",
        "summary": summary,
        "synthetic_fixture_audit": {
            "has_synthetic_marker": False,
            "has_fixture_marker": False,
            "has_smoke_label": False,
        },
        "artifact_paths": {
            "forecast_review_json": "reports/dean_os/domain_analyst_forecast_review_packet_current/latest.json",
        },
    }


def _template(**summary_overrides) -> dict:
    summary = {
        "candidate_status": "ready_for_manual_template_acceptance",
        "domain_id": "semiconductor_ai_infrastructure",
        "can_mark_template_accepted_now": False,
        "can_standardize_domain_template_after_manual_acceptance": True,
        "can_scale_to_other_domains_now": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "template_fixture",
        "mode": "domain_analyst_template_standardization_packet",
        "summary": summary,
    }


def _forecast(**summary_overrides) -> dict:
    summary = {
        "packet_status": "forecast_review_ready_with_cautions_pending_outcomes",
        "domain_id": "semiconductor_ai_infrastructure",
        "forecast_candidate_count": 1,
        "can_create_analyst_research_recommendation": True,
        "can_create_execution_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "forecast_fixture",
        "mode": "domain_analyst_forecast_review_packet",
        "summary": summary,
    }


def _registry(**summary_overrides) -> dict:
    summary = {
        "registry_status": "case_registry_ready_pending_outcomes",
        "domain_id": "semiconductor_ai_infrastructure",
        "case_count": 1,
        "expectation_case_count": 1,
        "can_create_analyst_learning_recommendation": True,
        "can_create_execution_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "registry_fixture",
        "mode": "domain_analyst_case_registry_packet",
        "summary": summary,
    }


def _portability(**summary_overrides) -> dict:
    summary = {
        "review_status": "domain_analyst_portability_review_ready",
        "source_domain_id": "semiconductor_ai_infrastructure",
        "profile_count": 5,
        "profiles_structurally_portable_count": 5,
        "can_clone_domain_profiles_now": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "portability_fixture",
        "mode": "domain_analyst_portability_review",
        "summary": summary,
    }


def _architecture(**summary_overrides) -> dict:
    summary = {
        "can_clone_domain_profiles_now": False,
        "can_generate_analyst_research_recommendations_now": True,
        "can_generate_execution_recommendations_now": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "architecture_fixture",
        "mode": "current_architecture_map",
        "summary": summary,
    }


def _write_inputs(tmp_path: Path, **overrides) -> dict[str, Path]:
    return {
        "vertical": _write_json(tmp_path / "vertical" / "latest.json", overrides.get("vertical") or _vertical()),
        "template": _write_json(tmp_path / "template" / "latest.json", overrides.get("template") or _template()),
        "forecast": _write_json(tmp_path / "forecast" / "latest.json", overrides.get("forecast") or _forecast()),
        "registry": _write_json(tmp_path / "registry" / "latest.json", overrides.get("registry") or _registry()),
        "portability": _write_json(tmp_path / "portability" / "latest.json", overrides.get("portability") or _portability()),
        "architecture": _write_json(tmp_path / "architecture" / "latest.json", overrides.get("architecture") or _architecture()),
    }


def _build(tmp_path: Path, paths: dict[str, Path], **kwargs) -> dict:
    return DomainAnalystTemplateDecisionPacket(tmp_path / "reports").build(
        vertical_slice_json=paths["vertical"],
        template_standardization_json=paths["template"],
        forecast_review_json=paths["forecast"],
        case_registry_json=paths["registry"],
        portability_review_json=paths["portability"],
        architecture_map_json=paths["architecture"],
        save=False,
        **kwargs,
    )


def test_template_decision_packet_keeps_pending_decision_closed(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = _build(tmp_path, paths)

    assert payload["summary"]["decision_status"] == "manual_template_decision_pending"
    assert payload["summary"]["template_accepted"] is False
    assert payload["summary"]["can_clone_one_next_domain_profile_candidate"] is False
    assert payload["summary"]["can_create_analyst_research_recommendation"] is True
    assert payload["summary"]["can_create_execution_recommendation"] is False
    assert payload["summary"]["can_trade"] is False
    assert any(check["code"] == "manual_decision_pending" and check["status"] == "warn" for check in payload["review_checks"])


def test_template_decision_packet_accepts_template_without_scoring_thesis(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = _build(
        tmp_path,
        paths,
        decision="accept_template",
        reviewer="operator",
        rationale="Reusable process accepted; thesis outcome remains pending.",
    )

    assert payload["summary"]["decision_status"] == "manual_template_accepted_review_only"
    assert payload["summary"]["template_accepted"] is True
    assert payload["summary"]["can_clone_one_next_domain_profile_candidate"] is True
    assert payload["summary"]["can_scale_all_domains_now"] is False
    assert payload["manual_decision"]["does_not_assert_thesis_truth"] is True
    assert payload["manual_decision"]["does_not_score_forecast_outcome"] is True
    assert any("exactly one next-domain clone candidate" in item for item in payload["allowed_after_decision"])


def test_template_decision_packet_requires_rationale_for_accept(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = _build(tmp_path, paths, decision="accept_template")

    assert payload["summary"]["decision_status"] == "manual_template_decision_blocked"
    assert payload["summary"]["template_accepted"] is False
    assert any(check["code"] == "decision_rationale_required" and check["status"] == "fail" for check in payload["review_checks"])


def test_template_decision_packet_records_rejection_without_clone(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = _build(
        tmp_path,
        paths,
        decision="reject_template",
        rationale="Template omitted material context needed for reuse.",
    )

    assert payload["summary"]["decision_status"] == "manual_template_rejected"
    assert payload["summary"]["template_rejected"] is True
    assert payload["summary"]["can_clone_one_next_domain_profile_candidate"] is False
    assert "do not clone from it" in payload["decision_interpretation"]["meaning"].lower()


def test_template_decision_packet_blocks_when_template_not_ready(tmp_path):
    paths = _write_inputs(tmp_path, template=_template(candidate_status="needs_more_template_review"))

    payload = _build(
        tmp_path,
        paths,
        decision="accept_template",
        rationale="Trying to accept too early.",
    )

    assert payload["summary"]["decision_status"] == "manual_template_decision_blocked"
    assert any(check["code"] == "template_ready_for_manual_decision" and check["status"] == "fail" for check in payload["review_checks"])


def test_template_decision_packet_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystTemplateDecisionPacket(tmp_path / "reports").build(
        vertical_slice_json=paths["vertical"],
        template_standardization_json=paths["template"],
        forecast_review_json=paths["forecast"],
        case_registry_json=paths["registry"],
        portability_review_json=paths["portability"],
        architecture_map_json=paths["architecture"],
        decision="accept_template",
        reviewer="operator",
        rationale="Reusable process accepted for one next-domain clone candidate.",
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Decision status: `manual_template_accepted_review_only`" in markdown
    assert "Can create analyst research recommendation: True" in markdown
    assert "Can create execution recommendation: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_template_decision_packet.py"),
            "--vertical-slice-json",
            str(paths["vertical"]),
            "--template-standardization-json",
            str(paths["template"]),
            "--forecast-review-json",
            str(paths["forecast"]),
            "--case-registry-json",
            str(paths["registry"]),
            "--portability-review-json",
            str(paths["portability"]),
            "--architecture-map-json",
            str(paths["architecture"]),
            "--decision",
            "pending_review",
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Decision status: manual_template_decision_pending" in result.stdout
    assert "Can create analyst research recommendation: True" in result.stdout
    assert "Can create execution recommendation: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
