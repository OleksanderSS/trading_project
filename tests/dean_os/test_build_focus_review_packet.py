from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.packets.build_focus_review_packet import BuildFocusReviewPacket


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _alignment(blocker_count: int = 0) -> dict:
    return {
        "mode": "current_system_alignment_review",
        "summary": {
            "alignment_status": "aligned_with_cautions" if blocker_count == 0 else "misaligned_blocked",
            "recommended_action": "continue_cached_source_review_path",
            "blocker_count": blocker_count,
            "can_trade": False,
        },
    }


def _template() -> dict:
    return {
        "mode": "domain_analyst_template_standardization_packet",
        "summary": {
            "candidate_status": "ready_for_manual_template_acceptance",
            "can_mark_template_accepted_now": False,
            "can_write_learning_memory": False,
            "can_trade": False,
        },
    }


def _case_registry() -> dict:
    return {
        "mode": "domain_analyst_case_registry_packet",
        "summary": {
            "registry_status": "case_registry_ready_pending_outcomes",
            "case_count": 1,
            "outcome_bucket_counts": {"pending_domain_outcome": 1},
            "can_write_learning_memory": False,
            "can_train_from_hits_only": False,
            "can_trade": False,
        },
    }


def _pipeline() -> dict:
    return {
        "mode": "pipeline_control_instance_contract",
        "summary": {
            "instance_status": "blocked_pipeline_control_instance",
            "blocked_metric_planes": ["data_quality", "replay_repeatability"],
            "caution_metric_planes": ["risk"],
            "can_write_learning_memory": False,
            "can_trade": False,
        },
    }


def _pipeline_review_ready_with_cautions() -> dict:
    return {
        "mode": "pipeline_control_instance_contract",
        "summary": {
            "instance_status": "pipeline_control_instance_review_ready_with_cautions",
            "blocked_metric_planes": [],
            "caution_metric_planes": ["risk", "validation", "feature_stability"],
            "can_write_learning_memory": False,
            "can_trade": False,
        },
    }


def _write_inputs(
    tmp_path: Path,
    *,
    alignment: dict | None = None,
    template: dict | None = None,
    case_registry: dict | None = None,
    pipeline: dict | None = None,
) -> dict[str, Path]:
    return {
        "alignment": _write_json(tmp_path / "alignment" / "latest.json", alignment or _alignment()),
        "template": _write_json(tmp_path / "template" / "latest.json", template or _template()),
        "case_registry": _write_json(tmp_path / "case_registry" / "latest.json", case_registry or _case_registry()),
        "pipeline": _write_json(tmp_path / "pipeline" / "latest.json", pipeline or _pipeline()),
    }


def test_build_focus_review_recommends_manual_review_or_pipeline_switch(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = BuildFocusReviewPacket(tmp_path / "reports").build(
        alignment_review_json=paths["alignment"],
        template_standardization_json=paths["template"],
        case_registry_json=paths["case_registry"],
        pipeline_control_instance_json=paths["pipeline"],
        save=False,
    )

    assert payload["summary"]["focus_status"] == "focus_review_ready"
    assert payload["summary"]["recommended_next_operation"] == "manual_template_acceptance_or_switch_to_pipeline_control_blockers"
    assert payload["summary"]["should_stop_adding_domain_template_gates"] is True
    assert payload["summary"]["should_switch_to_pipeline_control_blockers"] is True
    assert payload["summary"]["can_continue_domain_branch_only_for_outcome_lane"] is True
    assert payload["summary"]["can_trade"] is False
    assert any(check["code"] == "domain_branch_should_not_deepen_template_gates" and check["status"] == "pass" for check in payload["review_checks"])


def test_build_focus_review_handles_pipeline_cautions_without_blocker_switch(tmp_path):
    paths = _write_inputs(tmp_path, pipeline=_pipeline_review_ready_with_cautions())

    payload = BuildFocusReviewPacket(tmp_path / "reports").build(
        alignment_review_json=paths["alignment"],
        template_standardization_json=paths["template"],
        case_registry_json=paths["case_registry"],
        pipeline_control_instance_json=paths["pipeline"],
        save=False,
    )

    assert payload["summary"]["focus_status"] == "focus_review_ready"
    assert payload["summary"]["recommended_next_operation"] == "manual_template_acceptance_or_review_pipeline_cautions"
    assert payload["summary"]["should_switch_to_pipeline_control_blockers"] is False
    assert any(
        check["code"] == "pipeline_branch_review_ready_with_cautions" and check["status"] == "pass"
        for check in payload["review_checks"]
    )


def test_build_focus_review_blocks_when_alignment_has_blockers(tmp_path):
    paths = _write_inputs(tmp_path, alignment=_alignment(blocker_count=1))

    payload = BuildFocusReviewPacket(tmp_path / "reports").build(
        alignment_review_json=paths["alignment"],
        template_standardization_json=paths["template"],
        case_registry_json=paths["case_registry"],
        pipeline_control_instance_json=paths["pipeline"],
        save=False,
    )

    assert payload["summary"]["focus_status"] == "focus_blocked"
    assert payload["summary"]["recommended_next_operation"] == "fix_boundary_or_alignment_failures"
    assert any(check["code"] == "alignment_has_no_blockers" and check["status"] == "fail" for check in payload["review_checks"])


def test_build_focus_review_warns_when_case_registry_missing(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = BuildFocusReviewPacket(tmp_path / "reports").build(
        alignment_review_json=paths["alignment"],
        template_standardization_json=paths["template"],
        case_registry_json=None,
        pipeline_control_instance_json=paths["pipeline"],
        save=False,
    )

    assert payload["summary"]["focus_status"] == "focus_needs_more_review"
    assert payload["branch_assessment"]["domain_analyst_branch"]["status"] == "needs_case_registry_before_learning"
    assert any(check["code"] == "case_registry_prevents_hits_only_learning" and check["status"] == "warn" for check in payload["review_checks"])


def test_build_focus_review_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = BuildFocusReviewPacket(tmp_path / "reports").build(
        alignment_review_json=paths["alignment"],
        template_standardization_json=paths["template"],
        case_registry_json=paths["case_registry"],
        pipeline_control_instance_json=paths["pipeline"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Should stop adding domain template gates: True" in markdown
    assert "Can trade: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_build_focus_review_packet.py"),
            "--alignment-review-json",
            str(paths["alignment"]),
            "--template-standardization-json",
            str(paths["template"]),
            "--case-registry-json",
            str(paths["case_registry"]),
            "--pipeline-control-instance-json",
            str(paths["pipeline"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Focus status: focus_review_ready" in result.stdout
    assert "Stop domain template gates: True" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
