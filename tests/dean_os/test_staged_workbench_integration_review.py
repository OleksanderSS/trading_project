"""The draft bundle these tests classify was never committed to git.

`dean_os/draft/dean_os_after_245_full_context_bundle` existed only in the working
directory of an earlier agent session and is gone. That exposed the real defect:
the review reported `staged_workbench_review_ready` while reading nothing at all,
because `draft_bundle_found` was computed and then never allowed to affect
`review_status`. An empty read looked exactly like a clean review.

So the missing-input behaviour is asserted unconditionally below, and the
classification assertions skip when the bundle is absent rather than pinning the
suite to a directory that cannot be restored.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dean_os.staged_workbench_integration_review import StagedWorkbenchIntegrationReview

DRAFT_BUNDLE = Path(__file__).resolve().parents[2] / "dean_os" / "draft" / "dean_os_after_245_full_context_bundle"
DROPZONE = Path(__file__).resolve().parents[2] / "docs" / "research"

requires_bundle = pytest.mark.skipif(
    not DRAFT_BUNDLE.exists(),
    reason=f"staged draft bundle is not present: {DRAFT_BUNDLE}",
)


def test_missing_draft_bundle_blocks_the_review_instead_of_reporting_ready(tmp_path):
    payload = StagedWorkbenchIntegrationReview(tmp_path / "reports").build(
        draft_bundle=tmp_path / "no_such_bundle",
        dropzone=DROPZONE,
        save=False,
    )

    summary = payload["summary"]
    assert summary["review_status"] == "staged_workbench_review_blocked_missing_draft_bundle"
    assert summary["draft_bundle_found"] is False
    assert summary["staged_block_count"] == 0
    assert summary["can_trade"] is False
    assert summary["can_create_recommendation"] is False


def test_empty_draft_bundle_is_treated_as_missing(tmp_path):
    empty = tmp_path / "empty_bundle"
    empty.mkdir()

    payload = StagedWorkbenchIntegrationReview(tmp_path / "reports").build(
        draft_bundle=empty,
        dropzone=DROPZONE,
        save=False,
    )

    assert payload["summary"]["review_status"] == "staged_workbench_review_blocked_missing_draft_bundle"
    assert payload["summary"]["draft_bundle_found"] is False


@requires_bundle
def test_staged_workbench_review_classifies_blocks_and_keeps_safety_boundaries(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    payload = StagedWorkbenchIntegrationReview(tmp_path / "reports").build(
        draft_bundle=repo_root / "dean_os" / "draft" / "dean_os_after_245_full_context_bundle",
        dropzone=repo_root / "docs" / "research",
        save=False,
    )

    assert payload["summary"]["review_status"] == "staged_workbench_review_ready"
    assert payload["summary"]["staged_block_count"] >= 30
    assert payload["summary"]["integrate_candidate_count"] >= 3
    assert payload["summary"]["integrate_candidate_file_count"] < 20
    assert payload["summary"]["redundant_metadata_ladder_count"] >= 1
    assert payload["summary"]["can_trade"] is False
    assert payload["summary"]["can_create_recommendation"] is False
    by_block = {item["block_id"]: item for item in payload["staged_block_classifications"]}
    assert by_block["245_review_only_real_source_normalized_packet_fixture_v1"]["category"] == "A"
    assert by_block["245_review_only_real_source_normalized_packet_fixture_v1"]["can_promote_fixtures_as_evidence"] is False
    market_data_snapshots = [
        item
        for item in payload["staged_file_classifications"]
        if item["path"].endswith("dean_os/market_data_api.py")
    ]
    assert market_data_snapshots
    assert all(item["category"] != "A" for item in market_data_snapshots)
    assert payload["safety_boundary_audit"]["overall_status"] == "review_only_boundaries_preserved"


@requires_bundle
def test_staged_workbench_review_marks_vertical_slice_projection_gap(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    payload = StagedWorkbenchIntegrationReview(tmp_path / "reports").build(
        draft_bundle=repo_root / "dean_os" / "draft" / "dean_os_after_245_full_context_bundle",
        dropzone=repo_root / "docs" / "research",
        save=False,
    )

    vertical = payload["first_vertical_slice_viability"]
    assert vertical["slice_status"] in {
        "offline_vertical_slice_not_yet_viable",
        "offline_vertical_slice_viable_after_projection_adapter",
    }
    assert "normalized_packet_to_evidence_pack_projection_missing" in vertical["adapter_gaps"]
    assert any(step["step_id"] == "consumer_projection_read_model_preview" for step in vertical["steps"])
    assert payload["where_we_looped"]["loop_status"] == "loop_detected"


@requires_bundle
def test_staged_workbench_review_saves_markdown_and_cli_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    payload = StagedWorkbenchIntegrationReview(tmp_path / "reports").build(
        draft_bundle=repo_root / "dean_os" / "draft" / "dean_os_after_245_full_context_bundle",
        dropzone=repo_root / "docs" / "research",
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Staged Workbench Integration Review" in markdown
    assert "Can trade: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_staged_workbench_integration_review.py"),
            "--draft-bundle",
            str(repo_root / "dean_os" / "draft" / "dean_os_after_245_full_context_bundle"),
            "--dropzone",
            str(repo_root / "docs" / "research"),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Review status: staged_workbench_review_ready" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
