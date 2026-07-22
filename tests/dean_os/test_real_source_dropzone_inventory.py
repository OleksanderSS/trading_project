from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dean_os.real_source_dropzone_inventory import RealSourceDropzoneInventory


def test_dropzone_inventory_ignores_readme_and_reports_empty(tmp_path):
    dropzone = tmp_path / "research"
    dropzone.mkdir()
    (dropzone / "README.md").write_text("dropzone notes", encoding="utf-8")

    payload = RealSourceDropzoneInventory(output_dir=tmp_path / "reports").build(dropzone, save=False)

    assert payload["summary"]["dropzone_status"] == "empty_dropzone"
    assert payload["summary"]["supported_file_count"] == 0
    assert payload["summary"]["ignored_file_count"] == 1
    assert payload["summary"]["can_build_normalized_packet"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["commands"][0]["command_id"] == "add_operator_source_file"


def test_dropzone_inventory_lists_supported_and_unsupported_files(tmp_path):
    dropzone = tmp_path / "research"
    dropzone.mkdir()
    (dropzone / "semiconductor_report.md").write_text("Demand growth improved.", encoding="utf-8")
    (dropzone / "earnings_transcript.txt").write_text("Prepared remarks.", encoding="utf-8")
    (dropzone / "notes.csv").write_text("not,supported\n", encoding="utf-8")

    payload = RealSourceDropzoneInventory(output_dir=tmp_path / "reports").build(dropzone, save=False)

    assert payload["summary"]["dropzone_status"] == "ready_for_operator_source_review"
    assert payload["summary"]["supported_file_count"] == 2
    assert payload["summary"]["unsupported_file_count"] == 1
    assert payload["summary"]["can_build_normalized_packet"] is True
    assert payload["supported_files"][0]["source_type_hint"] == "transcript"
    assert payload["supported_files"][1]["source_type_hint"] == "report"
    assert payload["unsupported_files"][0]["reason"] == "unsupported_extension"
    assert payload["commands"][0]["command_id"] == "build_first_supported_normalized_packet"
    assert "run_agent_real_source_normalized_packet.py" in payload["commands"][0]["command"]


def test_dropzone_inventory_save_writes_review_artifacts(tmp_path):
    dropzone = tmp_path / "research"
    dropzone.mkdir()
    (dropzone / "market_news.txt").write_text("Demand growth improved.", encoding="utf-8")

    payload = RealSourceDropzoneInventory(output_dir=tmp_path / "reports").build(dropzone, save=True)

    assert payload["saved_paths"]["latest_json"].endswith("latest.json")
    assert payload["saved_paths"]["latest_markdown"].endswith("latest.md")


def test_dropzone_inventory_cli_smoke(tmp_path):
    dropzone = tmp_path / "research"
    dropzone.mkdir()
    (dropzone / "industry_report.md").write_text("Demand growth improved.", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_real_source_dropzone_inventory.py"),
            "--dropzone",
            str(dropzone),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Dropzone status: ready_for_operator_source_review" in result.stdout
    assert (tmp_path / "reports" / "latest.json").exists()
    assert (tmp_path / "reports" / "latest.md").exists()
