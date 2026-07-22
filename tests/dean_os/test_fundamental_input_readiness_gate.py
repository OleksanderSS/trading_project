from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.fundamental_input_readiness_gate import FundamentalInputReadinessGate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_fundamental_gate_accepts_simple_fundamentals_with_review_warnings(tmp_path):
    source_path = _write_json(
        tmp_path / "fundamentals.json",
        {
            "fundamentals": {
                "AMD": {
                    "pe": 42.0,
                    "pb": 3.2,
                    "fcf_yield": 0.025,
                }
            }
        },
    )

    payload = FundamentalInputReadinessGate(tmp_path / "reports").build(fundamentals_json=source_path, save=False)

    assert payload["summary"]["readiness_status"] == (
        "fundamental_input_structured_contract_blocked"
    )
    assert payload["summary"]["metric_count"] == 3
    assert payload["summary"]["ticker_count"] == 1
    assert payload["summary"]["can_enter_manual_fundamental_review"] is True
    assert payload["summary"]["can_feed_value_screening_after_manual_review"] is False
    assert payload["summary"]["can_compute_ratios_now"] is False
    assert payload["summary"]["can_create_recommendation"] is False
    assert payload["summary"]["can_trade"] is False
    assert any(check["code"].endswith("_source_citation_missing") for check in payload["readiness_checks"])
    assert any(check["code"].endswith("_period_missing") for check in payload["readiness_checks"])


def test_fundamental_gate_marks_cited_metric_rows_ready_for_manual_review(tmp_path):
    source_path = _write_json(
        tmp_path / "metric_rows.json",
        {
            "extracted_fundamental_metrics": [
                {
                    "ticker": "AMD",
                    "metric_name": "revenue",
                    "value": 25785000000,
                    "unit": "USD",
                    "period": "FY2025",
                    "available_at": "2026-02-04T21:00:00+00:00",
                    "source_citation": (
                        "https://example.test/filing_amd_10k#income_statement"
                    ),
                },
                {
                    "ticker": "AMD",
                    "metric_name": "shares",
                    "value": 1620000000,
                    "unit": "shares",
                    "period": "FY2025",
                    "available_at": "2026-02-04T21:00:00+00:00",
                    "source": (
                        "https://example.test/filing_amd_10k"
                    ),
                },
            ]
        },
    )

    payload = FundamentalInputReadinessGate(
        tmp_path / "reports"
    ).build(
        fundamentals_json=source_path,
        as_of="2026-06-30T12:00:00+00:00",
        save=False,
    )

    assert payload["summary"]["readiness_status"] == "fundamental_input_ready_for_manual_review"
    assert payload["summary"]["can_feed_value_screening_after_manual_review"] is True
    assert payload["decision_guidance"]["warning_count"] == 0
    assert payload["output_boundary"]["ratio_computation_performed_now"] is False
    assert payload["output_boundary"]["valuation_generated_now"] is False


def test_fundamental_gate_blocks_invalid_metric_values(tmp_path):
    source_path = _write_json(
        tmp_path / "bad_fundamentals.json",
        {"fundamentals": {"AMD": {"pe": "not-a-number", "period": "FY2025", "source": "manual_upload"}}},
    )

    payload = FundamentalInputReadinessGate(tmp_path / "reports").build(fundamentals_json=source_path, save=False)

    assert payload["summary"]["readiness_status"] == "blocked_fundamental_input"
    assert payload["summary"]["can_enter_manual_fundamental_review"] is False
    failed_codes = {check["code"] for check in payload["readiness_checks"] if check["status"] == "fail"}
    assert any(code.endswith("_numeric_value_present") for code in failed_codes)


def test_fundamental_gate_saves_markdown_and_cli_runs(tmp_path):
    source_path = _write_json(
        tmp_path / "fundamentals.json",
        {
            "fundamentals": {
                "AMD": {
                    "revenue": {
                        "value": 100.0,
                        "unit": "USD",
                        "period": "FY2025",
                        "available_at": "2026-02-04T21:00:00+00:00",
                        "source": "https://example.test/operator_file",
                    },
                }
            }
        },
    )

    payload = FundamentalInputReadinessGate(
        tmp_path / "reports"
    ).build(
        fundamentals_json=source_path,
        as_of="2026-06-30T12:00:00+00:00",
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can compute ratios now: False" in markdown
    assert "No ratio interpretation, valuation, recommendation, price target, allocation, or trade signal is generated." in markdown
    assert payload["saved_paths"]["latest_markdown"].endswith("latest.md")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_fundamental_input_readiness_gate.py"),
            "--fundamentals-json",
            str(source_path),
            "--output-dir",
            str(tmp_path / "cli_reports"),
            "--as-of",
            "2026-06-30T12:00:00+00:00",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Readiness: fundamental_input_ready_for_manual_review" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
