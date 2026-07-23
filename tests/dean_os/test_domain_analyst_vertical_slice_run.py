from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import pandas as pd

from dean_os.analyst_core.domain_analyst_vertical_slice_run import DomainAnalystVerticalSliceRun


def _write_local_inputs(tmp_path: Path) -> dict[str, Path]:
    news_path = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {
                "title": "AI infrastructure demand expands",
                "summary": "AI infrastructure demand and GPU accelerator orders increase across data center customers.",
                "published_at": "2026-01-05T00:00:00+00:00",
                "sector": "semiconductor",
            },
            {
                "title": "Hyperscaler capex supports chips",
                "summary": "Hyperscaler capex, cloud spending, and data center buildout remain supportive.",
                "published_at": "2026-01-06T00:00:00+00:00",
                "sector": "semiconductor",
            },
            {
                "title": "Advanced packaging update",
                "summary": "Foundry advanced packaging capacity and HBM memory supply remain tight.",
                "published_at": "2026-01-07T00:00:00+00:00",
                "sector": "semiconductor",
            },
            {
                "title": "Export control update",
                "summary": "Export control policy creates China and Taiwan geopolitical risk for chip equipment.",
                "published_at": "2026-01-08T00:00:00+00:00",
                "sector": "semiconductor",
            },
            {
                "title": "Sector market confirmation",
                "summary": "Semiconductor market relative strength and shares outperform broader risk assets.",
                "published_at": "2026-01-09T00:00:00+00:00",
                "sector": "semiconductor",
            },
        ]
    ).to_csv(news_path, index=False)
    macro_path = tmp_path / "macro.csv"
    pd.DataFrame(
        [
            {
                "series": "DGS10",
                "datetime": "2026-01-09T00:00:00+00:00",
                "value": 4.1,
                "summary": "Rates context is stable for growth and market confirmation review.",
            }
        ]
    ).to_csv(macro_path, index=False)
    return {"news": news_path, "macro": macro_path}


def test_domain_analyst_vertical_slice_runs_full_review_chain_from_local_tables(tmp_path):
    paths = _write_local_inputs(tmp_path)
    pipeline_context = tmp_path / "pipeline_context.json"
    pipeline_context.write_text(
        json.dumps(
            {
                "market_regime": "risk_off",
                "confidence": 0.66,
                "metrics": {
                    "vix": 28,
                    "inflation_yoy": 3.8,
                    "yield_curve_slope": -0.15,
                    "news_impact_score": -0.74,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = DomainAnalystVerticalSliceRun(tmp_path / "reports").build(
        news_data_paths=[paths["news"]],
        macro_data_paths=[paths["macro"]],
        pipeline_context_json=pipeline_context,
        sectors=["semiconductor"],
        tags=["ai_cycle"],
        sector_keywords=["AI infrastructure", "GPU", "capex", "foundry", "export control", "semiconductor"],
        save=False,
    )

    assert payload["summary"]["run_status"] == "domain_analyst_candidate_complete_pending_manual_acceptance"
    assert payload["summary"]["document_count"] >= 5
    assert payload["summary"]["event_interpretation_status"] in {
        "domain_analyst_event_interpretation_ready",
        "domain_analyst_event_interpretation_ready_with_review_items",
    }
    assert payload["summary"]["event_packet_count"] >= 5
    assert payload["summary"]["pipeline_context_supplied"] is True
    assert payload["summary"]["pipeline_context_status"] == "pipeline_context_overlay_ready"
    assert "pipeline_risk_off" in payload["summary"]["pipeline_context_tags"]
    assert payload["summary"]["pipeline_news_context_classified_count"] >= 1
    assert payload["summary"]["pipeline_crisis_pattern_event_count"] >= 1
    assert payload["summary"]["regime_scenario_status"] == "domain_analyst_regime_scenario_ready_with_review_items"
    assert payload["summary"]["scenario_node_count"] > 0
    assert payload["summary"]["scenario_probability_mass_valid"] is True
    assert payload["summary"]["can_create_regime_context_scenario_analysis"] is True
    assert payload["summary"]["can_create_detailed_data_news_analysis"] is True
    assert payload["summary"]["manual_acceptance_required"] is True
    assert payload["summary"]["forecast_candidate_count"] == 1
    assert payload["summary"]["analyst_control_plane_count"] >= 10
    assert payload["summary"]["can_create_analyst_research_recommendation"] is True
    assert payload["summary"]["can_create_execution_recommendation"] is False
    assert payload["summary"]["can_mark_template_accepted_now"] is False
    assert payload["summary"]["can_scale_to_other_domains_now"] is False
    assert payload["summary"]["can_create_recommendation"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["synthetic_fixture_audit"]["has_synthetic_marker"] is False
    assert payload["synthetic_fixture_audit"]["has_fixture_marker"] is False
    assert payload["artifact_paths"]["forecast_review_json"].endswith("latest.json")
    assert payload["artifact_paths"]["event_interpretation_json"].endswith("latest.json")
    assert payload["artifact_paths"]["regime_scenario_json"].endswith("latest.json")
    assert payload["artifact_paths"]["template_standardization_json"].endswith("latest.json")
    assert any(step["step_id"] == "event_interpretation" for step in payload["branch_readiness"]["steps"])
    assert any(step["step_id"] == "regime_scenario" for step in payload["branch_readiness"]["steps"])


def test_domain_analyst_vertical_slice_accepts_existing_local_evidence_pack(tmp_path):
    paths = _write_local_inputs(tmp_path)
    first = DomainAnalystVerticalSliceRun(tmp_path / "first").build(
        news_data_paths=[paths["news"]],
        macro_data_paths=[paths["macro"]],
        sectors=["semiconductor"],
        sector_keywords=["AI infrastructure", "GPU", "capex", "foundry", "export control", "semiconductor"],
    )

    payload = DomainAnalystVerticalSliceRun(tmp_path / "second").build(
        evidence_pack_json=first["artifact_paths"]["evidence_pack_json"],
        sectors=["semiconductor"],
        save=False,
    )

    assert payload["summary"]["evidence_source"] == "supplied_local_evidence_pack"
    assert payload["summary"]["run_status"] == "domain_analyst_candidate_complete_pending_manual_acceptance"
    assert payload["summary"]["can_trade"] is False


def test_domain_analyst_vertical_slice_cli_runs(tmp_path):
    paths = _write_local_inputs(tmp_path)
    pipeline_context = tmp_path / "pipeline_context.json"
    pipeline_context.write_text(
        json.dumps({"market_regime": "risk_off", "metrics": {"vix": 30, "news_impact_score": -0.8}}),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_vertical_slice.py"),
            "--news-data",
            str(paths["news"]),
            "--macro-data",
            str(paths["macro"]),
            "--pipeline-context-json",
            str(pipeline_context),
            "--sectors",
            "semiconductor",
            "--sector-keywords",
            "AI infrastructure",
            "GPU",
            "capex",
            "foundry",
            "export control",
            "semiconductor",
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Run status: domain_analyst_candidate_complete_pending_manual_acceptance" in result.stdout
    assert "Event interpretation: packets=" in result.stdout
    assert "Regime scenario:" in result.stdout
    assert "Pipeline context: pipeline_context_overlay_ready" in result.stdout
    assert "Pipeline crisis-pattern events:" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
