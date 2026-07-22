from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.domain_analyst_event_interpretation_packet import DomainAnalystEventInterpretationPacket


def test_event_interpretation_packet_preserves_detailed_news_analysis(tmp_path):
    evidence_pack = _write_json(tmp_path / "evidence_pack.json", _evidence_pack_fixture())

    payload = DomainAnalystEventInterpretationPacket(tmp_path / "reports").build(
        evidence_pack_json=evidence_pack,
        domain_id="semiconductor_ai_infrastructure",
        save=False,
    )

    assert payload["summary"]["packet_status"] == "domain_analyst_event_interpretation_ready_with_review_items"
    assert payload["summary"]["source_document_count"] == 4
    assert payload["summary"]["event_packet_count"] == 4
    assert payload["summary"]["context_snapshot_status"] == "context_snapshot_ready"
    assert payload["summary"]["can_create_context_conditioned_news_analysis"] is True
    assert "ai_capex_wave" in payload["summary"]["context_tags"]
    assert "inflation_or_rate_pressure" in payload["summary"]["context_tags"]
    assert "war_sanctions_tension" in payload["summary"]["context_tags"]
    assert payload["summary"]["high_materiality_count"] >= 1
    assert payload["summary"]["can_create_detailed_data_news_analysis"] is True
    assert payload["summary"]["can_create_event_interpretation"] is True
    assert payload["summary"]["pipeline_news_context_classified_count"] >= 1
    assert payload["summary"]["pipeline_crisis_pattern_event_count"] >= 1
    assert "technology" in payload["summary"]["pipeline_news_impact_class_counts"]
    assert payload["summary"]["can_create_mechanism_hypothesis"] is True
    assert payload["summary"]["can_create_evidence_gap_tasks"] is True
    assert payload["summary"]["can_create_analyst_research_recommendation"] is True
    assert payload["summary"]["can_create_execution_recommendation"] is False
    assert payload["summary"]["can_trade"] is False

    first = payload["event_interpretation_packets"][0]
    assert first["allowed_output"] == "hypothesis_for_review"
    assert first["pipeline_news_context"]["allowed_output"] == "pipeline_news_context_for_review"
    assert "prediction_adjustment" in first["pipeline_news_context"]["forbidden_outputs"]
    assert first["pipeline_news_context"]["context_tags"]
    assert first["mechanism_chain"]
    assert first["affected_value_chain"]
    assert first["intermediate_variables"]
    assert first["counterforces"]
    assert first["context_conditioned_interpretation"]["context_tags"]
    assert first["context_conditioned_interpretation"]["pipeline_news_context_summary"]["context_tags"]
    assert first["context_conditioned_interpretation"]["watch_metrics"]
    assert first["context_conditioned_interpretation"]["allowed_output"] == "context_conditioned_hypothesis_for_review"
    assert first["evidence_gaps"]
    assert "buy_sell_hold" in first["forbidden_outputs"]
    assert any(item["source_file"] == "NEWS_EVENT_INTERPRETATION_SCHEMA_TEMPLATE.json" for item in payload["after_385_harvest_decisions"])
    assert any(item["source_file"] == "src/config/context.yaml" for item in payload["pipeline_news_taxonomy_harvest_decisions"])


def test_event_interpretation_packet_warns_when_no_event_documents(tmp_path):
    evidence_pack = _write_json(
        tmp_path / "evidence_pack.json",
        {
            "mode": "analyst_evidence_pack",
            "inputs": {},
            "documents": [
                {
                    "document_id": "metric_1",
                    "title": "Internal metric",
                    "source_type": "metric",
                    "text": "Non-news metric payload.",
                }
            ],
        },
    )

    payload = DomainAnalystEventInterpretationPacket(tmp_path / "reports").build(
        evidence_pack_json=evidence_pack,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "domain_analyst_event_interpretation_ready"
    assert payload["summary"]["event_packet_count"] == 0
    assert payload["summary"]["pipeline_news_context_classified_count"] == 0
    assert payload["summary"]["context_snapshot_status"] == "context_snapshot_sparse"
    assert any(check["code"] == "event_interpretations_present" and check["status"] == "warn" for check in payload["review_checks"])
    assert payload["summary"]["can_create_execution_recommendation"] is False
    assert payload["summary"]["can_trade"] is False


def test_event_interpretation_packet_accepts_saved_pipeline_context_overlay(tmp_path):
    evidence_pack = _write_json(tmp_path / "evidence_pack.json", _evidence_pack_fixture())
    pipeline_context = _write_json(
        tmp_path / "pipeline_context.json",
        {
            "regime": "RISK_OFF",
            "confidence": 0.72,
            "context_tags": ["high_volatility", "credit_tightening"],
            "metrics": {
                "vix": 32,
                "inflation_yoy": 4.2,
                "yield_curve_slope": -0.35,
                "credit_spread": 2.1,
                "macro_score": -0.28,
                "news_impact_score": -0.85,
                "news_significance_level": "high",
                "news_quality_score": 0.82,
                "news_freshness_hours": 6,
                "nlp_sentiment_score": -0.31,
            },
        },
    )

    payload = DomainAnalystEventInterpretationPacket(tmp_path / "reports").build(
        evidence_pack_json=evidence_pack,
        pipeline_context_json=pipeline_context,
        domain_id="semiconductor_ai_infrastructure",
        save=False,
    )

    summary = payload["summary"]
    assert summary["pipeline_context_supplied"] is True
    assert summary["pipeline_context_status"] == "pipeline_context_overlay_ready"
    assert summary["pipeline_context_tag_count"] >= 6
    assert "pipeline_risk_off" in summary["context_tags"]
    assert "pipeline_inflation_pressure" in summary["context_tags"]
    assert "pipeline_volatility_high" in summary["context_tags"]
    assert "pipeline_credit_tightening" in summary["context_tags"]
    assert "pipeline_news_intensity_high" in summary["context_tags"]
    assert summary["can_create_execution_recommendation"] is False
    assert summary["can_trade"] is False

    overlay = payload["context_regime_snapshot"]["pipeline_context_overlay"]
    assert overlay["source_path"].endswith("pipeline_context.json")
    assert overlay["metric_count"] >= 8
    assert "news_impact_score" in overlay["watch_metrics"]
    assert overlay["review_only_rule"].startswith("The overlay uses saved pipeline context")

    first = payload["event_interpretation_packets"][0]["context_conditioned_interpretation"]
    assert "pipeline_risk_off" in first["context_tags"]
    assert "pipeline_macro_overlay_requires_scenario_split" in first["review_flags"]
    assert "news_impact_score" in first["watch_metrics"]
    assert first["pipeline_context_overlay"]["overlay_status"] == "pipeline_context_overlay_ready"
    assert any(check["code"] == "pipeline_context_is_review_only" and check["status"] == "pass" for check in payload["review_checks"])


def test_event_interpretation_packet_saves_markdown_and_cli_runs(tmp_path):
    evidence_pack = _write_json(tmp_path / "evidence_pack.json", _evidence_pack_fixture())
    pipeline_context = _write_json(
        tmp_path / "pipeline_context.json",
        {"regime": "risk_off", "metrics": {"vix": 31, "news_impact_score": -0.8}},
    )

    payload = DomainAnalystEventInterpretationPacket(tmp_path / "reports").build(
        evidence_pack_json=evidence_pack,
        pipeline_context_json=pipeline_context,
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can create detailed data/news analysis: True" in markdown
    assert "Pipeline context: `pipeline_context_overlay_ready`" in markdown
    assert "Can create execution recommendation: False" in markdown
    assert "NEWS_EVENT_INTERPRETATION_SCHEMA_TEMPLATE.json" in json.dumps(payload["after_385_harvest_decisions"])
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_event_interpretation_packet.py"),
            "--evidence-pack-json",
            str(evidence_pack),
            "--pipeline-context-json",
            str(pipeline_context),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Can create detailed data/news analysis: True" in result.stdout
    assert "Pipeline context: pipeline_context_overlay_ready" in result.stdout
    assert "Can create execution recommendation: False" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _evidence_pack_fixture() -> dict:
    return {
        "mode": "analyst_evidence_pack",
        "inputs": {"domain_id": "semiconductor_ai_infrastructure"},
        "documents": [
            {
                "document_id": "news_ai_demand",
                "title": "AI infrastructure demand expands GPU accelerator orders",
                "source_type": "news",
                "text": "Hyperscaler capex and data center AI demand increase GPU accelerator orders for semiconductor suppliers.",
                "uri": "local://news",
                "published_at": "2026-01-05T00:00:00+00:00",
                "tickers": [],
                "sectors": ["semiconductor"],
                "tags": ["news", "ai_cycle"],
                "metadata": {},
            },
            {
                "document_id": "news_export_controls",
                "title": "Export control update creates China semiconductor equipment risk",
                "source_type": "news",
                "text": "New export control restriction may affect China AI chip sales and equipment supply-chain routing.",
                "uri": "local://news",
                "published_at": "2026-01-06T00:00:00+00:00",
                "tickers": [],
                "sectors": ["semiconductor"],
                "tags": ["news", "policy"],
                "metadata": {},
            },
            {
                "document_id": "report_capacity",
                "title": "HBM packaging capacity remains tight",
                "source_type": "report",
                "text": "Foundry advanced packaging and HBM memory capacity remain bottlenecks for AI accelerator supply.",
                "uri": "local://report",
                "published_at": "2026-01-07T00:00:00+00:00",
                "tickers": [],
                "sectors": ["semiconductor"],
                "tags": ["report", "capacity"],
                "metadata": {},
            },
            {
                "document_id": "macro_rates",
                "title": "Inflation and rates remain high while credit conditions tighten",
                "source_type": "report",
                "text": "Inflation pressure, high rates, treasury yields, and credit tightening can delay long-duration capex response.",
                "uri": "local://macro",
                "published_at": "2026-01-08T00:00:00+00:00",
                "tickers": [],
                "sectors": ["semiconductor"],
                "tags": ["macro", "rates"],
                "metadata": {},
            },
        ],
    }
