from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.domain_analyst_event_interpretation_packet import DomainAnalystEventInterpretationPacket
from dean_os.domain_analyst_regime_scenario_packet import DomainAnalystRegimeScenarioPacket, REGIME_FIELDS


def test_regime_scenario_packet_builds_context_vector_and_graph(tmp_path):
    evidence_pack = _write_json(tmp_path / "evidence_pack.json", _evidence_pack_fixture())
    event_payload = DomainAnalystEventInterpretationPacket(tmp_path / "events").build(
        evidence_pack_json=evidence_pack,
        domain_id="semiconductor_ai_infrastructure",
    )

    payload = DomainAnalystRegimeScenarioPacket(tmp_path / "reports").build(
        event_interpretation_json=event_payload["saved_paths"]["latest_json"],
        domain_id="semiconductor_ai_infrastructure",
        save=False,
    )

    summary = payload["summary"]
    assert summary["packet_status"] == "domain_analyst_regime_scenario_ready_with_review_items"
    assert summary["source_event_packet_count"] == 4
    assert summary["regime_field_count"] == len(REGIME_FIELDS)
    assert summary["scenario_node_count"] > 0
    assert summary["probability_mass_valid"] is True
    assert summary["evidence_gap_count"] > 0
    assert summary["can_create_regime_context_vector"] is True
    assert summary["can_create_news_vs_regime_analysis"] is True
    assert summary["can_create_scenario_outcome_graph"] is True
    assert summary["can_call_gpt_or_finbert_now"] is False
    assert summary["can_create_execution_recommendation"] is False
    assert summary["can_trade"] is False

    fields = payload["regime_context_vector"]["fields"]
    assert set(REGIME_FIELDS).issubset(fields)
    assert fields["ai_tech_cycle"]["state"] in {"capex_boom", "memory_bottleneck", "infrastructure_bottleneck"}
    assert fields["geopolitical_state"]["evidence_ids"]
    assert payload["scenario_outcome_graph"]["probability_mass_check"]["sum"] == 1.0
    assert payload["scenario_outcome_graph"]["constraints"]
    assert payload["news_vs_regime_assessments"]
    assert payload["domain_analyst_report_extension"]["self_check_horizons"] == ["1d", "5d", "20d", "60d", "120d"]
    assert "buy_sell_hold" in payload["regime_context_vector"]["forbidden_outputs"]
    assert any(item["decision"] == "integrated_as_review_schema" for item in payload["thinking_harvest_decisions"])


def test_regime_scenario_packet_handles_empty_event_packet(tmp_path):
    event_path = _write_json(
        tmp_path / "events.json",
        {
            "mode": "domain_analyst_event_interpretation_packet",
            "summary": {"packet_status": "domain_analyst_event_interpretation_ready"},
            "context_regime_snapshot": {},
            "event_interpretation_packets": [],
        },
    )

    payload = DomainAnalystRegimeScenarioPacket(tmp_path / "reports").build(
        event_interpretation_json=event_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "domain_analyst_regime_scenario_ready_with_review_items"
    assert payload["summary"]["source_event_packet_count"] == 0
    assert payload["summary"]["probability_mass_valid"] is True
    assert payload["summary"]["can_trade"] is False
    assert any(check["code"] == "event_packets_present" and check["status"] == "warn" for check in payload["review_checks"])
    assert payload["evidence_gap_priorities"]


def test_regime_scenario_packet_saves_markdown_and_cli_runs(tmp_path):
    evidence_pack = _write_json(tmp_path / "evidence_pack.json", _evidence_pack_fixture())
    event_payload = DomainAnalystEventInterpretationPacket(tmp_path / "events").build(
        evidence_pack_json=evidence_pack,
        domain_id="semiconductor_ai_infrastructure",
    )

    payload = DomainAnalystRegimeScenarioPacket(tmp_path / "reports").build(
        event_interpretation_json=event_payload["saved_paths"]["latest_json"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Domain Analyst Regime Scenario Packet" in markdown
    assert "Can create execution recommendation: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_regime_scenario_packet.py"),
            "--event-interpretation-json",
            str(event_payload["saved_paths"]["latest_json"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Can create regime-context scenario analysis: True" in result.stdout
    assert "Can create execution recommendation: False" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()


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


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
