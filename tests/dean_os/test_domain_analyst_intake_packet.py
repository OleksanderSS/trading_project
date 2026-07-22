from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.domain_analyst_intake_packet import DomainAnalystIntakePacket


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _document(
    document_id: str,
    title: str,
    text: str,
    *,
    source_type: str = "news",
    tickers: list[str] | None = None,
    sectors: list[str] | None = None,
    tags: list[str] | None = None,
    published_at: str = "2026-05-01T00:00:00+00:00",
) -> dict:
    return {
        "document_id": document_id,
        "title": title,
        "source_type": source_type,
        "text": text,
        "uri": f"fixture://{document_id}",
        "published_at": published_at,
        "tickers": tickers or [],
        "sectors": sectors or ["semiconductor"],
        "tags": tags or ["ai_cycle"],
        "metadata": {},
    }


def _evidence_pack() -> dict:
    documents = [
        _document(
            "doc_sector_demand",
            "AI infrastructure demand expands",
            "AI infrastructure demand and GPU accelerator orders are increasing across data center customers.",
            source_type="article",
            tags=["ai_infrastructure"],
        ),
        _document(
            "doc_capex",
            "Hyperscaler capex supports chips",
            "Hyperscaler capex and cloud spending remain strong for data center buildout.",
            source_type="report",
            tags=["capex_cycle"],
        ),
        _document(
            "doc_supply",
            "Advanced packaging capacity update",
            "Foundry and advanced packaging capacity remain tight while HBM memory supply improves.",
            source_type="report",
            tags=["supply_chain"],
        ),
        _document(
            "doc_policy",
            "Export control update",
            "New export control policy creates China and Taiwan geopolitical risk for semiconductor equipment.",
            source_type="news",
            tags=["policy"],
        ),
        _document(
            "doc_market_amd",
            "AMD relative strength",
            "AMD shares outperform as market relative strength confirms stronger AI server expectations.",
            source_type="news",
            tickers=["AMD"],
            tags=["market_confirmation"],
        ),
    ]
    return {
        "run_id": "analyst_evidence_pack_fixture",
        "mode": "analyst_evidence_pack",
        "coverage": {
            "document_count": len(documents),
            "data_quality": "strong",
            "research_ready": True,
            "agent_lab_ready": True,
            "tickers": ["AMD", "NVDA"],
            "sectors": ["semiconductor"],
            "by_source_type": {"news": 3, "article": 1, "report": 2},
            "warning_count": 0,
            "dropped_count": 0,
        },
        "documents": documents,
    }


def _source_gate(**summary_overrides) -> dict:
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
        "validation_checks": [],
    }


def test_domain_analyst_intake_normalizes_source_pack_into_review_only_analyst_report(tmp_path):
    evidence_path = _write_json(tmp_path / "evidence" / "latest.json", _evidence_pack())
    gate_path = _write_json(tmp_path / "gate" / "latest.json", _source_gate())

    payload = DomainAnalystIntakePacket(tmp_path / "reports").build(
        evidence_pack_json=evidence_path,
        source_gate_json=gate_path,
        domain_id="semiconductor_ai_infrastructure",
        tickers=["AMD", "NVDA"],
        sectors=["semiconductor"],
        save=False,
    )

    assert payload["summary"]["intake_status"] == "domain_analyst_intake_ready"
    assert payload["summary"]["evidence_item_count"] == 5
    assert payload["summary"]["ticker_direct_count"] == 1
    assert payload["summary"]["sector_or_domain_count"] >= 3
    assert payload["summary"]["can_create_direct_ticker_thesis_without_bridge"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["analyst_report"]["recommendation"] == "partial_ready_for_review"
    assert payload["analyst_report"]["live_execution_allowed"] is False
    assert payload["analyst_report"]["ticker_basket"]["direct_ready_count"] == 1
    assert payload["analyst_report"]["ticker_basket"]["basket_candidate_count"] == 1
    assert payload["domain_profile_snapshot"]["source_registry_policy"]["policy_id"] == "default_domain_source_registry_policy_v1"
    assert payload["domain_profile_snapshot"]["evidence_scoring_policy"]["policy_id"] == "default_domain_evidence_scoring_policy_v1"
    assert "research_recommendation" in payload["domain_profile_snapshot"]["review_output_policy"]["allowed_review_outputs"]
    assert not payload["summary"]["required_evidence_missing"]


def test_domain_analyst_intake_blocks_when_source_gate_is_unsafe(tmp_path):
    evidence_path = _write_json(tmp_path / "evidence" / "latest.json", _evidence_pack())
    gate_path = _write_json(tmp_path / "gate" / "latest.json", _source_gate(can_trade=True))

    payload = DomainAnalystIntakePacket(tmp_path / "reports").build(
        evidence_pack_json=evidence_path,
        source_gate_json=gate_path,
        domain_id="semiconductor_ai_infrastructure",
        tickers=["AMD"],
        sectors=["semiconductor"],
        save=False,
    )

    assert payload["summary"]["intake_status"] == "blocked_domain_analyst_intake"
    assert payload["summary"]["analyst_report_created"] is False
    assert payload["summary"]["can_run_domain_analyst"] is False
    assert any(check["code"] == "source_gate_allows_domain_research" and check["status"] == "fail" for check in payload["review_checks"])
    assert payload["summary"]["can_trade"] is False


def test_domain_analyst_intake_classifies_capital_spending_as_capex_cycle(tmp_path):
    evidence_pack = _evidence_pack()
    evidence_pack["documents"] = [
        _document(
            "doc_capital_spending",
            "Capital spending rises for data center memory",
            "Cloud and AI infrastructure capital spending supports data center investment and memory demand.",
            source_type="news",
            tags=["ai_cycle"],
        )
    ]
    evidence_pack["coverage"]["document_count"] = 1
    evidence_pack["coverage"]["tickers"] = []
    evidence_pack["coverage"]["by_source_type"] = {"news": 1}
    evidence_path = _write_json(tmp_path / "evidence" / "latest.json", evidence_pack)
    gate_path = _write_json(tmp_path / "gate" / "latest.json", _source_gate())

    payload = DomainAnalystIntakePacket(tmp_path / "reports").build(
        evidence_pack_json=evidence_path,
        source_gate_json=gate_path,
        domain_id="semiconductor_ai_infrastructure",
        sectors=["semiconductor"],
        save=False,
    )

    assert payload["summary"]["ticker_direct_count"] == 0
    assert payload["evidence_type_summary"] == {"capex_cycle": 1}
    assert payload["evidence_items"][0]["directness"] == "sector"


def test_domain_analyst_intake_saves_markdown_and_cli_runs(tmp_path):
    evidence_path = _write_json(tmp_path / "evidence" / "latest.json", _evidence_pack())
    gate_path = _write_json(tmp_path / "gate" / "latest.json", _source_gate())

    payload = DomainAnalystIntakePacket(tmp_path / "reports").build(
        evidence_pack_json=evidence_path,
        source_gate_json=gate_path,
        domain_id="semiconductor_ai_infrastructure",
        tickers=["AMD", "NVDA"],
        sectors=["semiconductor"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can trade: False" in markdown
    assert "Ticker Bridge Guardrail" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_intake_packet.py"),
            "--evidence-pack-json",
            str(evidence_path),
            "--source-gate-json",
            str(gate_path),
            "--domain-id",
            "semiconductor_ai_infrastructure",
            "--tickers",
            "AMD",
            "NVDA",
            "--sectors",
            "semiconductor",
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Intake status: domain_analyst_intake_ready" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
