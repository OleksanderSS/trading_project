from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.domain_analyst_instance_contract import DomainAnalystInstanceContract


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evidence_pack() -> dict:
    return {
        "mode": "analyst_evidence_pack",
        "inputs": {
            "news_data_paths": ["news.parquet"],
            "macro_data_paths": ["macro.parquet"],
            "sector_keywords": ["semiconductor", "chip", "capital spending"],
        },
        "coverage": {
            "document_count": 12,
            "data_quality": "strong",
            "by_source_type": {"news": 8, "report": 4},
            "tickers": [],
            "sectors": ["semiconductor"],
            "warning_count": 0,
            "dropped_count": 0,
        },
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
    return {"mode": "source_evidence_validation_gate", "summary": summary}


def _domain_intake(**summary_overrides) -> dict:
    summary = {
        "intake_status": "domain_analyst_intake_ready_with_warnings",
        "domain_id": "semiconductor_ai_infrastructure",
        "document_count": 12,
        "evidence_item_count": 12,
        "ticker_direct_count": 0,
        "required_evidence_missing": [],
        "analyst_report_created": True,
        "can_create_direct_ticker_thesis_without_bridge": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "mode": "domain_analyst_intake_packet",
        "inputs": {"domain_id": "semiconductor_ai_infrastructure", "sectors": ["semiconductor"]},
        "summary": summary,
        "domain_profile_snapshot": {
            "domain_id": "semiconductor_ai_infrastructure",
            "required_evidence_types": ["sector_demand", "capex_cycle", "supply_chain", "policy_or_geopolitical", "market_confirmation"],
            "useful_evidence_types": ["earnings_guidance", "valuation_context"],
            "ticker_universe_hint": ["AMD", "NVDA", "TSM"],
        },
        "evidence_type_summary": {
            "sector_demand": 3,
            "capex_cycle": 2,
            "supply_chain": 2,
            "policy_or_geopolitical": 2,
            "market_confirmation": 3,
        },
        "directness_summary": {"sector": 8, "macro": 4},
        "analyst_report": {"recommendation": "partial_ready_for_review"},
    }


def _architecture_map() -> dict:
    return {
        "mode": "current_architecture_map",
        "summary": {
            "can_clone_domain_profiles_now": False,
            "can_trade": False,
        },
    }


def _write_inputs(tmp_path: Path, *, source_gate: dict | None = None, domain_intake: dict | None = None) -> dict[str, Path]:
    return {
        "evidence": _write_json(tmp_path / "evidence" / "latest.json", _evidence_pack()),
        "source_gate": _write_json(tmp_path / "gate" / "latest.json", source_gate or _source_gate()),
        "domain_intake": _write_json(tmp_path / "intake" / "latest.json", domain_intake or _domain_intake()),
        "architecture": _write_json(tmp_path / "architecture" / "latest.json", _architecture_map()),
    }


def test_domain_analyst_instance_contract_marks_template_review_ready(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystInstanceContract(tmp_path / "reports").build(
        evidence_pack_json=paths["evidence"],
        source_gate_json=paths["source_gate"],
        domain_intake_json=paths["domain_intake"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["instance_status"] == "domain_analyst_instance_review_ready"
    assert payload["summary"]["can_reuse_as_template_after_manual_review"] is True
    assert payload["summary"]["can_scale_to_other_domains_now"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["portable_template_slots"]["domain_id"] == "semiconductor_ai_infrastructure"
    assert "sector_keywords" in payload["portable_template_slots"]
    assert payload["portable_template_slots"]["source_registry_policy"]["policy_id"] == "default_domain_source_registry_policy_v1"
    assert payload["portable_template_slots"]["ingestion_filter_policy"]["policy_id"] == "default_domain_ingestion_filter_policy_v1"
    assert payload["portable_template_slots"]["evidence_scoring_policy"]["policy_id"] == "default_domain_evidence_scoring_policy_v1"
    assert "self_improvement_proposal" in payload["portable_template_slots"]["review_output_policy"]["allowed_review_outputs"]
    assert any(check["code"] == "sector_thesis_before_ticker_thesis" and check["status"] == "pass" for check in payload["review_checks"])


def test_domain_analyst_instance_contract_blocks_unsafe_source_gate(tmp_path):
    paths = _write_inputs(tmp_path, source_gate=_source_gate(can_trade=True))

    payload = DomainAnalystInstanceContract(tmp_path / "reports").build(
        evidence_pack_json=paths["evidence"],
        source_gate_json=paths["source_gate"],
        domain_intake_json=paths["domain_intake"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["instance_status"] == "blocked_domain_analyst_instance"
    assert payload["summary"]["can_reuse_as_template_after_manual_review"] is False
    assert any(check["code"] == "source_gate_downstream_actions_disabled" and check["status"] == "fail" for check in payload["review_checks"])


def test_domain_analyst_instance_contract_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystInstanceContract(tmp_path / "reports").build(
        evidence_pack_json=paths["evidence"],
        source_gate_json=paths["source_gate"],
        domain_intake_json=paths["domain_intake"],
        architecture_map_json=paths["architecture"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can trade: False" in markdown
    assert "Portable Slots" in markdown
    assert "Source policy: `default_domain_source_registry_policy_v1`" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_instance_contract.py"),
            "--evidence-pack-json",
            str(paths["evidence"]),
            "--source-gate-json",
            str(paths["source_gate"]),
            "--domain-intake-json",
            str(paths["domain_intake"]),
            "--architecture-map-json",
            str(paths["architecture"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Instance status: domain_analyst_instance_review_ready" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
