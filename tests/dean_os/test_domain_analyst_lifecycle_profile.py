from __future__ import annotations

import json
from pathlib import Path

from dean_os.analyst_core.domain_analyst_lifecycle_profile import (
    DomainAnalystLifecycleProfileCompiler,
    DomainAnalystLifecycleProfileReport,
)


def test_semiconductor_source_profile_is_structurally_and_data_ready():
    payload = DomainAnalystLifecycleProfileCompiler().compile(
        "semiconductor_ai_infrastructure"
    )

    assert payload["readiness"]["schema_valid"] is True
    assert payload["readiness"]["can_run_domain_analysis_now"] is True
    assert payload["readiness"]["can_activate_clone_now"] is False
    assert payload["readiness"]["can_trade"] is False


def test_energy_clone_materializes_but_does_not_claim_data_readiness():
    payload = DomainAnalystLifecycleProfileCompiler().compile("energy")

    assert payload["readiness"]["schema_valid"] is True
    assert payload["readiness"]["can_materialize_review_contract"] is True
    assert payload["readiness"]["can_run_domain_analysis_now"] is False
    assert set(payload["readiness"]["missing_context_bindings"]) == {
        "news",
        "official_policy",
        "macro",
        "fundamentals",
        "sector_market",
        "pipeline_context",
    }


def test_source_and_clone_share_fixed_contract_not_domain_overlay():
    compiler = DomainAnalystLifecycleProfileCompiler()
    source = compiler.compile("semiconductor_ai_infrastructure")
    clone = compiler.compile("energy")

    assert source["fixed_contract_sha256"] == clone["fixed_contract_sha256"]
    assert source["domain_overlay_sha256"] != clone["domain_overlay_sha256"]


def test_invalid_horizon_mixing_is_rejected(tmp_path):
    source = Path("dean_os/config/domain_analyst_lifecycle.template.json")
    template = json.loads(source.read_text(encoding="utf-8"))
    template["horizon_policy"]["event_response_days"] = [1, 5, 30, 60, 120]
    path = tmp_path / "bad_template.json"
    path.write_text(json.dumps(template), encoding="utf-8")

    payload = DomainAnalystLifecycleProfileCompiler(path).compile("energy")

    assert payload["readiness"]["schema_valid"] is False
    assert "horizon_contract_mismatch" in payload["readiness"]["structural_blockers"]
    assert (
        "sector_and_event_horizons_overlap"
        in payload["readiness"]["structural_blockers"]
    )


def test_report_records_dry_run_boundary_and_saves(tmp_path):
    payload = DomainAnalystLifecycleProfileReport(tmp_path / "reports").build()

    assert payload["summary"]["can_materialize_clone_contract"] is True
    assert payload["summary"]["can_run_clone_domain_analysis_now"] is False
    assert payload["summary"]["can_activate_clone_now"] is False
    assert payload["summary"]["can_trade"] is False
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")
    assert "Control clone: `energy`" in markdown
    assert "Can run clone analysis now: False" in markdown
