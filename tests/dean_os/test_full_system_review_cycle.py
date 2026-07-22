from __future__ import annotations

import json
from pathlib import Path

from dean_os.full_system_review_cycle import compose_full_system_review_cycle
from dean_os.schemas import PipelineReport


def test_composed_cycle_discloses_composite_execution_and_downstream_gap(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    manager = tmp_path / "manager.json"
    manager.write_text("{}", encoding="utf-8")
    report = PipelineReport(
        agent_name="semiconductor_pipeline_manager",
        agent_version="0.2.0",
        verdict="needs_more_data",
        confidence=0.0,
        data_quality_score=0.5,
        signal_strength=0.0,
        reasons=["review"],
        evidence=[],
        metrics_snapshot={
            "agent_role": "composite_domain_pipeline_manager",
            "evidence_count": 76,
            "artifact_count": 5,
            "lens_count": 5,
            "hypothesis_count": 3,
            "evidence_gap_count": 11,
            "recommendation": "needs_more_data",
            "stance": "insufficient_data",
            "errors": [],
            "saved_paths": {"latest_json": str(manager)},
            "pipeline_readiness": {
                "status": "pipeline_readiness_ready",
                "is_ready": True,
                "blocking_reasons": [],
            },
            "can_trade": False,
        },
    )
    payload = compose_full_system_review_cycle(
        report=report,
        system_manifest={
            "status": "observed_complete",
            "manifest_sha256": "a" * 64,
            "saved_paths": {"latest_json": str(tmp_path / "manifest.json")},
        },
        topology_path="dean_os/config/system_topology.yaml",
        authorization_ledger_path=tmp_path / "ledger.jsonl",
        artifact_paths={"news": source, "macro": source, "sector_market": source, "policy": source, "fundamental": source},
        timeframe_lane_readiness_path=source,
    )
    assert payload["summary"]["cycle_status"] == "analysis_cycle_completed_downstream_refresh_required"
    assert payload["summary"]["downstream_hash_bound_to_this_cycle"] is False
    assert payload["summary"]["downstream_refresh_required"] == [
        "world_model",
        "replay_evaluation",
        "governance_review",
    ]
    assert payload["summary"]["authorization_ledger_record_count"] == 0
    assert payload["safety"]["composite_execution_disclosed"] is True
    assert payload["safety"]["independent_branch_execution_claimed"] is False
    assert payload["safety"]["can_trade"] is False
    assert len(payload["branch_records"]) == 9
