from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from dean_os.draft.dean_os_agent_system_v7.dean_os.daily_agent_run import DailyAgentRun
from dean_os.draft.dean_os_agent_system_v7.dean_os.full_system_orchestrator import create_full_agent_system
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_stage03_bridge import PipelineStage03Bridge
from dean_os.schemas import MarketContext
from dean_os.draft.dean_os_agent_system_v7.dean_os.system_topology import (
    BranchId,
    BranchRunStatus,
    SystemTopology,
    load_default_system_topology,
)


AS_OF = "2026-07-12T12:00:00+00:00"


def test_default_topology_is_acyclic_and_covers_all_canonical_branches():
    topology = load_default_system_topology()
    order = [item.branch_id for item in topology.execution_order()]
    assert set(order) == {str(item) for item in BranchId}
    assert order.index("pipeline_stage03_intake") < order.index("pipeline_control")
    assert order.index("evidence_intelligence") < order.index("domain_analysis")
    assert order.index("domain_analysis") < order.index("world_model")
    assert order.index("world_model") < order.index("replay_evaluation")
    assert order.index("governance_review") < order.index("daily_audit")
    assert len(topology.topology_hash) == 64


def test_topology_rejects_dependency_cycles():
    raw = load_default_system_topology().model_dump(mode="json")
    raw["topology_hash"] = ""
    raw["branches"][0]["depends_on"] = ["daily_audit"]
    with pytest.raises(ValueError, match="cycle"):
        SystemTopology(**raw)


def test_stage03_bridge_normalizes_existing_pipeline_news_without_running_later_stages():
    result = {
        "status": "completed",
        "requested_stages": [0, 1, 2, 3, 4, 7],
        "results": {
            "raw_data": {"rss_news": [{"headline": "HBM supply expands", "timestamp": AS_OF}]},
            "processed_data": {"news": [{"title": "Processed HBM item", "published_at": AS_OF}]},
            "features_df": [{"ticker": "NVDA", "feature": 1.0}],
        },
    }
    packet = PipelineStage03Bridge().build_packet(
        result,
        as_of=AS_OF,
        knowledge_cutoff=AS_OF,
    )
    assert packet.stages_present == [0, 1, 2, 3]
    assert len(packet.news_items) == 2
    assert packet.safety["stages_above_3_allowed"] is False
    assert any("Ignored stage identifiers" in item for item in packet.warnings)
    assert len(packet.content_hash) == 64


class _CaptureSystem:
    def __init__(self):
        self.context = None

    async def run(self, context):
        self.context = context

        class _Result:
            status = "completed"
            run_id = "capture_run"
            world_model_event_learning = {"hypotheses": [], "scenario_outcome_graph": {}}
            world_state_snapshot = {}

            def model_dump(self, mode="json"):
                return {
                    "status": "completed",
                    "run_id": self.run_id,
                    "decision": {
                        "decision": "watchlist",
                        "confidence": 0.5,
                        "reasons": [],
                        "risks": [],
                    },
                    "agent_reports": [],
                    "context_grid": {},
                    "indicator_state_grid": {},
                    "world_model_event_learning": self.world_model_event_learning,
                    "world_state_snapshot": {},
                    "world_state_persistence": {"status": "disabled"},
                }

        return _Result()


@pytest.mark.asyncio
async def test_daily_runner_removes_rejected_future_news_before_agents_see_context():
    system = _CaptureSystem()
    runner = DailyAgentRun(
        system,
        domain_id="semiconductor_ai_infrastructure",
        required_coverage=[{"coverage_id": "supply", "label": "Supply"}],
    )
    future = (datetime.fromisoformat(AS_OF) + timedelta(days=1)).isoformat()
    context = MarketContext(
        as_of=AS_OF,
        metadata={"knowledge_cutoff": AS_OF},
        news=[
            {
                "evidence_id": "accepted",
                "source_type": "news",
                "source": "official_company_release",
                "title": "Accepted news",
                "text": "Known at cutoff",
                "available_at": AS_OF,
            },
            {
                "evidence_id": "future",
                "source_type": "news",
                "source": "future",
                "title": "Future news",
                "text": "Not known yet",
                "available_at": future,
                "ingested_at": future,
            },
        ],
    )
    result = await runner.run(context)
    assert [item["evidence_id"] for item in system.context.news] == ["accepted"]
    assert result.evidence_manifest.evidence_ids == ["accepted"]
    assert result.evidence_manifest.rejected_items[0]["reason"] == "evidence_available_after_knowledge_cutoff"


@pytest.mark.asyncio
async def test_full_orchestrator_runs_all_branch_skeletons_from_stage03_outputs(tmp_path):
    orchestrator = create_full_agent_system(
        project_root=".",
        domain_id="semiconductor_ai_infrastructure",
        soft_mode=True,
        persistence_enabled=True,
        reports_root=tmp_path / "reports",
        briefing_output_dir=tmp_path / "briefings",
    )
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA"],
        timeframes=["1d"],
        metadata={"knowledge_cutoff": AS_OF},
    )
    result = await orchestrator.run(
        context,
        pipeline_stage03_result={
            "status": "completed",
            "requested_stages": [0, 1, 2, 3],
            "results": {
                "raw_data": {
                    "rss_news": [
                        {
                            "evidence_id": "hbm_stage03",
                            "source": "official_company_release",
                            "headline": "HBM advanced packaging capacity expands",
                            "description": "Supplier announced additional capacity and capex.",
                            "timestamp": AS_OF,
                            "evidence_lanes": ["supply", "capex"],
                        }
                    ]
                },
                "processed_data": {"status": "completed"},
                "features_df": [{"ticker": "NVDA", "feature": 1.0}],
            },
        },
    )
    branch_ids = [item.branch_id for item in result.manifest.branch_records]
    assert branch_ids == [item.branch_id for item in result.topology.execution_order()]
    assert set(branch_ids) == {str(item) for item in BranchId}
    assert result.pipeline_stage03_packet.stages_present == [0, 1, 2, 3]
    control_record = next(
        item for item in result.manifest.branch_records if item.branch_id == "pipeline_control"
    )
    assert control_record.status == BranchRunStatus.COMPLETED
    assert result.daily_run.system_result["pipeline_execution_policy"]["allowed_stages"] == [0, 1, 2, 3]
    assert result.daily_run.persisted_run_record is not None
    assert result.safety["can_trade"] is False
    assert result.manifest.safety["can_write_learning_memory"] is False
    assert result.manifest.status in {
        BranchRunStatus.COMPLETED,
        BranchRunStatus.PARTIAL,
        BranchRunStatus.BLOCKED,
    }
    assert result.branch_outputs["domain_analysis"]["agent_reports"]
    assert len(result.manifest.content_hash) == 64
