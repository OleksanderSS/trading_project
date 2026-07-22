from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from dean_os.draft.dean_os_agent_system_v7.dean_os.briefing_contract import DailyBriefingBuilder
from dean_os.draft.dean_os_agent_system_v7.dean_os.daily_agent_run import DailyAgentRun
from dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_catalog import EvidenceCatalogBuilder, SQLiteEvidenceCatalog
from dean_os.draft.dean_os_agent_system_v7.dean_os.minimal_system import MinimalSystemRunResult
from dean_os.draft.dean_os_agent_system_v7.dean_os.replay_scheduler import ReplayScheduleItem, ReplayScheduler
from dean_os.schemas import ConsensusDecision, MarketContext


AS_OF = "2026-07-12T12:00:00+00:00"


def _decision() -> ConsensusDecision:
    return ConsensusDecision(
        decision_id="decision_test",
        decision="watchlist",
        final_score=0.1,
        confidence=0.6,
        reasons=["bounded analytical test"],
        risks=["uncalibrated probabilities"],
    )


def _system_result() -> MinimalSystemRunResult:
    return MinimalSystemRunResult(
        run_id="minimal_test",
        status="completed",
        domain_id="semiconductor_ai_infrastructure",
        decision=_decision(),
        context_grid={
            "schema_version": "dean_context_grid_v1",
            "domain_id": "semiconductor_ai_infrastructure",
            "as_of": AS_OF,
            "status": "completed",
            "nodes": [{"level": "global", "dimensions": {"economic_phase": {"state": "expansion"}}}],
        },
        indicator_state_grid={"schema_version": "dean_indicator_state_grid_v1", "observations": []},
        world_state_snapshot={
            "snapshot_id": "snapshot_test",
            "domain_id": "semiconductor_ai_infrastructure",
            "as_of": AS_OF,
        },
        world_model_event_learning={
            "classified_events": [{
                "event_id": "event_1",
                "event_type": "supply_expansion",
                "summary": "HBM packaging capacity expansion",
                "evidence_ids": ["evidence_hbm"],
            }],
            "evidence_gaps": [{"description": "Need verified shipment data"}],
            "scenario_outcome_graph": {
                "scenario_graph_id": "scenario_graph_1",
                "nodes": [
                    {"scenario_node_id": "base", "label": "Capacity catches up", "probability": 0.6},
                    {"scenario_node_id": "tight", "label": "Bottleneck persists", "probability": 0.4},
                ],
                "edges": [],
            },
            "replay_tasks": [{
                "task_id": "replay_h1_20d",
                "hypothesis_id": "h1",
                "scenario_graph_id": "scenario_graph_1",
                "as_of": AS_OF,
                "horizon_days": 20,
                "due_at": (datetime.fromisoformat(AS_OF) + timedelta(days=20)).isoformat(),
            }],
        },
    )


class FakeSystem:
    async def run(self, context: MarketContext) -> MinimalSystemRunResult:
        return _system_result()


def test_evidence_catalog_is_append_only_and_point_in_time(tmp_path):
    builder = EvidenceCatalogBuilder()
    record = builder.build_record(
        {
            "evidence_id": "evidence_hbm",
            "source_type": "news",
            "source": "official_company_release",
            "title": "HBM packaging capacity expansion",
            "text": "Capacity expansion announced.",
            "published_at": "2026-07-12T10:00:00+00:00",
            "available_at": "2026-07-12T10:05:00+00:00",
            "ingested_at": AS_OF,
            "evidence_lanes": ["supply", "capex"],
        },
        domain_id="semiconductor_ai_infrastructure",
        as_of=AS_OF,
    )
    catalog = SQLiteEvidenceCatalog(tmp_path / "evidence.sqlite3")
    assert catalog.append(record).status == "stored"
    assert catalog.append(record).status == "already_exists"
    assert catalog.get("evidence_hbm") == record
    assert catalog.list_records(domain_id=record.domain_id, available_before=AS_OF) == [record]


def test_future_evidence_is_marked_invalid():
    record = EvidenceCatalogBuilder().build_record(
        {
            "source_type": "news",
            "source": "test",
            "title": "Future item",
            "text": "not yet available",
            "available_at": "2026-07-13T00:00:00+00:00",
            "ingested_at": "2026-07-13T00:00:00+00:00",
        },
        domain_id="semiconductor_ai_infrastructure",
        as_of=AS_OF,
    )
    assert record.point_in_time_status == "invalid"


def test_briefing_enforces_coverage_gate_and_probability_mass():
    briefing = DailyBriefingBuilder().build(
        run_result=_system_result(),
        required_coverage=[
            {"coverage_id": "supply", "label": "Supply"},
            {"coverage_id": "logistics", "label": "Logistics"},
        ],
        evidence_records=[{"evidence_id": "evidence_hbm", "evidence_lanes": ["supply"], "sectors": []}],
    )
    statuses = {item.coverage_id: item.status for item in briefing.mandatory_coverage_gate}
    assert statuses == {"supply": "material_update", "logistics": "no_credible_material_update"}
    assert briefing.scenario_probabilities["probability_mass"] == pytest.approx(1.0)
    assert briefing.review_only is True


def test_replay_scheduler_only_returns_due_tasks():
    scheduler = ReplayScheduler()
    task = scheduler.build_from_run_result(_system_result())[0]
    assert scheduler.due([task], as_of=AS_OF) == []
    due = scheduler.due([task], as_of=(datetime.fromisoformat(AS_OF) + timedelta(days=21)).isoformat())
    assert due[0].status == "due_pending_evidence"
    assert due[0].can_write_learning_memory is False


@pytest.mark.asyncio
async def test_daily_agent_run_composes_manifest_briefing_and_replay(tmp_path):
    runner = DailyAgentRun(
        FakeSystem(),
        domain_id="semiconductor_ai_infrastructure",
        evidence_catalog=SQLiteEvidenceCatalog(tmp_path / "evidence.sqlite3"),
        required_coverage=[
            {"coverage_id": "supply", "label": "Supply"},
            {"coverage_id": "capex", "label": "Capex"},
            {"coverage_id": "logistics", "label": "Logistics"},
        ],
    )
    context = MarketContext(as_of=AS_OF, metadata={"knowledge_cutoff": AS_OF})
    result = await runner.run(
        context,
        evidence_payloads=[
            {
                "evidence_id": "evidence_hbm",
                "source_type": "news",
                "source": "official_company_release",
                "title": "HBM packaging capacity expansion",
                "text": "Capacity expansion announced.",
                "available_at": "2026-07-12T10:00:00+00:00",
                "ingested_at": AS_OF,
                "evidence_lanes": ["supply", "capex"],
            },
            {
                "evidence_id": "future_item",
                "source_type": "news",
                "source": "future",
                "title": "Future evidence",
                "text": "Future.",
                "available_at": "2026-07-13T10:00:00+00:00",
                "ingested_at": "2026-07-13T10:00:00+00:00",
            },
        ],
    )
    assert result.status == "partial"
    assert result.evidence_manifest.status == "partial"
    assert result.evidence_manifest.evidence_ids == ["evidence_hbm"]
    assert len(result.replay_schedule) == 1
    assert result.briefing.mandatory_coverage_gate[0].status == "material_update"
    assert result.safety["can_write_learning_memory"] is False

from dean_os.draft.dean_os_agent_system_v7.dean_os.daily_run_store import DailyRunRecordBuilder, SQLiteDailyRunStore


def test_daily_run_record_links_cross_artifact_hashes(tmp_path):
    result = {
        "daily_run_id": "daily_1",
        "status": "completed",
        "domain_id": "semiconductor_ai_infrastructure",
        "as_of": AS_OF,
        "knowledge_cutoff": AS_OF,
        "evidence_manifest": {"acquisition_run_id": "ev_1", "content_hash": "a" * 64},
        "system_result": {
            "run_id": "sys_1",
            "world_state_snapshot": {"snapshot_id": "ws_1", "integrity": {"content_hash": "b" * 64}},
        },
        "briefing": {"briefing_id": "brief_1", "domain_id": "semiconductor_ai_infrastructure"},
        "replay_schedule": [{"task_id": "task_1"}],
        "due_replay_tasks": [],
        "safety": {
            "can_trade": False,
            "can_write_production_config": False,
            "can_promote_model": False,
            "can_write_learning_memory": False,
            "human_review_required": True,
        },
    }
    record = DailyRunRecordBuilder().build(result)
    store = SQLiteDailyRunStore(tmp_path / "daily.sqlite3")
    assert store.append(record).status == "stored"
    assert store.append(record).status == "already_exists"
    assert store.get("daily_1") == record
