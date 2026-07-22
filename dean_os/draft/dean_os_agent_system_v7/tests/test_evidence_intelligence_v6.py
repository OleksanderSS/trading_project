from __future__ import annotations

from pathlib import Path

import pytest

from dean_os.draft.dean_os_agent_system_v7.dean_os.briefing_contract import DailyBriefingBuilder
from dean_os.draft.dean_os_agent_system_v7.dean_os.briefing_renderer import DailyBriefingRenderer
from dean_os.draft.dean_os_agent_system_v7.dean_os.collector_routing import DomainCollectorRouter
from dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_catalog import EvidenceCatalogBuilder
from dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_dedup import SemanticEvidenceDeduplicator
from dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_gap_planner_v2 import EvidenceGapPlanner
from dean_os.draft.dean_os_agent_system_v7.dean_os.operator_review_inbox_v2 import OperatorReviewInboxBuilder, SQLiteOperatorReviewInbox
from dean_os.draft.dean_os_agent_system_v7.dean_os.source_credibility import SourceCredibilityRegistry

from tests.test_daily_agent_run import AS_OF, _system_result


def test_source_credibility_is_domain_aware_and_not_truth_oracle():
    registry = SourceCredibilityRegistry.from_domain_profile("semiconductor_ai_infrastructure")
    official = registry.assess({"source": "official_company_release", "source_type": "report"}, point_in_time_status="valid")
    weak = registry.assess({"source": "stocktwits", "source_type": "news"}, point_in_time_status="valid")
    assert official.source_tier == "tier_1_core_evidence"
    assert official.credibility_score > weak.credibility_score
    assert weak.decision_use == "lead_only"


def test_catalog_uses_knowledge_cutoff_not_only_as_of():
    record = EvidenceCatalogBuilder().build_record(
        {
            "source": "official_company_release",
            "source_type": "report",
            "title": "Late source",
            "text": "Known after cutoff but before analysis run.",
            "available_at": "2026-07-12T11:30:00+00:00",
            "ingested_at": AS_OF,
        },
        domain_id="semiconductor_ai_infrastructure",
        as_of=AS_OF,
        knowledge_cutoff="2026-07-12T11:00:00+00:00",
    )
    assert record.point_in_time_status == "invalid"
    assert record.credibility_score == 0.0


def test_deduplicator_suppresses_same_source_but_retains_independent_corroboration():
    payloads = [
        {"source": "Reuters", "title": "HBM supply expands", "text": "HBM packaging supply expands sharply in 2026."},
        {"source": "Reuters", "title": "HBM supply expands", "text": "HBM packaging supply expands sharply in 2026."},
        {"source": "Bloomberg", "title": "HBM supply expands", "text": "HBM packaging supply expands sharply in 2026."},
    ]
    result = SemanticEvidenceDeduplicator().analyze(payloads)
    statuses = [item.status for item in result.decisions]
    assert statuses == ["unique", "exact_duplicate", "exact_duplicate"] or statuses == ["unique", "exact_duplicate", "independent_corroboration"]
    # Exact identical syndicated text is suppressed even across sources; independently worded corroboration is retained.
    varied = payloads[:2] + [{"source": "Bloomberg", "title": "More HBM output", "text": "Advanced packaging bottlenecks may ease as new capacity arrives."}]
    varied_result = SemanticEvidenceDeduplicator(corroboration_threshold=0.25).analyze(varied)
    assert varied_result.decisions[2].status in {"independent_corroboration", "unique"}


def test_weak_lane_becomes_evidence_gap_and_routes_are_non_executing():
    briefing = DailyBriefingBuilder().build(
        run_result=_system_result(),
        required_coverage=[{"coverage_id": "supply", "label": "Supply"}],
        evidence_records=[{
            "evidence_id": "weak_1",
            "evidence_lanes": ["supply"],
            "sectors": [],
            "credibility_score": 0.3,
            "point_in_time_status": "valid",
        }],
    )
    assert briefing.mandatory_coverage_gate[0].status == "evidence_gap"
    plan = EvidenceGapPlanner("semiconductor_ai_infrastructure").build(
        briefing=briefing,
        evidence_records=[{
            "evidence_id": "weak_1",
            "source_name": "stocktwits",
            "evidence_lanes": ["supply"],
            "credibility_score": 0.3,
        }],
    )
    assert plan.tasks
    assert all(route.network_execution_allowed is False for route in plan.tasks[0].collector_routes)
    assert plan.tasks[0].can_write_learning_memory is False


def test_briefing_renderer_and_review_inbox(tmp_path: Path):
    briefing = DailyBriefingBuilder().build(
        run_result=_system_result(),
        required_coverage=[{"coverage_id": "supply", "label": "Supply"}],
        evidence_records=[{
            "evidence_id": "evidence_hbm",
            "evidence_lanes": ["supply"],
            "sectors": [],
            "credibility_score": 0.95,
            "point_in_time_status": "valid",
        }],
    )
    plan = EvidenceGapPlanner("semiconductor_ai_infrastructure").build(briefing=briefing, evidence_records=[])
    md, html = DailyBriefingRenderer().save(briefing, tmp_path / "briefings", evidence_gap_plan=plan)
    assert "Mandatory Coverage Gate" in md.read_text(encoding="utf-8")
    assert "<!doctype html>" in html.read_text(encoding="utf-8")

    daily = {
        "daily_run_id": "daily_v6",
        "status": "completed",
        "domain_id": "semiconductor_ai_infrastructure",
        "as_of": AS_OF,
        "briefing": briefing.model_dump(mode="json"),
        "due_replay_tasks": [],
        "evidence_records": [],
    }
    items = OperatorReviewInboxBuilder().build(daily, evidence_gap_plan=plan)
    inbox = SQLiteOperatorReviewInbox(tmp_path / "review.sqlite3")
    for item in items:
        assert inbox.append(item) == "stored"
    assert inbox.list_pending(domain_id="semiconductor_ai_infrastructure")


def test_collector_router_is_configuration_only():
    routes = DomainCollectorRouter("semiconductor_ai_infrastructure").routes_for("market_confirmation")
    assert any(route.collector_kind == "market_data" for route in routes)
    assert all(route.review_only for route in routes)
    assert all(not route.network_execution_allowed for route in routes)

@pytest.mark.asyncio
async def test_daily_run_integrates_dedup_gap_plan_rendering_and_review_inbox(tmp_path: Path):
    from dean_os.draft.dean_os_agent_system_v7.dean_os.daily_agent_run import DailyAgentRun
    from dean_os.schemas import MarketContext
    from test_daily_agent_run import FakeSystem

    inbox = SQLiteOperatorReviewInbox(tmp_path / "operator_review.sqlite3")
    runner = DailyAgentRun(
        FakeSystem(),
        domain_id="semiconductor_ai_infrastructure",
        review_inbox=inbox,
        briefing_output_dir=str(tmp_path / "rendered"),
    )
    result = await runner.run(
        MarketContext(as_of=AS_OF, metadata={"knowledge_cutoff": AS_OF}),
        evidence_payloads=[
            {
                "evidence_id": "official_supply",
                "source": "official_company_release",
                "source_type": "report",
                "title": "HBM capacity expansion",
                "text": "New advanced packaging capacity is planned.",
                "available_at": "2026-07-12T10:00:00+00:00",
                "ingested_at": AS_OF,
                "evidence_lanes": ["supply", "capex"],
            },
            {
                "evidence_id": "duplicate_supply",
                "source": "official_company_release",
                "source_type": "report",
                "title": "HBM capacity expansion",
                "text": "New advanced packaging capacity is planned.",
                "available_at": "2026-07-12T10:00:00+00:00",
                "ingested_at": AS_OF,
                "evidence_lanes": ["supply", "capex"],
            },
        ],
    )
    assert result.evidence_dedup is not None
    assert result.evidence_dedup.suppressed_indices == [1]
    assert result.evidence_manifest.suppressed_items
    assert result.evidence_gap_plan is not None
    assert result.review_inbox_items
    assert Path(result.rendered_artifacts["markdown"]).exists()
    assert inbox.list_pending(domain_id="semiconductor_ai_infrastructure")

@pytest.mark.asyncio
async def test_daily_run_record_binds_v6_artifacts(tmp_path: Path):
    from dean_os.draft.dean_os_agent_system_v7.dean_os.daily_agent_run import DailyAgentRun
    from dean_os.draft.dean_os_agent_system_v7.dean_os.daily_run_store import SQLiteDailyRunStore
    from dean_os.schemas import MarketContext
    from test_daily_agent_run import FakeSystem

    run_store = SQLiteDailyRunStore(tmp_path / "daily_runs.sqlite3")
    runner = DailyAgentRun(
        FakeSystem(),
        domain_id="semiconductor_ai_infrastructure",
        daily_run_store=run_store,
        briefing_output_dir=str(tmp_path / "rendered"),
    )
    result = await runner.run(
        MarketContext(as_of=AS_OF, metadata={"knowledge_cutoff": AS_OF}),
        evidence_payloads=[{
            "evidence_id": "official_supply_v6",
            "source": "official_company_release",
            "source_type": "report",
            "title": "HBM capacity expansion",
            "text": "New advanced packaging capacity is planned.",
            "available_at": "2026-07-12T10:00:00+00:00",
            "ingested_at": AS_OF,
            "evidence_lanes": ["supply"],
        }],
    )
    record = result.persisted_run_record
    assert record is not None
    assert record.evidence_dedup_hash
    assert record.evidence_gap_plan_hash
    assert record.review_inbox_item_ids
    assert set(record.rendered_artifact_hashes) == {"markdown", "html"}
    assert run_store.get(result.daily_run_id) == record
