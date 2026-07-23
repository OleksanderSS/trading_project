from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dean_os.agents.working_domain_analyst import WorkingDomainAnalystAgent
from dean_os.analyst_core.analyst_knowledge_readiness import AnalystKnowledgeReadiness
from dean_os.analyst_knowledge.schemas import (
    KnowledgeItem,
    KnowledgePack,
    KnowledgeQuery,
    KnowledgeSource,
)
from dean_os.analyst_knowledge.store import LocalKnowledgeStore


DOMAIN_ID = "semiconductor_ai_infrastructure"
SOURCE_HASH = "a" * 64


def _source(
    source_id: str,
    *,
    published_at: str = "2026-04-01T10:00:00+00:00",
    retrieved_at: str = "2026-04-01T10:05:00+00:00",
    content_sha256: str | None = SOURCE_HASH,
) -> KnowledgeSource:
    return KnowledgeSource(
        source_id=source_id,
        title=f"Source {source_id}",
        source_type="filing",
        reference=f"https://example.test/{source_id}",
        published_at=published_at,
        retrieved_at=retrieved_at,
        content_sha256=content_sha256,
        reliability="high",
        allowed_uses=["context", "evidence", "review"],
    )


def _item(
    item_id: str,
    source_id: str | None,
    *,
    updated_at: str = "2026-04-02T00:00:00+00:00",
) -> KnowledgeItem:
    return KnowledgeItem(
        item_id=item_id,
        domain_id=DOMAIN_ID,
        item_type="driver",
        title=f"AMD capex driver {item_id}",
        body="AMD data-center capex demand is relevant to this test.",
        tickers=["AMD"],
        tags=["capex", "datacenter"],
        source_ids=[source_id] if source_id else [],
        confidence=0.8,
        importance=4,
        updated_at=updated_at,
    )


def _pack(
    *,
    pack_id: str = "pack_test",
    version: str = "1.0.0",
    sources: list[KnowledgeSource],
    items: list[KnowledgeItem],
) -> KnowledgePack:
    return KnowledgePack(
        pack_id=pack_id,
        domain_id=DOMAIN_ID,
        name="Point-in-time test pack",
        version=version,
        sources=sources,
        items=items,
    )


def _strict_query() -> KnowledgeQuery:
    return KnowledgeQuery(
        query="AMD capex data center",
        domain_id=DOMAIN_ID,
        tickers=["AMD"],
        as_of="2026-06-01T00:00:00+00:00",
        require_point_in_time=True,
        require_source_provenance=True,
        intended_use="evidence",
    )


def test_pack_rejects_dangling_source_reference():
    with pytest.raises(ValidationError, match="unknown source_ids"):
        _pack(sources=[], items=[_item("dangling", "missing_source")])


def test_strict_retrieval_keeps_lineage_and_excludes_future_knowledge(tmp_path):
    store = LocalKnowledgeStore(tmp_path / "store")
    pack = _pack(
        sources=[
            _source("eligible_source"),
            _source(
                "future_source",
                published_at="2026-07-01T00:00:00+00:00",
                retrieved_at="2026-07-01T00:05:00+00:00",
            ),
            _source("future_item_source"),
        ],
        items=[
            _item("eligible_item", "eligible_source"),
            _item("future_source_item", "future_source"),
            _item(
                "future_authored_item",
                "future_item_source",
                updated_at="2026-07-02T00:00:00+00:00",
            ),
            _item("source_free_item", None),
        ],
    )
    store.add_pack(pack)

    result = store.search(_strict_query())

    assert [hit.item.item_id for hit in result.hits] == ["eligible_item"]
    assert result.hits[0].point_in_time["status"] == "point_in_time_compatible"
    assert result.hits[0].sources[0].source_id == "eligible_source"
    assert len(result.hits[0].lineage["pack_sha256"]) == 64
    assert store.list_packs()["pack_test"]["source_count"] == 3
    assert len(store.list_packs()["pack_test"]["pack_sha256"]) == 64

    exclusions = {item.item_id: item for item in result.exclusions}
    assert "item_source_ids_missing" in exclusions["source_free_item"].reasons
    assert "item_updated_after_as_of" in exclusions["future_authored_item"].reasons
    assert any(
        "published_after_as_of" in reason
        for reason in exclusions["future_source_item"].reasons
    )
    assert result.audit["status"] == "eligible_with_exclusions"


def test_strict_retrieval_rejects_source_not_yet_retrieved(tmp_path):
    store = LocalKnowledgeStore(tmp_path / "store")
    store.add_pack(
        _pack(
            sources=[
                _source(
                    "late_retrieval",
                    published_at="2026-04-01T00:00:00+00:00",
                    retrieved_at="2026-06-02T00:00:00+00:00",
                )
            ],
            items=[_item("late_retrieval_item", "late_retrieval")],
        )
    )

    result = store.search(_strict_query())

    assert not result.hits
    assert result.audit["status"] == "blocked_no_point_in_time_eligible_hits"
    assert any(
        "retrieved_after_as_of" in reason
        for reason in result.exclusions[0].reasons
    )


def test_non_strict_retrieval_remains_compatible_with_source_free_items(tmp_path):
    store = LocalKnowledgeStore(tmp_path / "store")
    store.add_pack(_pack(sources=[], items=[_item("legacy_item", None)]))

    result = store.search(
        KnowledgeQuery(
            query="AMD capex",
            domain_id=DOMAIN_ID,
            tickers=["AMD"],
        )
    )

    assert [hit.item.item_id for hit in result.hits] == ["legacy_item"]
    assert result.hits[0].point_in_time["strict"] is False


def test_store_requires_pack_version_bump_for_changed_item(tmp_path):
    store = LocalKnowledgeStore(tmp_path / "store")
    store.add_pack(
        _pack(
            sources=[_source("stable_source")],
            items=[_item("stable_item", "stable_source")],
        )
    )
    changed = _item("stable_item", "stable_source")
    changed.body = "Changed body without a pack version bump."

    with pytest.raises(ValueError, match="without a version bump"):
        store.add_pack(
            _pack(
                sources=[_source("stable_source")],
                items=[changed],
            )
        )


def test_query_requires_timezone_for_strict_point_in_time():
    with pytest.raises(ValidationError, match="must include a timezone"):
        KnowledgeQuery(
            query="capex",
            as_of="2026-06-01T00:00:00",
            require_point_in_time=True,
        )


def test_working_domain_analyst_propagates_exact_source_provenance(tmp_path):
    pack = _pack(
        sources=[_source("analyst_source")],
        items=[_item("analyst_item", "analyst_source")],
    )
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(
        json.dumps(pack.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )

    payload = WorkingDomainAnalystAgent(
        store_dir=tmp_path / "store",
        output_dir=tmp_path / "reports",
    ).run(
        question="How does data-center capex affect AMD?",
        domain_id=DOMAIN_ID,
        tickers=["AMD"],
        pack_paths=[pack_path],
        as_of="2026-06-01T00:00:00+00:00",
        save=False,
    )

    assert payload["knowledge_retrieval_gate"]["review_eligible"] is True
    assert payload["retrieval"]["audit"]["strict"] is True
    evidence = payload["analyst_report"]["evidence"][0]
    assert evidence["published_at"] == "2026-04-01T10:00:00+00:00"
    assert evidence["point_in_time"]["status"] == "point_in_time_compatible"
    assert evidence["provenance"]["source_ids"] == ["analyst_source"]
    assert evidence["provenance"]["required_lane_eligible"] is False
    assert evidence["provenance"]["ticker_thesis_eligible"] is False
    assert len(evidence["provenance"]["pack_sha256"]) == 64
    assert payload["safety"]["live_execution_allowed"] is False


def test_knowledge_readiness_reports_empty_store_without_fabricating_readiness(
    tmp_path,
):
    payload = AnalystKnowledgeReadiness(
        store_dir=tmp_path / "empty_store",
        output_dir=tmp_path / "reports",
    ).build(
        as_of="2026-06-01T00:00:00+00:00",
        save=False,
    )

    assert payload["status"] == "knowledge_store_empty_blocked"
    assert payload["summary"]["eligible_item_count"] == 0
    assert payload["summary"]["can_feed_review_only_analyst"] is False
    assert payload["summary"]["can_influence_pipeline_prediction"] is False
    assert payload["summary"]["can_trade"] is False


def test_knowledge_readiness_separates_eligible_and_future_items(tmp_path):
    store_dir = tmp_path / "store"
    store = LocalKnowledgeStore(store_dir)
    store.add_pack(
        _pack(
            sources=[
                _source("ready_source"),
                _source(
                    "future_source",
                    published_at="2026-07-01T00:00:00+00:00",
                    retrieved_at="2026-07-01T00:05:00+00:00",
                ),
            ],
            items=[
                _item("ready_item", "ready_source"),
                _item("future_item", "future_source"),
            ],
        )
    )

    payload = AnalystKnowledgeReadiness(
        store_dir=store_dir,
        output_dir=tmp_path / "reports",
    ).build(
        as_of="2026-06-01T00:00:00+00:00",
        save=False,
    )

    assert payload["status"] == "knowledge_review_ready_with_exclusions"
    assert payload["summary"]["eligible_item_count"] == 1
    assert payload["summary"]["blocked_item_count"] == 1
    assert payload["summary"]["can_feed_review_only_analyst"] is True
    assert payload["integration_contract"]["blocked_shortcuts"]
