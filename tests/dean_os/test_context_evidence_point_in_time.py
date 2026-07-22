from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest

from dean_os.analysts.context_adapter import (
    MarketContextEvidenceAdapter,
)
from dean_os.agents.domain_research import NewsCatalystAgent
from dean_os.agents.research_agents import material_documents
from dean_os.context_evidence_provenance import (
    audit_news_records,
    audit_research_documents,
)
from dean_os.context_evidence_review_packet import (
    ContextEvidenceReviewPacket,
)
from dean_os.pipeline_adapter import HybridPipelineAdapter
from dean_os.schemas import (
    MarketContext,
    ResearchDocument,
    ResearchNote,
    SourceCitation,
)


AS_OF = "2026-06-01T12:00:00+00:00"
DOMAIN_ID = "semiconductor_ai_infrastructure"


class _LocalOrchestrator:
    def __init__(self, result):
        self.result = result

    async def run_local_pipeline(self, **kwargs):
        return self.result


def _news(
    title: str,
    *,
    published_at: str = "2026-05-31T12:00:00+00:00",
    url: str | None = "https://example.test/news/1",
    tickers: list[str] | None = None,
) -> dict:
    payload = {
        "title": title,
        "published_at": published_at,
        "tickers": tickers or [],
    }
    if url is not None:
        payload["url"] = url
    return payload


def test_news_audit_excludes_future_missing_unstructured_and_duplicates():
    eligible = _news(
        "AMD data center capex growth",
        tickers=["AMD"],
    )
    result = audit_news_records(
        [
            eligible,
            eligible,
            _news(
                "Future semiconductor event",
                published_at="2026-06-02T00:00:00+00:00",
                url="https://example.test/news/future",
            ),
            {"title": "Missing timestamp", "url": "https://example.test/missing"},
            _news("Missing locator", url=None),
            "unstructured headline",
        ],
        as_of=AS_OF,
        requested_tickers=["AMD"],
    )

    assert result["accepted_count"] == 1
    assert result["excluded_count"] == 5
    assert result["accepted"][0]["_dean_context_provenance"][
        "direct_requested_tickers"
    ] == ["AMD"]
    assert result["reason_counts"]["duplicate_news_record"] == 1
    assert result["reason_counts"]["publication_after_as_of"] == 1
    assert result["reason_counts"][
        "publication_timestamp_missing_or_invalid"
    ] == 1
    assert result["reason_counts"]["stable_source_locator_missing"] == 1
    assert result["reason_counts"]["news_record_not_structured"] == 1


def test_plain_text_ticker_name_does_not_create_direct_ticker_evidence():
    result = audit_news_records(
        [_news("AMD semiconductor demand expands", tickers=[])],
        as_of=AS_OF,
        requested_tickers=["AMD"],
    )

    assert result["accepted_count"] == 1
    assert result["accepted"][0]["_dean_context_provenance"][
        "direct_requested_tickers"
    ] == []


def test_context_adapter_propagates_real_time_and_exclusion_audit():
    valid_note = ResearchNote(
        note_id="note_valid",
        agent_name="specialist",
        topic="semiconductor capex",
        thesis="Data center spending supports the sector.",
        tickers=["AMD"],
        sectors=["semiconductors"],
        confidence=0.7,
        created_at="2026-05-31T13:00:00+00:00",
        citations=[
            SourceCitation(
                source_id="source_1",
                source_type="news",
                title="Source",
                uri="https://example.test/source/1",
                timestamp="2026-05-31T10:00:00+00:00",
            )
        ],
    )
    future_note = valid_note.model_copy(
        update={
            "note_id": "note_future",
            "created_at": "2026-06-02T00:00:00+00:00",
        }
    )
    context = MarketContext(
        as_of=AS_OF,
        tickers=["AMD"],
        news=[
            _news(
                "AMD data center capex growth",
                tickers=["AMD"],
            ),
            _news(
                "Semiconductor demand expands",
                url="https://example.test/news/sector",
            ),
            _news(
                "Future export control",
                published_at="2026-06-02T00:00:00+00:00",
                url="https://example.test/news/future",
            ),
        ],
        research_notes=[valid_note, future_note],
        pipeline_result={"status": "completed"},
    )

    payload = MarketContextEvidenceAdapter(DOMAIN_ID).adapt(
        context,
        as_of=AS_OF,
    )

    assert payload["status"] == "review_context_ready_with_exclusions"
    assert payload["summary"]["news_accepted_count"] == 2
    assert payload["summary"]["news_excluded_count"] == 1
    ticker_items = [
        item for item in payload["evidence"] if item.directness == "ticker"
    ]
    assert any(
        item.published_at == "2026-05-31T12:00:00+00:00"
        for item in ticker_items
    )
    assert any(
        item.provenance.get("note_id") == "note_valid"
        for item in ticker_items
    )
    assert any(
        "research_note_created_after_as_of" in item["reasons"]
        for item in payload["exclusions"]
    )
    assert any(
        item["family"] == "pipeline_result"
        for item in payload["exclusions"]
    )
    assert payload["summary"]["can_influence_pipeline_prediction"] is False
    assert payload["summary"]["can_trade"] is False


def test_pipeline_adapter_quarantines_future_and_missing_news():
    news_frame = pd.DataFrame(
        [
            {
                "title": "Past AMD capex",
                "published_date": "2026-05-31T12:00:00+00:00",
                "link": "https://example.test/past",
                "tickers": ["AMD"],
            },
            {
                "title": "Future AMD capex",
                "published_date": "2026-06-02T12:00:00+00:00",
                "link": "https://example.test/future",
                "tickers": ["AMD"],
            },
            {
                "title": "No date",
                "link": "https://example.test/no-date",
                "tickers": ["AMD"],
            },
        ]
    )
    result = {
        "status": "completed",
        "as_of": AS_OF,
        "results": {
            "features_df": pd.DataFrame({"return": [0.01]}),
            "news_data": news_frame,
        },
    }
    context = MarketContext(tickers=["AMD"], timeframe="1d")

    asyncio.run(
        HybridPipelineAdapter(
            mode="local",
            orchestrator=_LocalOrchestrator(result),
        )(context)
    )

    assert context.as_of == AS_OF
    assert len(context.news) == 1
    assert context.news[0]["title"] == "Past AMD capex"
    audit = context.metadata["news_point_in_time_audit"]
    assert audit["accepted_count"] == 1
    assert audit["excluded_count"] == 2
    assert context.dataframes["news"] is news_frame


def test_pipeline_adapter_quarantines_all_news_without_context_as_of():
    result = {
        "status": "completed",
        "results": {
            "features_df": pd.DataFrame({"return": [0.01]}),
            "news_data": pd.DataFrame(
                [
                    {
                        "title": "News without context cutoff",
                        "published_at": "2026-05-31T00:00:00+00:00",
                        "url": "https://example.test/news",
                    }
                ]
            ),
        },
    }
    context = MarketContext(tickers=["AMD"], timeframe="1d")

    asyncio.run(
        HybridPipelineAdapter(
            mode="local",
            orchestrator=_LocalOrchestrator(result),
        )(context)
    )

    assert context.news == []
    assert context.metadata["news_point_in_time_audit"]["status"] == (
        "blocked_context_as_of_missing"
    )


def test_news_audit_requires_timezone_aware_as_of():
    with pytest.raises(ValueError, match="timezone-aware"):
        audit_news_records(
            [_news("AMD capex", tickers=["AMD"])],
            as_of="2026-06-01T12:00:00",
            requested_tickers=["AMD"],
        )


def test_context_evidence_review_packet_is_review_only(tmp_path):
    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            {
                "as_of": AS_OF,
                "tickers": ["AMD"],
                "news": [
                    _news(
                        "AMD data center capex growth",
                        tickers=["AMD"],
                    ),
                    _news(
                        "Future AMD event",
                        published_at="2026-06-02T00:00:00+00:00",
                        url="https://example.test/future",
                        tickers=["AMD"],
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = ContextEvidenceReviewPacket(
        context_json=context_path,
        domain_id=DOMAIN_ID,
        output_dir=tmp_path / "reports",
    ).build(save=False)

    assert payload["status"] == "review_context_ready_with_exclusions"
    assert payload["summary"]["evidence_count"] >= 1
    assert payload["summary"]["can_influence_pipeline_prediction"] is False
    assert payload["integration_boundary"]["can_become_stage5_feature"] is False
    assert payload["safety"]["pipeline_run"] is False
    assert payload["safety"]["can_trade"] is False
    assert len(payload["fingerprint"]) == 64


def test_context_evidence_review_packet_blocks_missing_as_of(tmp_path):
    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            {
                "tickers": ["AMD"],
                "news": [
                    _news(
                        "AMD data center capex growth",
                        tickers=["AMD"],
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = ContextEvidenceReviewPacket(
        context_json=context_path,
        domain_id=DOMAIN_ID,
        output_dir=tmp_path / "reports",
    ).build(save=False)

    assert payload["status"] == "blocked_context_as_of_missing"
    assert payload["evidence"] == []
    assert payload["summary"]["can_trade"] is False


def test_direct_keyword_agent_cannot_bypass_future_news_cutoff():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["AMD"],
        news=[
            _news(
                "AMD partnership growth and upgrade",
                published_at="2026-06-02T00:00:00+00:00",
                tickers=["AMD"],
            )
        ],
    )

    report = asyncio.run(
        NewsCatalystAgent(
            name="news_catalyst",
            config={},
        ).run(context)
    )

    assert report.verdict == "needs_more_data"
    assert context.metadata["news_point_in_time_audit"][
        "accepted_count"
    ] == 0
    assert context.metadata["news_point_in_time_audit"][
        "excluded_count"
    ] == 1


def test_material_documents_does_not_convert_news_without_context_as_of():
    context = MarketContext(
        tickers=["AMD"],
        news=[
            _news(
                "AMD data center capex growth",
                tickers=["AMD"],
            )
        ],
    )

    documents = material_documents(context)

    assert documents == []
    assert context.metadata["news_point_in_time_audit"]["status"] == (
        "blocked_context_as_of_missing"
    )


def test_research_document_audit_distinguishes_publication_ingestion_and_replay():
    eligible = ResearchDocument(
        document_id="doc_eligible",
        title="AMD filing",
        source_type="filing",
        text="AMD data center capex increased.",
        uri="https://example.test/filing",
        published_at="2026-05-30T00:00:00+00:00",
        ingested_at="2026-05-31T00:00:00+00:00",
        tickers=["AMD"],
    )
    unknown_publication_report = ResearchDocument(
        document_id="doc_report",
        title="Operator report",
        source_type="report",
        text="Semiconductor capacity context.",
        uri="file:///operator-report",
        published_at=None,
        ingested_at="2026-05-31T00:00:00+00:00",
    )
    replay_document = ResearchDocument(
        document_id="doc_replay",
        title="Historical article",
        source_type="article",
        text="Historical semiconductor demand.",
        uri="https://example.test/historical",
        published_at="2026-05-20T00:00:00+00:00",
        ingested_at="2026-06-30T00:00:00+00:00",
        metadata={"point_in_time_replay": {"as_of": AS_OF}},
    )
    future = ResearchDocument(
        document_id="doc_future",
        title="Future filing",
        source_type="filing",
        text="Future evidence.",
        uri="https://example.test/future-filing",
        published_at="2026-06-02T00:00:00+00:00",
        ingested_at="2026-05-31T00:00:00+00:00",
    )
    duplicate = eligible.model_copy(
        update={
            "document_id": "doc_duplicate",
            "uri": "https://example.test/duplicate",
        }
    )

    audit = audit_research_documents(
        [
            eligible,
            unknown_publication_report,
            replay_document,
            future,
            duplicate,
        ],
        as_of=AS_OF,
    )

    assert audit["accepted_count"] == 3
    assert audit["excluded_count"] == 2
    assert audit["reason_counts"]["document_published_after_as_of"] == 1
    assert audit["reason_counts"]["duplicate_research_document"] == 1
    assert unknown_publication_report.metadata[
        "_dean_document_provenance"
    ]["availability_basis"] == "ingested_at_publication_unknown"
    assert replay_document.metadata["_dean_document_provenance"][
        "replay_reconstruction"
    ] is True


def test_context_adapter_emits_document_provenance_and_excludes_future_document():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["AMD"],
        research_documents=[
            ResearchDocument(
                document_id="doc_past",
                title="AMD filing",
                source_type="filing",
                text="AMD data center capex growth supports demand.",
                uri="https://example.test/past-filing",
                published_at="2026-05-30T00:00:00+00:00",
                ingested_at="2026-05-31T00:00:00+00:00",
                tickers=["AMD"],
            ),
            ResearchDocument(
                document_id="doc_future",
                title="Future filing",
                source_type="filing",
                text="AMD data center capex growth.",
                uri="https://example.test/future-filing",
                published_at="2026-06-02T00:00:00+00:00",
                ingested_at="2026-05-31T00:00:00+00:00",
                tickers=["AMD"],
            ),
        ],
    )

    payload = MarketContextEvidenceAdapter(DOMAIN_ID).adapt(
        context,
        as_of=AS_OF,
    )

    document_items = [
        item
        for item in payload["evidence"]
        if item.provenance.get("document_id") == "doc_past"
    ]
    assert document_items
    assert all(item.directness == "ticker" for item in document_items)
    assert all(
        len(item.provenance["content_sha256"]) == 64
        for item in document_items
    )
    assert any(
        "document_published_after_as_of" in item["reasons"]
        for item in payload["exclusions"]
    )
