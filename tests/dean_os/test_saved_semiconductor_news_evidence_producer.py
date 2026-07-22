from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dean_os.analysts.base import BaseAnalystAgent
from dean_os.analysts.context_adapter import MarketContextEvidenceAdapter
from dean_os.analysts._producers.news import (
    SavedSemiconductorNewsEvidenceProducer,
    load_verified_semiconductor_news_context_fragment,
)
from dean_os.schemas import MarketContext


AS_OF = "2026-06-30T21:00:00+00:00"


def _source(tmp_path: Path) -> Path:
    rows = [
        {
            "title": "Nvidia sees strong AI demand for chips",
            "description": "Nvidia sees strong AI demand for chips",
            "published_date": "2026-06-01T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://reuters.test/nvidia-demand",
            "url": None,
            "source": "Reuters",
        },
        {
            "title": "AMD sales forecast rises with AI demand",
            "description": None,
            "published_date": "2026-06-02T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://bloomberg.test/amd-demand",
            "url": None,
            "source": "Bloomberg.com",
        },
        {
            "title": "Microsoft capital spending rises on memory chips",
            "description": None,
            "published_date": "2026-06-03T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://cnbc.test/chip-capex",
            "url": None,
            "source": "CNBC",
        },
        {
            "title": "TSMC foundry capacity constraints remain",
            "description": None,
            "published_date": "2026-06-04T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://seekingalpha.test/tsmc",
            "url": None,
            "source": "Seeking Alpha",
        },
        {
            "title": None,
            "description": None,
            "published_date": None,
            "publishedAt": None,
            "link": None,
            "url": None,
            "source": None,
        },
        {
            "title": "Nvidia AI demand after cutoff",
            "description": None,
            "published_date": "2026-07-02T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://reuters.test/future",
            "url": None,
            "source": "Reuters",
        },
    ]
    path = tmp_path / "news.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_saved_news_producer_requires_independent_strong_sources(
    tmp_path,
):
    source = _source(tmp_path)
    output = tmp_path / "output"
    payload = SavedSemiconductorNewsEvidenceProducer(
        output_dir=output
    ).build(
        source_path=source,
        as_of=AS_OF,
    )

    assert payload["status"] == (
        "semiconductor_news_evidence_ready_with_gaps"
    )
    assert payload["summary"]["ready_required_lanes"] == [
        "sector_demand"
    ]
    assert payload["summary"]["missing_required_lanes"] == [
        "capex_cycle",
        "supply_chain",
        "policy_or_geopolitical",
        "market_confirmation",
    ]
    demand = payload["lane_review"][0]
    assert demand["independent_strong_source_count"] == 2
    assert set(demand["independent_strong_sources"]) == {
        "reuters",
        "bloomberg",
    }
    assert payload["summary"]["can_close_lane_from_keyword_only"] is False
    assert payload["summary"]["future_row_count"] == 1
    nvidia = next(
        item
        for item in payload["candidates"]
        if item["source_locator"]
        == "https://reuters.test/nvidia-demand"
    )
    assert nvidia["summary"] == (
        "Nvidia sees strong AI demand for chips"
    )
    requests = payload["evidence_acquisition_requests"]
    assert {
        item["evidence_type"] for item in requests
    } == {
        "capex_cycle",
        "supply_chain",
        "policy_or_geopolitical",
        "market_confirmation",
    }
    assert all(
        item["automatic_collection_authorized"] is False
        for item in requests
    )

    fragment = load_verified_semiconductor_news_context_fragment(
        output / "latest.json",
        expected_as_of=AS_OF,
    )
    eligible = [
        record
        for record in fragment["news"]
        if record["_dean_semantic_evidence"][
            "required_lane_eligible"
        ]
    ]
    assert len(eligible) == 2
    assert {
        record["_dean_semantic_evidence"]["evidence_type"]
        for record in eligible
    } == {"sector_demand"}
    adapted = MarketContextEvidenceAdapter(
        "semiconductor_ai_infrastructure"
    ).adapt(
        MarketContext(
            as_of=AS_OF,
            tickers=["NVDA", "AMD", "INTC", "TSM"],
            news=fragment["news"],
        ),
        as_of=AS_OF,
    )
    reuters_summary = next(
        item.summary
        for item in adapted["evidence"]
        if item.source == "https://reuters.test/nvidia-demand"
    )
    assert reuters_summary == (
        "Nvidia sees strong AI demand for chips"
    )


def test_saved_news_loader_detects_source_change(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "output"
    SavedSemiconductorNewsEvidenceProducer(
        output_dir=output
    ).build(
        source_path=source,
        as_of=AS_OF,
    )
    source.write_bytes(source.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="source_provenance hash"):
        load_verified_semiconductor_news_context_fragment(
            output / "latest.json"
        )


def test_unverified_keyword_news_cannot_close_required_lane():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA", "AMD", "INTC", "TSM"],
        news=[
            {
                "title": "Nvidia sees strong AI demand",
                "published_at": "2026-06-01T12:00:00+00:00",
                "url": "https://example.test/unverified",
            }
        ],
    )
    packet = MarketContextEvidenceAdapter(
        "semiconductor_ai_infrastructure"
    ).adapt(context, as_of=AS_OF)
    news_item = packet["evidence"][0]
    assert news_item.evidence_type == "sector_demand"
    assert news_item.provenance["required_lane_eligible"] is False

    report = BaseAnalystAgent(
        "semiconductor_ai_infrastructure"
    ).run(packet["evidence"], as_of=AS_OF)
    assert "sector_demand" in report.thesis.blind_spots
    assert report.recommendation == "needs_more_data"


def test_saved_news_producer_accepts_cached_summary_timestamp_schema_as_candidate(tmp_path):
    rows = [
        {
            "title": "$NVDA Nvidia AI demand for chips rises",
            "summary": (
                "$NVDA Nvidia AI demand for chips rises as data center "
                "customers add accelerator orders https://t.co/nvda-ai"
            ),
            "ticker": "NVDA",
            "source": "HF/twitter-financial-news",
            "timestamp": "2026-06-30T18:00:00+00:00",
        }
    ]
    source = tmp_path / "cached_news.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    payload = SavedSemiconductorNewsEvidenceProducer(
        output_dir=tmp_path / "output"
    ).build(
        source_path=source,
        as_of=AS_OF,
        save=False,
    )

    assert payload["status"] == "semiconductor_news_evidence_ready_with_gaps"
    assert payload["summary"]["usable_source_row_count"] == 1
    assert payload["summary"]["domain_candidate_count"] == 1
    assert payload["summary"]["accepted_news_record_count"] == 1
    assert payload["summary"]["ready_required_lanes"] == []
    candidate = payload["candidates"][0]
    assert candidate["source_locator"] == "https://t.co/nvda-ai"
    assert candidate["source_tier"] == "tier_4_weak_or_unverified"
    assert candidate["evidence_type"] == "sector_demand"


def test_saved_news_producer_routes_plural_chip_export_policy_terms(tmp_path):
    source = tmp_path / "policy_news.parquet"
    pd.DataFrame(
        [
            {
                "title": "Easing chip export controls raises semiconductor risk",
                "summary": "Debate covers chip exports to China.",
                "source": "Bloomberg Politics",
                "published_date": "2026-05-14T18:53:54Z",
                "link": "https://bloomberg.test/chip-export-controls",
            }
        ]
    ).to_parquet(source, index=False)

    payload = SavedSemiconductorNewsEvidenceProducer(
        output_dir=tmp_path / "output"
    ).build(source_path=source, as_of=AS_OF, save=False)

    policy = [
        item
        for item in payload["candidates"]
        if item["evidence_type"] == "policy_or_geopolitical"
    ]
    assert len(policy) == 1
    assert policy[0]["source_tier"] == "tier_2_strong_context"
    assert set(policy[0]["matched_terms"]) == {
        "chip exports",
        "export controls",
    }


def test_saved_news_producer_uses_word_boundaries_and_market_confirmation(tmp_path):
    rows = [
        {
            "title": "Intelsat downgraded after satellite weakness",
            "summary": (
                "Intelsat downgraded after satellite weakness "
                "https://t.co/intelsat"
            ),
            "ticker": "",
            "source": "HF/twitter-financial-news",
            "timestamp": "2026-06-30T18:00:00+00:00",
        },
        {
            "title": "$NVDA Nvidia stock climbs after analyst upgrade",
            "summary": (
                "$NVDA Nvidia stock climbs after analyst upgrade "
                "https://t.co/nvda-upgrade"
            ),
            "ticker": "NVDA",
            "source": "HF/twitter-financial-news",
            "timestamp": "2026-06-30T18:00:00+00:00",
        },
    ]
    source = tmp_path / "market_news.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    payload = SavedSemiconductorNewsEvidenceProducer(
        output_dir=tmp_path / "output"
    ).build(
        source_path=source,
        as_of=AS_OF,
        save=False,
    )

    assert payload["summary"]["usable_source_row_count"] == 2
    assert payload["summary"]["domain_candidate_count"] == 1
    assert payload["summary"]["classified_candidate_count"] == 1
    assert payload["summary"]["accepted_news_record_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["source_locator"] == "https://t.co/nvda-upgrade"
    assert candidate["evidence_type"] == "market_confirmation"
    assert candidate["source_tier"] == "tier_4_weak_or_unverified"


def test_narrow_capex_and_supply_terms_can_be_corroborated(tmp_path):
    rows = [
        {
            "title": "Microsoft capital spending rises on memory chips",
            "description": None,
            "published_date": "2026-06-01T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://cnbc.test/capex",
            "url": None,
            "source": "CNBC",
        },
        {
            "title": "Meta makes $200 billion data center bet",
            "description": None,
            "published_date": "2026-06-02T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://bloomberg.test/capex",
            "url": None,
            "source": "Bloomberg.com",
        },
        {
            "title": "Soaring memory prices pressure AI chips",
            "description": None,
            "published_date": "2026-06-03T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://cnbc.test/supply",
            "url": None,
            "source": "CNBC",
        },
        {
            "title": "Nvidia faces supply constraints and memory crunch",
            "description": None,
            "published_date": "2026-06-04T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://bloomberg.test/supply",
            "url": None,
            "source": "Bloomberg.com",
        },
    ]
    source = tmp_path / "news.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    payload = SavedSemiconductorNewsEvidenceProducer(
        output_dir=tmp_path / "output"
    ).build(
        source_path=source,
        as_of=AS_OF,
        save=False,
    )

    assert {
        lane["evidence_type"]
        for lane in payload["lane_review"]
        if lane["status"] == "eligible"
    } == {"capex_cycle", "supply_chain"}
