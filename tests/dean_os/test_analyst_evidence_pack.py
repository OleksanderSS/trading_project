from __future__ import annotations

import json

import pandas as pd

from dean_os.analyst_evidence_pack import (
    AnalystEvidencePackRunner,
    documents_from_evidence_pack,
    render_analyst_evidence_pack_markdown,
)


def test_analyst_evidence_pack_builds_documents_from_materials_news_and_macro(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "amd_ai_article.md").write_text(
        "# AMD AI Report\nAMD benefits from AI accelerator demand and data center compute investment.",
        encoding="utf-8",
    )
    news_path = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {
                "title": "AMD wins cloud AI accelerator order",
                "summary": "AMD backlog improves as cloud AI data center demand expands.",
                "published_at": "2026-01-05T00:00:00+00:00",
                "ticker": "AMD",
            },
            {
                "title": "Unrelated consumer story",
                "summary": "No requested ticker here.",
                "published_at": "2026-01-06T00:00:00+00:00",
                "ticker": "XYZ",
            },
        ]
    ).to_csv(news_path, index=False)
    macro_path = tmp_path / "macro.csv"
    pd.DataFrame(
        [
            {
                "indicator": "rates",
                "date": "2026-01-04",
                "value": 4.25,
                "summary": "Fed policy easing expectations support liquidity.",
            }
        ]
    ).to_csv(macro_path, index=False)

    payload = AnalystEvidencePackRunner(output_dir=tmp_path / "reports").run(
        materials_paths=[materials],
        news_data_paths=[news_path],
        macro_data_paths=[macro_path],
        tickers=["AMD"],
        sectors=["semiconductor"],
        tags=["ai_cycle"],
        start_at="2026-01-01T00:00:00+00:00",
        end_at="2026-02-01T00:00:00+00:00",
    )

    assert payload["coverage"]["document_count"] == 3
    assert payload["coverage"]["by_source_type"]["article"] == 1
    assert payload["coverage"]["by_source_type"]["news"] == 1
    assert payload["coverage"]["by_source_type"]["report"] == 1
    assert payload["coverage"]["by_ticker"]["AMD"] >= 2
    assert payload["coverage"]["agent_lab_ready"] is True
    assert payload["analyst_inputs"]["base_analyst"]["ready"] is True
    assert payload["analyst_inputs"]["manager_plan"]["mode"] == "single_base_then_specialize"
    assert payload["analyst_inputs"]["manager_plan"]["active_profiles"] == ["generalist_base_analyst"]
    assert "evidence-pack-json" in payload["analyst_inputs"]["base_analyst"]["agent_lab_command_preview"]
    assert (tmp_path / "reports" / "latest.json").exists()


def test_documents_from_evidence_pack_round_trips_research_documents(tmp_path):
    payload = AnalystEvidencePackRunner(output_dir=tmp_path / "reports").run(
        news_data_paths=[],
        macro_data_paths=[],
        materials_paths=[],
        tickers=["AMD"],
        save=False,
    )
    assert payload["documents"] == []

    pack_path = tmp_path / "pack.json"
    pack_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "doc1",
                        "title": "One document",
                        "source_type": "article",
                        "text": "AMD AI compute cycle evidence.",
                        "tickers": ["AMD"],
                        "sectors": ["semiconductor"],
                        "tags": ["ai_cycle"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    documents = documents_from_evidence_pack(pack_path)

    assert len(documents) == 1
    assert documents[0].title == "One document"
    assert documents[0].tickers == ["AMD"]


def test_analyst_evidence_pack_filters_published_date_future_rows(tmp_path):
    news_path = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {
                "title": "Visible AMD AI order",
                "summary": "AMD AI data center demand was visible before the cutoff.",
                "published_date": "2026-01-04T00:00:00+00:00",
                "ticker": "AMD",
            },
            {
                "title": "Future AMD earnings result",
                "summary": "This future row must not be visible in the old-data replay evidence pack.",
                "published_date": "2026-01-08T00:00:00+00:00",
                "ticker": "AMD",
            },
        ]
    ).to_csv(news_path, index=False)

    payload = AnalystEvidencePackRunner(output_dir=tmp_path / "reports").run(
        news_data_paths=[news_path],
        tickers=["AMD"],
        start_at="2026-01-01T00:00:00+00:00",
        as_of="2026-01-05T00:00:00+00:00",
    )

    titles = [document["title"] for document in payload["documents"]]
    assert titles == ["Visible AMD AI order"]
    assert payload["coverage"]["date_range"]["end"] == "2026-01-04T00:00:00+00:00"


def test_analyst_evidence_pack_sector_keywords_filter_news_without_ticker_forcing(tmp_path):
    news_path = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {
                "title": "Semiconductor equipment demand improves",
                "summary": "Chip equipment orders and AI accelerator supply chain demand improve.",
                "published_at": "2026-01-05T00:00:00+00:00",
            },
            {
                "title": "Retail store expansion",
                "summary": "Consumer retailer opens new stores with no sector relevance.",
                "published_at": "2026-01-06T00:00:00+00:00",
            },
        ]
    ).to_csv(news_path, index=False)

    payload = AnalystEvidencePackRunner(output_dir=tmp_path / "reports").run(
        news_data_paths=[news_path],
        sectors=["semiconductor"],
        sector_keywords=["semiconductor", "chip", "accelerator"],
    )

    assert payload["coverage"]["document_count"] == 1
    assert payload["coverage"]["tickers"] == []
    assert payload["coverage"]["sectors"] == ["semiconductor"]
    assert payload["documents"][0]["title"] == "Semiconductor equipment demand improves"
    assert payload["inputs"]["sector_keywords"] == ["accelerator", "chip", "semiconductor"]
    assert "- Sector keywords: accelerator, chip, semiconductor" in render_analyst_evidence_pack_markdown(payload)
