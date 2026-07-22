from __future__ import annotations

import json
import os

from dean_os.analyst_core.research_corpus_evidence_loader import (
    ResearchCorpusEvidenceLoader,
)
from dean_os.research_corpus import ResearchCorpus
from dean_os.schemas import ResearchDocument


def test_research_corpus_loader_is_context_only_and_point_in_time(tmp_path):
    path = tmp_path / "research.sqlite"
    corpus = ResearchCorpus(path)
    corpus.add_document(
        ResearchDocument(
            document_id="capex_doc",
            title="Data center capex",
            source_type="article",
            text="Hyperscaler data center capex supports semiconductor demand.",
            uri="https://example.test/capex",
            published_at="2026-07-01T00:00:00+00:00",
            tickers=["NVDA"],
            sectors=["semiconductors"],
            tags=["capex"],
        )
    )
    os.utime(path, (1782950400, 1782950400))

    items = ResearchCorpusEvidenceLoader().load(
        path,
        domain_id="semiconductor_ai_infrastructure",
        as_of="2026-07-03T00:00:00+00:00",
        query="hyperscaler data center capex",
        tickers=["NVDA"],
        top_k=5,
    )

    assert len(items) == 1
    assert items[0].evidence_type == "capex_cycle"
    assert items[0].provenance["required_lane_eligible"] is False
    assert items[0].provenance["ticker_thesis_eligible"] is False
    assert len(items[0].provenance["content_sha256"]) == 64


def test_research_corpus_loader_excludes_future_document(tmp_path):
    path = tmp_path / "research.sqlite"
    corpus = ResearchCorpus(path)
    corpus.add_document(
        ResearchDocument(
            document_id="future_doc",
            title="Future capex",
            source_type="article",
            text="Future hyperscaler capex update.",
            published_at="2026-07-05T00:00:00+00:00",
        )
    )
    os.utime(path, (1782950400, 1782950400))

    items = ResearchCorpusEvidenceLoader().load(
        path,
        domain_id="semiconductor_ai_infrastructure",
        as_of="2026-07-03T00:00:00+00:00",
        query="hyperscaler capex",
    )

    assert items == []
