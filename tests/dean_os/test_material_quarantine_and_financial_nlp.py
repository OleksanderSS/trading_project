from __future__ import annotations

from dean_os.financial_nlp import RuleBasedFinancialNLP
from dean_os.intake_normalizer import normalize_and_chunk
from dean_os.material_loaders import annotate_quarantine, filter_quarantined_text, load_research_document
from dean_os.research_corpus import ResearchCorpus
from dean_os.schemas import ResearchDocument


def test_material_loader_marks_typical_disclaimer_blocks(monkeypatch, tmp_path):
    monkeypatch.setenv("DEAN_PROJECT_ROOT", str(tmp_path))
    material_path = tmp_path / "amd_sector_note.md"
    material_path.write_text(
        "\n".join(
            [
                "# AMD Sector Note",
                "",
                "Demand growth and backlog improved after a contract win.",
                "",
                "Forward-looking statements involve risks and uncertainties.",
                "Actual results may differ materially.",
            ]
        ),
        encoding="utf-8",
    )

    document = load_research_document(material_path, source_type="report", tickers=["AMD"])

    assert document.title == "AMD Sector Note"
    assert document.quality_precheck == "quarantine_detected"
    assert "legal_disclaimer" in document.quarantine_flags
    assert document.metadata["quarantine_block_count"] >= 1
    assert document.metadata["quarantine_blocks"][0]["quarantine_flags"] == ["legal_disclaimer"]


def test_rule_based_financial_nlp_ignores_quarantined_disclaimer_terms():
    document = ResearchDocument(
        title="Safe Text Sentiment",
        source_type="report",
        text=(
            "Demand growth and backlog support margin expansion. "
            "Forward-looking statements involve risks and uncertainties, lawsuit, recession and delay. "
            "Actual results may differ materially."
        ),
        quarantine_flags=["legal_disclaimer"],
    )

    result = RuleBasedFinancialNLP().analyze_document(document, "financial_nlp")

    assert result.tone == "positive"
    assert result.sentiment_score == 1.0
    assert result.risk_score == 0.0
    assert "lawsuit" not in result.key_terms
    assert result.metadata["sentiment_source"] == "quarantine_filtered_text"
    assert result.metadata["removed_quarantine_flags"] == ["legal_disclaimer"]
    assert result.extracted_facts == []
    assert result.extracted_events == []


def test_filter_returns_empty_safe_text_when_document_is_only_quarantine():
    text = "Forward-looking statements involve risks and uncertainties. Actual results may differ materially."

    safe_text, removed = filter_quarantined_text(text)
    result = RuleBasedFinancialNLP().analyze_document(
        ResearchDocument(title="Disclaimer Only", source_type="report", text=text),
        "financial_nlp",
    )

    assert safe_text == ""
    assert len(removed) == 1
    assert result.tone == "neutral"
    assert result.sentiment_score == 0.0
    assert result.risk_score == 0.0
    assert result.key_terms == []
    assert result.metadata["sentiment_text_empty"] is True


def test_intake_normalizer_preserves_quarantine_flags_on_chunks():
    document = ResearchDocument(
        title="Chunk Quarantine",
        source_type="report",
        text=(
            "Demand growth and backlog improved. "
            "Safe harbor: forward-looking statements involve risks and uncertainties."
        ),
    )

    chunks = normalize_and_chunk(document, chunk_size=500)

    assert any(chunk.quality_precheck == "passed" for chunk in chunks)
    assert any("legal_disclaimer" in chunk.quarantine_flags for chunk in chunks)


def test_research_corpus_restores_quarantine_flags_from_metadata(tmp_path):
    document = annotate_quarantine(
        ResearchDocument(
            title="Persisted Quarantine",
            source_type="report",
            text="Demand growth improved. Safe harbor: forward-looking statements involve risks and uncertainties.",
        )
    )
    corpus = ResearchCorpus(tmp_path / "research_corpus.sqlite")

    corpus.add_document(document, chunk_size=500)
    loaded_document = corpus.list_documents()[0]
    loaded_chunks = corpus.search_chunks("forward-looking", limit=5)

    assert loaded_document.quality_precheck == "quarantine_detected"
    assert loaded_document.quarantine_flags == ["legal_disclaimer"]
    assert any("legal_disclaimer" in chunk.quarantine_flags for chunk in loaded_chunks)
