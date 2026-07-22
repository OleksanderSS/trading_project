from __future__ import annotations

import pytest

from dean_os.analysts.context_adapter import MarketContextEvidenceAdapter
from dean_os.analysts.domain_feeder import DomainDataFeeder
from dean_os.context_evidence_provenance import audit_research_documents
from dean_os.schemas import MarketContext, ResearchDocument

AS_OF = "2026-07-01T12:00:00+00:00"
DOMAIN_ID = "semiconductor_ai_infrastructure"


def test_domain_data_feeder_routes_history_through_loader_and_adapter(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("DEAN_PROJECT_ROOT", str(tmp_path))
    material_path = tmp_path / "semiconductor_cycle_history.md"
    material_path.write_text(
        "\n".join(
            [
                "# Semiconductor Cycle History",
                "",
                "Historical capex digestion often follows a memory oversupply cycle.",
                "This is sector-level context, not a ticker trade signal.",
            ]
        ),
        encoding="utf-8",
    )
    context = MarketContext(as_of=AS_OF, tickers=["AMD"])

    DomainDataFeeder(DOMAIN_ID).feed_history(context, material_path)

    document = context.research_documents[0]
    assert document.source_type == "book"
    assert document.uri == str(material_path.resolve())
    assert "historical_analog" in document.tags
    assert document.metadata["_dean_document_provenance"][
        "loader"
    ] == "dean_os.material_loaders.load_research_document"

    payload = MarketContextEvidenceAdapter(DOMAIN_ID).adapt(
        context,
        as_of=AS_OF,
    )

    document_items = [
        item
        for item in payload["evidence"]
        if item.source == str(material_path.resolve())
    ]
    assert len(document_items) == 1
    item = document_items[0]
    assert item.evidence_type == "historical_analog"
    assert item.source_type == "book"
    assert item.directness == "sector"
    assert item.provenance["evidence_type"] == "historical_analog"
    assert item.provenance["availability_basis"] == (
        "ingested_at_publication_unknown"
    )
    assert item.provenance["source_declared_availability_basis"] == (
        "user_fed_context"
    )
    assert "User-provided historical_analog material." in item.limitations
    assert "publication_timestamp_unknown" in item.limitations


def test_domain_data_feeder_preserves_material_quarantine_and_custom_type(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("DEAN_PROJECT_ROOT", str(tmp_path))
    material_path = tmp_path / "idea_template.md"
    material_path.write_text(
        "\n".join(
            [
                "# Idea Template",
                "",
                "Track wafer starts, foundry utilization, and end-market inventory.",
                "",
                "Forward-looking statements involve risks and uncertainties.",
                "Actual results may differ materially.",
            ]
        ),
        encoding="utf-8",
    )
    context = MarketContext(as_of=AS_OF)

    document = DomainDataFeeder(DOMAIN_ID).feed_material(
        context,
        material_path,
        evidence_type="idea_template",
        source_type="report",
    )

    assert document.quality_precheck == "quarantine_detected"
    assert "legal_disclaimer" in document.quarantine_flags
    assert "idea_template" in document.tags

    payload = MarketContextEvidenceAdapter(DOMAIN_ID).adapt(
        context,
        as_of=AS_OF,
    )

    document_items = [
        item
        for item in payload["evidence"]
        if item.source == str(material_path.resolve())
    ]
    assert len(document_items) == 1
    assert document_items[0].evidence_type == "idea_template"
    assert document_items[0].provenance["evidence_type"] == "idea_template"


@pytest.mark.parametrize("as_of", [None, "2026-07-01T12:00:00"])
def test_domain_data_feeder_requires_timezone_aware_as_of(
    monkeypatch,
    tmp_path,
    as_of,
):
    monkeypatch.setenv("DEAN_PROJECT_ROOT", str(tmp_path))
    material_path = tmp_path / "stats.json"
    material_path.write_text(
        '{"title": "Industry stats", "summary": "Foundry utilization context"}',
        encoding="utf-8",
    )
    context = MarketContext(as_of=as_of)

    with pytest.raises(ValueError, match="timezone-aware"):
        DomainDataFeeder(DOMAIN_ID).feed_stats(context, material_path)


def test_research_document_audit_preserves_declared_evidence_type():
    document = ResearchDocument(
        title="User idea note",
        source_type="report",
        text="Template says monitor channel inventory and utilization.",
        uri="file:///idea-note",
        ingested_at=AS_OF,
        metadata={
            "_dean_document_provenance": {
                "availability_at": AS_OF,
                "availability_basis": "user_fed_material",
                "limitations": ["User-provided idea template."],
                "evidence_type": "idea_template",
                "domain_id": DOMAIN_ID,
            }
        },
    )

    audit = audit_research_documents([document], as_of=AS_OF)

    assert audit["accepted_count"] == 1
    provenance = document.metadata["_dean_document_provenance"]
    assert provenance["evidence_type"] == "idea_template"
    assert provenance["domain_id"] == DOMAIN_ID
    assert provenance["source_declared_availability_basis"] == (
        "user_fed_material"
    )
    assert "User-provided idea template." in provenance["limitations"]
    assert "publication_timestamp_unknown" in provenance["limitations"]
