from __future__ import annotations

from pathlib import Path

from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.draft.dean_os_agent_system_v7.dean_os.material_loaders import ResearchSourceType, load_research_document
from dean_os.schemas import MarketContext, ResearchDocument


class DomainDataFeeder:
    """Feeds raw files (theory, history, stats) into a MarketContext.

    Raw domain materials should not bypass the shared material loader: books,
    notes, reports, templates, and user-fed ideas need the same text cleanup,
    quarantine flags, source metadata, and point-in-time provenance as cached
    news and research artifacts.
    """

    def __init__(self, domain_id: str):
        self.domain_id = domain_id

    def feed_theory(self, context: MarketContext, file_path: str | Path) -> None:
        """Injects economic or domain theory from a text file."""
        self._inject_file(context, file_path, "economic_theory", "article")

    def feed_history(self, context: MarketContext, file_path: str | Path) -> None:
        """Injects historical analogs or historical context from a text file."""
        self._inject_file(context, file_path, "historical_analog", "book")

    def feed_stats(self, context: MarketContext, file_path: str | Path) -> None:
        """Injects industry data/statistics from a JSON or text file."""
        self._inject_material(
            context,
            file_path,
            evidence_type="industry_statistics",
            source_type="report",
            availability_basis="user_fed_stats",
            limitations=["User-provided industry statistics."],
        )

    def feed_material(
        self,
        context: MarketContext,
        file_path: str | Path,
        *,
        evidence_type: str,
        source_type: ResearchSourceType = "report",
        limitations: list[str] | None = None,
    ) -> ResearchDocument:
        """Inject a generic user-provided material as typed analyst evidence.

        Use this for templates, idea notes, saved research briefs, or other
        domain materials that do not fit the theory/history/stat helpers.
        """
        return self._inject_material(
            context,
            file_path,
            evidence_type=evidence_type,
            source_type=source_type,
            availability_basis="user_fed_material",
            limitations=limitations or [f"User-provided {evidence_type} material."],
        )

    def _inject_file(
        self,
        context: MarketContext,
        file_path: str | Path,
        evidence_tag: str,
        source_type: ResearchSourceType,
    ) -> ResearchDocument:
        return self._inject_material(
            context,
            file_path,
            evidence_type=evidence_tag,
            source_type=source_type,
            availability_basis="user_fed_context",
            limitations=[f"User-provided {evidence_tag} material."],
        )

    def _inject_material(
        self,
        context: MarketContext,
        file_path: str | Path,
        *,
        evidence_type: str,
        source_type: ResearchSourceType,
        availability_basis: str,
        limitations: list[str],
    ) -> ResearchDocument:
        as_of = self._require_point_in_time_as_of(context)
        path = Path(file_path)
        document = load_research_document(
            path,
            source_type=source_type,
            sectors=[self.domain_id],
            tags=[self.domain_id, evidence_type],
        )
        published_at, publication_limitations = self._publication_timestamp(
            document,
            source_type,
            as_of,
        )
        metadata = dict(document.metadata or {})
        metadata["_dean_document_provenance"] = {
            **dict(metadata.get("_dean_document_provenance") or {}),
            "availability_at": as_of,
            "availability_basis": availability_basis,
            "limitations": [*limitations, *publication_limitations],
            "evidence_type": evidence_type,
            "domain_id": self.domain_id,
            "source_path": str(path),
            "loader": "dean_os.draft.dean_os_agent_system_v7.dean_os.material_loaders.load_research_document",
        }
        fed_document = document.model_copy(
            update={
                "source_type": source_type,
                "uri": str(path.resolve()),
                "published_at": published_at,
                "ingested_at": as_of,
                "sectors": sorted({*document.sectors, self.domain_id}),
                "tags": sorted({*document.tags, self.domain_id, evidence_type}),
                "metadata": metadata,
            }
        )
        context.research_documents.append(fed_document)
        return fed_document

    @staticmethod
    def _require_point_in_time_as_of(context: MarketContext) -> str:
        if parse_timezone_aware(context.as_of) is None:
            raise ValueError(
                "DomainDataFeeder requires context.as_of to be a timezone-aware ISO-8601 timestamp"
            )
        return str(context.as_of)

    @staticmethod
    def _publication_timestamp(
        document: ResearchDocument,
        source_type: ResearchSourceType,
        as_of: str,
    ) -> tuple[str | None, list[str]]:
        if document.published_at:
            return document.published_at, []
        if source_type in {"news", "article", "filing", "transcript"}:
            return as_of, ["publication_timestamp_defaulted_to_context_as_of"]
        return None, ["publication_timestamp_unknown"]
