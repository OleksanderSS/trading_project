from __future__ import annotations

from dean_os.analyst_knowledge.schemas import KnowledgeQuery, KnowledgeRetrievalResult
from dean_os.analyst_knowledge.store import LocalKnowledgeStore


class KnowledgeRetriever:
    """Retrieval wrapper used by analyst agents.

    It deliberately avoids LLM calls. It returns evidence-like knowledge items
    that the analyst can cite in a report.
    """

    def __init__(self, store: LocalKnowledgeStore):
        self.store = store

    def retrieve(
        self,
        query: str,
        *,
        domain_id: str | None = None,
        tickers: list[str] | None = None,
        tags: list[str] | None = None,
        item_types: list[str] | None = None,
        top_k: int = 8,
        as_of: str | None = None,
        require_point_in_time: bool = False,
        require_source_provenance: bool = False,
        intended_use: str = "evidence",
    ) -> KnowledgeRetrievalResult:
        return self.store.search(
            KnowledgeQuery(
                query=query,
                domain_id=domain_id,
                tickers=tickers or [],
                tags=tags or [],
                item_types=item_types or [],
                top_k=top_k,
                as_of=as_of,
                require_point_in_time=require_point_in_time,
                require_source_provenance=require_source_provenance,
                intended_use=intended_use,
            )
        )

    def retrieve_for_context(self, context: dict, *, top_k: int = 8) -> KnowledgeRetrievalResult:
        query_parts = [
            str(context.get("question") or ""),
            " ".join(context.get("tickers") or []),
            " ".join(context.get("tags") or []),
            str(context.get("event") or ""),
        ]
        return self.retrieve(
            " ".join(part for part in query_parts if part).strip(),
            domain_id=context.get("domain_id"),
            tickers=context.get("tickers") or [],
            tags=context.get("tags") or [],
            top_k=top_k,
            as_of=context.get("as_of"),
            require_point_in_time=bool(context.get("require_point_in_time", False)),
            require_source_provenance=bool(context.get("require_source_provenance", False)),
            intended_use=str(context.get("intended_use") or "evidence"),
        )


def render_retrieval_markdown(result: KnowledgeRetrievalResult) -> str:
    lines = [
        "# Analyst Knowledge Retrieval",
        "",
        f"- Query: `{result.query.query}`",
        f"- Domain: `{result.query.domain_id}`",
        f"- Tickers: `{', '.join(result.query.tickers)}`",
        f"- Hits: `{len(result.hits)}`",
        f"- Retrieval status: `{result.audit.get('status')}`",
        f"- As of: `{result.query.as_of}`",
        "",
    ]

    if not result.hits:
        lines.append("No knowledge items matched.")
        for exclusion in result.exclusions:
            lines.append(
                f"- Excluded `{exclusion.item_id}`: {', '.join(exclusion.reasons)}"
            )
        return "\n".join(lines) + "\n"

    for idx, hit in enumerate(result.hits, start=1):
        item = hit.item
        lines.extend(
            [
                f"## {idx}. {item.title}",
                "",
                f"- Score: `{hit.score}`",
                f"- Type: `{item.item_type}`",
                f"- Tags: `{', '.join(item.tags)}`",
                f"- Tickers: `{', '.join(item.tickers)}`",
                f"- Match reasons: `{'; '.join(hit.match_reasons)}`",
                "",
                item.body,
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"
