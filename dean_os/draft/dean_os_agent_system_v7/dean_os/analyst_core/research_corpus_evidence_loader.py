from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.analysts.schemas import AnalystEvidenceItem
from dean_os.utils import sha256_json


DEFAULT_RESEARCH_QUERY = (
    "semiconductor demand capex hyperscaler data center supply chain "
    "foundry hbm advanced packaging capacity export control"
)


class ResearchCorpusEvidenceLoader:
    """Load local full-text corpus matches as weak, context-only evidence."""

    def load(
        self,
        corpus_path: str | Path,
        *,
        domain_id: str,
        as_of: str,
        query: str = DEFAULT_RESEARCH_QUERY,
        tickers: list[str] | None = None,
        top_k: int = 20,
    ) -> list[AnalystEvidenceItem]:
        path = Path(corpus_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        cutoff = _timestamp(as_of, "as_of")
        snapshot_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if snapshot_at > cutoff:
            raise ValueError(
                "Research corpus snapshot is future evidence: "
                f"{snapshot_at.isoformat()} > {cutoff.isoformat()}"
            )
        terms = _query_terms(query)
        if not terms:
            raise ValueError("research query must contain meaningful terms")
        top_k = max(1, min(int(top_k), 50))
        rows = _search_rows(path, terms, candidate_limit=top_k * 30)
        corpus_sha256 = _sha256_file(path)
        requested_tickers = sorted(
            {str(value).strip().upper() for value in tickers or [] if str(value).strip()}
        )
        candidates: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            published_at = _optional_timestamp(row["published_at"])
            if published_at is None or published_at > cutoff:
                continue
            haystack = f"{row['title']} {row['text']}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                candidates.append((score, row))
        candidates.sort(key=lambda value: (-value[0], value[1]["document_id"]))

        evidence: list[AnalystEvidenceItem] = []
        for score, row in candidates[:top_k]:
            text = str(row["text"] or "").strip()
            content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            evidence_type = _classify_evidence_type(text)
            doc_tickers = _document_tickers(row.get("tickers"), requested_tickers)
            evidence.append(
                AnalystEvidenceItem(
                    evidence_id="research_corpus_" + sha256_json(
                        {
                            "corpus_sha256": corpus_sha256,
                            "document_id": row["document_id"],
                            "content_sha256": content_sha256,
                        }
                    )[:24],
                    source_type="research_document",
                    source=(
                        str(row.get("uri") or "").strip()
                        or f"{path}#document_id={row['document_id']}"
                    ),
                    published_at=str(row["published_at"]),
                    as_of=as_of,
                    domain_id=domain_id,
                    tickers=doc_tickers,
                    sectors=[domain_id],
                    evidence_type=evidence_type,
                    summary=text[:800],
                    stance_hint="unknown",
                    strength=min(0.4, 0.18 + score * 0.025),
                    freshness_score=0.35,
                    directness="sector",
                    reliability_score=0.4 if row.get("uri") else 0.3,
                    limitations=[
                        "Context-only corpus match; cannot close a required evidence lane.",
                        "Corpus ingestion did not preserve original per-document retrieval time.",
                        *(
                            []
                            if row.get("uri")
                            else ["Original external source locator is missing."]
                        ),
                    ],
                    provenance={
                        "contract": "dean_research_corpus_context_v1",
                        "corpus_path": str(path),
                        "corpus_sha256": corpus_sha256,
                        "corpus_snapshot_at": snapshot_at.isoformat(),
                        "availability_basis": "corpus_file_mtime_conservative",
                        "document_id": row["document_id"],
                        "content_sha256": content_sha256,
                        "query_match_count": score,
                        "required_lane_eligible": False,
                        "ticker_thesis_eligible": False,
                    },
                    point_in_time={
                        "status": "context_only_snapshot_compatible",
                        "published_at": row["published_at"],
                        "snapshot_at": snapshot_at.isoformat(),
                        "as_of": as_of,
                    },
                )
            )
        return evidence


def _search_rows(
    path: Path,
    terms: list[str],
    *,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    clauses = " OR ".join(
        "LOWER(title || ' ' || text) LIKE ?" for _ in terms
    )
    query = (
        "SELECT document_id, title, source_type, uri, published_at, "
        "tickers, sectors, tags, metadata, text FROM documents "
        "WHERE LENGTH(published_at) > 0 AND (" + clauses + ") LIMIT ?"
    )
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            query,
            [f"%{term}%" for term in terms] + [candidate_limit],
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def _query_terms(query: str) -> list[str]:
    normalized = " ".join(str(query).lower().split())
    phrases = [
        "data center", "supply chain", "advanced packaging", "export control"
    ]
    terms = [phrase for phrase in phrases if phrase in normalized]
    terms.extend(
        token
        for token in normalized.split()
        if len(token) >= 4 and token not in {"data", "center", "supply", "chain", "advanced", "packaging", "export", "control"}
    )
    return list(dict.fromkeys(terms))


def _classify_evidence_type(text: str) -> str:
    value = text.lower()
    if any(term in value for term in ("capex", "hyperscaler", "data center")):
        return "capex_cycle"
    if any(term in value for term in ("supply chain", "foundry", "hbm", "advanced packaging", "capacity")):
        return "supply_chain"
    if any(term in value for term in ("export control", "sanction", "policy")):
        return "policy_or_geopolitical"
    return "sector_demand"


def _document_tickers(raw: Any, requested: list[str]) -> list[str]:
    import json

    try:
        values = json.loads(raw or "[]")
    except (TypeError, ValueError):
        values = []
    normalized = {str(value).strip().upper() for value in values if str(value).strip()}
    return sorted(normalized.intersection(requested))


def _optional_timestamp(value: Any) -> datetime | None:
    try:
        return _timestamp(value, "published_at")
    except ValueError:
        return None


def _timestamp(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["DEFAULT_RESEARCH_QUERY", "ResearchCorpusEvidenceLoader"]
