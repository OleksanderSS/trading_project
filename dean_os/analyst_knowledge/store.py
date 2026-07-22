from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from dean_os.analyst_knowledge.schemas import (
    KnowledgeItem,
    KnowledgePack,
    KnowledgeQuery,
    KnowledgeRetrievalExclusion,
    KnowledgeRetrievalHit,
    KnowledgeRetrievalResult,
)

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9_+.-]+")


class LocalKnowledgeStore:
    """Small local knowledge store for domain analyst packs.

    This is intentionally simple: JSONL + lexical search. It is not a vector DB
    and does not call an LLM. The point is to make the analyst usable while
    keeping dependencies and side effects low.
    """

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "knowledge_items.jsonl"
        self.sources_path = self.store_dir / "knowledge_sources.jsonl"
        self.item_lineage_path = self.store_dir / "knowledge_item_lineage.json"
        self.pack_manifest_path = self.store_dir / "packs.json"
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        if self.index_path.exists():
            self.index_path.unlink()
        if self.sources_path.exists():
            self.sources_path.unlink()
        if self.item_lineage_path.exists():
            self.item_lineage_path.unlink()
        if self.pack_manifest_path.exists():
            self.pack_manifest_path.unlink()

    def add_pack(self, pack: KnowledgePack) -> None:
        pack_payload = pack.model_dump(mode="json")
        pack_sha256 = sha256(
            json.dumps(
                pack_payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        existing = {item.item_id: item for item in self.iter_items()}
        existing_sources = {source.source_id: source for source in self.iter_sources()}
        lineage = self._load_item_lineage()
        for item in pack.items:
            previous_item = existing.get(item.item_id)
            previous_lineage = lineage.get(item.item_id) or {}
            if previous_item is not None and previous_item != item:
                if previous_lineage.get("pack_id") != pack.pack_id:
                    raise ValueError(
                        f"Knowledge item_id collision across packs: {item.item_id}"
                    )
                if previous_lineage.get("pack_version") == pack.version:
                    raise ValueError(
                        f"Knowledge pack {pack.pack_id} changed item {item.item_id} "
                        "without a version bump"
                    )
            existing[item.item_id] = item
            lineage[item.item_id] = {
                "pack_id": pack.pack_id,
                "pack_version": pack.version,
                "pack_sha256": pack_sha256,
                "domain_id": pack.domain_id,
                "source_ids": item.source_ids,
            }
        for source in pack.sources:
            previous_source = existing_sources.get(source.source_id)
            if previous_source is not None and previous_source != source:
                raise ValueError(
                    f"Knowledge source_id collision requires a versioned source_id: {source.source_id}"
                )
            existing_sources[source.source_id] = source

        with self.index_path.open("w", encoding="utf-8") as f:
            for item in sorted(existing.values(), key=lambda value: value.item_id):
                f.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
        with self.sources_path.open("w", encoding="utf-8") as f:
            for source in sorted(existing_sources.values(), key=lambda value: value.source_id):
                f.write(json.dumps(source.model_dump(mode="json"), ensure_ascii=False) + "\n")
        self.item_lineage_path.write_text(
            json.dumps(lineage, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        manifest = self._load_manifest()
        manifest[pack.pack_id] = {
            "pack_id": pack.pack_id,
            "domain_id": pack.domain_id,
            "name": pack.name,
            "version": pack.version,
            "item_count": len(pack.items),
            "source_count": len(pack.sources),
            "tickers": pack.tickers,
            "tags": pack.tags,
            "pack_sha256": pack_sha256,
        }
        self.pack_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_packs(self) -> dict:
        return self._load_manifest()

    def iter_items(self) -> Iterable[KnowledgeItem]:
        if not self.index_path.exists():
            return []
        items: list[KnowledgeItem] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(KnowledgeItem(**json.loads(line)))
        return items

    def iter_sources(self) -> Iterable[Any]:
        from dean_os.analyst_knowledge.schemas import KnowledgeSource

        if not self.sources_path.exists():
            return []
        sources: list[KnowledgeSource] = []
        for line in self.sources_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sources.append(KnowledgeSource(**json.loads(line)))
        return sources

    def search(self, query: KnowledgeQuery | str, **kwargs) -> KnowledgeRetrievalResult:
        q = query if isinstance(query, KnowledgeQuery) else KnowledgeQuery(query=query, **kwargs)
        query_terms = _tokens(q.query)
        hits: list[KnowledgeRetrievalHit] = []
        exclusions: list[KnowledgeRetrievalExclusion] = []
        sources_by_id = {source.source_id: source for source in self.iter_sources()}
        lineage_by_item = self._load_item_lineage()

        for item in self.iter_items():
            if q.domain_id and item.domain_id != q.domain_id:
                continue
            if q.tickers and not set(q.tickers).intersection(item.tickers):
                continue
            if q.tags and not set(q.tags).intersection(item.tags):
                continue
            if q.item_types and item.item_type not in q.item_types:
                continue

            item_text = " ".join(
                [
                    item.title,
                    item.body,
                    " ".join(item.tags),
                    " ".join(item.tickers),
                    " ".join(item.sectors),
                    " ".join(item.metrics),
                    item.item_type,
                ]
            )
            item_terms = _tokens(item_text)
            overlap = sorted(query_terms.intersection(item_terms))
            score = 0.0
            reasons: list[str] = []

            if overlap:
                score += len(overlap) * 1.5
                reasons.append(f"term overlap: {', '.join(overlap[:8])}")

            if q.tickers and set(q.tickers).intersection(item.tickers):
                score += 4.0
                reasons.append("ticker filter match")

            if q.tags and set(q.tags).intersection(item.tags):
                score += 2.5
                reasons.append("tag filter match")

            if q.domain_id and q.domain_id == item.domain_id:
                score += 1.0
                reasons.append("domain filter match")

            score += item.confidence * 0.75
            score += item.importance * 0.2

            # keep weak but filtered results, drop unrelated unfiltered results
            if score > 0.7:
                sources = [
                    sources_by_id[source_id]
                    for source_id in item.source_ids
                    if source_id in sources_by_id
                ]
                lineage = dict(lineage_by_item.get(item.item_id) or {})
                point_in_time = _point_in_time_audit(
                    item=item,
                    sources=sources,
                    lineage=lineage,
                    query=q,
                )
                if not point_in_time["eligible"]:
                    exclusions.append(
                        KnowledgeRetrievalExclusion(
                            item_id=item.item_id,
                            title=item.title,
                            reasons=point_in_time["reasons"],
                            point_in_time=point_in_time,
                        )
                    )
                    continue
                hits.append(
                    KnowledgeRetrievalHit(
                        item=item,
                        score=round(score, 4),
                        matched_terms=overlap,
                        match_reasons=reasons or ["metadata/filter match"],
                        sources=sources,
                        lineage=lineage,
                        point_in_time=point_in_time,
                    )
                )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        exclusions.sort(key=lambda exclusion: exclusion.item_id)
        selected_hits = hits[: q.top_k]
        return KnowledgeRetrievalResult(
            query=q,
            hits=selected_hits,
            exclusions=exclusions,
            audit=_retrieval_audit(
                query=q,
                eligible_count=len(hits),
                selected_count=len(selected_hits),
                excluded_count=len(exclusions),
            ),
        )

    def audit_point_in_time(
        self,
        *,
        as_of: str,
        intended_use: str = "evidence",
    ) -> list[dict[str, Any]]:
        """Audit every stored item without depending on lexical query relevance."""

        query = KnowledgeQuery(
            query="point in time audit",
            as_of=as_of,
            require_point_in_time=True,
            require_source_provenance=True,
            intended_use=intended_use,
        )
        sources_by_id = {source.source_id: source for source in self.iter_sources()}
        lineage_by_item = self._load_item_lineage()
        records: list[dict[str, Any]] = []
        for item in self.iter_items():
            sources = [
                sources_by_id[source_id]
                for source_id in item.source_ids
                if source_id in sources_by_id
            ]
            audit = _point_in_time_audit(
                item=item,
                sources=sources,
                lineage=dict(lineage_by_item.get(item.item_id) or {}),
                query=query,
            )
            records.append(
                {
                    "item_id": item.item_id,
                    "domain_id": item.domain_id,
                    "title": item.title,
                    "source_ids": item.source_ids,
                    **audit,
                }
            )
        return sorted(records, key=lambda record: record["item_id"])

    def _load_manifest(self) -> dict:
        if not self.pack_manifest_path.exists():
            return {}
        return json.loads(self.pack_manifest_path.read_text(encoding="utf-8"))

    def _load_item_lineage(self) -> dict[str, dict[str, Any]]:
        if not self.item_lineage_path.exists():
            return {}
        data = json.loads(self.item_lineage_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w+.-]+", text or "", flags=re.UNICODE)
        if len(token) > 1
    }


def _point_in_time_audit(
    *,
    item: KnowledgeItem,
    sources: list[Any],
    lineage: dict[str, Any],
    query: KnowledgeQuery,
) -> dict[str, Any]:
    reasons: list[str] = []
    as_of = _parse_timestamp(query.as_of)
    source_ids = {source.source_id for source in sources}
    missing_source_ids = sorted(set(item.source_ids).difference(source_ids))

    if query.require_point_in_time:
        item_updated_at = _parse_timestamp(item.updated_at)
        if item_updated_at is None:
            reasons.append("item_updated_at_missing_or_invalid")
        elif as_of is not None and item_updated_at > as_of:
            reasons.append("item_updated_after_as_of")

    if query.require_source_provenance:
        if not item.source_ids:
            reasons.append("item_source_ids_missing")
        if missing_source_ids:
            reasons.append("referenced_sources_missing_from_store")
        if not lineage.get("pack_id") or not _valid_sha256(lineage.get("pack_sha256")):
            reasons.append("pack_lineage_missing_or_invalid")

    source_audits: list[dict[str, Any]] = []
    for source in sources:
        source_reasons: list[str] = []
        published_at = _parse_timestamp(source.published_at)
        retrieved_at = _parse_timestamp(source.retrieved_at)

        if query.require_point_in_time:
            if published_at is None:
                source_reasons.append("published_at_missing_or_invalid")
            elif as_of is not None and published_at > as_of:
                source_reasons.append("published_after_as_of")
            if retrieved_at is None:
                source_reasons.append("retrieved_at_missing_or_invalid")
            elif as_of is not None and retrieved_at > as_of:
                source_reasons.append("retrieved_after_as_of")

        if query.require_source_provenance:
            if not _valid_sha256(source.content_sha256):
                source_reasons.append("content_sha256_missing_or_invalid")
            if not (source.reference or source.raw_storage_path):
                source_reasons.append("source_locator_missing")
            if query.intended_use not in source.allowed_uses:
                source_reasons.append("intended_use_not_allowed")

        if source_reasons:
            reasons.append(f"source:{source.source_id}:" + ",".join(source_reasons))
        source_audits.append(
            {
                "source_id": source.source_id,
                "published_at": source.published_at,
                "retrieved_at": source.retrieved_at,
                "content_sha256": source.content_sha256,
                "allowed_uses": source.allowed_uses,
                "status": "eligible" if not source_reasons else "blocked",
                "reasons": source_reasons,
            }
        )

    strict = query.require_point_in_time or query.require_source_provenance
    return {
        "contract": "dean_analyst_knowledge_point_in_time_v1",
        "strict": strict,
        "as_of": query.as_of,
        "intended_use": query.intended_use,
        "eligible": not reasons,
        "status": "point_in_time_compatible" if not reasons else "blocked",
        "reasons": reasons,
        "source_audits": source_audits,
        "item_updated_at": item.updated_at,
        "pack_id": lineage.get("pack_id"),
        "pack_sha256": lineage.get("pack_sha256"),
    }


def _retrieval_audit(
    *,
    query: KnowledgeQuery,
    eligible_count: int,
    selected_count: int,
    excluded_count: int,
) -> dict[str, Any]:
    strict = query.require_point_in_time or query.require_source_provenance
    if selected_count:
        status = "eligible_with_exclusions" if excluded_count else "eligible"
    elif excluded_count:
        status = "blocked_no_point_in_time_eligible_hits"
    else:
        status = "no_matching_items"
    return {
        "contract": "dean_analyst_knowledge_point_in_time_v1",
        "status": status,
        "strict": strict,
        "as_of": query.as_of,
        "intended_use": query.intended_use,
        "eligible_count_before_top_k": eligible_count,
        "selected_count": selected_count,
        "excluded_count": excluded_count,
    }


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _valid_sha256(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return len(normalized) == 64 and all(char in "0123456789abcdef" for char in normalized)
