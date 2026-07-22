from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class EvidenceDedupDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_index: int
    cluster_id: str
    status: Literal["unique", "exact_duplicate", "near_duplicate", "independent_corroboration"]
    canonical_index: int
    similarity: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class EvidenceDedupResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    decisions: list[EvidenceDedupDecision]
    accepted_indices: list[int]
    suppressed_indices: list[int]
    cluster_count: int


class SemanticEvidenceDeduplicator:
    """Conservative lexical deduplicator.

    Exact or same-source near duplicates are suppressed. Similar items from
    independent sources are retained as corroboration, because deduplication
    must not erase source diversity.
    """

    def __init__(self, *, near_duplicate_threshold: float = 0.86, corroboration_threshold: float = 0.72):
        self.near_duplicate_threshold = near_duplicate_threshold
        self.corroboration_threshold = corroboration_threshold

    def analyze(self, payloads: list[dict[str, Any]]) -> EvidenceDedupResult:
        decisions: list[EvidenceDedupDecision] = []
        accepted: list[int] = []
        suppressed: list[int] = []
        clusters: dict[int, str] = {}
        fingerprints = [self._fingerprint(item) for item in payloads]

        for index, payload in enumerate(payloads):
            best_index: int | None = None
            best_similarity = 0.0
            exact_match = False
            for candidate_index in accepted:
                left = fingerprints[index]
                right = fingerprints[candidate_index]
                exact = bool(left["content_hash"] and left["content_hash"] == right["content_hash"])
                similarity = 1.0 if exact else self._similarity(left["tokens"], right["tokens"])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_index = candidate_index
                    exact_match = exact

            if best_index is None:
                cluster = f"evidence_cluster_{uuid4().hex}"
                clusters[index] = cluster
                accepted.append(index)
                decisions.append(EvidenceDedupDecision(
                    input_index=index,
                    cluster_id=cluster,
                    status="unique",
                    canonical_index=index,
                    similarity=1.0,
                    reasons=["first item in cluster"],
                ))
                continue

            same_source = self._source(payload) == self._source(payloads[best_index])
            cluster = clusters[best_index]
            clusters[index] = cluster
            if exact_match:
                suppressed.append(index)
                decisions.append(EvidenceDedupDecision(
                    input_index=index,
                    cluster_id=cluster,
                    status="exact_duplicate",
                    canonical_index=best_index,
                    similarity=1.0,
                    reasons=["identical normalized content hash"],
                ))
            elif best_similarity >= self.near_duplicate_threshold and same_source:
                suppressed.append(index)
                decisions.append(EvidenceDedupDecision(
                    input_index=index,
                    cluster_id=cluster,
                    status="near_duplicate",
                    canonical_index=best_index,
                    similarity=round(best_similarity, 4),
                    reasons=["high lexical overlap from the same source"],
                ))
            elif best_similarity >= self.corroboration_threshold and not same_source:
                accepted.append(index)
                decisions.append(EvidenceDedupDecision(
                    input_index=index,
                    cluster_id=cluster,
                    status="independent_corroboration",
                    canonical_index=best_index,
                    similarity=round(best_similarity, 4),
                    reasons=["similar claim from a distinct source retained as corroboration"],
                ))
            else:
                new_cluster = f"evidence_cluster_{uuid4().hex}"
                clusters[index] = new_cluster
                accepted.append(index)
                decisions.append(EvidenceDedupDecision(
                    input_index=index,
                    cluster_id=new_cluster,
                    status="unique",
                    canonical_index=index,
                    similarity=round(best_similarity, 4),
                    reasons=["similarity below deduplication threshold"],
                ))

        return EvidenceDedupResult(
            decisions=decisions,
            accepted_indices=accepted,
            suppressed_indices=suppressed,
            cluster_count=len(set(clusters.values())),
        )

    def _fingerprint(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = " ".join(str(payload.get(key) or "") for key in ("title", "summary", "text", "content"))
        normalized = " ".join(_TOKEN_RE.findall(text.lower()))
        tokens = set(normalized.split())
        return {
            "tokens": tokens,
            "content_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else "",
        }

    @staticmethod
    def _similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _source(payload: dict[str, Any]) -> str:
        return " ".join(str(payload.get("source") or payload.get("source_name") or "unknown").lower().split())


__all__ = ["EvidenceDedupDecision", "EvidenceDedupResult", "SemanticEvidenceDeduplicator"]
