from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dean_os.analysts.profiles import get_domain_profile


class CollectorRoute(BaseModel):
    model_config = ConfigDict(frozen=True)

    route_id: str
    evidence_lane: str
    collector_kind: Literal["official_feed", "filing_feed", "news_feed", "research_corpus", "market_data", "manual_upload", "pipeline_artifact"]
    priority: int = Field(ge=1, le=5)
    source_types: list[str] = Field(default_factory=list)
    query_terms: list[str] = Field(default_factory=list)
    review_only: bool = True
    network_execution_allowed: bool = False


class DomainCollectorRouter:
    """Builds bounded collector instructions; it never executes collectors."""

    def __init__(self, domain_id: str):
        self.domain_id = domain_id
        self.profile = get_domain_profile(domain_id)

    def routes_for(self, evidence_lane: str) -> list[CollectorRoute]:
        configured = (self.profile.source_registry_policy or {}).get("collector_routes", {})
        lane_config = configured.get(evidence_lane) if isinstance(configured, dict) else None
        if lane_config:
            return [CollectorRoute.model_validate(item) for item in lane_config]
        return self._default_routes(evidence_lane)

    def _default_routes(self, lane: str) -> list[CollectorRoute]:
        terms = list((self.profile.evidence_keywords or {}).get(lane, []))[:12]
        routes = [
            CollectorRoute(
                route_id=f"{self.domain_id}:{lane}:official",
                evidence_lane=lane,
                collector_kind="official_feed",
                priority=1,
                source_types=["filing", "report", "document", "dataset"],
                query_terms=terms,
            ),
            CollectorRoute(
                route_id=f"{self.domain_id}:{lane}:news",
                evidence_lane=lane,
                collector_kind="news_feed",
                priority=2,
                source_types=["news", "article", "transcript"],
                query_terms=terms,
            ),
            CollectorRoute(
                route_id=f"{self.domain_id}:{lane}:corpus",
                evidence_lane=lane,
                collector_kind="research_corpus",
                priority=3,
                source_types=["book", "research_note", "report"],
                query_terms=terms,
            ),
        ]
        if lane in {"market_confirmation", "pricing", "relative_strength"}:
            routes.insert(0, CollectorRoute(
                route_id=f"{self.domain_id}:{lane}:market_data",
                evidence_lane=lane,
                collector_kind="market_data",
                priority=1,
                source_types=["metric", "dataset"],
                query_terms=terms,
            ))
        return routes


__all__ = ["CollectorRoute", "DomainCollectorRouter"]
