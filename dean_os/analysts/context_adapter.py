from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dean_os.context_evidence_provenance import (
    CONTEXT_EVIDENCE_CONTRACT,
    audit_news_records,
    audit_research_documents,
    parse_timezone_aware,
)
from dean_os.schemas import MarketContext
from dean_os.structured_context_provenance import (
    STRUCTURED_CONTEXT_CONTRACT,
    audit_structured_context,
)

from .profiles import get_domain_profile
from .schemas import AnalystEvidenceItem, DomainProfile


def _load_evidence_keywords(domain_id: str) -> dict[str, tuple[str, ...]]:
    path = (
        Path(__file__).resolve().parent.parent.parent
        / "config"
        / "domain_profiles"
        / domain_id
        / "evidence_keywords.yaml"
    )
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data: dict[str, list[str]] = yaml.safe_load(f) or {}
    return {k: tuple(v) for k, v in data.items()}


# Keyed by registry context_key, not the raw FRED series id (e.g. "cpi", not
# "CPIAUCSL"). SavedMacroEvidenceProducer's market_context_fragment.macro dict
# -- and therefore audit_structured_context's observation["name"], which is
# what `_structured_context_evidence` below actually reads as `series_id` --
# is keyed by context_key (dean_os/config/macro_series_registry.yaml). A
# first version of this map used the raw FRED series id and silently matched
# nothing against live data: every macro observation, for every domain,
# fell through to the generic domain-default evidence_type regardless of
# which series it actually was. Caught by an end-to-end smoke test against a
# real SavedMacroEvidenceProducer artifact (same failure mode already found
# and fixed in RegimeContextLens's MACRO_SERIES_TO_DIMENSION).
MACRO_SERIES_EVIDENCE_MAP: dict[str, dict[str, str]] = {
    # context_key -> domain_id -> evidence_type
    "cpi": {"macro_policy": "inflation", "energy": "policy_or_geopolitical", "real_estate": "market_confirmation", "liquidity_credit": "inflation"},
    "pce_price_index": {"macro_policy": "inflation", "liquidity_credit": "inflation"},
    "core_pce_price_index": {"macro_policy": "inflation", "liquidity_credit": "inflation"},
    "producer_price_index": {"macro_policy": "inflation", "energy": "policy_or_geopolitical", "agriculture": "supply", "logistics": "market_confirmation"},
    "fed_funds_rate": {"macro_policy": "rates_policy", "energy": "market_confirmation", "real_estate": "policy_or_geopolitical", "liquidity_credit": "rates_policy"},
    "treasury_10y_daily": {"macro_policy": "rates_policy", "energy": "market_confirmation", "real_estate": "market_confirmation", "liquidity_credit": "rates_policy"},
    "treasury_10y_monthly": {"macro_policy": "rates_policy", "real_estate": "market_confirmation", "liquidity_credit": "rates_policy"},
    "treasury_2y_monthly": {"macro_policy": "rates_policy", "liquidity_credit": "rates_policy"},
    "yield_curve_10y_2y": {"macro_policy": "rates_policy", "energy": "market_confirmation", "liquidity_credit": "market_confirmation", "real_estate": "market_confirmation"},
    "federal_reserve_total_assets": {"macro_policy": "rates_policy", "liquidity_credit": "policy_or_geopolitical"},
    "unemployment_rate": {"macro_policy": "labor_market", "real_estate": "demand", "liquidity_credit": "market_confirmation"},
    "nonfarm_payrolls": {"macro_policy": "labor_market", "energy": "demand", "real_estate": "demand"},
    "manufacturing_employment": {"macro_policy": "labor_market", "energy": "demand", "logistics": "demand"},
    "continued_claims": {"macro_policy": "labor_market"},
    "industrial_production": {"macro_policy": "growth", "energy": "demand", "logistics": "demand", "semiconductor_ai_infrastructure": "demand"},
    "retail_sales": {"macro_policy": "growth", "energy": "demand", "logistics": "demand", "real_estate": "demand"},
    "housing_starts": {"macro_policy": "growth", "energy": "demand", "real_estate": "supply", "logistics": "demand"},
    "building_permits": {"macro_policy": "growth", "real_estate": "supply"},
    "durable_goods_orders": {"macro_policy": "growth", "semiconductor_ai_infrastructure": "demand", "logistics": "demand"},
    "real_disposable_personal_income": {"macro_policy": "growth", "real_estate": "demand"},
    "total_vehicle_sales": {"macro_policy": "growth", "energy": "demand", "logistics": "demand"},
    "consumer_sentiment": {"macro_policy": "growth", "energy": "market_confirmation", "real_estate": "market_confirmation", "liquidity_credit": "market_confirmation"},
    "usd_cny": {"macro_policy": "growth", "agriculture": "market_confirmation"},
    "usd_eur": {"macro_policy": "growth"},
    "vix": {"macro_policy": "market_confirmation", "energy": "market_confirmation", "liquidity_credit": "market_confirmation", "geopolitics": "market_confirmation", "real_estate": "market_confirmation"},
    "high_yield_oas": {"macro_policy": "market_confirmation", "energy": "market_confirmation", "liquidity_credit": "market_confirmation", "real_estate": "market_confirmation"},
    "wti_crude_oil": {"macro_policy": "inflation", "energy": "supply", "logistics": "supply", "agriculture": "supply", "geopolitics": "market_confirmation"},
}


def _macro_series_evidence_type(context_key: str, domain_id: str) -> str | None:
    mapping = MACRO_SERIES_EVIDENCE_MAP.get(context_key)
    if mapping is None:
        return None
    return mapping.get(domain_id)


_POSITIVE_TERMS = {
    "accelerating",
    "growth",
    "upgrade",
    "strong",
    "tailwind",
    "demand",
    "recovery",
    "expansion",
    "easing",
    "supportive",
    "improving",
    "beat",
    "raised guidance",
    "capex growth",
}

_NEGATIVE_TERMS = {
    "slowdown",
    "downgrade",
    "weak",
    "headwind",
    "risk",
    "restriction",
    "sanction",
    "shortage",
    "margin pressure",
    "recession",
    "tightening",
    "miss",
    "cut guidance",
}


class MarketContextEvidenceAdapter:
    """Builds review-only analyst evidence from the existing MarketContext.

    This adapter is intentionally rule-based. It does not call an LLM, FinBERT,
    network sources, or collectors.
    """

    def __init__(self, domain_id: str):
        self.profile: DomainProfile = get_domain_profile(domain_id)
        self.domain_id = domain_id
        self.keyword_map = _load_evidence_keywords(domain_id)

    def from_context(
        self,
        context: MarketContext,
        as_of: str,
    ) -> list[AnalystEvidenceItem]:
        return self.adapt(context, as_of=as_of)["evidence"]

    def adapt(
        self,
        context: MarketContext,
        *,
        as_of: str,
    ) -> dict[str, Any]:
        news_audit = audit_news_records(
            list(context.news or []),
            as_of=as_of,
            requested_tickers=context.tickers,
        )
        news_evidence = self._news_evidence(
            context,
            news_audit["accepted"],
            as_of,
        )
        document_audit = audit_research_documents(
            list(context.research_documents or []),
            as_of=as_of,
        )
        document_evidence = self._document_evidence(
            context,
            document_audit["accepted"],
            as_of,
        )
        structured_evidence, structured_exclusions = (
            self._structured_context_evidence(context, as_of)
        )
        note_evidence, note_exclusions = self._research_note_evidence(
            context,
            as_of,
        )
        evidence = [
            *news_evidence,
            *document_evidence,
            *structured_evidence,
            *note_evidence,
        ]
        exclusions = [
            *[
                {"family": "news", **item}
                for item in news_audit["exclusions"]
            ],
            *[
                {"family": "research_document", **item}
                for item in document_audit["exclusions"]
            ],
            *structured_exclusions,
            *note_exclusions,
        ]
        return {
            "contract": CONTEXT_EVIDENCE_CONTRACT,
            "status": (
                "review_context_ready_with_exclusions"
                if evidence and exclusions
                else "review_context_ready"
                if evidence
                else "blocked_no_point_in_time_context_evidence"
            ),
            "as_of": news_audit["as_of"],
            "evidence": evidence,
            "exclusions": exclusions,
            "summary": {
                "evidence_count": len(evidence),
                "excluded_count": len(exclusions),
                "news_input_count": news_audit["input_count"],
                "news_accepted_count": news_audit[
                    "accepted_count"
                ],
                "news_excluded_count": news_audit[
                    "excluded_count"
                ],
                "document_input_count": document_audit[
                    "input_count"
                ],
                "document_accepted_count": document_audit[
                    "accepted_count"
                ],
                "document_excluded_count": document_audit[
                    "excluded_count"
                ],
                "ticker_direct_count": sum(
                    item.directness == "ticker" for item in evidence
                ),
                "can_influence_pipeline_prediction": False,
                "can_trade": False,
            },
            "safety": {
                "review_only": True,
                "future_evidence_excluded": True,
                "missing_timestamp_excluded": True,
                "plain_text_ticker_promotion_allowed": False,
                "pipeline_decision_influence": False,
                "live_execution_allowed": False,
            },
        }

    def _news_evidence(
        self,
        context: MarketContext,
        records: list[dict[str, Any]],
        as_of: str,
    ) -> list[AnalystEvidenceItem]:
        items: list[AnalystEvidenceItem] = []
        as_of_dt = parse_timezone_aware(as_of)
        for idx, raw in enumerate(records):
            text = _text_from_news(raw)
            lower = text.lower()
            if not lower:
                continue
            semantic = raw.get("_dean_semantic_evidence")
            if not isinstance(semantic, dict):
                semantic = {}
            explicit_type = str(
                semantic.get("evidence_type") or ""
            )
            if explicit_type in self.keyword_map:
                classifications = [
                    (
                        explicit_type,
                        list(semantic.get("matched_terms") or []),
                    )
                ]
            else:
                classifications = [
                    (
                        evidence_type,
                        [
                            keyword
                            for keyword in keywords
                            if keyword in lower
                        ],
                    )
                    for evidence_type, keywords in self.keyword_map.items()
                ]

            for evidence_type, hits in classifications:
                if not hits:
                    continue
                provenance = {
                    **dict(
                        raw.get("_dean_context_provenance") or {}
                    ),
                    "required_lane_eligible": (
                        semantic.get("required_lane_eligible") is True
                    ),
                    "semantic_evidence_contract": semantic.get(
                        "producer_contract"
                    ),
                    "source_tier": semantic.get("source_tier"),
                    "source_identity": semantic.get("source_identity"),
                    "candidate_sha256": semantic.get(
                        "candidate_sha256"
                    ),
                }
                direct_tickers = list(
                    provenance.get("direct_requested_tickers") or []
                )
                published_at = provenance.get("published_at")
                source_tier = str(
                    semantic.get("source_tier") or ""
                )
                reliability = {
                    "tier_1_core_evidence": 0.9,
                    "tier_2_strong_context": 0.75,
                    "tier_3_event_context": 0.55,
                    "tier_4_weak_or_unverified": 0.3,
                }.get(source_tier, 0.4)
                items.append(
                    AnalystEvidenceItem(
                        source_type="news",
                        source=str(
                            provenance.get("source_locator")
                            or f"context.news[{idx}]"
                        ),
                        published_at=published_at,
                        as_of=as_of,
                        domain_id=self.domain_id,
                        tickers=direct_tickers,
                        sectors=[self.domain_id],
                        evidence_type=evidence_type,
                        summary=_shorten(text),
                        stance_hint=str(
                            semantic.get("stance_hint")
                            or _stance_hint(lower)
                        ),
                        strength=min(
                            1.0,
                            0.35
                            + 0.1 * len(hits)
                            + (
                                0.15
                                if semantic.get(
                                    "required_lane_eligible"
                                )
                                is True
                                else 0.0
                            ),
                        ),
                        freshness_score=_freshness_score(
                            published_at,
                            as_of_dt,
                        ),
                        directness=(
                            "ticker" if direct_tickers else "sector"
                        ),
                        reliability_score=reliability,
                        limitations=[
                            (
                                "Rule-based news evidence; no LLM/FinBERT "
                                "validation in MVP."
                            ),
                            (
                                "Ticker directness requires explicit ticker "
                                "metadata or a cashtag."
                            ),
                            (
                                "Keyword-only news cannot close a required "
                                "evidence lane."
                                if semantic.get(
                                    "required_lane_eligible"
                                )
                                is not True
                                else (
                                    "Lane eligibility reflects source "
                                    "corroboration, not thesis truth."
                                )
                            ),
                        ],
                        provenance=provenance,
                        point_in_time={
                            "contract": CONTEXT_EVIDENCE_CONTRACT,
                            "status": "point_in_time_compatible",
                            "published_at": published_at,
                            "as_of": as_of,
                            "future_evidence": False,
                        },
                    )
                )
        return items

    def _document_evidence(
        self,
        context: MarketContext,
        documents: list[Any],
        as_of: str,
    ) -> list[AnalystEvidenceItem]:
        items: list[AnalystEvidenceItem] = []
        requested = {
            str(ticker).upper() for ticker in context.tickers
        }
        as_of_dt = parse_timezone_aware(as_of)
        for document in documents:
            text = f"{document.title} {document.text}".strip()
            lower = text.lower()
            provenance = dict(
                document.metadata.get(
                    "_dean_document_provenance",
                    {},
                )
            )
            document_tickers = {
                str(ticker).upper() for ticker in document.tickers
            }
            direct_tickers = sorted(
                requested.intersection(document_tickers)
            )
            
            explicit_evidence_type = provenance.get("evidence_type")
            if explicit_evidence_type:
                evidence_types = [explicit_evidence_type]
            else:
                evidence_types = [
                    evidence_type
                    for evidence_type, keywords in self.keyword_map.items()
                    if any(keyword in lower for keyword in keywords)
                ]
            
            if not evidence_types:
                continue
            for evidence_type in evidence_types:
                items.append(
                    AnalystEvidenceItem(
                        source_type=document.source_type,
                        source=str(document.uri),
                        published_at=provenance.get(
                            "availability_at"
                        ),
                        as_of=as_of,
                        domain_id=self.domain_id,
                        tickers=direct_tickers,
                        sectors=(
                            document.sectors or [self.domain_id]
                        ),
                        evidence_type=evidence_type,
                        summary=_shorten(document.text),
                        stance_hint=_stance_hint(lower),
                        strength=0.6,
                        freshness_score=_freshness_score(
                            provenance.get("availability_at"),
                            as_of_dt,
                        ),
                        directness=(
                            "ticker" if direct_tickers else "sector"
                        ),
                        reliability_score=_document_reliability(
                            document.source_type
                        ),
                        limitations=[
                            "Document evidence remains review-only.",
                            *provenance.get("limitations", []),
                        ],
                        provenance=provenance,
                        point_in_time={
                            "contract": CONTEXT_EVIDENCE_CONTRACT,
                            "status": "point_in_time_compatible",
                            "availability_at": provenance.get(
                                "availability_at"
                            ),
                            "availability_basis": provenance.get(
                                "availability_basis"
                            ),
                            "as_of": as_of,
                        },
                    )
                )
        return items

    def _structured_context_evidence(
        self,
        context: MarketContext,
        as_of: str,
    ) -> tuple[list[AnalystEvidenceItem], list[dict[str, Any]]]:
        items: list[AnalystEvidenceItem] = []
        audit = audit_structured_context(
            fundamentals=context.fundamentals,
            macro=context.macro,
            sector_data=context.sector_data,
            as_of=as_of,
        )
        exclusions = list(audit["exclusions"])
        as_of_dt = parse_timezone_aware(as_of)
        macro_domains = {
            "macro_policy",
            "liquidity_credit",
            "energy",
            "semiconductor_ai_infrastructure",
        }

        for observation in audit["accepted_observations"]:
            family = observation["family"]
            if family == "macro" and self.domain_id not in macro_domains:
                exclusions.append(
                    {
                        "family": family,
                        "scope": observation["scope"],
                        "name": observation["name"],
                        "status": "excluded",
                        "reasons": [
                            "structured_family_not_relevant_to_domain"
                        ],
                    }
                )
                continue
            if family == "fundamental":
                tickers = [str(observation["scope"]).upper()]
                source_type = "fundamental"
                directness = "ticker"
                evidence_type = str(
                    observation.get("evidence_type")
                    or "fundamental_context"
                )
                strength = 0.6
                reliability = 0.6
            elif family == "macro":
                tickers = []
                source_type = "macro"
                directness = "macro"
                # observation["name"] is the registry context_key (e.g.
                # "fed_funds_rate"), not the raw FRED series id -- see
                # MACRO_SERIES_EVIDENCE_MAP's docstring above.
                context_key = str(observation.get("name", ""))
                # NOT self._first_required_or(...): every domain profile sets
                # macro_evidence_type: "macro_context" as a deliberate sentinel
                # -- EventClassifierLens._classify_macro_observation special-
                # cases evidence_type == "macro_context" to keyword-classify
                # the series by name (cpi/pce -> inflation_observation,
                # fed_funds/central_bank -> liquidity_observation) instead of
                # falling through to generic text detection. Routing an
                # unmapped macro series through _first_required_or instead
                # silently replaced that sentinel with the domain's first
                # required_evidence_types entry (non-empty for every real
                # domain profile), which both defeated the classifier's
                # macro_context branch entirely and let irrelevant macro
                # noise falsely "satisfy" an unrelated required lane (e.g. a
                # random unmapped FRED series counting as semiconductor
                # sector-demand evidence).
                evidence_type = str(
                    observation.get("evidence_type")
                    or _macro_series_evidence_type(
                        context_key, self.domain_id
                    )
                    or self.profile.macro_evidence_type
                    or "market_confirmation"
                )
                strength = 0.55
                reliability = 0.55
            else:
                tickers = []
                source_type = "sector"
                directness = "sector"
                evidence_type = str(
                    observation.get("evidence_type")
                    or "sector_context"
                )
                strength = (
                    0.7
                    if observation.get("required_lane_eligible") is True
                    else 0.55
                )
                reliability = strength
            summary = (
                f"{family} observation {observation['name']}="
                f"{observation['value']} {observation['unit']} for "
                f"{observation['period']}."
            )
            structured_provenance = {
                **observation,
                "required_lane_eligible": (
                    observation.get("required_lane_eligible", True) is not False
                ),
                "ticker_thesis_eligible": False,
            }
            items.append(
                AnalystEvidenceItem(
                    source_type=source_type,
                    source=observation["source_locator"],
                    published_at=observation["available_at"],
                    as_of=as_of,
                    domain_id=self.domain_id,
                    tickers=tickers,
                    sectors=[self.domain_id],
                    evidence_type=evidence_type,
                    summary=summary,
                    stance_hint=str(
                        observation.get("stance_hint") or "unknown"
                    ),
                    strength=strength,
                    freshness_score=_freshness_score(
                        observation["available_at"],
                        as_of_dt,
                    ),
                    directness=directness,
                    reliability_score=reliability,
                    limitations=[
                        (
                            "Structured context is review-only supporting "
                            "evidence."
                        )
                    ],
                    provenance=structured_provenance,
                    point_in_time={
                        "contract": STRUCTURED_CONTEXT_CONTRACT,
                        "status": "point_in_time_compatible",
                        "available_at": observation["available_at"],
                        "period": observation["period"],
                        "unit": observation["unit"],
                        "as_of": as_of,
                    },
                )
            )

        if context.pipeline_result:
            exclusions.append(
                {
                    "family": "pipeline_result",
                    "status": "excluded",
                    "reasons": [
                        (
                            "pipeline_result_requires_separate_exact_context_"
                            "review_contract"
                        )
                    ],
                }
            )
        return items, exclusions

    def _research_note_evidence(
        self,
        context: MarketContext,
        as_of: str,
    ) -> tuple[list[AnalystEvidenceItem], list[dict[str, Any]]]:
        items: list[AnalystEvidenceItem] = []
        exclusions: list[dict[str, Any]] = []
        as_of_dt = parse_timezone_aware(as_of)
        for idx, note in enumerate(context.research_notes or []):
            topic = str(getattr(note, "topic", "") or "")
            thesis = str(getattr(note, "thesis", "") or "")
            text = f"{topic} {thesis}".strip()
            if not text:
                exclusions.append(
                    {
                        "family": f"research_note[{idx}]",
                        "status": "excluded",
                        "reasons": ["research_note_text_missing"],
                    }
                )
                continue
            created_at = parse_timezone_aware(
                getattr(note, "created_at", None)
            )
            citations = list(getattr(note, "citations", []) or [])
            citation_payloads = [
                (
                    citation.model_dump(mode="json")
                    if hasattr(citation, "model_dump")
                    else dict(citation)
                )
                for citation in citations
                if hasattr(citation, "model_dump")
                or isinstance(citation, dict)
            ]
            citation_times = [
                parse_timezone_aware(citation.get("timestamp"))
                for citation in citation_payloads
            ]
            reasons: list[str] = []
            if created_at is None:
                reasons.append(
                    "research_note_created_at_missing_or_invalid"
                )
            elif as_of_dt is not None and created_at > as_of_dt:
                reasons.append("research_note_created_after_as_of")
            if not citation_payloads:
                reasons.append("research_note_citations_missing")
            elif any(value is None for value in citation_times):
                reasons.append(
                    "research_note_citation_timestamp_missing_or_invalid"
                )
            elif as_of_dt is not None and any(
                value > as_of_dt
                for value in citation_times
                if value is not None
            ):
                reasons.append(
                    "research_note_citation_after_as_of"
                )
            if reasons:
                exclusions.append(
                    {
                        "family": f"research_note[{idx}]",
                        "status": "excluded",
                        "reasons": reasons,
                        "note_id": getattr(note, "note_id", None),
                    }
                )
                continue
            note_tickers = {
                str(ticker).upper()
                for ticker in getattr(note, "tickers", [])
            }
            requested_tickers = {
                str(ticker).upper() for ticker in context.tickers
            }
            direct_tickers = sorted(
                note_tickers.intersection(requested_tickers)
            )
            provenance = {
                "contract": CONTEXT_EVIDENCE_CONTRACT,
                "note_id": getattr(note, "note_id", None),
                "agent_name": getattr(note, "agent_name", None),
                "created_at": created_at.isoformat(),
                "citation_count": len(citation_payloads),
                "citations": citation_payloads,
                "as_of": as_of,
            }
            items.append(
                AnalystEvidenceItem(
                    source_type="research_note",
                    source=str(
                        getattr(note, "note_id", None)
                        or f"context.research_notes[{idx}]"
                    ),
                    published_at=created_at.isoformat(),
                    as_of=as_of,
                    domain_id=self.domain_id,
                    tickers=direct_tickers,
                    sectors=list(
                        getattr(note, "sectors", []) or [self.domain_id]
                    ),
                    evidence_type=self._first_required_or("sector_demand"),
                    summary=_shorten(text),
                    stance_hint=_stance_hint(text.lower()),
                    strength=float(getattr(note, "confidence", 0.5) or 0.5),
                    freshness_score=_freshness_score(
                        created_at.isoformat(),
                        as_of_dt,
                    ),
                    directness=(
                        "ticker" if direct_tickers else "sector"
                    ),
                    reliability_score=0.6,
                    limitations=[
                        (
                            "Research note is derived review context; cited "
                            "source evidence remains authoritative."
                        )
                    ],
                    provenance=provenance,
                    point_in_time={
                        "contract": CONTEXT_EVIDENCE_CONTRACT,
                        "status": "point_in_time_compatible",
                        "created_at": created_at.isoformat(),
                        "latest_citation_at": max(
                            value
                            for value in citation_times
                            if value is not None
                        ).isoformat(),
                        "as_of": as_of,
                    },
                )
            )
        return items, exclusions

    def _first_required_or(self, fallback: str) -> str:
        return self.profile.required_evidence_types[0] if self.profile.required_evidence_types else fallback


def _text_from_news(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        parts: list[str] = []
        seen: set[str] = set()
        for key in (
            "title",
            "headline",
            "summary",
            "description",
            "content",
            "text",
        ):
            value = " ".join(str(raw.get(key) or "").split())
            normalized = value.casefold()
            if not value or normalized in seen:
                continue
            seen.add(normalized)
            parts.append(value)
        return " ".join(parts)
    return str(raw)


def _tickers_from_text(lower_text: str, context_tickers: list[str]) -> list[str]:
    found = []
    for ticker in context_tickers:
        normalized = str(ticker).upper().strip()
        if normalized and normalized.lower() in lower_text:
            found.append(normalized)
    return sorted(set(found))


def _stance_hint(lower_text: str) -> str:
    positive = sum(term in lower_text for term in _POSITIVE_TERMS)
    negative = sum(term in lower_text for term in _NEGATIVE_TERMS)
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "unknown"


def _shorten(text: str, limit: int = 260) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _freshness_score(
    published_at: str | None,
    as_of: Any,
) -> float:
    published = parse_timezone_aware(published_at)
    reference = (
        as_of
        if hasattr(as_of, "tzinfo")
        else parse_timezone_aware(as_of)
    )
    if published is None or reference is None:
        return 0.0
    age_days = max(
        0.0,
        (reference - published).total_seconds() / 86400.0,
    )
    return max(0.0, min(1.0, 1.0 - (age_days / 90.0)))


def _document_reliability(source_type: str) -> float:
    return {
        "filing": 0.9,
        "transcript": 0.75,
        "report": 0.7,
        "book": 0.65,
        "article": 0.55,
        "news": 0.45,
    }.get(str(source_type).lower(), 0.4)
