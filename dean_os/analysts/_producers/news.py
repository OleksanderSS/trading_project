from __future__ import annotations

__all__ = [
    'DEFAULT_REGISTRY',
    'DOMAIN_TERMS',
    'LANE_TERMS',
    'SAVED_SEMICONDUCTOR_NEWS_EVIDENCE_CONTRACT',
    'SavedSemiconductorNewsEvidenceProducer',
    'load_verified_semiconductor_news_context_fragment',
    'render_saved_news_markdown',
]

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import yaml

from dean_os.analysts.profiles import get_domain_profile
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import (
    audit_news_records,
    parse_timezone_aware,
)
from dean_os.schemas import utc_now_iso

SAVED_SEMICONDUCTOR_NEWS_EVIDENCE_CONTRACT = (
    "dean_saved_semiconductor_news_evidence_producer_v1"
)
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "semiconductor_news_source_registry.yaml"
)
_DOMAIN_ID = "semiconductor_ai_infrastructure"
_DEFAULT_TICKERS = list(get_domain_profile(_DOMAIN_ID).ticker_universe_hint)
LANE_TERMS = {
    "sector_demand": (
        "ai demand",
        "data center demand",
        "accelerator demand",
        "gpu demand",
        "chip demand",
        "semiconductor demand",
        "hbm demand",
        "memory demand",
        "ai orders",
        "sales forecast",
    ),
    "capex_cycle": (
        "capex",
        "capital expenditure",
        "capital spending",
        "data center spending",
        "data center investment",
        "cloud capex",
        "hyperscaler spending",
        "ai infrastructure spending",
        "data center bet",
    ),
    "supply_chain": (
        "foundry",
        "capacity constraint",
        "capacity expansion",
        "advanced packaging",
        "supply chain",
        "wafer",
        "packaging capacity",
        "chip shortage",
        "hbm supply",
        "memory supply",
        "chip fab",
        "semiconductor fab",
        "memory crunch",
        "supply constraints",
        "soaring memory prices",
    ),
    "policy_or_geopolitical": (
        "export control",
        "export controls",
        "export restriction",
        "chip exports",
        "chip ban",
        "chip sanction",
        "chip tariff",
        "semiconductor tariff",
        "taiwan strait",
        "taiwan tension",
        "taiwan risk",
        "china chip restriction",
        "china sales restriction",
        "commerce department chip",
        "trade restriction on chip",
        "geopolitical risk",
    ),
    "market_confirmation": (
        "relative strength",
        "breakout",
        "momentum",
        "drawdown",
        "price action",
        "price target",
        "target raised",
        "target cut",
        "upgrade",
        "upgraded",
        "downgrade",
        "downgraded",
        "outperform",
        "underperform",
        "market perform",
        "earnings preview",
        "beat eps",
        "revenue",
        "stock climbs",
        "shares",
    ),
}
DOMAIN_TERMS = (
    "semiconductor",
    "chipmaker",
    "chip maker",
    "chipmaking",
    " chip",
    "chips",
    "gpu",
    "nvidia",
    "advanced micro devices",
    " amd ",
    "intel",
    "tsmc",
    "taiwan semiconductor",
    "foundry",
    "wafer",
    "hbm",
    "high-bandwidth memory",
    "memory",
    "data center",
    "ai accelerator",
    "advanced packaging",
)


class SavedSemiconductorNewsEvidenceProducer:
    """Extract strict review-only lane candidates from saved news."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/"
            "saved_semiconductor_news_evidence_producer"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        source_path: str | Path,
        as_of: str,
        registry_path: str | Path = DEFAULT_REGISTRY,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError(
                "news producer as_of must be a timezone-aware ISO-8601 "
                "timestamp"
            )
        source = Path(source_path)
        registry_source = Path(registry_path)
        run_id = _run_id()
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "saved_semiconductor_news_evidence_producer",
            "producer_contract": (
                SAVED_SEMICONDUCTOR_NEWS_EVIDENCE_CONTRACT
            ),
            "inputs": {
                "source_path": str(source),
                "registry_path": str(registry_source),
                "as_of": as_of_dt.isoformat(),
                "domain_id": "semiconductor_ai_infrastructure",
            },
            "source_provenance": _file_reference(source),
            "registry": _file_reference(registry_source),
        }
        if not source.exists():
            payload.update(_blocked("news_source_missing"))
            return self._finish(payload, save=save)
        if not registry_source.exists():
            payload.update(_blocked("news_source_registry_missing"))
            return self._finish(payload, save=save)
        try:
            registry = yaml.safe_load(
                registry_source.read_text(encoding="utf-8")
            )
            frame = pd.read_parquet(source)
        except (OSError, ValueError, ImportError, yaml.YAMLError) as exc:
            payload["load_error"] = str(exc)
            payload.update(_blocked("news_source_or_registry_unreadable"))
            return self._finish(payload, save=save)

        normalized = _normalize_records(
            frame,
            as_of=as_of_dt,
            registry=registry,
        )
        candidates = normalized["candidates"]
        lane_review = _lane_review(candidates, registry=registry)
        lane_ready = {
            lane["evidence_type"]: lane["status"] == "eligible"
            for lane in lane_review
        }
        records = []
        for candidate in candidates:
            eligible = (
                lane_ready.get(candidate["evidence_type"], False)
                and candidate["source_tier"]
                in _eligible_tiers(registry)
            )
            records.append(
                {
                    "title": candidate["title"],
                    "summary": candidate["summary"],
                    "source": candidate["source"],
                    "published_at": candidate["published_at"],
                    "url": candidate["source_locator"],
                    "_dean_semantic_evidence": {
                        "producer_contract": (
                            SAVED_SEMICONDUCTOR_NEWS_EVIDENCE_CONTRACT
                        ),
                        "evidence_type": candidate["evidence_type"],
                        "required_lane_eligible": eligible,
                        "source_tier": candidate["source_tier"],
                        "source_identity": candidate["source_identity"],
                        "matched_terms": candidate["matched_terms"],
                        "domain_terms": candidate["domain_terms"],
                        "candidate_sha256": candidate[
                            "candidate_sha256"
                        ],
                        "stance_hint": "unknown",
                    },
                }
            )
        news_audit = audit_news_records(
            records,
            as_of=as_of_dt.isoformat(),
            requested_tickers=_DEFAULT_TICKERS,
        )
        accepted_records = [
            {
                key: value
                for key, value in record.items()
                if key != "_dean_context_provenance"
            }
            for record in news_audit["accepted"]
        ]
        fingerprint = _canonical_sha256(accepted_records)
        ready_lanes = [
            item["evidence_type"]
            for item in lane_review
            if item["status"] == "eligible"
        ]
        missing_lanes = [
            item["evidence_type"]
            for item in lane_review
            if item["status"] != "eligible"
        ]
        status = (
            "semiconductor_news_evidence_ready"
            if ready_lanes and not missing_lanes
            else "semiconductor_news_evidence_ready_with_gaps"
            if accepted_records
            else "blocked_no_semiconductor_news_evidence"
        )
        payload.update(
            {
                "status": status,
                "summary": {
                    "source_row_count": len(frame),
                    "usable_source_row_count": normalized[
                        "usable_source_row_count"
                    ],
                    "orphan_or_invalid_row_count": normalized[
                        "orphan_or_invalid_row_count"
                    ],
                    "future_row_count": normalized["future_row_count"],
                    "domain_candidate_count": normalized[
                        "domain_candidate_count"
                    ],
                    "classified_candidate_count": len(candidates),
                    "duplicate_candidate_count": normalized[
                        "duplicate_candidate_count"
                    ],
                    "accepted_news_record_count": len(
                        accepted_records
                    ),
                    "accepted_fingerprint": fingerprint,
                    "ready_required_lanes": ready_lanes,
                    "missing_required_lanes": missing_lanes,
                    "ready_required_lane_count": len(ready_lanes),
                    "can_enter_market_context_review": bool(
                        accepted_records
                    ),
                    "can_close_lane_from_keyword_only": False,
                    "can_influence_ticker_prediction": False,
                    "can_train": False,
                    "can_trade": False,
                },
                "lane_review": lane_review,
                "evidence_acquisition_requests": (
                    _evidence_acquisition_requests(
                        lane_review,
                        as_of=as_of_dt.isoformat(),
                    )
                ),
                "source_tier_counts": dict(
                    sorted(
                        Counter(
                            item["source_tier"] for item in candidates
                        ).items()
                    )
                ),
                "candidates": candidates,
                "exclusions": normalized["exclusions"],
                "news_point_in_time_audit": {
                    key: value
                    for key, value in news_audit.items()
                    if key != "accepted"
                },
                "market_context_fragment": {
                    "as_of": as_of_dt.isoformat(),
                    "news": accepted_records,
                    "metadata": {
                        "saved_semiconductor_news_run_id": run_id,
                        "saved_semiconductor_news_source_sha256": payload[
                            "source_provenance"
                        ]["sha256"],
                        "saved_semiconductor_news_registry_sha256": payload[
                            "registry"
                        ]["sha256"],
                        "saved_semiconductor_news_fingerprint": (
                            fingerprint
                        ),
                        "ready_required_lanes": ready_lanes,
                        "missing_required_lanes": missing_lanes,
                    },
                },
                "integration_boundary": {
                    "review_only": True,
                    "headline_and_description_classification_only": True,
                    "full_article_body_used_for_lane_routing": False,
                    "sentiment_column_used": False,
                    "keyword_hit_is_lane_completion": False,
                    "independent_strong_sources_required": True,
                    "plain_text_ticker_promotion_allowed": False,
                    "pipeline_feature_promotion_allowed": False,
                    "training_allowed": False,
                    "automatic_trading_allowed": False,
                },
                "safety": _safety(),
            }
        )
        return self._finish(payload, save=save)

    def _finish(
        self,
        payload: dict[str, Any],
        *,
        save: bool,
    ) -> dict[str, Any]:
        payload.setdefault("safety", _safety())
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_saved_news_markdown(payload),
                run_id=payload["run_id"],
            )
        return payload


def load_verified_semiconductor_news_context_fragment(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("producer_contract")
        != SAVED_SEMICONDUCTOR_NEWS_EVIDENCE_CONTRACT
    ):
        raise ValueError("unsupported semiconductor news contract")
    if payload.get("status") not in {
        "semiconductor_news_evidence_ready",
        "semiconductor_news_evidence_ready_with_gaps",
    }:
        raise ValueError("semiconductor news artifact is not ready")
    summary = payload.get("summary", {})
    safety = payload.get("safety", {})
    if (
        summary.get("can_enter_market_context_review") is not True
        or summary.get("can_trade") is not False
        or summary.get("can_train") is not False
        or summary.get("can_close_lane_from_keyword_only") is not False
        or safety.get("review_only") is not True
        or safety.get("pipeline_run_performed") is not False
        or safety.get("live_execution_performed") is not False
    ):
        raise ValueError("semiconductor news safety boundary invalid")
    fragment = payload.get("market_context_fragment")
    if not isinstance(fragment, dict):
        raise ValueError("semiconductor news fragment missing")
    fragment_as_of = parse_timezone_aware(fragment.get("as_of"))
    if fragment_as_of is None:
        raise ValueError("semiconductor news fragment as_of invalid")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != fragment_as_of:
            raise ValueError("semiconductor news expected as_of mismatch")
    for key in ("source_provenance", "registry"):
        reference = payload.get(key, {})
        source = Path(str(reference.get("path") or ""))
        if (
            not source.exists()
            or _sha256_file(source) != reference.get("sha256")
        ):
            raise ValueError(
                f"semiconductor news {key} hash mismatch"
            )
    records = fragment.get("news")
    if not isinstance(records, list):
        raise ValueError("semiconductor news records missing")
    audit = audit_news_records(
        records,
        as_of=fragment_as_of.isoformat(),
        requested_tickers=_DEFAULT_TICKERS,
    )
    stripped = [
        {
            key: value
            for key, value in record.items()
            if key != "_dean_context_provenance"
        }
        for record in audit["accepted"]
    ]
    if (
        audit["excluded_count"] != 0
        or audit["accepted_count"]
        != summary.get("accepted_news_record_count")
        or _canonical_sha256(stripped)
        != summary.get("accepted_fingerprint")
    ):
        raise ValueError("semiconductor news fragment fingerprint mismatch")
    return {
        "as_of": fragment_as_of.isoformat(),
        "news": stripped,
        "metadata": {
            **dict(fragment.get("metadata", {})),
            "saved_semiconductor_news_artifact_path": str(path),
            "saved_semiconductor_news_artifact_sha256": (
                _sha256_file(path)
            ),
            "saved_semiconductor_news_verified": True,
        },
    }


def _normalize_records(
    frame: pd.DataFrame,
    *,
    as_of: Any,
    registry: dict[str, Any],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen: set[str] = set()
    usable = 0
    future = 0
    domain_candidates = 0
    duplicate_count = 0
    max_age_days = int(
        registry.get("lane_eligibility", {}).get(
            "max_age_days", 120
        )
    )
    required_columns = (
        "title",
        "description",
        "summary",
        "published_date",
        "publishedAt",
        "timestamp",
        "link",
        "url",
        "source",
    )
    available = [
        column for column in required_columns if column in frame.columns
    ]
    positions = {
        column: index for index, column in enumerate(available)
    }
    for row_index, values in enumerate(
        frame[available].itertuples(index=False, name=None)
    ):
        def value(column: str) -> Any:
            position = positions.get(column)
            return values[position] if position is not None else None

        title = _clean(value("title"))
        description = _clean(value("description"))
        if not description:
            description = _clean(value("summary"))
        summary = _combine_news_text(title, description)
        published_raw = (
            value("published_date")
            if not _missing(value("published_date"))
            else value("publishedAt")
        )
        if _missing(published_raw):
            published_raw = value("timestamp")
        published_at = parse_timezone_aware(published_raw)
        locator = (
            value("link")
            if not _missing(value("link"))
            else value("url")
        )
        locator = _canonical_url(locator)
        if not locator:
            locator = _first_url(summary)
        source = _clean(value("source"))
        reasons: list[str] = []
        if not summary:
            reasons.append("news_text_missing")
        if published_at is None:
            reasons.append("publication_timestamp_missing_or_invalid")
        elif published_at > as_of:
            reasons.append("publication_after_as_of")
            future += 1
        elif (as_of - published_at).total_seconds() / 86400 > max_age_days:
            reasons.append("news_record_stale_for_lane_review")
        if not locator:
            reasons.append("stable_source_locator_missing")
        if not source:
            reasons.append("news_source_identity_missing")
        if reasons:
            exclusions.append(
                {
                    "index": row_index,
                    "status": "excluded",
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        usable += 1
        lower = _matching_text(summary)
        domain_hits = sorted(
            {term.strip() for term in DOMAIN_TERMS if _term_in_text(term, lower)}
        )
        if not domain_hits:
            continue
        domain_candidates += 1
        source_identity, source_tier = _source_tier(source, registry)
        for evidence_type, terms in LANE_TERMS.items():
            matched = sorted(
                {term for term in terms if _term_in_text(term, lower)}
            )
            if not matched:
                continue
            identity = _canonical_sha256(
                {
                    "title": _normalized_title(title),
                    "published_at": published_at.isoformat(),
                    "evidence_type": evidence_type,
                }
            )
            if identity in seen:
                duplicate_count += 1
                exclusions.append(
                    {
                        "index": row_index,
                        "status": "excluded",
                        "reasons": ["duplicate_lane_candidate"],
                        "evidence_type": evidence_type,
                    }
                )
                continue
            seen.add(identity)
            candidate = {
                "title": title,
                "summary": summary,
                "source": source,
                "source_identity": source_identity,
                "source_tier": source_tier,
                "published_at": published_at.isoformat(),
                "source_locator": locator,
                "evidence_type": evidence_type,
                "matched_terms": matched,
                "domain_terms": domain_hits,
            }
            candidate["candidate_sha256"] = _canonical_sha256(
                candidate
            )
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            item["evidence_type"],
            item["published_at"],
            item["source_identity"],
            item["candidate_sha256"],
        )
    )
    return {
        "candidates": candidates,
        "exclusions": exclusions,
        "usable_source_row_count": usable,
        "orphan_or_invalid_row_count": len(frame) - usable,
        "future_row_count": future,
        "domain_candidate_count": domain_candidates,
        "duplicate_candidate_count": duplicate_count,
    }


def _lane_review(
    candidates: list[dict[str, Any]],
    *,
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    policy = registry.get("lane_eligibility", {})
    min_sources = int(policy.get("min_independent_sources", 2))
    eligible_tiers = _eligible_tiers(registry)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[item["evidence_type"]].append(item)
    result = []
    for evidence_type in LANE_TERMS:
        items = grouped.get(evidence_type, [])
        strong = [
            item
            for item in items
            if item["source_tier"] in eligible_tiers
        ]
        identities = sorted(
            {item["source_identity"] for item in strong}
        )
        status = (
            "eligible"
            if len(identities) >= min_sources
            else "needs_more_independent_strong_sources"
        )
        result.append(
            {
                "evidence_type": evidence_type,
                "status": status,
                "candidate_count": len(items),
                "strong_candidate_count": len(strong),
                "independent_strong_source_count": len(identities),
                "independent_strong_sources": identities,
                "minimum_independent_sources": min_sources,
                "keyword_hit_alone_is_eligible": False,
            }
        )
    return result


def _evidence_acquisition_requests(
    lane_review: list[dict[str, Any]],
    *,
    as_of: str,
) -> list[dict[str, Any]]:
    templates = {
        "capex_cycle": {
            "preferred_primary_sources": [
                "hyperscaler earnings release or filed report",
                "company capital-expenditure guidance",
            ],
            "strong_context_sources": [
                "Reuters",
                "Bloomberg",
                "CNBC",
                "Financial Times",
                "Wall Street Journal",
            ],
            "search_concepts": [
                "AI infrastructure capital spending",
                "hyperscaler data center capex",
                "semiconductor fab investment",
            ],
        },
        "supply_chain": {
            "preferred_primary_sources": [
                "foundry capacity or packaging disclosure",
                "HBM or memory supplier earnings release",
                "company supply-constraint guidance",
            ],
            "strong_context_sources": [
                "Reuters",
                "Bloomberg",
                "CNBC",
                "Financial Times",
                "Wall Street Journal",
            ],
            "search_concepts": [
                "advanced packaging capacity constraint",
                "HBM supply shortage",
                "semiconductor foundry capacity",
            ],
        },
        "policy_or_geopolitical": {
            "preferred_primary_sources": [
                "US Bureau of Industry and Security rule or notice",
                "US Department of Commerce announcement",
                "official Taiwan or allied-government policy notice",
            ],
            "strong_context_sources": [
                "Reuters",
                "Bloomberg",
                "CNBC",
                "Financial Times",
                "Wall Street Journal",
            ],
            "search_concepts": [
                "semiconductor export controls China",
                "advanced computing chip licensing rule",
                "Taiwan semiconductor geopolitical supply risk",
            ],
        },
    }
    requests = []
    for lane in lane_review:
        if lane.get("status") == "eligible":
            continue
        evidence_type = str(lane.get("evidence_type"))
        template = templates.get(evidence_type, {})
        minimum = int(lane.get("minimum_independent_sources", 2))
        current = int(
            lane.get("independent_strong_source_count", 0)
        )
        requests.append(
            {
                "request_id": (
                    f"semiconductor:{evidence_type}:"
                    "independent_source_gap"
                ),
                "evidence_type": evidence_type,
                "status": "open",
                "priority": "high",
                "as_of": as_of,
                "current_independent_strong_sources": (
                    lane.get("independent_strong_sources", [])
                ),
                "minimum_independent_sources": minimum,
                "additional_independent_sources_needed": max(
                    0, minimum - current
                ),
                **template,
                "acceptance_criteria": [
                    (
                        "Publication or official availability timestamp "
                        "must be no later than the analysis as_of."
                    ),
                    "A stable source locator and source identity are required.",
                    (
                        "At least two independent tier-1/tier-2 sources "
                        "must support the lane."
                    ),
                    (
                        "The source must state a lane-specific mechanism; "
                        "a generic AI, China, or ticker mention is not enough."
                    ),
                ],
                "automatic_collection_authorized": False,
                "automatic_lane_promotion_authorized": False,
            }
        )
    return requests


def _source_tier(
    source: str,
    registry: dict[str, Any],
) -> tuple[str, str]:
    normalized = re.sub(r"\s+", " ", source.strip().lower())
    for tier, names in registry.get("source_tiers", {}).items():
        for name in names or []:
            target = str(name).strip().lower()
            if (
                normalized == target
                or normalized.startswith(f"{target}.")
                or normalized.startswith(f"{target}_")
                or target == "bloomberg"
                and normalized.startswith("bloomberg")
            ):
                return target, str(tier)
    return normalized, str(
        registry.get("default_tier")
        or "tier_4_weak_or_unverified"
    )


def _eligible_tiers(registry: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in registry.get("lane_eligibility", {}).get(
            "eligible_source_tiers", []
        )
    }


def _matching_text(value: str) -> str:
    without_urls = re.sub(r"https?://\S+", " ", value.lower())
    normalized = re.sub(r"[^a-z0-9.+-]+", " ", without_urls)
    return f" {re.sub(r'\s+', ' ', normalized).strip()} "


def _term_in_text(term: str, lower_text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(term).strip().lower())
    if not cleaned:
        return False
    pattern = re.escape(cleaned).replace(r"\ ", r"\s+")
    return re.search(
        rf"(?<![a-z0-9]){pattern}(?![a-z0-9])",
        lower_text,
    ) is not None


def _canonical_url(value: Any) -> str | None:
    if _missing(value):
        return None
    text = str(value).strip()
    try:
        split = urlsplit(text)
    except ValueError:
        return None
    if split.scheme not in {"http", "https"} or not split.netloc:
        return None
    return urlunsplit(
        (
            split.scheme.lower(),
            split.netloc.lower(),
            split.path,
            split.query,
            "",
        )
    )


def _first_url(value: str) -> str | None:
    if not value:
        return None
    for match in re.findall(r"https?://[^\s<>\]\)\"']+", value):
        locator = _canonical_url(match.rstrip(".,;:!?"))
        if locator:
            return locator
    return None


def _normalized_title(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _combine_news_text(title: str, description: str) -> str:
    if not title:
        return description
    if not description:
        return title
    normalized_title = _normalized_title(title)
    normalized_description = _normalized_title(description)
    if normalized_title == normalized_description:
        return title
    if normalized_description.startswith(normalized_title):
        return description
    if normalized_title.startswith(normalized_description):
        return title
    return f"{title} {description}"


def _clean(value: Any) -> str:
    if _missing(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if any(marker in text for marker in ("â€", "Ã", "Â")):
        try:
            repaired = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        else:
            text = repaired
    return text


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _file_reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256_file(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "status": "blocked_semiconductor_news_evidence",
        "summary": {
            "reason_counts": {reason: 1},
            "ready_required_lanes": [],
            "missing_required_lanes": list(LANE_TERMS),
            "can_enter_market_context_review": False,
            "can_close_lane_from_keyword_only": False,
            "can_influence_ticker_prediction": False,
            "can_train": False,
            "can_trade": False,
        },
        "market_context_fragment": {"news": []},
        "safety": _safety(),
    }


def _safety() -> dict[str, Any]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "collector_run_performed": False,
        "pipeline_run_performed": False,
        "training_run_performed": False,
        "tuning_run_performed": False,
        "learning_write_performed": False,
        "production_config_write_performed": False,
        "broker_access_performed": False,
        "live_execution_performed": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_id() -> str:
    return (
        "saved_semiconductor_news_evidence_"
        + utc_now_iso().replace(":", "").replace("+", "Z")
    )


def render_saved_news_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    return "\n".join(
        [
            "# Saved Semiconductor News Evidence",
            "",
            f"- Status: `{payload.get('status')}`",
            f"- As of: `{payload.get('inputs', {}).get('as_of')}`",
            (
                "- Classified candidates: `"
                + str(summary.get("classified_candidate_count", 0))
                + "`"
            ),
            (
                "- Ready lanes: `"
                + (
                    ", ".join(summary.get("ready_required_lanes", []))
                    or "none"
                )
                + "`"
            ),
            (
                "- Missing lanes: `"
                + (
                    ", ".join(
                        summary.get("missing_required_lanes", [])
                    )
                    or "none"
                )
                + "`"
            ),
            "",
            "Keyword matches create candidates only. A required lane needs "
            "independent strong sources and remains review-only.",
            "",
        ]
    )
