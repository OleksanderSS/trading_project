from __future__ import annotations

__all__ = [
    'DEFAULT_ISSUER_REGISTRY',
    'DEFAULT_TICKERS',
    'SAVED_TICKER_SPECIFIC_EVIDENCE_CONTRACT',
    'SavedTickerSpecificEvidenceProducer',
    'load_verified_ticker_specific_evidence_fragment',
    'render_saved_ticker_evidence_markdown',
]

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from dean_os.analysts._producers.news import (
    load_verified_semiconductor_news_context_fragment,
)
from dean_os.analysts.profiles import get_domain_profile
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

SAVED_TICKER_SPECIFIC_EVIDENCE_CONTRACT = (
    "dean_saved_ticker_specific_evidence_v1"
)
DEFAULT_ISSUER_REGISTRY = (
    Path(__file__).resolve().parent
    / "config"
    / "semiconductor_issuer_identity_registry.yaml"
)
_DOMAIN_ID = "semiconductor_ai_infrastructure"
DEFAULT_TICKERS = tuple(get_domain_profile(_DOMAIN_ID).ticker_universe_hint)

_POSITIVE_TERMS = (
    "above expectations",
    "beats estimates",
    "beat estimates",
    "strong demand",
    "strong sales",
    "sales forecast",
    "profit rises",
    "record high",
    "record run",
    "soar",
    "jump",
    "growth",
    "optimism",
)
_NEGATIVE_TERMS = (
    "supply constraints",
    "uncertainty",
    "restriction",
    "export control",
    "shortage",
    "margin pressure",
    "weak demand",
    "sales decline",
    "profit falls",
    "cut forecast",
    "misses estimates",
    "risk",
)


class SavedTickerSpecificEvidenceProducer:
    """Build source-bound company mechanism evidence from verified news.

    Exact reviewed issuer aliases can establish company identity. Directional
    eligibility still requires two independent strong sources with a
    consistent stance in the same mechanism lane. This artifact never creates
    a ticker forecast by itself.
    """

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/"
            "saved_ticker_specific_evidence_producer_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        news_artifact_path: str | Path,
        as_of: str,
        tickers: list[str] | None = None,
        registry_path: str | Path = DEFAULT_ISSUER_REGISTRY,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError("as_of must be timezone-aware")
        requested = [
            str(ticker).upper().strip()
            for ticker in (tickers or list(DEFAULT_TICKERS))
            if str(ticker).strip()
        ]
        if not requested:
            raise ValueError("at least one ticker is required")

        news_path = Path(news_artifact_path)
        verified_news = (
            load_verified_semiconductor_news_context_fragment(
                news_path,
                expected_as_of=as_of_dt.isoformat(),
            )
        )
        news_payload = _load_json(news_path)
        registry_source = Path(registry_path)
        registry = _load_registry(registry_source)
        issuer_map = _issuer_map(registry, requested)
        base_records = _company_records(
            news_payload.get("candidates") or [],
            issuer_map=issuer_map,
            requested=requested,
            strong_tiers=set(
                registry.get("eligibility", {}).get(
                    "strong_source_tiers", []
                )
            ),
        )
        lane_review = _lane_review(
            base_records,
            requested=requested,
            minimum_sources=int(
                registry.get("eligibility", {}).get(
                    "minimum_independent_strong_sources", 2
                )
            ),
        )
        lane_by_key = {
            (item["ticker"], item["evidence_type"]): item
            for item in lane_review
        }
        records = []
        for item in base_records:
            lane = lane_by_key[
                (item["ticker"], item["evidence_type"])
            ]
            eligible = bool(
                item["strong_source"]
                and lane["status"] == "corroborated"
                and item["stance_hint"]
                == lane["dominant_stance"]
            )
            records.append(
                {
                    **item,
                    "ticker_thesis_eligible": eligible,
                    "allowed_use": (
                        "source_bound_company_mechanism_context"
                    ),
                    "can_create_ticker_forecast": False,
                }
            )
        accepted_fingerprint = _canonical_sha256(records)
        eligible_lanes = [
            item
            for item in lane_review
            if item["status"] == "corroborated"
        ]
        eligible_tickers = sorted(
            {item["ticker"] for item in eligible_lanes}
        )
        missing_tickers = sorted(
            set(requested).difference(eligible_tickers)
        )
        if len(eligible_tickers) == len(requested):
            status = "ticker_specific_evidence_ready"
        elif records:
            status = "ticker_specific_evidence_ready_with_gaps"
        else:
            status = "ticker_specific_evidence_needs_sources"
        run_id = (
            "saved_ticker_specific_evidence_"
            + utc_now_iso().replace(":", "").replace("+", "Z")
        )
        payload = json_ready(
            {
                "run_id": run_id,
                "created_at": utc_now_iso(),
                "mode": "saved_ticker_specific_evidence_producer",
                "producer_contract": (
                    SAVED_TICKER_SPECIFIC_EVIDENCE_CONTRACT
                ),
                "status": status,
                "inputs": {
                    "news_artifact_path": str(news_path),
                    "news_artifact_sha256": _sha256_file(news_path),
                    "news_artifact_run_id": news_payload.get("run_id"),
                    "registry_path": str(registry_source),
                    "registry_sha256": _sha256_file(registry_source),
                    "as_of": as_of_dt.isoformat(),
                    "tickers": requested,
                },
                "source_verification": {
                    "verified_news_metadata": verified_news.get(
                        "metadata", {}
                    ),
                    "issuer_registry_id": registry.get("registry_id"),
                    "issuer_registry_review_status": registry.get(
                        "review_status"
                    ),
                },
                "summary": {
                    "requested_ticker_count": len(requested),
                    "requested_tickers": requested,
                    "company_candidate_count": len(records),
                    "strong_company_candidate_count": sum(
                        1 for item in records if item["strong_source"]
                    ),
                    "corroborated_lane_count": len(eligible_lanes),
                    "ticker_thesis_eligible_record_count": sum(
                        1
                        for item in records
                        if item["ticker_thesis_eligible"]
                    ),
                    "eligible_tickers": eligible_tickers,
                    "missing_corroborated_tickers": missing_tickers,
                    "accepted_fingerprint": accepted_fingerprint,
                    "can_enter_ticker_evidence_review": bool(records),
                    "can_create_ticker_thesis": False,
                    "can_create_ticker_forecast": False,
                    "can_feed_prediction_directly": False,
                    "can_train": False,
                    "can_tune": False,
                    "can_trade": False,
                },
                "ticker_summary": _ticker_summary(
                    records,
                    lane_review,
                    requested,
                ),
                "lane_review": lane_review,
                "records": records,
                "evidence_acquisition_requests": (
                    _evidence_acquisition_requests(
                        lane_review,
                        requested=requested,
                        as_of=as_of_dt.isoformat(),
                    )
                ),
                "ticker_context_fragment": {
                    "as_of": as_of_dt.isoformat(),
                    "domain_id": registry.get("domain_id"),
                    "records": records,
                    "metadata": {
                        "producer_contract": (
                            SAVED_TICKER_SPECIFIC_EVIDENCE_CONTRACT
                        ),
                        "accepted_fingerprint": accepted_fingerprint,
                        "issuer_registry_id": registry.get(
                            "registry_id"
                        ),
                    },
                },
                "integration_boundary": {
                    "issuer_identity_requires_reviewed_exact_alias": True,
                    "plain_substring_match_allowed": False,
                    "two_independent_strong_sources_required": True,
                    "consistent_lane_stance_required": True,
                    "raw_fundamentals_are_directional_evidence": False,
                    "sector_context_can_close_ticker_lane": False,
                    "ticker_evidence_alone_can_create_forecast": False,
                    "pipeline_feature_promotion_allowed": False,
                    "automatic_training_allowed": False,
                    "automatic_tuning_allowed": False,
                    "automatic_trading_allowed": False,
                },
                "safety": _safety(),
            }
        )
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_saved_ticker_evidence_markdown(
                    payload
                ),
                run_id=run_id,
            )
        return payload


def load_verified_ticker_specific_evidence_fragment(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = _load_json(path)
    if payload.get("producer_contract") != (
        SAVED_TICKER_SPECIFIC_EVIDENCE_CONTRACT
    ):
        raise ValueError("unsupported ticker-specific evidence contract")
    if payload.get("status") not in {
        "ticker_specific_evidence_ready",
        "ticker_specific_evidence_ready_with_gaps",
    }:
        raise ValueError("ticker-specific evidence artifact not ready")
    summary = payload.get("summary") or {}
    safety = payload.get("safety") or {}
    if (
        summary.get("can_enter_ticker_evidence_review") is not True
        or summary.get("can_create_ticker_forecast") is not False
        or summary.get("can_train") is not False
        or summary.get("can_trade") is not False
        or safety.get("review_only") is not True
        or safety.get("live_execution_performed") is not False
    ):
        raise ValueError("ticker-specific evidence safety invalid")
    inputs = payload.get("inputs") or {}
    news_path = Path(str(inputs.get("news_artifact_path") or ""))
    registry_path = Path(str(inputs.get("registry_path") or ""))
    if (
        not news_path.is_file()
        or _sha256_file(news_path)
        != inputs.get("news_artifact_sha256")
    ):
        raise ValueError("ticker-specific source news hash mismatch")
    if (
        not registry_path.is_file()
        or _sha256_file(registry_path)
        != inputs.get("registry_sha256")
    ):
        raise ValueError("ticker-specific registry hash mismatch")
    load_verified_semiconductor_news_context_fragment(
        news_path,
        expected_as_of=inputs.get("as_of"),
    )
    fragment = payload.get("ticker_context_fragment") or {}
    fragment_as_of = parse_timezone_aware(fragment.get("as_of"))
    if fragment_as_of is None:
        raise ValueError("ticker-specific fragment as_of invalid")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != fragment_as_of:
            raise ValueError(
                "ticker-specific expected as_of mismatch"
            )
    records = fragment.get("records")
    if not isinstance(records, list):
        raise ValueError("ticker-specific records missing")
    if _canonical_sha256(records) != summary.get(
        "accepted_fingerprint"
    ):
        raise ValueError(
            "ticker-specific evidence fingerprint mismatch"
        )
    return {
        "as_of": fragment_as_of.isoformat(),
        "domain_id": fragment.get("domain_id"),
        "records": records,
        "ticker_summary": payload.get("ticker_summary", []),
        "lane_review": payload.get("lane_review", []),
        "metadata": {
            **dict(fragment.get("metadata") or {}),
            "artifact_path": str(path),
            "artifact_sha256": _sha256_file(path),
            "verified": True,
        },
    }


def render_saved_ticker_evidence_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# DEAN-OS Saved Ticker-Specific Evidence",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Company candidates: {summary.get('company_candidate_count')}",
        f"- Strong candidates: {summary.get('strong_company_candidate_count')}",
        f"- Corroborated lanes: {summary.get('corroborated_lane_count')}",
        f"- Eligible tickers: `{', '.join(summary.get('eligible_tickers', [])) or 'none'}`",
        f"- Missing corroboration: `{', '.join(summary.get('missing_corroborated_tickers', [])) or 'none'}`",
        f"- Can create ticker thesis: {summary.get('can_create_ticker_thesis')}",
        f"- Can create ticker forecast: {summary.get('can_create_ticker_forecast')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Ticker Summary",
        "",
    ]
    for item in payload.get("ticker_summary", []):
        lines.append(
            f"- `{item.get('ticker')}` status=`{item.get('status')}` "
            f"candidates={item.get('candidate_count')} "
            f"strong={item.get('strong_candidate_count')} "
            f"corroborated={item.get('corroborated_lane_count')}"
        )
    lines.extend(["", "## Lane Review", ""])
    for item in payload.get("lane_review", []):
        lines.append(
            f"- `{item.get('ticker')}/{item.get('evidence_type')}` "
            f"status=`{item.get('status')}` "
            f"sources={item.get('independent_strong_source_count')} "
            f"stance=`{item.get('dominant_stance')}`"
        )
    lines.extend(["", "## Boundary", ""])
    lines.extend(
        [
            "- Exact reviewed issuer aliases establish company identity; plain substring matching is forbidden.",
            "- Two independent strong sources with consistent stance are required for ticker-thesis eligibility in one mechanism lane.",
            "- This artifact supplies evidence context only. It cannot create a forecast, train, tune, or trade.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _company_records(
    candidates: list[dict[str, Any]],
    *,
    issuer_map: dict[str, dict[str, Any]],
    requested: list[str],
    strong_tiers: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        text = str(
            candidate.get("title")
            or candidate.get("summary")
            or ""
        )
        for ticker in requested:
            issuer = issuer_map[ticker]
            matched_alias = _match_alias(
                text,
                issuer.get("aliases", []),
            )
            if not matched_alias:
                continue
            stance = _stance_hint(text)
            record = {
                "ticker": ticker,
                "legal_name": issuer.get("legal_name"),
                "matched_alias": matched_alias,
                "identity_match_type": "reviewed_exact_alias",
                "title": candidate.get("title"),
                "summary": candidate.get("summary"),
                "source": candidate.get("source"),
                "source_identity": candidate.get(
                    "source_identity"
                ),
                "source_tier": candidate.get("source_tier"),
                "strong_source": candidate.get("source_tier")
                in strong_tiers,
                "published_at": candidate.get("published_at"),
                "source_locator": candidate.get(
                    "source_locator"
                ),
                "evidence_type": candidate.get(
                    "evidence_type"
                ),
                "matched_terms": candidate.get(
                    "matched_terms", []
                ),
                "stance_hint": stance,
                "candidate_sha256": candidate.get(
                    "candidate_sha256"
                ),
            }
            record["record_sha256"] = _canonical_sha256(record)
            records.append(record)
    records.sort(
        key=lambda item: (
            item["ticker"],
            item["evidence_type"],
            item["published_at"],
            item["source_identity"],
        )
    )
    return records


def _lane_review(
    records: list[dict[str, Any]],
    *,
    requested: list[str],
    minimum_sources: int,
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for item in records:
        grouped[(item["ticker"], item["evidence_type"])].append(
            item
        )
    review = []
    for (ticker, evidence_type), items in sorted(grouped.items()):
        strong = [item for item in items if item["strong_source"]]
        sources = sorted(
            {
                str(item["source_identity"])
                for item in strong
                if item.get("source_identity")
            }
        )
        directional = Counter(
            item["stance_hint"]
            for item in strong
            if item["stance_hint"] in {"positive", "negative"}
        )
        if directional["positive"] and directional["negative"]:
            dominant = "conflicted"
        elif directional["positive"]:
            dominant = "positive"
        elif directional["negative"]:
            dominant = "negative"
        else:
            dominant = "unknown"
        corroborated = (
            len(sources) >= minimum_sources
            and dominant in {"positive", "negative"}
        )
        review.append(
            {
                "ticker": ticker,
                "evidence_type": evidence_type,
                "status": (
                    "corroborated"
                    if corroborated
                    else (
                        "conflicting_strong_sources"
                        if dominant == "conflicted"
                        else "needs_more_independent_strong_sources"
                    )
                ),
                "candidate_count": len(items),
                "strong_candidate_count": len(strong),
                "independent_strong_source_count": len(sources),
                "independent_strong_sources": sources,
                "minimum_independent_sources": minimum_sources,
                "strong_stance_counts": dict(
                    sorted(directional.items())
                ),
                "dominant_stance": dominant,
            }
        )
    return review


def _ticker_summary(
    records: list[dict[str, Any]],
    lane_review: list[dict[str, Any]],
    requested: list[str],
) -> list[dict[str, Any]]:
    by_ticker = defaultdict(list)
    for item in records:
        by_ticker[item["ticker"]].append(item)
    lane_by_ticker = defaultdict(list)
    for item in lane_review:
        lane_by_ticker[item["ticker"]].append(item)
    result = []
    for ticker in requested:
        items = by_ticker[ticker]
        lanes = lane_by_ticker[ticker]
        corroborated = [
            item
            for item in lanes
            if item["status"] == "corroborated"
        ]
        result.append(
            {
                "ticker": ticker,
                "status": (
                    "company_mechanism_corroborated"
                    if corroborated
                    else (
                        "company_evidence_needs_corroboration"
                        if items
                        else "no_company_evidence"
                    )
                ),
                "candidate_count": len(items),
                "strong_candidate_count": sum(
                    1 for item in items if item["strong_source"]
                ),
                "corroborated_lane_count": len(corroborated),
                "corroborated_lanes": [
                    {
                        "evidence_type": item["evidence_type"],
                        "dominant_stance": item[
                            "dominant_stance"
                        ],
                        "sources": item[
                            "independent_strong_sources"
                        ],
                    }
                    for item in corroborated
                ],
                "ticker_thesis_eligible_record_count": sum(
                    1
                    for item in items
                    if item.get("ticker_thesis_eligible")
                ),
                "can_create_ticker_forecast": False,
            }
        )
    return result


def _evidence_acquisition_requests(
    lane_review: list[dict[str, Any]],
    *,
    requested: list[str],
    as_of: str,
) -> list[dict[str, Any]]:
    by_ticker = defaultdict(list)
    for item in lane_review:
        by_ticker[item["ticker"]].append(item)
    requests = []
    for ticker in requested:
        lanes = by_ticker[ticker]
        if any(item["status"] == "corroborated" for item in lanes):
            continue
        strongest = max(
            lanes,
            key=lambda item: item[
                "independent_strong_source_count"
            ],
            default=None,
        )
        requests.append(
            {
                "request_id": (
                    f"ticker:{ticker}:independent_company_mechanism"
                ),
                "ticker": ticker,
                "status": "open",
                "priority": "high",
                "as_of": as_of,
                "preferred_evidence_type": (
                    strongest.get("evidence_type")
                    if strongest
                    else None
                ),
                "current_independent_strong_sources": (
                    strongest.get(
                        "independent_strong_sources", []
                    )
                    if strongest
                    else []
                ),
                "acceptance_criteria": [
                    "Exact reviewed issuer alias in the source title or source metadata.",
                    "Stable source locator and publication timestamp no later than as_of.",
                    "Independent strong source corroborating the same company mechanism and stance.",
                    "Explicit company mechanism; a generic sector statement is insufficient.",
                ],
                "automatic_collection_authorized": False,
                "automatic_ticker_promotion_authorized": False,
            }
        )
    return requests


def _issuer_map(
    registry: dict[str, Any],
    requested: list[str],
) -> dict[str, dict[str, Any]]:
    issuers = registry.get("issuers")
    if not isinstance(issuers, dict):
        raise ValueError("issuer registry issuers missing")
    missing = [ticker for ticker in requested if ticker not in issuers]
    if missing:
        raise ValueError(
            "issuer registry missing tickers: " + ", ".join(missing)
        )
    result = {}
    for ticker in requested:
        issuer = issuers[ticker]
        aliases = issuer.get("aliases") if isinstance(issuer, dict) else None
        if not aliases:
            raise ValueError(f"issuer aliases missing: {ticker}")
        result[ticker] = {
            "legal_name": issuer.get("legal_name"),
            "aliases": [
                str(alias).strip()
                for alias in aliases
                if str(alias).strip()
            ],
        }
    return result


def _match_alias(text: str, aliases: list[str]) -> str | None:
    for alias in sorted(aliases, key=len, reverse=True):
        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(alias)
            + r"(?![A-Za-z0-9])"
        )
        if re.search(pattern, text, flags=re.IGNORECASE):
            return alias
    return None


def _stance_hint(text: str) -> str:
    lower = text.casefold()
    positive = sum(term in lower for term in _POSITIVE_TERMS)
    negative = sum(term in lower for term in _NEGATIVE_TERMS)
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "unknown"


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"issuer registry missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("issuer registry must be an object")
    if payload.get("registry_id") != (
        "dean_semiconductor_issuer_identity_registry_v1"
    ):
        raise ValueError("unsupported issuer registry")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            json_ready(value),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _safety() -> dict[str, bool]:
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
