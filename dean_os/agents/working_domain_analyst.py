from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.analyst_knowledge.pack_loader import load_knowledge_pack
from dean_os.analyst_knowledge.retriever import KnowledgeRetriever
from dean_os.analyst_knowledge.schemas import KnowledgeRetrievalHit, KnowledgeRetrievalResult
from dean_os.analyst_knowledge.store import LocalKnowledgeStore
from dean_os.analysts import BaseAnalystAgent
from dean_os.analysts.schemas import AnalystEvidenceItem
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import clamp, json_ready


class WorkingDomainAnalystAgent:
    """Review-only analyst loop over local domain knowledge packs.

    Purpose:
        feed knowledge -> retrieve relevant knowledge -> produce analyst report.

    This class deliberately has no authority to trade, call brokers, train/tune
    models, promote model artifacts, write production config, or write autonomous
    learning memory. It only reads local knowledge packs/store data and writes
    local review artifacts.
    """

    version = "0.2.0"
    mode = "working_domain_analyst"

    def __init__(
        self,
        *,
        store_dir: str | Path = "data/dean_os/analyst_knowledge",
        output_dir: str | Path = "reports/dean_os/working_domain_analyst",
        agent_name: str = "working_domain_analyst",
    ) -> None:
        self.store_dir = Path(store_dir)
        self.output_dir = Path(output_dir)
        self.agent_name = agent_name

    def run(
        self,
        *,
        question: str,
        domain_id: str,
        tickers: list[str] | None = None,
        context: dict[str, Any] | None = None,
        pack_paths: list[str | Path] | None = None,
        top_k: int = 8,
        horizon_days: int | None = None,
        reset_store: bool = False,
        save: bool = True,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        """Run the local evidence retrieval + review-only analyst report cycle."""

        as_of = as_of or utc_now_iso()
        question = (question or "").strip()
        domain_id = (domain_id or "").strip()
        tickers = _normalize_tickers(tickers or [])
        context = dict(context or {})
        top_k = int(clamp(float(top_k), 1.0, 50.0))

        if not question:
            raise ValueError("question cannot be empty")
        if not domain_id:
            raise ValueError("domain_id cannot be empty")

        store = LocalKnowledgeStore(self.store_dir)
        if reset_store:
            store.reset()

        loaded_packs = []
        for pack_path in pack_paths or []:
            pack = load_knowledge_pack(pack_path)
            store.add_pack(pack)
            stored_manifest = store.list_packs().get(pack.pack_id) or {}
            loaded_packs.append(
                {
                    "pack_id": pack.pack_id,
                    "domain_id": pack.domain_id,
                    "name": pack.name,
                    "version": pack.version,
                    "item_count": len(pack.items),
                    "tickers": pack.tickers,
                    "path": str(pack_path),
                    "pack_sha256": stored_manifest.get("pack_sha256"),
                    "source_count": stored_manifest.get("source_count", len(pack.sources)),
                }
            )

        retrieval = self._retrieve(
            store=store,
            question=question,
            domain_id=domain_id,
            tickers=tickers,
            context=context,
            top_k=top_k,
            as_of=as_of,
        )
        evidence = [
            _hit_to_analyst_evidence(
                hit,
                domain_id=domain_id,
                requested_tickers=tickers,
                as_of=as_of,
            )
            for hit in retrieval.hits
        ]

        analyst = BaseAnalystAgent(domain_id=domain_id, agent_name=self.agent_name)
        analyst_report = analyst.run(
            evidence=evidence,
            tickers=tickers,
            horizon_days=horizon_days,
            as_of=as_of,
        )

        evidence_scope_summary = _evidence_scope_summary(evidence, requested_tickers=tickers)
        knowledge_retrieval_gate = _knowledge_retrieval_gate(retrieval)
        missing_evidence = _missing_evidence_notes(
            requested_tickers=tickers,
            evidence_scope_summary=evidence_scope_summary,
            analyst_report=analyst_report.model_dump(mode="json"),
        )
        conclusion = _build_conclusion(
            analyst_report=analyst_report.model_dump(mode="json"),
            evidence_scope_summary=evidence_scope_summary,
            missing_evidence=missing_evidence,
        )

        run_id = _run_id(domain_id=domain_id, tickers=tickers)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "mode": self.mode,
            "agent_name": self.agent_name,
            "agent_version": self.version,
            "created_at": as_of,
            "question": question,
            "domain_id": domain_id,
            "tickers": tickers,
            "context": json_ready(context),
            "loaded_packs": loaded_packs,
            "retrieval": _retrieval_to_payload(retrieval),
            "knowledge_retrieval_gate": knowledge_retrieval_gate,
            "evidence_scope_summary": evidence_scope_summary,
            "missing_evidence": missing_evidence,
            "conclusion": conclusion,
            "analyst_report": analyst_report.model_dump(mode="json"),
            "safety": _safety_flags(),
        }

        if save:
            markdown = render_working_domain_analyst_markdown(payload)
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=markdown,
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths

        return json_ready(payload)

    def _retrieve(
        self,
        *,
        store: LocalKnowledgeStore,
        question: str,
        domain_id: str,
        tickers: list[str],
        context: dict[str, Any],
        top_k: int,
        as_of: str,
    ) -> KnowledgeRetrievalResult:
        """Run broad + ticker-aware retrieval and merge results.

        The existing store's ticker filter is intentionally strict. That is good
        for direct evidence, but a working analyst also needs sector/domain rules
        with empty ticker lists. Therefore this method runs both paths and merges
        by item_id.
        """

        retriever = KnowledgeRetriever(store)
        query = _build_query(question=question, tickers=tickers, context=context)
        broad = retriever.retrieve(
            query,
            domain_id=domain_id,
            top_k=top_k,
            as_of=as_of,
            require_point_in_time=True,
            require_source_provenance=True,
            intended_use="evidence",
        )
        if not tickers:
            return broad

        ticker = retriever.retrieve(
            query,
            domain_id=domain_id,
            tickers=tickers,
            top_k=top_k,
            as_of=as_of,
            require_point_in_time=True,
            require_source_provenance=True,
            intended_use="evidence",
        )
        merged_hits = _merge_hits([ticker, broad], top_k=top_k)
        exclusions = {}
        for result in (ticker, broad):
            for exclusion in result.exclusions:
                exclusions[exclusion.item_id] = exclusion
        return KnowledgeRetrievalResult(
            query=broad.query,
            hits=merged_hits,
            exclusions=sorted(exclusions.values(), key=lambda value: value.item_id),
            audit=_merged_retrieval_audit(
                hits=merged_hits,
                exclusions=list(exclusions.values()),
                as_of=as_of,
            ),
        )


def render_working_domain_analyst_markdown(payload: dict[str, Any]) -> str:
    report = payload.get("analyst_report") or {}
    thesis = report.get("thesis") or {}
    basket = report.get("ticker_basket") or {}
    scope = payload.get("evidence_scope_summary") or {}
    conclusion = payload.get("conclusion") or {}
    retrieval_gate = payload.get("knowledge_retrieval_gate") or {}

    lines: list[str] = [
        "# DEAN-OS Working Domain Analyst",
        "",
        f"- Mode: `{payload.get('mode')}`",
        f"- Agent: `{payload.get('agent_name')}` `{payload.get('agent_version')}`",
        f"- Domain: `{payload.get('domain_id')}`",
        f"- Tickers: `{', '.join(payload.get('tickers') or [])}`",
        f"- As of: `{payload.get('created_at')}`",
        f"- Question: {payload.get('question')}",
        f"- Recommendation: `{report.get('recommendation')}`",
        f"- Review required: `{report.get('review_required', True)}`",
        f"- Live execution allowed: `{report.get('live_execution_allowed', False)}`",
        f"- Knowledge retrieval gate: `{retrieval_gate.get('status')}`",
        f"- Point-in-time eligible: `{retrieval_gate.get('review_eligible', False)}`",
        "",
        "## Thesis",
        "",
        f"- Stance: `{thesis.get('stance')}`",
        f"- Expected direction: `{thesis.get('expected_direction')}`",
        f"- Confidence: `{thesis.get('confidence')}`",
        f"- Data quality: `{thesis.get('data_quality')}`",
        "",
        str(thesis.get("thesis") or ""),
        "",
        "## Evidence Scope",
        "",
        f"- Total evidence: `{scope.get('total', 0)}`",
        f"- Ticker-specific: `{scope.get('ticker', 0)}`",
        f"- Sector-level: `{scope.get('sector', 0)}`",
        f"- Domain-level: `{scope.get('domain', 0)}`",
        f"- Macro/policy/market/geopolitical: `{scope.get('external_context', 0)}`",
        "",
        "## Ticker / Basket Conclusion",
        "",
        f"- Conclusion type: `{conclusion.get('conclusion_type')}`",
        f"- Basket status: `{basket.get('basket_status')}`",
        f"- Direct ready count: `{basket.get('direct_ready_count')}`",
        f"- Basket candidate count: `{basket.get('basket_candidate_count')}`",
        f"- Blocked count: `{basket.get('blocked_count')}`",
        "",
        str(conclusion.get("summary") or ""),
        "",
    ]

    candidates = basket.get("candidates") or []
    if candidates:
        lines.extend(["| Ticker | Status | Confidence | Missing | Blocked reasons |", "|---|---:|---:|---|---|"])
        for candidate in candidates:
            lines.append(
                "| {ticker} | {status} | {confidence} | {missing} | {blocked} |".format(
                    ticker=candidate.get("ticker"),
                    status=candidate.get("candidate_status"),
                    confidence=candidate.get("confidence"),
                    missing=", ".join(candidate.get("required_missing_evidence") or []),
                    blocked=", ".join(candidate.get("blocked_reasons") or []),
                )
            )
        lines.append("")

    lines.extend(["## Missing Evidence", ""])
    for item in payload.get("missing_evidence") or []:
        lines.append(f"- {item}")
    if not payload.get("missing_evidence"):
        lines.append("- No missing-evidence blocker was generated by the working analyst wrapper.")

    excluded = (payload.get("retrieval") or {}).get("exclusions") or []
    if excluded:
        lines.extend(["", "## Excluded Knowledge", ""])
        for item in excluded:
            lines.append(
                f"- `{item.get('item_id')}`: {', '.join(item.get('reasons') or [])}"
            )

    lines.extend(["", "## Retrieved Evidence", ""])
    for idx, hit in enumerate((payload.get("retrieval") or {}).get("hits") or [], start=1):
        item = hit.get("item") or {}
        lines.extend(
            [
                f"### {idx}. {item.get('title')}",
                "",
                f"- Score: `{hit.get('score')}`",
                f"- Item type: `{item.get('item_type')}`",
                f"- Tickers: `{', '.join(item.get('tickers') or [])}`",
                f"- Tags: `{', '.join(item.get('tags') or [])}`",
                f"- Match reasons: `{'; '.join(hit.get('match_reasons') or [])}`",
                "",
                str(item.get("body") or ""),
                "",
            ]
        )

    lines.extend(["## Safety", ""])
    for key, value in sorted((payload.get("safety") or {}).items()):
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Operator Note",
            "",
            "Review-only artifact. A sector/domain thesis is not automatically a ticker thesis. Direct ticker thesis requires direct ticker evidence.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _hit_to_analyst_evidence(
    hit: KnowledgeRetrievalHit,
    *,
    domain_id: str,
    requested_tickers: list[str],
    as_of: str,
) -> AnalystEvidenceItem:
    item = hit.item
    directness = _classify_directness(item, requested_tickers=requested_tickers)
    evidence_type = _classify_evidence_type(item)
    strength = clamp((float(item.confidence) * 0.65) + (float(item.importance) / 5.0 * 0.35), 0.0, 1.0)
    source_reliabilities = [_quality_to_score(str(source.reliability)) for source in hit.sources]
    reliability = (
        sum(source_reliabilities) / len(source_reliabilities)
        if source_reliabilities
        else _quality_to_score(str(item.metadata.get("reliability") or "unverified"))
    )

    limitations: list[str] = []
    if directness != "ticker" and requested_tickers:
        limitations.append("Sector/domain evidence only; cannot support a direct ticker thesis by itself.")
    if item.stance_hint in {"unknown", "mixed"}:
        limitations.append("Stance is mixed/unknown; requires corroborating evidence before conviction.")
    for source in hit.sources:
        limitations.extend(source.known_limitations)

    published_at = _latest_timestamp_value(
        [source.published_at for source in hit.sources if source.published_at]
    )

    return AnalystEvidenceItem(
        source_type="knowledge_pack",
        source=item.item_id,
        published_at=published_at,
        as_of=as_of,
        domain_id=domain_id,
        tickers=item.tickers,
        sectors=item.sectors,
        evidence_type=evidence_type,
        summary=item.body,
        stance_hint=item.stance_hint,
        strength=strength,
        freshness_score=0.75,
        directness=directness,
        reliability_score=reliability,
        limitations=sorted(set(limitations)),
        blocked_windows=[],
        provenance={
            "contract": "dean_analyst_knowledge_point_in_time_v1",
            "required_lane_eligible": (
                item.metadata.get("required_lane_eligible") is True
            ),
            "ticker_thesis_eligible": False,
            "item_id": item.item_id,
            "pack_id": hit.lineage.get("pack_id"),
            "pack_version": hit.lineage.get("pack_version"),
            "pack_sha256": hit.lineage.get("pack_sha256"),
            "source_ids": [source.source_id for source in hit.sources],
            "sources": [
                {
                    "source_id": source.source_id,
                    "source_type": source.source_type,
                    "reference": source.reference,
                    "raw_storage_path": source.raw_storage_path,
                    "anchor": source.anchor,
                    "published_at": source.published_at,
                    "event_at": source.event_at,
                    "retrieved_at": source.retrieved_at,
                    "content_sha256": source.content_sha256,
                    "reliability": source.reliability,
                }
                for source in hit.sources
            ],
        },
        point_in_time=hit.point_in_time,
    )


def _classify_directness(item: Any, *, requested_tickers: list[str]) -> str:
    requested = {ticker.upper() for ticker in requested_tickers}
    item_tickers = {ticker.upper() for ticker in item.tickers}
    tags = {tag.lower() for tag in item.tags}
    title_body = f"{item.title} {item.body}".upper()

    if requested and requested.intersection(item_tickers):
        if item.item_type == "ticker":
            return "ticker"
        if any(ticker.lower() in tags for ticker in requested):
            return "ticker"
        if any(ticker in title_body for ticker in requested):
            return "ticker"

    if tags.intersection({"geopolitics", "export_controls", "china", "taiwan", "sanctions"}):
        return "geopolitical"
    if tags.intersection({"policy", "regulation", "export_control", "export_controls"}):
        return "policy"
    if tags.intersection({"market", "relative_strength", "price", "valuation", "multiple"}):
        return "market"
    if tags.intersection({"macro", "rates", "inflation", "liquidity", "credit"}):
        return "macro"
    if item.sectors:
        return "sector"
    return "domain"


def _classify_evidence_type(item: Any) -> str:
    tags = {tag.lower() for tag in item.tags}
    metrics = {metric.lower() for metric in item.metrics}
    item_type = str(item.item_type)

    if tags.intersection({"capex", "hyperscaler", "datacenter"}) or metrics.intersection({"capex_guidance", "datacenter_revenue"}):
        return "capex_cycle"
    if tags.intersection({"supply_chain", "foundry", "advanced_packaging", "hbm", "capacity"}):
        return "supply_chain"
    if tags.intersection({"export_controls", "geopolitics", "policy", "china", "taiwan"}):
        return "policy_or_geopolitical"
    if tags.intersection({"market", "relative_strength", "price", "valuation"}) or metrics.intersection({"relative_strength", "evidence_directness"}):
        return "market_confirmation"
    if item_type == "risk":
        return "risk"
    if item_type == "driver":
        return "sector_demand"
    if item_type == "ticker":
        return "ticker_specific"
    if item_type == "metric":
        return "metric"
    if item_type == "thesis_rule":
        return "thesis_rule"
    return item_type


def _build_query(*, question: str, tickers: list[str], context: dict[str, Any]) -> str:
    parts = [question, " ".join(tickers)]
    for key in ("event", "thesis", "risk", "tags", "notes"):
        value = context.get(key)
        if isinstance(value, list):
            parts.append(" ".join(str(item) for item in value))
        elif value:
            parts.append(str(value))
    return " ".join(part for part in parts if part).strip()


def _merge_hits(results: list[KnowledgeRetrievalResult], *, top_k: int) -> list[KnowledgeRetrievalHit]:
    merged: dict[str, KnowledgeRetrievalHit] = {}
    for result in results:
        for hit in result.hits:
            item_id = hit.item.item_id
            existing = merged.get(item_id)
            if existing is None or hit.score > existing.score:
                merged[item_id] = hit
            elif existing is not None:
                existing.match_reasons = sorted(set(existing.match_reasons + hit.match_reasons))
                existing.matched_terms = sorted(set(existing.matched_terms + hit.matched_terms))
    return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[:top_k]


def _retrieval_to_payload(result: KnowledgeRetrievalResult) -> dict[str, Any]:
    return {
        "query": result.query.model_dump(mode="json"),
        "created_at": result.created_at,
        "hit_count": len(result.hits),
        "audit": result.audit,
        "hits": [
            {
                "score": hit.score,
                "matched_terms": hit.matched_terms,
                "match_reasons": hit.match_reasons,
                "item": hit.item.model_dump(mode="json"),
                "sources": [source.model_dump(mode="json") for source in hit.sources],
                "lineage": hit.lineage,
                "point_in_time": hit.point_in_time,
            }
            for hit in result.hits
        ],
        "exclusions": [
            exclusion.model_dump(mode="json")
            for exclusion in result.exclusions
        ],
        "safety": result.safety,
    }


def _knowledge_retrieval_gate(result: KnowledgeRetrievalResult) -> dict[str, Any]:
    hit_statuses = [hit.point_in_time.get("status") for hit in result.hits]
    review_eligible = bool(result.hits) and all(
        status == "point_in_time_compatible" for status in hit_statuses
    )
    if review_eligible and result.exclusions:
        status = "review_eligible_with_exclusions"
    elif review_eligible:
        status = "review_eligible"
    elif result.exclusions:
        status = "blocked_no_point_in_time_eligible_knowledge"
    else:
        status = "blocked_no_matching_knowledge"
    return {
        "contract": "dean_analyst_knowledge_point_in_time_v1",
        "status": status,
        "review_eligible": review_eligible,
        "strict_point_in_time_required": True,
        "strict_source_provenance_required": True,
        "eligible_hit_count": len(result.hits),
        "excluded_hit_count": len(result.exclusions),
        "as_of": result.query.as_of,
        "does_not_authorize_prediction_or_trade": True,
    }


def _merged_retrieval_audit(
    *,
    hits: list[KnowledgeRetrievalHit],
    exclusions: list[Any],
    as_of: str,
) -> dict[str, Any]:
    if hits:
        status = "eligible_with_exclusions" if exclusions else "eligible"
    elif exclusions:
        status = "blocked_no_point_in_time_eligible_hits"
    else:
        status = "no_matching_items"
    return {
        "contract": "dean_analyst_knowledge_point_in_time_v1",
        "status": status,
        "strict": True,
        "as_of": as_of,
        "intended_use": "evidence",
        "selected_count": len(hits),
        "excluded_count": len(exclusions),
    }


def _evidence_scope_summary(evidence: list[AnalystEvidenceItem], *, requested_tickers: list[str]) -> dict[str, Any]:
    counts = Counter(item.directness for item in evidence)
    requested = {ticker.upper() for ticker in requested_tickers}
    direct_by_ticker = {
        ticker: sum(1 for item in evidence if item.directness == "ticker" and ticker in item.tickers)
        for ticker in sorted(requested)
    }
    external = sum(counts[key] for key in ("macro", "policy", "market", "geopolitical"))
    return {
        "total": len(evidence),
        "ticker": counts["ticker"],
        "sector": counts["sector"],
        "domain": counts["domain"],
        "macro": counts["macro"],
        "policy": counts["policy"],
        "market": counts["market"],
        "geopolitical": counts["geopolitical"],
        "external_context": external,
        "direct_by_ticker": direct_by_ticker,
    }


def _missing_evidence_notes(
    *,
    requested_tickers: list[str],
    evidence_scope_summary: dict[str, Any],
    analyst_report: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    direct_by_ticker = evidence_scope_summary.get("direct_by_ticker") or {}
    for ticker in requested_tickers:
        if int(direct_by_ticker.get(ticker, 0)) == 0:
            notes.append(f"Missing direct ticker-specific evidence for {ticker}; only basket/sector/domain review is allowed.")

    thesis = analyst_report.get("thesis") or {}
    for missing in thesis.get("blind_spots") or []:
        notes.append(f"Missing required domain evidence: {missing}.")

    quality_gates = analyst_report.get("quality_gates") or {}
    if quality_gates.get("required_evidence_complete") is False:
        notes.append("Required evidence gate is incomplete; recommendation should remain needs_more_data/blocked.")

    if not evidence_scope_summary.get("total"):
        notes.append("No retrieved knowledge evidence; analyst report is not usable.")

    return sorted(set(notes))


def _build_conclusion(
    *,
    analyst_report: dict[str, Any],
    evidence_scope_summary: dict[str, Any],
    missing_evidence: list[str],
) -> dict[str, Any]:
    basket = analyst_report.get("ticker_basket") or {}
    recommendation = analyst_report.get("recommendation")
    basket_status = basket.get("basket_status")
    direct_count = int(basket.get("direct_ready_count") or 0)
    basket_count = int(basket.get("basket_candidate_count") or 0)

    if missing_evidence and direct_count == 0:
        conclusion_type = "needs_more_evidence"
    elif direct_count > 0 and recommendation in {"ready_for_review", "partial_ready_for_review"}:
        conclusion_type = "ticker_review_candidate"
    elif basket_count > 0 or evidence_scope_summary.get("sector", 0) > 0:
        conclusion_type = "basket_review_candidate"
    else:
        conclusion_type = "domain_context_only"

    summary = (
        f"Working analyst conclusion: {conclusion_type}. "
        f"Recommendation={recommendation}, basket_status={basket_status}. "
        "This is review-only and does not authorize execution."
    )
    return {
        "conclusion_type": conclusion_type,
        "recommendation": recommendation,
        "basket_status": basket_status,
        "summary": summary,
    }


def _safety_flags() -> dict[str, bool]:
    return {
        "review_only": True,
        "trade_signal_generated": False,
        "live_execution_allowed": False,
        "live_execution_performed": False,
        "broker_access_performed": False,
        "production_config_write_performed": False,
        "training_performed": False,
        "tuning_performed": False,
        "model_promotion_performed": False,
        "learning_write_performed": False,
        "network_access_performed": False,
    }


def _normalize_tickers(tickers: list[str]) -> list[str]:
    return sorted({str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()})


def _quality_to_score(value: str) -> float:
    return {"high": 0.9, "medium": 0.65, "low": 0.35, "unverified": 0.25}.get(value.lower(), 0.65)


def _latest_timestamp_value(values: list[str]) -> str | None:
    parsed: list[tuple[datetime, str]] = []
    for value in values:
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        parsed.append((timestamp.astimezone(UTC), value))
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[0])[1]


def _run_id(*, domain_id: str, tickers: list[str]) -> str:
    suffix = "_".join(tickers) if tickers else "domain"
    stamp = utc_now_iso().replace(":", "").replace("+", "Z")
    return f"working_domain_analyst_{domain_id}_{suffix}_{stamp}"
