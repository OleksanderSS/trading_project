from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.analyst_core.domain_analyst_pipeline_news_taxonomy import classify_pipeline_news_context
from dean_os.schemas import utc_now_iso
from dean_os.utils import clamp, json_ready

DEFAULT_EVIDENCE_PACK_JSON = "reports/dean_os/analyst_evidence_pack_semiconductor_sector_only_strict_current/latest.json"
DEFAULT_DOMAIN_ID = "semiconductor_ai_infrastructure"


EVENT_KEYWORDS = {
    "policy_subsidy": ("subsidy", "chips act", "industrial policy", "grant", "incentive"),
    "demand_driver": ("demand", "ai", "gpu", "accelerator", "data center", "hyperscaler", "cloud", "orders"),
    "supply_disruption": ("shortage", "supply disruption", "bottleneck", "earthquake", "fire", "delay", "tight"),
    "capex_signal": ("capex", "capital expenditure", "spending", "investment", "buildout"),
    "rate_policy": ("fed", "rate", "inflation", "yield", "treasury"),
    "tariff": ("tariff", "customs", "trade war"),
    "sanctions": ("sanction", "export control", "blacklist", "restriction"),
    "capacity_change": ("capacity", "fab", "foundry", "packaging", "hbm", "wafer", "lithography"),
    "regulation": ("regulation", "regulator", "compliance", "law", "rule"),
}

POSITIVE_WORDS = ("growth", "increase", "expands", "strong", "support", "beat", "surge", "upgrade", "accelerate")
NEGATIVE_WORDS = ("risk", "decline", "weak", "shortage", "delay", "restriction", "sanction", "tariff", "cut")


class DomainAnalystEventInterpretationPacket:
    """Offline review-only news/event interpretation layer for the domain analyst.

    It turns local evidence-pack documents into structured hypotheses for review.
    It does not call external services, produce final conclusions, recommend
    trades, write learning memory, or mutate production config.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_event_interpretation_packet_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        evidence_pack_json: str | Path = DEFAULT_EVIDENCE_PACK_JSON,
        pipeline_context_json: str | Path | None = None,
        domain_id: str = DEFAULT_DOMAIN_ID,
        max_events: int = 80,
        save: bool = True,
    ) -> dict[str, Any]:
        evidence_pack = _load_json(evidence_pack_json)
        documents = [item for item in evidence_pack.get("documents", []) if isinstance(item, dict)]
        pipeline_context = _load_json(pipeline_context_json) if pipeline_context_json else None
        context_snapshot = _context_regime_snapshot(
            documents,
            pipeline_context=pipeline_context,
            pipeline_context_json=pipeline_context_json,
        )
        packets = [_interpret_document(item, domain_id=domain_id, context_snapshot=context_snapshot) for item in documents]
        packets = [item for item in packets if item is not None]
        packets = sorted(packets, key=lambda item: item["materiality_score"], reverse=True)[: max(1, int(max_events))]
        checks = _review_checks(evidence_pack=evidence_pack, packets=packets, context_snapshot=context_snapshot)
        status = _packet_status(checks, packets)
        payload = {
            "run_id": _run_id("domain_analyst_event_interpretation_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_event_interpretation_packet",
            "inputs": {
                "evidence_pack_json": str(evidence_pack_json),
                "pipeline_context_json": str(pipeline_context_json) if pipeline_context_json else None,
                "domain_id": domain_id,
                "max_events": max_events,
            },
            "summary": _summary(status, evidence_pack, packets, context_snapshot),
            "context_regime_snapshot": context_snapshot,
            "interpretation_contract": _interpretation_contract(),
            "event_interpretation_packets": packets,
            "review_checks": checks,
            "after_385_harvest_decisions": _after_385_harvest_decisions(),
            "pipeline_news_taxonomy_harvest_decisions": _pipeline_news_taxonomy_harvest_decisions(),
            "operator_next_steps": _operator_next_steps(status, packets),
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_event_interpretation_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_event_interpretation_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Domain Analyst Event Interpretation Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Source documents: {summary.get('source_document_count')}",
        f"- Event packets: {summary.get('event_packet_count')}",
        f"- Review required: {summary.get('review_required_count')}",
        f"- High materiality: {summary.get('high_materiality_count')}",
        f"- Context snapshot: `{summary.get('context_snapshot_status')}`",
        f"- Pipeline context: `{summary.get('pipeline_context_status')}` tags={summary.get('pipeline_context_tag_count')}",
        f"- Pipeline crisis-pattern events: {summary.get('pipeline_crisis_pattern_event_count')}",
        f"- Can create detailed data/news analysis: {summary.get('can_create_detailed_data_news_analysis')}",
        f"- Can create analyst research recommendation: {summary.get('can_create_analyst_research_recommendation')}",
        f"- Can create execution recommendation: {summary.get('can_create_execution_recommendation')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Top Event Interpretations",
        "",
    ]
    for item in payload.get("event_interpretation_packets", [])[:20]:
        lines.extend(
            [
                f"### `{item.get('event_id')}`",
                "",
                f"- Title: {item.get('title')}",
                f"- Type: `{item.get('event_type')}` directness=`{item.get('directness')}` materiality=`{item.get('materiality_label')}`",
                f"- Causal patterns: `{', '.join(item.get('candidate_causal_patterns') or [])}`",
                f"- Pipeline news tags: `{', '.join(item.get('pipeline_news_context', {}).get('context_tags') or [])}`",
                f"- Context-conditioned note: {item.get('context_conditioned_interpretation', {}).get('context_note')}",
                f"- Mechanism: {item.get('mechanism_summary')}",
                f"- Counterforces: {', '.join(item.get('counterforces') or []) or 'none'}",
                f"- Evidence gaps: {', '.join(item.get('evidence_gaps') or []) or 'none'}",
                "",
            ]
        )
    if not payload.get("event_interpretation_packets"):
        lines.append("- none")
    lines.extend(["## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _interpret_document(document: dict[str, Any], *, domain_id: str, context_snapshot: dict[str, Any]) -> dict[str, Any] | None:
    source_type = str(document.get("source_type") or "unknown").lower()
    if source_type not in {"news", "report", "article", "filing", "transcript", "document"}:
        return None
    title = str(document.get("title") or "")
    text = f"{title} {document.get('text') or ''}".strip()
    if not text:
        return None
    lower = text.lower()
    event_type = _event_type(lower)
    directness = _directness(document, lower)
    sentiment = _sentiment(lower)
    pipeline_news_context = classify_pipeline_news_context(text, sentiment_label=sentiment)
    patterns = _causal_patterns(event_type, lower)
    materiality_score = _materiality_score(event_type, source_type, directness, lower, document)
    materiality_score = clamp(materiality_score + _pipeline_news_materiality_adjustment(pipeline_news_context), 0.0, 1.0)
    materiality_label = _materiality_label(materiality_score)
    evidence_gaps = _evidence_gaps(document, directness, event_type)
    counterforces = _counterforces(event_type, lower)
    context_conditioned = _context_conditioned_interpretation(
        event_type,
        directness,
        lower,
        context_snapshot,
        pipeline_news_context=pipeline_news_context,
    )
    event_date = document.get("published_at")
    return {
        "event_id": f"event:{document.get('document_id') or _stable_slug(title)}",
        "source_id": document.get("document_id"),
        "source_type": _source_type(source_type, lower),
        "trust_tier": _trust_tier(source_type, lower),
        "published_at": document.get("published_at"),
        "event_date": event_date,
        "title": title[:240],
        "summary": _summary_text(text),
        "entities": _entities(document, text),
        "sectors": document.get("sectors") or [],
        "geography": _geography(lower),
        "event_type": event_type,
        "directness": directness,
        "sentiment": {
            "label": sentiment,
            "confidence": 0.55 if sentiment in {"positive", "negative"} else 0.35,
            "note": "Sentiment is weak context and never sufficient for an analytical conclusion.",
        },
        "candidate_causal_patterns": patterns,
        "pipeline_news_context": pipeline_news_context,
        "mechanism_summary": _mechanism_summary(event_type, directness),
        "mechanism_chain": _mechanism_chain(event_type),
        "affected_value_chain": _affected_value_chain(lower),
        "intermediate_variables": _intermediate_variables(event_type),
        "counterforces": counterforces,
        "context_conditioned_interpretation": context_conditioned,
        "evidence_gaps": evidence_gaps,
        "next_collection_tasks": _next_collection_tasks(
            evidence_gaps,
            context_conditioned.get("watch_metrics", []) + pipeline_news_context.get("watch_metrics", []),
        ),
        "materiality_score": round(materiality_score, 3),
        "materiality_label": materiality_label,
        "confidence_score": round(_confidence_score(document, source_type, directness, evidence_gaps), 3),
        "time_horizon": _time_horizon(event_type, lower),
        "allowed_output": "hypothesis_for_review",
        "review_required": materiality_label in {"watchlist_high", "review_required"} or bool(pipeline_news_context.get("crisis_pattern_matches")),
        "forbidden_outputs": _forbidden_outputs(),
        "source_anchor": {
            "uri": document.get("uri"),
            "tags": document.get("tags") or [],
            "metadata": document.get("metadata") or {},
        },
        "domain_id": domain_id,
    }


def _event_type(lower: str) -> str:
    matches = []
    for event_type, keywords in EVENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lower)
        if score:
            matches.append((score, event_type))
    if not matches:
        return "other"
    return sorted(matches, reverse=True)[0][1]


def _directness(document: dict[str, Any], lower: str) -> str:
    if document.get("tickers"):
        return "direct"
    if any(word in lower for word in ("semiconductor", "chip", "gpu", "hbm", "foundry", "wafer", "asml", "tsmc")):
        return "indirect"
    return "contextual"


def _sentiment(lower: str) -> str:
    positive = sum(1 for word in POSITIVE_WORDS if word in lower)
    negative = sum(1 for word in NEGATIVE_WORDS if word in lower)
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "unknown"


def _causal_patterns(event_type: str, lower: str) -> list[str]:
    patterns = {
        "demand_driver": "ai_capex_demand_cycle",
        "capex_signal": "hyperscaler_capex_to_semiconductor_demand",
        "capacity_change": "capacity_bottleneck_or_relief",
        "supply_disruption": "supply_chain_disruption",
        "sanctions": "export_control_geopolitical_constraint",
        "tariff": "trade_policy_margin_or_demand_shock",
        "regulation": "policy_regulation_compliance_shift",
        "rate_policy": "discount_rate_growth_factor_pressure",
        "policy_subsidy": "industrial_policy_capacity_incentive",
    }
    result = [patterns.get(event_type, "general_event_hypothesis")]
    if "taiwan" in lower or "china" in lower:
        result.append("geopolitical_supply_chain_risk")
    if "hbm" in lower or "packaging" in lower:
        result.append("advanced_packaging_memory_constraint")
    return sorted(set(result))


def _materiality_score(event_type: str, source_type: str, directness: str, lower: str, document: dict[str, Any]) -> float:
    score = 0.18
    if source_type == "report":
        score += 0.16
    if directness == "direct":
        score += 0.22
    elif directness == "indirect":
        score += 0.14
    if event_type in {"demand_driver", "capex_signal", "capacity_change", "sanctions", "tariff", "policy_subsidy"}:
        score += 0.22
    if any(word in lower for word in ("ai", "gpu", "hbm", "data center", "export control", "foundry", "asml", "tsmc")):
        score += 0.18
    if document.get("published_at"):
        score += 0.08
    return clamp(score, 0.0, 1.0)


def _pipeline_news_materiality_adjustment(pipeline_news_context: dict[str, Any]) -> float:
    adjustment = 0.0
    impact_classes = pipeline_news_context.get("impact_classifications", [])
    crisis_matches = pipeline_news_context.get("crisis_pattern_matches", [])
    if any(item.get("impact_strength") == "high" for item in impact_classes):
        adjustment += 0.06
    if any(int(item.get("severity") or 0) >= 7 for item in crisis_matches):
        adjustment += 0.08
    if pipeline_news_context.get("learned_pattern_matches"):
        adjustment += 0.03
    return clamp(adjustment, 0.0, 0.14)


def _materiality_label(score: float) -> str:
    if score >= 0.78:
        return "review_required"
    if score >= 0.62:
        return "watchlist_high"
    if score >= 0.45:
        return "watchlist_medium"
    if score >= 0.28:
        return "watchlist_low"
    return "archive"


def _evidence_gaps(document: dict[str, Any], directness: str, event_type: str) -> list[str]:
    gaps = []
    if not document.get("published_at"):
        gaps.append("missing_publication_or_event_date")
    if directness != "direct":
        gaps.append("needs_direct_company_or_ticker_evidence_before_ticker_thesis")
    if event_type == "other":
        gaps.append("event_type_unclear_needs_human_or_extractor_review")
    gaps.append("needs_corroborating_source_before_final_domain_conclusion")
    return gaps


def _counterforces(event_type: str, lower: str) -> list[str]:
    counterforces = {
        "demand_driver": ["capacity constraints", "customer capex slowdown", "valuation or rate pressure"],
        "capex_signal": ["capex pull-forward risk", "inventory correction", "supplier bottlenecks"],
        "capacity_change": ["demand mismatch", "ramp delays", "margin pressure from overcapacity"],
        "supply_disruption": ["inventory buffers", "supplier substitution", "policy response"],
        "sanctions": ["licensing carveouts", "domestic substitution", "demand rerouting"],
        "tariff": ["pass-through pricing", "supply-chain relocation", "demand elasticity"],
        "rate_policy": ["earnings growth offset", "liquidity support", "positioning unwind"],
    }
    result = list(counterforces.get(event_type, ["source uncertainty", "market expectations already priced"]))
    if "risk" in lower:
        result.append("risk may already be reflected in market narrative")
    return sorted(set(result))


def _context_regime_snapshot(
    documents: list[dict[str, Any]],
    *,
    pipeline_context: dict[str, Any] | None = None,
    pipeline_context_json: str | Path | None = None,
) -> dict[str, Any]:
    text = " ".join(
        f"{document.get('title') or ''} {document.get('text') or ''}"
        for document in documents
        if isinstance(document, dict)
    ).lower()
    dimensions = {
        "growth_regime": _dimension(
            text,
            positive=("growth", "expands", "expansion", "strong demand", "capex", "spending", "buildout"),
            negative=("recession", "slowdown", "contraction", "weak demand", "cuts", "decline"),
            positive_label="growth_expansion",
            negative_label="growth_slowdown",
        ),
        "inflation_rates_credit": _dimension(
            text,
            positive=("inflation", "rate", "rates", "yield", "treasury", "credit tightening", "higher for longer"),
            negative=("rate cut", "easing", "disinflation", "lower yields", "liquidity support"),
            positive_label="inflation_or_rate_pressure",
            negative_label="easing_or_disinflation",
        ),
        "war_geopolitical_context": _dimension(
            text,
            positive=("war", "sanction", "export control", "taiwan", "china", "geopolitical", "conflict"),
            negative=("peace", "ceasefire", "deescalation", "detente"),
            positive_label="war_sanctions_tension",
            negative_label="peace_or_deescalation",
        ),
        "commodity_energy_context": _dimension(
            text,
            positive=("oil shock", "energy shock", "gas prices", "commodity spike", "power shortage"),
            negative=("energy prices fall", "commodity relief", "lower oil"),
            positive_label="commodity_energy_stress",
            negative_label="commodity_energy_relief",
        ),
        "market_risk_appetite": _dimension(
            text,
            positive=("outperform", "momentum", "risk-on", "rally", "strong shares", "upgrade"),
            negative=("selloff", "risk-off", "volatility", "drawdown", "weak shares", "downgrade"),
            positive_label="risk_on_or_momentum",
            negative_label="risk_off_or_volatility",
        ),
        "technology_capex_context": _dimension(
            text,
            positive=("ai", "gpu", "accelerator", "data center", "hyperscaler", "hbm", "advanced packaging"),
            negative=("capex delay", "overcapacity", "inventory correction", "order cancellation"),
            positive_label="ai_capex_wave",
            negative_label="tech_capex_cooling",
        ),
    }
    tags = [
        _dimension_tag(dimension_id, value["label"])
        for dimension_id, value in dimensions.items()
        if value["label"] != "unknown"
    ]
    pipeline_overlay = _pipeline_context_overlay(
        pipeline_context,
        source_path=str(pipeline_context_json) if pipeline_context_json else None,
    )
    tags.extend(pipeline_overlay.get("context_tags", []))
    narrative = _narrative_context(tags, text)
    return {
        "snapshot_id": "context_regime_snapshot_from_evidence_pack_v1",
        "snapshot_status": "context_snapshot_ready" if tags else "context_snapshot_sparse",
        "method": "deterministic_keyword_context_slice_from_local_evidence_pack_with_optional_saved_pipeline_overlay",
        "dimensions": dimensions,
        "context_tags": sorted(set(tags)),
        "pipeline_context_overlay": pipeline_overlay,
        "narrative_context": narrative,
        "interpretation_rule": "The same news can imply different hypotheses under different macro/geopolitical/growth/risk regimes.",
        "confidence_boundary": "Context tags are review scaffolding, not final macro truth or trade signal.",
    }


def _dimension(
    text: str,
    *,
    positive: tuple[str, ...],
    negative: tuple[str, ...],
    positive_label: str,
    negative_label: str,
) -> dict[str, Any]:
    positive_hits = sorted({word for word in positive if word in text})
    negative_hits = sorted({word for word in negative if word in text})
    if positive_hits and negative_hits:
        label = "mixed"
    elif positive_hits:
        label = positive_label
    elif negative_hits:
        label = negative_label
    else:
        label = "unknown"
    total_hits = len(positive_hits) + len(negative_hits)
    confidence = clamp(0.2 + min(total_hits, 5) * 0.12, 0.0, 0.8) if total_hits else 0.0
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "supporting_terms": positive_hits + negative_hits,
    }


def _dimension_tag(dimension_id: str, label: str) -> str:
    if label == "mixed":
        return f"{dimension_id}_mixed"
    return label


def _pipeline_context_overlay(pipeline_context: dict[str, Any] | None, *, source_path: str | None) -> dict[str, Any]:
    if not pipeline_context:
        return {
            "overlay_status": "pipeline_context_not_supplied",
            "source_path": source_path,
            "supplied": False,
            "context_tags": [],
            "derived_context_tags": [],
            "provided_context_tags": [],
            "metrics": {},
            "watch_metrics": [],
            "warnings": [],
            "review_only_rule": "Optional saved pipeline context may condition interpretation but cannot trigger execution.",
        }

    metrics = _pipeline_metrics(pipeline_context)
    provided_tags = _provided_pipeline_tags(pipeline_context)
    regime_label = _string_value(_first_key_value(pipeline_context, ("regime", "market_regime", "current_regime", "regime_label")))
    volatility_regime = _string_value(_first_key_value(pipeline_context, ("volatility_regime", "vol_regime")))
    confidence = _numeric_value(_first_key_value(pipeline_context, ("confidence", "regime_confidence", "market_regime_confidence")))
    derived_tags = _derive_pipeline_context_tags(
        metrics=metrics,
        regime_label=regime_label,
        volatility_regime=volatility_regime,
        provided_tags=provided_tags,
    )
    watch_metrics = _pipeline_watch_metrics(metrics, derived_tags)
    warnings = []
    if not derived_tags and not provided_tags:
        warnings.append("pipeline_context_supplied_but_no_review_tags_derived")
    if confidence is not None and confidence < 0.35:
        warnings.append("pipeline_regime_confidence_low")
    return {
        "overlay_status": "pipeline_context_overlay_ready" if derived_tags or provided_tags else "pipeline_context_overlay_sparse",
        "source_path": source_path,
        "supplied": True,
        "raw_regime_label": regime_label,
        "raw_volatility_regime": volatility_regime,
        "confidence": confidence,
        "context_tags": sorted(set(derived_tags + provided_tags)),
        "derived_context_tags": sorted(set(derived_tags)),
        "provided_context_tags": sorted(set(provided_tags)),
        "metric_count": len(metrics),
        "metrics": metrics,
        "watch_metrics": watch_metrics,
        "warnings": warnings,
        "accepted_pipeline_fields": [
            "regime",
            "market_regime",
            "macro_score",
            "yield_curve_slope",
            "yield_curve_inverted",
            "vix",
            "volatility",
            "volatility_ratio",
            "credit_spread",
            "inflation_yoy",
            "fed_funds",
            "news_impact_score",
            "news_significance_level",
            "news_quality_score",
            "news_freshness_hours",
            "news_intensity",
            "nlp_sentiment_score",
            "sentiment_score",
        ],
        "source_pipeline_modules": [
            "src/analytics/context/market_regime_analyzer.py",
            "src/analytics/context/macro_context_analyzer.py",
            "src/features/enrichers/market_context_enricher.py",
            "src/features/enrichers/news_impact_enricher.py",
            "src/features/enrichers/news_quality_enricher.py",
            "src/features/enrichers/sentiment_features_enricher.py",
            "src/features/enrichers/nlp_features_enricher.py",
        ],
        "review_only_rule": "The overlay uses saved pipeline context as interpretation scaffolding only; it does not run collectors, models, tuning, or trading.",
    }


def _pipeline_metrics(payload: Any, *, prefix: str = "") -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    accepted = {
        "macro_score",
        "market_regime",
        "vix",
        "volatility",
        "volatility_5d",
        "volatility_20d",
        "volatility_ratio",
        "trend_5d",
        "trend_20d",
        "trend_alignment",
        "yield_curve_slope",
        "yield_curve_inverted",
        "fed_funds",
        "fed_funds_trend",
        "fed_funds_velocity",
        "credit_spread",
        "credit_spreads",
        "inflation",
        "inflation_yoy",
        "cpi_yoy",
        "news_impact_score",
        "news_significance_level",
        "news_quality_score",
        "news_source_count",
        "news_freshness_hours",
        "news_intensity",
        "nlp_sentiment_score",
        "sentiment_score",
        "sentiment_velocity",
        "sentiment_available",
        "market_context_volatility_ratio",
        "market_context_yield_curve_slope",
        "market_context_yield_curve_inverted",
    }
    containers = {
        "metrics",
        "latest_metrics",
        "latest_values",
        "market_context",
        "market_context_vector",
        "macro_features",
        "features",
        "context_vector",
        "summary",
    }
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized_key = _normalize_metric_key(prefix + key)
            if normalized_key in accepted or key.lower().startswith(("market_context_", "FRED_".lower())):
                scalar = _json_scalar(value)
                if scalar is not None:
                    metrics[normalized_key] = scalar
            if isinstance(value, dict) and key in containers:
                metrics.update(_pipeline_metrics(value))
            elif isinstance(value, dict) and prefix == "":
                nested = _pipeline_metrics(value, prefix="")
                for nested_key, nested_value in nested.items():
                    metrics.setdefault(nested_key, nested_value)
    return dict(sorted(metrics.items()))


def _provided_pipeline_tags(payload: Any) -> list[str]:
    tags: list[str] = []
    for key in ("context_tags", "tags", "regime_tags"):
        value = _first_key_value(payload, (key,))
        if isinstance(value, list):
            tags.extend(f"pipeline_{_slug_tag(item)}" for item in value if str(item).strip())
    return sorted(set(tags))


def _derive_pipeline_context_tags(
    *,
    metrics: dict[str, Any],
    regime_label: str | None,
    volatility_regime: str | None,
    provided_tags: list[str],
) -> list[str]:
    tags: list[str] = []
    regime = f"{regime_label or ''} {' '.join(provided_tags)}".lower()
    if any(word in regime for word in ("risk_off", "risk-off", "bear", "contraction", "stress", "crash")):
        tags.append("pipeline_risk_off")
    if any(word in regime for word in ("risk_on", "risk-on", "bull", "expansion", "growth")):
        tags.append("pipeline_risk_on")
    if any(word in regime for word in ("high_volatility", "high volatility", "volatile")):
        tags.append("pipeline_volatility_high")
    if any(word in regime for word in ("low_volatility", "low volatility", "calm")):
        tags.append("pipeline_volatility_low")

    volatility = _first_numeric_metric(metrics, ("vix", "volatility", "volatility_20d", "market_context_volatility_ratio", "volatility_ratio"))
    if volatility is not None:
        if volatility >= 25 or _metric_value(metrics, "volatility_ratio", default=0) >= 1.4 or _metric_value(metrics, "market_context_volatility_ratio", default=0) >= 1.4:
            tags.append("pipeline_volatility_high")
        elif 0 < volatility <= 14:
            tags.append("pipeline_volatility_low")
    if volatility_regime and "high" in volatility_regime.lower():
        tags.append("pipeline_volatility_high")

    macro_score = _numeric_value(metrics.get("macro_score"))
    if macro_score is not None:
        if macro_score >= 0.2:
            tags.append("pipeline_macro_expansion")
        elif macro_score <= -0.2:
            tags.append("pipeline_macro_contraction")
    inflation = _first_numeric_metric(metrics, ("inflation_yoy", "cpi_yoy", "inflation"))
    if inflation is not None and inflation >= 3.0:
        tags.append("pipeline_inflation_pressure")
    rate_level = _first_numeric_metric(metrics, ("fed_funds", "fed_funds_trend", "fed_funds_velocity"))
    if rate_level is not None and abs(rate_level) >= 4.0:
        tags.append("pipeline_rate_pressure")
    yield_slope = _first_numeric_metric(metrics, ("yield_curve_slope", "market_context_yield_curve_slope"))
    if yield_slope is not None and yield_slope < 0:
        tags.append("pipeline_yield_curve_inverted")
    if _truthy_metric(metrics, "yield_curve_inverted") or _truthy_metric(metrics, "market_context_yield_curve_inverted"):
        tags.append("pipeline_yield_curve_inverted")
    credit_spread = _first_numeric_metric(metrics, ("credit_spread", "credit_spreads"))
    if credit_spread is not None and credit_spread >= 1.5:
        tags.append("pipeline_credit_tightening")

    news_impact = _numeric_value(metrics.get("news_impact_score"))
    news_intensity = _numeric_value(metrics.get("news_intensity"))
    if news_intensity is not None and news_intensity >= 0.7:
        tags.append("pipeline_news_intensity_high")
    if news_impact is not None:
        if abs(news_impact) >= 0.8:
            tags.append("pipeline_news_intensity_high")
        if news_impact <= -0.2:
            tags.append("pipeline_negative_news_tone")
        elif news_impact >= 0.2:
            tags.append("pipeline_positive_news_tone")
    significance = metrics.get("news_significance_level")
    if str(significance).lower() in {"high", "3", "2"}:
        tags.append("pipeline_news_intensity_high")
    sentiment = _first_numeric_metric(metrics, ("nlp_sentiment_score", "sentiment_score", "sentiment_velocity"))
    if sentiment is not None:
        if sentiment <= -0.2:
            tags.append("pipeline_negative_news_tone")
        elif sentiment >= 0.2:
            tags.append("pipeline_positive_news_tone")
    news_quality = _numeric_value(metrics.get("news_quality_score"))
    if news_quality is not None:
        if news_quality < 0.4:
            tags.append("pipeline_news_quality_weak")
        elif news_quality >= 0.7:
            tags.append("pipeline_news_quality_strong")
    freshness = _numeric_value(metrics.get("news_freshness_hours"))
    if freshness is not None and freshness >= 72:
        tags.append("pipeline_news_stale")
    return sorted(set(tags))


def _pipeline_watch_metrics(metrics: dict[str, Any], tags: list[str]) -> list[str]:
    watch = set()
    for key in metrics:
        if key in {
            "macro_score",
            "vix",
            "volatility",
            "volatility_ratio",
            "yield_curve_slope",
            "credit_spread",
            "inflation_yoy",
            "news_impact_score",
            "news_quality_score",
            "news_freshness_hours",
            "nlp_sentiment_score",
        }:
            watch.add(key)
    if any(tag in tags for tag in ("pipeline_risk_off", "pipeline_volatility_high")):
        watch.update({"vix", "volatility_ratio", "sector_relative_performance"})
    if any(tag in tags for tag in ("pipeline_inflation_pressure", "pipeline_rate_pressure", "pipeline_credit_tightening")):
        watch.update({"inflation_yoy", "real_rates", "credit_spreads", "yield_curve_slope"})
    if "pipeline_news_intensity_high" in tags:
        watch.update({"news_impact_score", "news_significance_level", "news_source_count"})
    if "pipeline_news_quality_weak" in tags or "pipeline_news_stale" in tags:
        watch.update({"news_quality_score", "news_freshness_hours", "source_count"})
    return sorted(watch)


def _normalize_metric_key(key: str) -> str:
    cleaned = key.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned.startswith("fred_"):
        aliases = {
            "fred_dgs10": "ten_year_yield",
            "fred_dgs2": "two_year_yield",
            "fred_fedfunds": "fed_funds",
            "fred_cpi": "inflation",
            "fred_cpiacu": "inflation",
        }
        return aliases.get(cleaned, cleaned)
    return cleaned


def _first_key_value(payload: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(payload, dict):
        return None
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    for value in payload.values():
        if isinstance(value, dict):
            found = _first_key_value(value, keys)
            if found is not None:
                return found
    return None


def _first_numeric_metric(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _numeric_value(metrics.get(key))
        if value is not None:
            return value
    return None


def _metric_value(metrics: dict[str, Any], key: str, *, default: float) -> float:
    value = _numeric_value(metrics.get(key))
    return default if value is None else value


def _truthy_metric(metrics: dict[str, Any], key: str) -> bool:
    value = metrics.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "yes", "1", "inverted"}


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip().replace("%", "")
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _slug_tag(value: Any) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "unknown"


def _narrative_context(tags: list[str], text: str) -> dict[str, Any]:
    ai_intensity = text.count("ai") + text.count("gpu") + text.count("data center")
    if "ai_capex_wave" in tags or ("technology_capex_context_mixed" in tags and ai_intensity >= 3):
        label = "AI_everything"
    elif "inflation_or_rate_pressure" in tags:
        label = "higher_for_longer"
    elif "war_sanctions_tension" in tags:
        label = "deglobalization_security_of_supply"
    else:
        label = "unknown"
    dominance = "high" if ai_intensity >= 8 else "medium" if label != "unknown" else "unknown"
    crowdedness = "high" if label == "AI_everything" and dominance == "high" else "unknown"
    return {
        "label": label,
        "dominance": dominance,
        "momentum": "rising" if label != "unknown" else "unknown",
        "crowdedness": crowdedness,
        "reversal_risk": "medium" if crowdedness == "high" else "unknown",
        "rule": "Narrative context affects interpretation but is not a final signal.",
    }


def _context_conditioned_interpretation(
    event_type: str,
    directness: str,
    lower: str,
    context_snapshot: dict[str, Any],
    *,
    pipeline_news_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dimensions = context_snapshot.get("dimensions", {})
    tags = list(context_snapshot.get("context_tags", []))
    pipeline_overlay = context_snapshot.get("pipeline_context_overlay", {})
    pipeline_tags = set(pipeline_overlay.get("context_tags", [])) if isinstance(pipeline_overlay, dict) else set()
    pipeline_news_tags = set(pipeline_news_context.get("context_tags", [])) if isinstance(pipeline_news_context, dict) else set()
    dimension_labels = {key: value.get("label") for key, value in dimensions.items() if isinstance(value, dict)}
    amplifiers: list[str] = []
    dampeners: list[str] = []
    watch_metrics: list[str] = []
    review_flags: list[str] = []

    growth_label = dimension_labels.get("growth_regime")
    inflation_label = dimension_labels.get("inflation_rates_credit")
    geopolitical_label = dimension_labels.get("war_geopolitical_context")
    commodity_label = dimension_labels.get("commodity_energy_context")
    risk_label = dimension_labels.get("market_risk_appetite")
    tech_label = dimension_labels.get("technology_capex_context")

    if growth_label == "growth_expansion" and event_type in {"demand_driver", "capex_signal"}:
        amplifiers.append("growth expansion can make demand/capex news more plausible")
        watch_metrics.extend(["orders", "backlog", "utilization", "capex_guidance"])
    if growth_label == "growth_slowdown" and event_type in {"demand_driver", "capex_signal"}:
        dampeners.append("growth slowdown can weaken demand/capex transmission")
        watch_metrics.extend(["cancellations", "inventory", "customer_budget_cuts"])
    if growth_label == "mixed" and event_type in {"demand_driver", "capex_signal"}:
        review_flags.append("growth_context_mixed_requires_scenario_split")
        watch_metrics.extend(["orders", "inventory", "capex_guidance"])
    if inflation_label in {"inflation_or_rate_pressure", "mixed"}:
        if event_type in {"demand_driver", "capex_signal", "capacity_change"}:
            dampeners.append("inflation/rate pressure can delay long-duration capex response")
            watch_metrics.extend(["real_rates", "credit_spreads", "discount_rate_sensitivity"])
        if event_type == "rate_policy":
            amplifiers.append("rate-policy news is more material inside an inflation/rates regime")
        if inflation_label == "mixed":
            review_flags.append("inflation_rates_context_mixed_requires_expectation_gap_review")
    if geopolitical_label in {"war_sanctions_tension", "mixed"}:
        if event_type in {"sanctions", "tariff", "supply_disruption", "capacity_change"}:
            amplifiers.append("war/sanctions context raises security-of-supply and routing importance")
            watch_metrics.extend(["export_license_changes", "supply_chain_routing", "regional_revenue_exposure"])
        review_flags.append("geopolitical_context_requires_human_review")
    if commodity_label == "commodity_energy_stress":
        dampeners.append("energy/commodity stress can pressure margins and logistics")
        watch_metrics.extend(["energy_costs", "input_prices", "shipping_costs"])
    if risk_label in {"risk_on_or_momentum", "mixed"}:
        review_flags.append("market narrative may already price some good news")
        watch_metrics.extend(["sector_relative_performance", "earnings_revisions", "news_intensity"])
        if risk_label == "mixed":
            review_flags.append("risk_appetite_context_mixed")
    if tech_label in {"ai_capex_wave", "mixed"} and event_type in {"demand_driver", "capex_signal", "capacity_change"}:
        amplifiers.append("AI capex narrative can reinforce semiconductor demand mechanisms")
        watch_metrics.extend(["hyperscaler_capex", "gpu_lead_times", "hbm_supply", "advanced_packaging_capacity"])
        if tech_label == "mixed":
            review_flags.append("ai_capex_context_mixed_check_overcapacity_or_inventory")
    if "pipeline_macro_expansion" in pipeline_tags and event_type in {"demand_driver", "capex_signal"}:
        amplifiers.append("saved pipeline macro context points to expansion support")
        watch_metrics.extend(["macro_score", "orders", "capex_guidance"])
    if "pipeline_macro_contraction" in pipeline_tags and event_type in {"demand_driver", "capex_signal", "capacity_change"}:
        dampeners.append("saved pipeline macro context points to contraction risk")
        watch_metrics.extend(["macro_score", "inventory", "customer_budget_cuts"])
    if pipeline_tags.intersection({"pipeline_inflation_pressure", "pipeline_rate_pressure", "pipeline_credit_tightening", "pipeline_yield_curve_inverted"}):
        dampeners.append("saved pipeline macro/rates context can pressure long-duration growth mechanisms")
        watch_metrics.extend(["inflation_yoy", "real_rates", "credit_spreads", "yield_curve_slope"])
        review_flags.append("pipeline_macro_overlay_requires_scenario_split")
    if "pipeline_risk_off" in pipeline_tags:
        dampeners.append("saved pipeline risk-off context can reduce tolerance for positive narrative surprises")
        watch_metrics.extend(["vix", "sector_relative_performance", "liquidity"])
        review_flags.append("pipeline_risk_off_overlay")
    if "pipeline_risk_on" in pipeline_tags:
        review_flags.append("pipeline_risk_on_overlay_check_crowded_positioning")
        watch_metrics.extend(["sector_relative_performance", "earnings_revisions"])
    if "pipeline_volatility_high" in pipeline_tags:
        dampeners.append("saved pipeline volatility context raises interpretation uncertainty")
        watch_metrics.extend(["vix", "volatility_ratio"])
        review_flags.append("pipeline_high_volatility_overlay")
    if "pipeline_news_intensity_high" in pipeline_tags:
        review_flags.append("pipeline_news_intensity_high_check_narrative_crowding")
        watch_metrics.extend(["news_impact_score", "news_significance_level", "news_source_count"])
    if pipeline_tags.intersection({"pipeline_negative_news_tone", "pipeline_positive_news_tone"}):
        review_flags.append("pipeline_sentiment_overlay_is_context_not_signal")
        watch_metrics.extend(["nlp_sentiment_score", "news_impact_score"])
    if pipeline_tags.intersection({"pipeline_news_quality_weak", "pipeline_news_stale"}):
        dampeners.append("saved pipeline news-quality context weakens confidence in uncorroborated interpretation")
        watch_metrics.extend(["news_quality_score", "news_freshness_hours", "source_count"])
    if pipeline_news_tags:
        watch_metrics.extend(pipeline_news_context.get("watch_metrics", []) if isinstance(pipeline_news_context, dict) else [])
        review_flags.extend(pipeline_news_context.get("review_flags", []) if isinstance(pipeline_news_context, dict) else [])
    if pipeline_news_tags.intersection({"pipeline_market_crisis_context", "pipeline_linguistic_high_impact"}):
        amplifiers.append("pipeline news taxonomy marks this as crisis/high-impact context for review")
        review_flags.append("pipeline_crisis_or_high_impact_context")
    if "pipeline_tech_news_inside_crisis_context" in pipeline_news_tags and event_type in {"demand_driver", "capex_signal", "capacity_change"}:
        review_flags.append("technology_news_inside_crisis_context_requires_scenario_split")
        watch_metrics.extend(["valuation_multiples", "orders", "credit_spreads", "supply_constraints"])
    if "pipeline_news_geopolitical" in pipeline_news_tags and event_type in {"sanctions", "tariff", "supply_disruption", "capacity_change"}:
        amplifiers.append("pipeline news taxonomy flags geopolitical context")
        watch_metrics.extend(["export_controls", "regional_revenue_exposure", "substitution_path"])
    if "pipeline_learned_pattern_monetary_policy_shift" in pipeline_news_tags:
        dampeners.append("pipeline learned-pattern analogy flags monetary-policy sensitivity")
        watch_metrics.extend(["real_rates", "duration_sensitivity", "liquidity"])

    if directness != "direct":
        review_flags.append("context_conditioned_but_not_direct_ticker_evidence")
    if "tariff" in lower or "sanction" in lower or "export control" in lower:
        review_flags.append("policy_or_geopolitical_interpretation")

    note_parts = []
    if amplifiers:
        note_parts.append("amplified by " + "; ".join(amplifiers[:2]))
    if dampeners:
        note_parts.append("dampened by " + "; ".join(dampeners[:2]))
    if not note_parts:
        note_parts.append("context available but no strong context adjustment was detected")

    return {
        "context_tags": sorted(set(tags).union(pipeline_news_tags)),
        "context_dimensions": {key: value.get("label") for key, value in dimensions.items()},
        "amplifiers": sorted(set(amplifiers)),
        "dampeners": sorted(set(dampeners)),
        "watch_metrics": sorted(set(watch_metrics)),
        "review_flags": sorted(set(review_flags)),
        "narrative_context": context_snapshot.get("narrative_context", {}),
        "pipeline_context_overlay": _event_pipeline_overlay_summary(pipeline_overlay),
        "pipeline_news_context_summary": _event_pipeline_news_context_summary(pipeline_news_context),
        "context_note": "; ".join(note_parts),
        "allowed_output": "context_conditioned_hypothesis_for_review",
        "forbidden_outputs": _forbidden_outputs(),
    }


def _event_pipeline_news_context_summary(pipeline_news_context: Any) -> dict[str, Any]:
    if not isinstance(pipeline_news_context, dict):
        return {"adapter_id": None, "context_tags": []}
    return {
        "adapter_id": pipeline_news_context.get("adapter_id"),
        "dominant_impact_classification": pipeline_news_context.get("dominant_impact_classification"),
        "dominant_crisis_pattern": pipeline_news_context.get("dominant_crisis_pattern"),
        "context_tags": pipeline_news_context.get("context_tags", []),
        "watch_metrics": pipeline_news_context.get("watch_metrics", []),
        "review_flags": pipeline_news_context.get("review_flags", []),
        "historical_analogy_rule": pipeline_news_context.get("historical_analogy_rule"),
    }


def _event_pipeline_overlay_summary(pipeline_overlay: Any) -> dict[str, Any]:
    if not isinstance(pipeline_overlay, dict):
        return {"overlay_status": "pipeline_context_not_supplied", "context_tags": []}
    return {
        "overlay_status": pipeline_overlay.get("overlay_status"),
        "source_path": pipeline_overlay.get("source_path"),
        "context_tags": pipeline_overlay.get("context_tags", []),
        "watch_metrics": pipeline_overlay.get("watch_metrics", []),
        "warnings": pipeline_overlay.get("warnings", []),
        "review_only_rule": pipeline_overlay.get("review_only_rule"),
    }


def _next_collection_tasks(gaps: list[str], watch_metrics: list[str] | None = None) -> list[str]:
    tasks = []
    if "needs_direct_company_or_ticker_evidence_before_ticker_thesis" in gaps:
        tasks.append("collect_company_filings_transcripts_or_press_releases_for_direct_evidence")
    if "missing_publication_or_event_date" in gaps:
        tasks.append("repair_source_timestamp_or_event_date_metadata")
    if "needs_corroborating_source_before_final_domain_conclusion" in gaps:
        tasks.append("find_second_independent_source_or_primary_confirmation")
    if "event_type_unclear_needs_human_or_extractor_review" in gaps:
        tasks.append("route_to_human_event_type_review")
    for metric in watch_metrics or []:
        tasks.append(f"collect_or_check_watch_metric:{metric}")
    return sorted(set(tasks))


def _confidence_score(document: dict[str, Any], source_type: str, directness: str, gaps: list[str]) -> float:
    score = 0.35
    if source_type == "report":
        score += 0.16
    if document.get("published_at"):
        score += 0.12
    if directness == "direct":
        score += 0.18
    elif directness == "indirect":
        score += 0.08
    score -= min(len(gaps) * 0.04, 0.18)
    return clamp(score, 0.0, 1.0)


def _mechanism_summary(event_type: str, directness: str) -> str:
    return f"{event_type} interpreted as a {directness} review-only hypothesis; mechanism must be corroborated before conclusion."


def _mechanism_chain(event_type: str) -> list[str]:
    chains = {
        "demand_driver": ["event suggests demand impulse", "check order/capex evidence", "map impact to capacity, pricing, and margins"],
        "capex_signal": ["capex signal appears", "check supplier exposure", "watch backlog, lead times, and cancellation risk"],
        "capacity_change": ["capacity signal appears", "check ramp timing", "compare supply addition with demand path"],
        "supply_disruption": ["disruption appears", "check affected nodes", "watch substitution, inventory, and pricing"],
        "sanctions": ["policy restriction appears", "identify exposed entities", "watch licensing, rerouting, and substitution"],
        "tariff": ["trade cost shock appears", "map pass-through and elasticity", "watch margin and demand response"],
        "rate_policy": ["macro rate signal appears", "map discount-rate pressure", "watch growth sensitivity and liquidity"],
    }
    return chains.get(event_type, ["event observed", "classify mechanism", "collect corroborating evidence"])


def _affected_value_chain(lower: str) -> list[str]:
    nodes = []
    if any(word in lower for word in ("gpu", "accelerator", "data center", "hyperscaler", "cloud")):
        nodes.extend(["accelerators", "data_centers", "hyperscaler_capex"])
    if any(word in lower for word in ("hbm", "memory", "dram")):
        nodes.extend(["memory", "hbm"])
    if any(word in lower for word in ("foundry", "tsmc", "fab", "wafer")):
        nodes.extend(["foundry", "wafer_fabrication"])
    if any(word in lower for word in ("asml", "lithography", "equipment")):
        nodes.extend(["semiconductor_equipment", "lithography"])
    if any(word in lower for word in ("china", "taiwan", "export control", "sanction")):
        nodes.extend(["geopolitical_exposure", "supply_chain_routing"])
    return sorted(set(nodes)) or ["domain_context"]


def _intermediate_variables(event_type: str) -> list[str]:
    variables = {
        "demand_driver": ["order growth", "backlog", "utilization", "pricing", "customer capex"],
        "capex_signal": ["capex guidance", "lead times", "supplier backlog", "cancellation risk"],
        "capacity_change": ["capacity ramp timing", "yield", "utilization", "inventory"],
        "supply_disruption": ["affected node", "duration", "inventory buffer", "substitution"],
        "sanctions": ["license availability", "revenue exposure", "substitution path", "policy duration"],
        "tariff": ["pass-through", "margin impact", "demand elasticity", "relocation timing"],
        "rate_policy": ["discount rates", "liquidity", "growth multiple sensitivity"],
    }
    return variables.get(event_type, ["materiality", "source quality", "corroboration"])


def _time_horizon(event_type: str, lower: str) -> str:
    if event_type in {"rate_policy", "supply_disruption", "tariff", "sanctions"}:
        return "immediate"
    if event_type in {"demand_driver", "capex_signal"}:
        return "3_6_months"
    if event_type in {"capacity_change", "policy_subsidy"}:
        return "1_2_years"
    if "structural" in lower:
        return "structural_long_term"
    return "unclear"


def _source_type(source_type: str, lower: str) -> str:
    if "regulator" in lower or "government" in lower:
        return "government"
    if "press release" in lower:
        return "company_release"
    if source_type == "report":
        return "trade_press"
    return "news"


def _trust_tier(source_type: str, lower: str) -> str:
    if "regulator" in lower or "government" in lower:
        return "tier_1"
    if source_type in {"report", "filing", "transcript"}:
        return "tier_2"
    if source_type == "news":
        return "tier_3"
    return "tier_4"


def _entities(document: dict[str, Any], text: str) -> list[str]:
    entities = set(str(ticker).upper() for ticker in document.get("tickers") or [])
    known = ("TSMC", "ASML", "AMD", "NVDA", "NVIDIA", "INTEL", "TSM", "CHINA", "TAIWAN", "FED")
    upper = text.upper()
    for item in known:
        if item in upper:
            entities.add(item)
    return sorted(entities)


def _geography(lower: str) -> list[str]:
    geo = []
    for label, words in {
        "Taiwan": ("taiwan", "tsmc"),
        "China": ("china", "export control"),
        "United States": ("u.s.", "us ", "united states", "fed"),
        "Europe": ("europe", "eu ", "asml"),
    }.items():
        if any(word in lower for word in words):
            geo.append(label)
    return sorted(set(geo))


def _summary_text(text: str) -> str:
    return " ".join(text.split())[:600]


def _stable_slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)[:80].strip("_")
    return cleaned or "unknown"


def _review_checks(evidence_pack: dict[str, Any], packets: list[dict[str, Any]], context_snapshot: dict[str, Any]) -> list[dict[str, str]]:
    documents = evidence_pack.get("documents", [])
    pipeline_overlay = context_snapshot.get("pipeline_context_overlay", {})
    pipeline_supplied = bool(isinstance(pipeline_overlay, dict) and pipeline_overlay.get("supplied"))
    pipeline_tags = pipeline_overlay.get("context_tags", []) if isinstance(pipeline_overlay, dict) else []
    checks = [
        _check("pass" if evidence_pack.get("mode") == "analyst_evidence_pack" else "fail", "evidence_pack_artifact_type", str(evidence_pack.get("mode"))),
        _check("pass" if isinstance(documents, list) and documents else "fail", "source_documents_present", str(len(documents) if isinstance(documents, list) else 0)),
        _check("pass" if packets else "warn", "event_interpretations_present", str(len(packets))),
        _check("pass" if context_snapshot.get("context_tags") else "warn", "context_regime_snapshot_present", ", ".join(context_snapshot.get("context_tags", [])) or "No context tags inferred."),
        _check(
            "pass" if not pipeline_supplied or pipeline_tags else "warn",
            "pipeline_context_overlay_optional",
            ", ".join(pipeline_tags) if pipeline_tags else "Optional saved pipeline context not supplied or no tags derived.",
        ),
        _check("pass", "pipeline_context_is_review_only", "Saved pipeline context may condition analyst interpretation but cannot run collectors, train, tune, recommend, allocate, or trade."),
        _check("pass", "detailed_analysis_outputs_allowed", "Event interpretation, mechanisms, counterforces, evidence gaps, and watch metrics are allowed review outputs."),
        _check("pass", "execution_outputs_blocked", "Execution, buy/sell/hold, sizing, allocation, orders, and trading remain blocked."),
    ]
    missing_required = [
        packet.get("event_id")
        for packet in packets
        if not all(key in packet for key in ("event_type", "directness", "mechanism_chain", "context_conditioned_interpretation", "evidence_gaps", "allowed_output", "forbidden_outputs"))
    ]
    checks.append(_check("pass" if not missing_required else "fail", "event_packet_schema_fields_present", ", ".join(missing_required) or "All packets include required schema fields."))
    missing_pipeline_news_context = [packet.get("event_id") for packet in packets if "pipeline_news_context" not in packet]
    checks.append(_check("pass" if not missing_pipeline_news_context else "fail", "pipeline_news_taxonomy_fields_present", ", ".join(missing_pipeline_news_context) or "All packets include pipeline news taxonomy fields."))
    return checks


def _packet_status(checks: list[dict[str, str]], packets: list[dict[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "domain_analyst_event_interpretation_blocked"
    if any(packet.get("review_required") for packet in packets):
        return "domain_analyst_event_interpretation_ready_with_review_items"
    return "domain_analyst_event_interpretation_ready"


def _summary(status: str, evidence_pack: dict[str, Any], packets: list[dict[str, Any]], context_snapshot: dict[str, Any]) -> dict[str, Any]:
    documents = evidence_pack.get("documents", [])
    pipeline_overlay = context_snapshot.get("pipeline_context_overlay", {})
    pipeline_context_tags = pipeline_overlay.get("context_tags", []) if isinstance(pipeline_overlay, dict) else []
    pipeline_news_contexts = [packet.get("pipeline_news_context", {}) for packet in packets if isinstance(packet.get("pipeline_news_context"), dict)]
    crisis_pattern_event_count = sum(1 for item in pipeline_news_contexts if item.get("crisis_pattern_matches"))
    learned_pattern_event_count = sum(1 for item in pipeline_news_contexts if item.get("learned_pattern_matches"))
    return {
        "packet_status": status,
        "domain_id": evidence_pack.get("inputs", {}).get("domain_id") or DEFAULT_DOMAIN_ID,
        "context_snapshot_status": context_snapshot.get("snapshot_status"),
        "context_tags": context_snapshot.get("context_tags", []),
        "narrative_label": context_snapshot.get("narrative_context", {}).get("label"),
        "pipeline_context_supplied": bool(isinstance(pipeline_overlay, dict) and pipeline_overlay.get("supplied")),
        "pipeline_context_status": pipeline_overlay.get("overlay_status") if isinstance(pipeline_overlay, dict) else "pipeline_context_not_supplied",
        "pipeline_context_tags": pipeline_context_tags,
        "pipeline_context_tag_count": len(pipeline_context_tags),
        "pipeline_context_metric_count": pipeline_overlay.get("metric_count", 0) if isinstance(pipeline_overlay, dict) else 0,
        "pipeline_news_context_classified_count": sum(1 for item in pipeline_news_contexts if item.get("impact_classifications")),
        "pipeline_crisis_pattern_event_count": crisis_pattern_event_count,
        "pipeline_learned_pattern_event_count": learned_pattern_event_count,
        "pipeline_news_impact_class_counts": _pipeline_news_class_counts(pipeline_news_contexts),
        "pipeline_crisis_pattern_counts": _pipeline_crisis_pattern_counts(pipeline_news_contexts),
        "source_document_count": len(documents) if isinstance(documents, list) else 0,
        "event_packet_count": len(packets),
        "review_required_count": sum(1 for packet in packets if packet.get("review_required")),
        "high_materiality_count": sum(1 for packet in packets if packet.get("materiality_label") in {"watchlist_high", "review_required"}),
        "event_type_counts": _count(packets, "event_type"),
        "materiality_counts": _count(packets, "materiality_label"),
        "can_create_detailed_data_news_analysis": True,
        "can_create_event_interpretation": True,
        "can_create_context_conditioned_news_analysis": True,
        "can_create_mechanism_hypothesis": True,
        "can_create_evidence_gap_tasks": True,
        "can_create_watch_metric_requests": True,
        "can_create_analyst_research_recommendation": True,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_write_learning_memory": False,
        "can_update_production_config": False,
        "can_trade": False,
    }


def _interpretation_contract() -> dict[str, Any]:
    return {
        "contract_id": "domain_analyst_news_event_interpretation_v1",
        "source_templates": [
            "NEWS_EVENT_INTERPRETATION_SCHEMA_TEMPLATE.json",
            "ANALYST_NEWS_INTERPRETATION_PROMPT_TEMPLATE.md",
            "CAUSAL_PATTERN_SCHEMA_TEMPLATE.yaml",
            "SAFE_AUTOMATION_BOUNDARY_TEMPLATE.yaml",
            "src/config/context.yaml",
            "src/config/news_impact_classification.yaml",
            "src/patterns/pattern_recognition_adjustment.py",
        ],
        "allowed_outputs": [
            "hypothesis_for_review",
            "event_interpretation",
            "context_regime_snapshot",
            "pipeline_context_overlay",
            "pipeline_news_context_for_review",
            "crisis_pattern_analogy_for_review",
            "news_impact_classification_for_review",
            "context_conditioned_hypothesis_for_review",
            "mechanism_chain",
            "affected_value_chain",
            "intermediate_variables",
            "counterforces",
            "evidence_gap",
            "next_collection_task",
            "watchlist_materiality_label",
            "review_queue_item",
        ],
        "forbidden_outputs": _forbidden_outputs(),
        "rule": "News is interpreted as mechanism/hypothesis evidence, not as a sentiment-only signal or final trading conclusion.",
        "context_rule": "News must be interpreted inside growth, inflation/rates/credit, geopolitical, commodity, risk-appetite, narrative, and optional saved pipeline-context slices.",
        "pipeline_taxonomy_rule": "Pipeline news/crisis taxonomies are harvested as review-only context classifiers; prediction-adjustment and trading uses are explicitly excluded.",
        "pipeline_overlay_rule": "Pipeline context is read only from supplied local JSON artifacts and never triggers live pipeline execution.",
    }


def _after_385_harvest_decisions() -> list[dict[str, str]]:
    return [
        _harvest("NEWS_EVENT_INTERPRETATION_SCHEMA_TEMPLATE.json", "adapted_to_executable_packet", "Used as the event interpretation packet shape."),
        _harvest("ANALYST_NEWS_INTERPRETATION_PROMPT_TEMPLATE.md", "adapted_to_allowed_outputs", "Used mechanism, counterforce, evidence-gap, and watchlist output requirements."),
        _harvest("CAUSAL_PATTERN_SCHEMA_TEMPLATE.yaml", "adapted_partially", "Used causal pattern fields without importing domain-specific heavy-industry patterns."),
        _harvest("MACRO_REGIME_CONTEXT_ARCHITECTURE.md", "adapted_to_context_snapshot", "Used growth, inflation/rates/credit, war/geopolitical, commodity, and risk-appetite context slices."),
        _harvest("NARRATIVE_REGIME_TRACKING_TEMPLATE.yaml", "adapted_to_context_snapshot", "Used narrative label, dominance, crowdedness, and reversal-risk context as interpretation scaffolding."),
        _harvest("POLICY_IMPACT_ASSESSMENT_TEMPLATE.md", "adapted_partially", "Used policy/regime context requirements for sanctions, tariffs, regulation, and subsidies."),
        _harvest("DAILY_DOMAIN_LEARNING_RUN_TEMPLATE.yaml", "deferred", "Daily automation remains blocked until offline event packets and review queues are accepted."),
        _harvest("SAFE_AUTOMATION_BOUNDARY_TEMPLATE.yaml", "adapted_to_non_actions", "Used forbidden automation and review requirements."),
    ]


def _pipeline_news_taxonomy_harvest_decisions() -> list[dict[str, str]]:
    return [
        _harvest("src/config/context.yaml", "adapted_to_review_taxonomy", "Historical crisis events, severity, and indicators are used as analogy prompts, not proof."),
        _harvest("src/config/news_impact_classification.yaml", "adapted_to_review_taxonomy", "News impact classes, affected scope, timeframes, and strength are used for review priority and evidence routing."),
        _harvest("src/patterns/pattern_recognition_adjustment.py", "adapted_partially", "Learned news patterns are used as analogy labels only; prediction adjustments are excluded."),
        _harvest("src/features/enrichers/context_map_enricher.py", "adapted_to_context_language", "Context fingerprint, pattern sequence, and velocity vocabulary are used for future saved-context snapshots."),
        _harvest("src/features/analysis/market_conditions_analyzer.py", "adapted_to_regime_vocabulary", "Normal, volatile, trend, and crisis vocabulary is used as review-only regime language."),
    ]


def _operator_next_steps(status: str, packets: list[dict[str, Any]]) -> list[str]:
    if status == "domain_analyst_event_interpretation_blocked":
        return ["Fix evidence-pack shape before using news/event interpretation."]
    if packets:
        return [
            "Review high-materiality event interpretations before using them in thesis updates.",
            "Use evidence gaps as collection tasks, not as conclusions.",
            "Keep this packet separate from ticker recommendations and trading workflows.",
        ]
    return ["Provide local evidence-pack documents with news/report text before event interpretation."]


def _explicit_non_actions() -> list[str]:
    return [
        "No live fetch, external API, GPT, or FinBERT call is made.",
        "No final thesis truth, price target, buy/sell/hold, sizing, allocation, order, broker call, paper trade, or live trade is generated.",
        "No pipeline pattern-based prediction adjustment is applied.",
        "No learning memory, pattern memory, model, prompt, source registry, or production config is written.",
        "No daily automation is enabled by this packet.",
    ]


def _forbidden_outputs() -> list[str]:
    return [
        "buy_sell_hold",
        "price_target",
        "trade_signal",
        "autonomous_portfolio_action",
        "broker_order",
        "paper_trade",
        "live_trade",
        "production_config_mutation",
    ]


def _count(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _pipeline_news_class_counts(contexts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for context in contexts:
        for item in context.get("impact_classifications", []):
            class_id = str(item.get("classification_id") or "unknown")
            counts[class_id] = counts.get(class_id, 0) + 1
    return dict(sorted(counts.items()))


def _pipeline_crisis_pattern_counts(contexts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for context in contexts:
        for item in context.get("crisis_pattern_matches", []):
            pattern_id = str(item.get("pattern_id") or "unknown")
            counts[pattern_id] = counts.get(pattern_id, 0) + 1
    return dict(sorted(counts.items()))


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _harvest(source_file: str, classification: str, decision: str) -> dict[str, str]:
    return {"source_file": source_file, "classification": classification, "decision": decision}


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
