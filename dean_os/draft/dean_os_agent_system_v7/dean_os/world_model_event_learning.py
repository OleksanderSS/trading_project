from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from dean_os.analyst_core import (
    OUTCOME_HORIZONS,
    AnalysisPacket,
    Confidence,
    EvidenceGap,
    HypothesisLedgerEntry,
    Priority,
    ScenarioEdge,
    ScenarioNode,
    ScenarioOutcomeGraph,
)
from dean_os.analyst_core.lenses.event_classifier_lens import EventClassifierLens
from dean_os.analyst_core.lenses.historical_analog_lens import HistoricalAnalogLens
from dean_os.analyst_core.lenses.hypothesis_ledger_lens import HypothesisLedgerLens
from dean_os.analysts.context_adapter import MarketContextEvidenceAdapter
from dean_os.analysts.profiles import get_domain_profile
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import MarketContext, utc_now_iso
from dean_os.utils import json_ready

WORLD_MODEL_EVENT_LEARNING_CONTRACT = "dean_world_model_event_learning_packet_v1"
WORLD_MODEL_PRINCIPLES_SOURCE = (
    "dean_os/draft/DEAN_OS_World_Model_Architecture_Principles_v2(1).md"
)


class WorldModelEventLearningPacket:
    """Review-only event learning packet for the world-model loop.

    This is the bounded bridge from verified context/news/material evidence into
    structured hypotheses, scenario graphs, evidence gaps, and replay tasks. It
    intentionally does not write learning memory, update model weights, produce
    trades, or mutate the agent registry.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/world_model_event_learning_packet",
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        context: MarketContext,
        *,
        domain_id: str,
        as_of: str | None = None,
        max_events: int = 12,
        save: bool = True,
    ) -> dict[str, Any]:
        resolved_as_of = _require_as_of(as_of or context.as_of)
        knowledge_cutoff = _require_as_of(
            str((context.metadata or {}).get("knowledge_cutoff") or resolved_as_of)
        )
        if parse_timezone_aware(knowledge_cutoff) > parse_timezone_aware(resolved_as_of):
            raise ValueError("knowledge_cutoff cannot be later than as_of")
        profile = get_domain_profile(domain_id)
        adaptation = MarketContextEvidenceAdapter(domain_id).adapt(
            context,
            as_of=knowledge_cutoff,
        )
        pipeline_context = _pipeline_indicator_context(context)
        event_evidence = _event_evidence_records(
            adaptation.get("evidence", []),
            max_events=max_events,
        )
        packet = AnalysisPacket(
            packet_id=_run_id("world_model_event_packet"),
            as_of_date=resolved_as_of,
            source_packet_ids=[WORLD_MODEL_PRINCIPLES_SOURCE],
            event_records=event_evidence,
        )
        delta_trail: list[dict[str, Any]] = []

        if event_evidence:
            config = {
                "domain_id": domain_id,
                "sector_keywords": profile.sector_keywords,
                "ticker_universe": profile.ticker_universe_hint,
                "default_horizon_days": profile.horizon_days_default,
                "checkpoint_horizons": OUTCOME_HORIZONS,
            }
            event_delta = EventClassifierLens().analyze(packet, config)
            packet.classified_events.extend(event_delta.classified_events_added)
            packet.review_notes.extend(event_delta.review_notes_added)
            delta_trail.append(event_delta.model_dump(mode="json"))

            analog_delta = HistoricalAnalogLens().analyze(
                packet,
                {"max_analogs_per_event": 3},
            )
            packet.watch_signals.extend(analog_delta.watch_signals_added)
            packet.review_notes.extend(analog_delta.review_notes_added)
            delta_trail.append(analog_delta.model_dump(mode="json"))

            hypothesis_delta = HypothesisLedgerLens().analyze(packet, config)
            packet.hypotheses.extend(hypothesis_delta.hypotheses_added)
            packet.review_notes.extend(hypothesis_delta.review_notes_added)
            delta_trail.append(hypothesis_delta.model_dump(mode="json"))

        scenario_graph = _scenario_graph_from_packet(
            packet,
            domain_id=domain_id,
            as_of=resolved_as_of,
            pipeline_context=pipeline_context,
        )
        packet.scenario_graph = scenario_graph
        packet.evidence_gaps.extend(
            _evidence_gaps(
                context,
                packet,
                adaptation,
                domain_id=domain_id,
                pipeline_context=pipeline_context,
            )
        )
        replay_tasks = _replay_tasks(
            packet.hypotheses,
            scenario_graph,
            as_of=resolved_as_of,
            pipeline_context=pipeline_context,
        )
        status = _packet_status(packet, replay_tasks)
        payload = {
            "run_id": packet.packet_id,
            "created_at": utc_now_iso(),
            "mode": "world_model_event_learning_packet",
            "contract": WORLD_MODEL_EVENT_LEARNING_CONTRACT,
            "architecture_source": WORLD_MODEL_PRINCIPLES_SOURCE,
            "inputs": {
                "domain_id": domain_id,
                "as_of": resolved_as_of,
                "knowledge_cutoff": knowledge_cutoff,
                "max_events": max_events,
                "context_news_count": len(context.news or []),
                "context_research_document_count": len(
                    context.research_documents or []
                ),
                "context_macro_available": bool(context.macro),
                "context_pipeline_result_available": bool(context.pipeline_result),
                "pipeline_indicator_context_status": pipeline_context["status"],
                "expectation_context_available": pipeline_context[
                    "expectation_context_available"
                ],
            },
            "summary": _summary(
                status,
                adaptation,
                packet,
                replay_tasks,
                domain_id=domain_id,
                pipeline_context=pipeline_context,
            ),
            "world_model_boundary": _world_model_boundary(),
            "pipeline_indicator_context": pipeline_context,
            "historical_world_state_analogs": list(
                pipeline_context.get("historical_world_state_analogs", []) or []
            ),
            "source_evidence_audit": {
                "status": adaptation.get("status"),
                "accepted_evidence_count": len(adaptation.get("evidence", [])),
                "excluded_count": len(adaptation.get("exclusions", [])),
                "exclusions": adaptation.get("exclusions", [])[:20],
            },
            "analysis_packet": {
                "packet_id": packet.packet_id,
                "as_of_date": packet.as_of_date,
                "review_only": packet.review_only,
                "forbidden_outputs": packet.forbidden_outputs,
            },
            "classified_events": packet.classified_events,
            "historical_analog_candidates": packet.watch_signals,
            "hypotheses": [
                item.model_dump(mode="json") for item in packet.hypotheses
            ],
            "scenario_outcome_graph": (
                scenario_graph.model_dump(mode="json")
                if scenario_graph
                else None
            ),
            "evidence_gaps": [
                item.model_dump(mode="json") for item in packet.evidence_gaps
            ],
            "replay_tasks": replay_tasks,
            "delta_trail": delta_trail,
            "operator_next_steps": _operator_next_steps(status, packet),
            "safety": _safety(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_world_model_event_learning_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_world_model_event_learning_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    pipeline_context = payload.get("pipeline_indicator_context", {})
    lines = [
        "# DEAN-OS World Model Event Learning Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('packet_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- As-of: `{summary.get('as_of')}`",
        f"- Classified events: {summary.get('classified_event_count')}",
        f"- Hypotheses: {summary.get('hypothesis_count')}",
        f"- Scenario probability mass valid: {summary.get('scenario_probability_mass_valid')}",
        f"- Pipeline/indicator context: `{summary.get('pipeline_indicator_context_status')}`",
        f"- Historical world-state analogs: {summary.get('historical_world_state_analog_count')}",
        f"- Indicator metrics: {summary.get('indicator_metric_count')}",
        f"- Expectation context: {summary.get('expectation_context_available')}",
        f"- Replay tasks: {summary.get('replay_task_count')}",
        f"- Can trade: {summary.get('can_trade')}",
        f"- Can write learning memory now: {summary.get('can_write_learning_memory')}",
        "",
        "## Boundary",
        "",
        payload.get("world_model_boundary", {}).get("summary", ""),
        "",
        "## Hypotheses",
        "",
    ]
    for hypothesis in payload.get("hypotheses", []):
        lines.extend(
            [
                f"- `{hypothesis.get('hypothesis_id')}`: {hypothesis.get('hypothesis')}",
                f"  - horizons: {hypothesis.get('horizons_to_check')}",
                f"  - invalidation: {', '.join(hypothesis.get('invalidation_signals') or [])}",
            ]
        )
    if not payload.get("hypotheses"):
        lines.append("- none")
    lines.extend(["", "## Evidence Gaps", ""])
    for gap in payload.get("evidence_gaps", []):
        lines.append(f"- `{gap.get('priority')}`: {gap.get('description')}")
    if not payload.get("evidence_gaps"):
        lines.append("- none")
    lines.extend(["", "## Pipeline / Indicator Context", ""])
    lines.append(f"- Status: `{pipeline_context.get('status')}`")
    lines.append(f"- Regime: `{pipeline_context.get('regime_background', {}).get('label')}`")
    lines.append(
        "- Context tags: "
        + (
            ", ".join(pipeline_context.get("context_tags") or [])
            if pipeline_context.get("context_tags")
            else "none"
        )
    )
    lines.append(
        "- Watch metrics: "
        + (
            ", ".join(pipeline_context.get("watch_metrics") or [])
            if pipeline_context.get("watch_metrics")
            else "none"
        )
    )
    lines.extend(["", "## Replay Tasks", ""])
    for task in payload.get("replay_tasks", [])[:20]:
        lines.append(
            f"- `{task.get('horizon_days')}d` due `{task.get('due_at')}`: "
            f"{task.get('hypothesis_id')}"
        )
    if not payload.get("replay_tasks"):
        lines.append("- none")
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _event_evidence_records(
    evidence: list[Any],
    *,
    max_events: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        evidence,
        key=lambda item: (
            item.provenance.get("required_lane_eligible") is True,
            float(item.strength or 0.0),
            float(item.reliability_score or 0.0),
            float(item.freshness_score or 0.0),
        ),
        reverse=True,
    )
    records: list[dict[str, Any]] = []
    for item in ranked[:max_events]:
        record = item.model_dump(mode="json")
        record.update(
            {
                "event_id": item.evidence_id,
                "id": item.evidence_id,
                "source_id": item.source,
                "title": item.summary[:160],
                "summary": item.summary,
                "required_lane_eligible": (
                    item.provenance.get("required_lane_eligible") is True
                ),
            }
        )
        records.append(record)
    return records


def _pipeline_indicator_context(context: MarketContext) -> dict[str, Any]:
    metadata = dict(context.metadata or {})
    supplied_payloads = [
        metadata.get("pipeline_context"),
        metadata.get("indicator_state_grid"),
        metadata.get("pipeline_metric_input_readiness"),
        metadata.get("stage5_prediction_review"),
        metadata.get("stage7_regime_review"),
        context.pipeline_result,
        context.macro,
    ]
    supplied_any = any(isinstance(payload, dict) and bool(payload) for payload in supplied_payloads)
    metrics: dict[str, Any] = {}
    for payload in supplied_payloads:
        metrics.update(_extract_metrics(payload))
    regime = _extract_regime_background(metadata, context.pipeline_result)
    expectation = _extract_expectation_context(metadata, context.pipeline_result)
    historical_world_state_analogs = [
        dict(item)
        for item in (metadata.get("historical_world_state_analogs") or [])
        if isinstance(item, dict)
    ]
    context_tags = sorted(
        set(
            _provided_context_tags(metadata)
            + _derive_context_tags(metrics, regime, expectation)
        )
    )
    watch_metrics = _watch_metrics(metrics, context_tags, expectation)
    warnings: list[str] = []
    if supplied_any and not metrics and not regime.get("label"):
        warnings.append("pipeline_context_supplied_but_no_metric_or_regime_fields_extracted")
    if not expectation["available"]:
        warnings.append("expectation_context_missing")
    status = (
        "pipeline_indicator_context_ready"
        if metrics or regime.get("label") or context_tags or expectation["available"]
        else "pipeline_indicator_context_not_supplied"
    )
    return {
        "status": status,
        "review_only": True,
        "indicator_state_grid": {
            "available": bool(metrics),
            "metric_count": len(metrics),
            "metrics": metrics,
            "watch_metrics": watch_metrics,
        },
        "regime_background": regime,
        "expectation_context": expectation,
        "expectation_context_available": expectation["available"],
        "context_tags": context_tags,
        "watch_metrics": watch_metrics,
        "historical_world_state_analogs": historical_world_state_analogs,
        "historical_world_state_analog_count": len(historical_world_state_analogs),
        "warnings": warnings,
        "accepted_pipeline_fields": sorted(_ACCEPTED_METRIC_FIELDS),
        "review_only_rule": (
            "Pipeline/indicator context conditions world-model interpretation "
            "only; it cannot trigger execution, model promotion, or learning "
            "memory writes."
        ),
    }


_ACCEPTED_METRIC_FIELDS = {
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
    "market_context_volatility_ratio",
    "market_context_yield_curve_slope",
    "market_context_yield_curve_inverted",
    "pipeline_lane_available_count",
    "pipeline_lane_exact_context_count",
    "pipeline_lane_missing_count",
    "stage3_shard_count",
    "stage4_exact_context_count",
    "stage5_context_count",
    "stage5_complete_context_count",
    "metric_clear_plane_count",
    "metric_caution_plane_count",
    "metric_blocked_plane_count",
}


def _extract_metrics(payload: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return metrics
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
        "indicator_state_grid",
    "pipeline_context",
    "indicators",
    "indicator_metrics",
    }
    observations = payload.get("selected_observations") or payload.get("observations")
    if isinstance(observations, list):
        for item in observations:
            if not isinstance(item, dict):
                continue
            key = _normalize_metric_key(
                str(
                    item.get("metric")
                    or item.get("metric_name")
                    or item.get("series_id")
                    or item.get("name")
                    or ""
                )
            )
            value = _json_scalar(item.get("value"))
            if key and value is not None:
                metrics[key] = value
    for key, value in payload.items():
        normalized = _normalize_metric_key(str(key))
        if normalized in _ACCEPTED_METRIC_FIELDS or normalized.startswith("fred_"):
            scalar = _json_scalar(value)
            if scalar is not None:
                metrics[normalized] = scalar
        if isinstance(value, dict) and (key in containers or key in {"summary", "metadata"}):
            metrics.update(_extract_metrics(value))
    return dict(sorted(metrics.items()))


def _extract_regime_background(
    metadata: dict[str, Any],
    pipeline_result: dict[str, Any],
) -> dict[str, Any]:
    sources = [
        metadata.get("regime_context"),
        metadata.get("stage7_regime_review"),
        metadata.get("pipeline_context"),
        pipeline_result,
    ]
    label = None
    confidence = None
    as_of = None
    source = None
    for payload in sources:
        if not isinstance(payload, dict):
            continue
        if payload.get("contexts") and isinstance(payload["contexts"], list):
            first = next((item for item in payload["contexts"] if isinstance(item, dict)), None)
            if first:
                label = label or _string_value(first.get("regime"))
                confidence = confidence if confidence is not None else _numeric_value(first.get("confidence"))
                as_of = as_of or first.get("as_of")
                source = source or payload.get("schema_version") or "stage7_regime_review"
        label = label or _string_value(
            _first_present(payload, ("regime", "market_regime", "current_regime", "regime_label"))
        )
        confidence = confidence if confidence is not None else _numeric_value(
            _first_present(payload, ("confidence", "regime_confidence", "market_regime_confidence"))
        )
        as_of = as_of or _first_present(payload, ("as_of", "timestamp", "created_at"))
        if label and not source:
            source = "pipeline_context"
    return {
        "available": bool(label),
        "label": label,
        "confidence": confidence,
        "as_of": as_of,
        "source": source,
        "review_note": (
            "Regime background conditions event interpretation; it is not an "
            "execution signal."
        ),
    }


def _extract_expectation_context(
    metadata: dict[str, Any],
    pipeline_result: dict[str, Any],
) -> dict[str, Any]:
    payload = metadata.get("expectation_context")
    if not isinstance(payload, dict):
        payload = metadata.get("expectation_gap")
    if not isinstance(payload, dict):
        payload = pipeline_result.get("expectation_context") if isinstance(pipeline_result, dict) else None
    if not isinstance(payload, dict):
        pipeline_context = metadata.get("pipeline_context")
        payload = pipeline_context.get("expectation_context") if isinstance(pipeline_context, dict) else None
    if not isinstance(payload, dict):
        return {
            "available": False,
            "status": "expectation_context_not_supplied",
            "expectation_tags": [],
            "watch_metrics": [],
            "review_note": (
                "Consensus, positioning, market-implied probability, and "
                "crowdedness were not supplied."
            ),
        }
    tags = [
        _slug_tag(item)
        for item in payload.get("expectation_tags", payload.get("tags", [])) or []
        if str(item).strip()
    ]
    metrics = _extract_metrics(payload)
    crowdedness = _numeric_value(
        _first_present(payload, ("crowdedness", "positioning_score"))
    )
    surprise = _numeric_value(
        _first_present(payload, ("surprise_magnitude", "expectation_gap"))
    )
    watch = set(payload.get("watch_metrics", []) or [])
    if crowdedness is not None:
        watch.add("crowdedness")
    if surprise is not None:
        watch.add("surprise_magnitude")
    return {
        "available": True,
        "status": "expectation_context_ready",
        "expectation_tags": sorted(set(tags)),
        "metrics": metrics,
        "crowdedness": crowdedness,
        "surprise_magnitude": surprise,
        "watch_metrics": sorted(str(item) for item in watch if str(item).strip()),
        "review_note": (
            "Expectation context separates market reaction from fundamental "
            "change."
        ),
    }


def _provided_context_tags(metadata: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for value in metadata.get("context_tags", []) or []:
        if str(value).strip():
            tags.append(_slug_tag(value))
    for container_key in ("pipeline_context", "indicator_state_grid", "regime_context"):
        payload = metadata.get(container_key)
        if not isinstance(payload, dict):
            continue
        for key in ("context_tags", "tags", "regime_tags"):
            value = payload.get(key)
            if isinstance(value, list):
                tags.extend(_slug_tag(item) for item in value if str(item).strip())
    return sorted(set(tags))


def _derive_context_tags(
    metrics: dict[str, Any],
    regime: dict[str, Any],
    expectation: dict[str, Any],
) -> list[str]:
    tags: list[str] = []
    label = str(regime.get("label") or "").lower()
    if any(term in label for term in ("risk_off", "risk-off", "bear", "stress")):
        tags.append("pipeline_risk_off")
    if any(term in label for term in ("risk_on", "risk-on", "bull", "growth")):
        tags.append("pipeline_risk_on")
    volatility = _first_numeric_metric(metrics, ("vix", "volatility", "volatility_ratio", "market_context_volatility_ratio"))
    if volatility is not None:
        if volatility >= 25 or _metric_value(metrics, "volatility_ratio", default=0.0) >= 1.4:
            tags.append("pipeline_volatility_high")
        elif 0 < volatility <= 14:
            tags.append("pipeline_volatility_low")
    macro_score = _numeric_value(metrics.get("macro_score"))
    if macro_score is not None:
        tags.append("pipeline_macro_expansion" if macro_score >= 0.2 else "pipeline_macro_contraction" if macro_score <= -0.2 else "pipeline_macro_neutral")
    inflation = _first_numeric_metric(metrics, ("inflation_yoy", "cpi_yoy", "inflation"))
    if inflation is not None and inflation >= 3.0:
        tags.append("pipeline_inflation_pressure")
    slope = _first_numeric_metric(metrics, ("yield_curve_slope", "market_context_yield_curve_slope"))
    if slope is not None and slope < 0:
        tags.append("pipeline_yield_curve_inverted")
    credit = _first_numeric_metric(metrics, ("credit_spread", "credit_spreads"))
    if credit is not None and credit >= 1.5:
        tags.append("pipeline_credit_tightening")
    news_impact = _numeric_value(metrics.get("news_impact_score"))
    if news_impact is not None:
        if abs(news_impact) >= 0.8:
            tags.append("pipeline_news_intensity_high")
        tags.append("pipeline_negative_news_tone" if news_impact < -0.2 else "pipeline_positive_news_tone" if news_impact > 0.2 else "pipeline_news_tone_neutral")
    if expectation.get("crowdedness") is not None and float(expectation["crowdedness"]) >= 0.7:
        tags.append("expectation_crowded")
    tags.extend(expectation.get("expectation_tags", []))
    return sorted(set(tags))


def _watch_metrics(
    metrics: dict[str, Any],
    tags: list[str],
    expectation: dict[str, Any],
) -> list[str]:
    watch = {
        key
        for key in metrics
        if key
        in {
            "macro_score",
            "vix",
            "volatility",
            "volatility_ratio",
            "yield_curve_slope",
            "credit_spread",
            "credit_spreads",
            "inflation_yoy",
            "news_impact_score",
            "news_quality_score",
            "news_freshness_hours",
            "nlp_sentiment_score",
            "pipeline_lane_available_count",
            "pipeline_lane_missing_count",
            "pipeline_lane_exact_context_count",
            "stage3_shard_count",
            "stage4_exact_context_count",
            "stage5_context_count",
            "stage5_complete_context_count",
        }
    }
    if any(tag in tags for tag in ("pipeline_risk_off", "pipeline_volatility_high")):
        watch.update({"vix", "volatility_ratio", "sector_relative_performance"})
    if any(tag in tags for tag in ("pipeline_inflation_pressure", "pipeline_credit_tightening", "pipeline_yield_curve_inverted")):
        watch.update({"inflation_yoy", "real_rates", "credit_spreads", "yield_curve_slope"})
    if "pipeline_news_intensity_high" in tags:
        watch.update({"news_impact_score", "news_significance_level", "news_source_count"})
    watch.update(expectation.get("watch_metrics", []) or [])
    return sorted(str(item) for item in watch if str(item).strip())


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


def _first_present(payload: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(payload, dict):
        return None
    lowered = {
        _normalize_metric_key(str(key)): value
        for key, value in payload.items()
    }
    for key in keys:
        value = lowered.get(_normalize_metric_key(key))
        if value is not None and value != "":
            return value
    return None


def _first_numeric_metric(
    metrics: dict[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    for key in keys:
        value = _numeric_value(metrics.get(key))
        if value is not None:
            return value
    return None


def _metric_value(metrics: dict[str, Any], key: str, *, default: float) -> float:
    value = _numeric_value(metrics.get(key))
    return default if value is None else value


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
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in str(value)
    ).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "unknown"


def _scenario_graph_from_packet(
    packet: AnalysisPacket,
    *,
    domain_id: str,
    as_of: str,
    pipeline_context: dict[str, Any],
) -> ScenarioOutcomeGraph | None:
    if not packet.classified_events:
        return None
    event = max(
        packet.classified_events,
        key=lambda item: float(item.get("materiality_score", 0.0) or 0.0),
    )
    event_class = str(event.get("event_class") or "other")
    event_label = f"{event_class}: {event.get('title') or event.get('text_preview') or 'event'}"
    evidence_ids = [
        str(event.get("evidence_id") or event.get("event_id") or "").strip()
    ]
    evidence_ids = [item for item in evidence_ids if item]
    confidence = _confidence_from_event(event)
    indicator_grid = pipeline_context.get("indicator_state_grid", {}) or {}
    regime = pipeline_context.get("regime_background", {}) or {}
    expectation_context = pipeline_context.get("expectation_context", {}) or {}
    context_tags = list(pipeline_context.get("context_tags", []) or [])
    watch_metrics = list(pipeline_context.get("watch_metrics", []) or [])
    pipeline_ready = pipeline_context.get("status") == "pipeline_indicator_context_ready"
    world_state_analogs = list(
        pipeline_context.get("historical_world_state_analogs", []) or []
    )
    regime_label = regime.get("label") or "unknown"
    root_description = (
        "Review-only prior context using supplied pipeline/indicator "
        f"background: regime={regime_label}, "
        f"indicator_metrics={indicator_grid.get('metric_count', 0)}, "
        f"context_tags={context_tags[:10]}, watch_metrics={watch_metrics[:10]}, "
        f"historical_world_state_analogs={len(world_state_analogs)}."
        if pipeline_ready
        else (
            "Review-only prior context. Indicator and qualitative grids remain "
            "separate until evidence-backed merge."
        )
    )
    root = ScenarioNode(
        node_type="regime_state",
        label=f"{domain_id}: {regime_label} world state prior",
        description=root_description,
        as_of=as_of,
        confidence=Confidence.MEDIUM if pipeline_ready else Confidence.LOW,
    )
    event_node = ScenarioNode(
        node_type="event",
        label=event_label[:200],
        description=str(event.get("text_preview") or event.get("summary") or ""),
        as_of=as_of,
        confidence=confidence,
        evidence_ids=evidence_ids,
    )
    channel = ScenarioNode(
        node_type="transmission_channel",
        label=_transmission_channel(event_class),
        description=(
            "Candidate transmission channel; requires replay before it can "
            "be treated as learned causal memory."
        ),
        as_of=as_of,
        confidence=confidence,
        evidence_ids=evidence_ids,
    )
    if expectation_context.get("available"):
        expectation = ScenarioNode(
            node_type="expectation_gap",
            label="expectation_context_supplied",
            description=(
                "Supplied expectation context conditions the reaction path: "
                f"tags={expectation_context.get('expectation_tags', [])}, "
                f"crowdedness={expectation_context.get('crowdedness')}, "
                f"surprise={expectation_context.get('surprise_magnitude')}."
            ),
            as_of=as_of,
            confidence=Confidence.MEDIUM,
        )
        expectation_edge_weight = 0.5
        expectation_edge_confidence = Confidence.MEDIUM
        expectation_rationale = (
            "Market reaction is conditioned on supplied expectation context; "
            "still requires fixed-horizon replay."
        )
    else:
        expectation = ScenarioNode(
            node_type="expectation_gap",
            label="expectation_gap_unknown",
            description=(
                "Consensus, positioning, and market-implied probability were not "
                "proven inside this packet."
            ),
            as_of=as_of,
            confidence=Confidence.LOW,
        )
        expectation_edge_weight = 0.3
        expectation_edge_confidence = Confidence.LOW
        expectation_rationale = (
            "Market reaction depends on expectations; expectation evidence is "
            "missing until supplied."
        )
    scenarios = [
        ScenarioNode(
            node_type="scenario",
            label="context_only_or_priced_in",
            description=(
                "Event remains useful context but does not materially change "
                "the domain path."
            ),
            as_of=as_of,
            probability=0.50,
            confidence=Confidence.LOW,
            evidence_ids=evidence_ids,
            uncertainty_notes="Coarse review prior, not calibrated probability.",
        ),
        ScenarioNode(
            node_type="scenario",
            label="transmission_channel_activates",
            description=(
                "Event propagates through the candidate channel and changes "
                "future evidence observations."
            ),
            as_of=as_of,
            probability=0.30,
            confidence=confidence,
            evidence_ids=evidence_ids,
            uncertainty_notes="Requires outcome replay at fixed horizons.",
        ),
        ScenarioNode(
            node_type="scenario",
            label="signal_fades_or_false_positive",
            description=(
                "Initial interpretation weakens, source was noisy, or impact "
                "fails to appear."
            ),
            as_of=as_of,
            probability=0.20,
            confidence=Confidence.LOW,
            evidence_ids=evidence_ids,
            uncertainty_notes="False-signal scenario is explicit by design.",
        ),
    ]
    nodes = [root, event_node, channel, expectation, *scenarios]
    edges = [
        ScenarioEdge(
            source_node_id=root.node_id,
            target_node_id=event_node.node_id,
            edge_type="conditional_update",
            weight=0.6,
            rationale="News/material update is interpreted against prior world state.",
            evidence_ids=evidence_ids,
            confidence=confidence,
        ),
        ScenarioEdge(
            source_node_id=event_node.node_id,
            target_node_id=channel.node_id,
            edge_type="causal_channel",
            weight=0.5,
            rationale="Candidate event-to-world transmission path.",
            evidence_ids=evidence_ids,
            confidence=confidence,
        ),
        ScenarioEdge(
            source_node_id=event_node.node_id,
            target_node_id=expectation.node_id,
            edge_type="conditional_update",
            weight=expectation_edge_weight,
            rationale=expectation_rationale,
            evidence_ids=evidence_ids,
            confidence=expectation_edge_confidence,
        ),
        *[
            ScenarioEdge(
                source_node_id=channel.node_id,
                target_node_id=scenario.node_id,
                edge_type="leads_to",
                weight=0.4,
                rationale="Coarse scenario branch for review and future calibration.",
                evidence_ids=evidence_ids,
                confidence=scenario.confidence,
            )
            for scenario in scenarios
        ],
    ]
    graph_gaps = ["probability_is_review_prior_not_calibrated"]
    if expectation_context.get("available"):
        graph_gaps.append("expectation_context_supplied_but_not_calibrated")
    else:
        graph_gaps.append("expectation_gap_unknown")
    if indicator_grid.get("available"):
        graph_gaps.append("indicator_state_grid_supplied_review_only")
    else:
        graph_gaps.append("indicator_state_grid_missing_or_incomplete")
    if world_state_analogs:
        graph_gaps.append(
            "historical_world_state_analogs_supplied_review_only="
            + str(len(world_state_analogs))
        )
    else:
        graph_gaps.append("historical_world_state_analogs_missing")
    if context_tags:
        graph_gaps.append("pipeline_context_tags=" + ",".join(context_tags[:12]))
    return ScenarioOutcomeGraph(
        as_of=as_of,
        event_id=str(event.get("event_id") or ""),
        nodes=nodes,
        edges=edges,
        evidence_gaps=graph_gaps,
    )


def _evidence_gaps(
    context: MarketContext,
    packet: AnalysisPacket,
    adaptation: dict[str, Any],
    *,
    domain_id: str,
    pipeline_context: dict[str, Any],
) -> list[EvidenceGap]:
    gaps: list[EvidenceGap] = []
    indicator_grid = pipeline_context.get("indicator_state_grid", {}) or {}
    if not indicator_grid.get("available"):
        gaps.append(
            EvidenceGap(
                description=(
                    "Indicator State Grid is not supplied; event interpretation "
                    "cannot be merged with macro/market metric context yet."
                ),
                importance_to_scenario_probability=Priority.HIGH,
                expected_source_type="macro_or_pipeline_artifact",
                priority=Priority.HIGH,
            )
        )
    if not pipeline_context.get("expectation_context_available"):
        gaps.append(
            EvidenceGap(
                description=(
                    "Expectation Graph missing: no consensus, positioning, "
                    "market-implied probability, or crowdedness evidence supplied."
                ),
                importance_to_scenario_probability=Priority.HIGH,
                expected_source_type="expectation_context",
                priority=Priority.HIGH,
            )
        )
    if not packet.watch_signals and not pipeline_context.get(
        "historical_world_state_analogs"
    ):
        gaps.append(
            EvidenceGap(
                description=(
                    "Historical analog cluster missing or weak; current packet "
                    "cannot use analog base rates."
                ),
                importance_to_scenario_probability=Priority.MEDIUM,
                expected_source_type="historical_analog_library",
                priority=Priority.MEDIUM,
            )
        )
    if adaptation.get("exclusions"):
        gaps.append(
            EvidenceGap(
                description=(
                    f"{len(adaptation.get('exclusions', []))} source records were "
                    "excluded by point-in-time/source audit."
                ),
                importance_to_scenario_probability=Priority.MEDIUM,
                expected_source_type="source_quality_remediation",
                priority=Priority.MEDIUM,
            )
        )
    if not packet.hypotheses and packet.classified_events:
        gaps.append(
            EvidenceGap(
                description=(
                    f"No falsifiable hypothesis template matched classified "
                    f"{domain_id} events; add or review template coverage."
                ),
                importance_to_scenario_probability=Priority.MEDIUM,
                expected_source_type="hypothesis_template",
                priority=Priority.MEDIUM,
            )
        )
    return gaps


def _replay_tasks(
    hypotheses: list[HypothesisLedgerEntry],
    scenario_graph: ScenarioOutcomeGraph | None,
    *,
    as_of: str,
    pipeline_context: dict[str, Any],
) -> list[dict[str, Any]]:
    as_of_dt = parse_timezone_aware(as_of)
    if as_of_dt is None:
        raise ValueError("replay task as_of must be timezone-aware")
    indicator_grid = pipeline_context.get("indicator_state_grid", {}) or {}
    regime = pipeline_context.get("regime_background", {}) or {}
    context_snapshot = {
        "pipeline_indicator_context_status": pipeline_context.get("status"),
        "indicator_metric_count": indicator_grid.get("metric_count", 0),
        "regime_label": regime.get("label"),
        "regime_confidence": regime.get("confidence"),
        "expectation_context_available": pipeline_context.get(
            "expectation_context_available"
        ),
        "context_tags": list(pipeline_context.get("context_tags", []) or []),
        "watch_metrics": list(pipeline_context.get("watch_metrics", []) or []),
    }
    tasks: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        for horizon in hypothesis.horizons_to_check:
            tasks.append(
                {
                    "task_id": f"replay_{hypothesis.hypothesis_id}_{horizon}d",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "scenario_graph_id": (
                        scenario_graph.scenario_graph_id
                        if scenario_graph
                        else None
                    ),
                    "as_of": as_of,
                    "horizon_days": horizon,
                    "due_at": (as_of_dt + timedelta(days=horizon)).isoformat(),
                    "registration_status": "candidate_pending_manual_review",
                    "manual_review_gate_required": True,
                    "pipeline_context_snapshot": context_snapshot,
                    "review_action": "observe_outcome_and_score_hypothesis",
                    "allowed_update_after_review": [
                        "mark_hypothesis_confirmed_weakened_falsified_or_unresolved",
                        "record_false_analog_risk",
                        "propose_collector_or_template_improvement",
                    ],
                    "forbidden_update": [
                        "trade_signal",
                        "position_sizing",
                        "model_promotion_without_review",
                        "learning_memory_write_without_review",
                    ],
                }
            )
    return tasks


def _summary(
    status: str,
    adaptation: dict[str, Any],
    packet: AnalysisPacket,
    replay_tasks: list[dict[str, Any]],
    *,
    domain_id: str,
    pipeline_context: dict[str, Any],
) -> dict[str, Any]:
    scenario = packet.scenario_graph
    indicator_grid = pipeline_context.get("indicator_state_grid", {}) or {}
    regime = pipeline_context.get("regime_background", {}) or {}
    watch_metrics = list(pipeline_context.get("watch_metrics", []) or [])
    return {
        "packet_status": status,
        "domain_id": domain_id,
        "as_of": packet.as_of_date,
        "accepted_evidence_count": len(adaptation.get("evidence", [])),
        "event_record_count": len(packet.event_records),
        "classified_event_count": len(packet.classified_events),
        "historical_analog_candidate_count": len(packet.watch_signals),
        "historical_world_state_analog_count": pipeline_context.get(
            "historical_world_state_analog_count", 0
        ),
        "hypothesis_count": len(packet.hypotheses),
        "evidence_gap_count": len(packet.evidence_gaps),
        "scenario_graph_available": scenario is not None,
        "scenario_probability_mass_valid": (
            scenario.probability_mass_check if scenario else None
        ),
        "pipeline_indicator_context_status": pipeline_context.get("status"),
        "indicator_metric_count": indicator_grid.get("metric_count", 0),
        "regime_label": regime.get("label"),
        "expectation_context_available": pipeline_context.get(
            "expectation_context_available"
        ),
        "pipeline_context_tags": list(pipeline_context.get("context_tags", []) or []),
        "watch_metric_count": len(watch_metrics),
        "watch_metrics": watch_metrics,
        "replay_task_count": len(replay_tasks),
        "manual_review_required": True,
        "manual_review_gate": "world_model_replay_review_gate_required",
        "can_register_replay_after_manual_review": bool(replay_tasks),
        "can_write_learning_memory": False,
        "can_promote_model": False,
        "can_write_config": False,
        "can_trade": False,
    }


def _packet_status(
    packet: AnalysisPacket,
    replay_tasks: list[dict[str, Any]],
) -> str:
    if not packet.event_records:
        return "blocked_no_point_in_time_event_evidence"
    if not packet.classified_events:
        return "blocked_no_classified_events"
    if replay_tasks and packet.scenario_graph:
        return "world_model_event_learning_ready_pending_replay"
    return "world_model_event_learning_ready_with_gaps"


def _operator_next_steps(status: str, packet: AnalysisPacket) -> list[str]:
    steps = [
        "Review generated hypotheses before registering replay tasks.",
        "Supply Indicator State Grid and Expectation Graph inputs when available.",
        "Do not treat scenario probabilities as calibrated until replay outcomes exist.",
    ]
    if status.startswith("blocked"):
        steps.insert(
            0,
            "Fix source/audit inputs first: no usable event evidence reached the world-model packet.",
        )
    if packet.evidence_gaps:
        steps.append("Use evidence gaps as collector/template backlog candidates.")
    return steps


def _world_model_boundary() -> dict[str, Any]:
    return {
        "summary": (
            "This packet updates review-only world-model reasoning: event "
            "classification, analog candidates, hypotheses, scenario branches, "
            "evidence gaps, and replay tasks. It is not a trade signal."
        ),
        "allowed_outputs": [
            "context_update",
            "scenario_branch",
            "hypothesis_for_review",
            "historical_analog_candidate",
            "evidence_gap",
            "replay_task_for_manual_review",
        ],
        "blocked_outputs": [
            "buy_sell_hold",
            "position_sizing",
            "live_order",
            "production_price_target",
            "model_promotion",
            "learning_memory_write_without_review",
        ],
        "probability_policy": (
            "Scenario probabilities are coarse review priors until calibrated "
            "with fixed-horizon replay. Probability, impact, confidence, and "
            "market reaction must remain separate."
        ),
    }


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "can_trade": False,
        "can_write_production_config": False,
        "can_promote_model": False,
        "can_write_learning_memory": False,
        "network_access_performed": False,
        "live_execution_performed": False,
        "broker_access_performed": False,
        "production_config_write_performed": False,
        "model_promotion_performed": False,
        "learning_memory_write_performed": False,
        "outcome_registration_performed": False,
    }


def _transmission_channel(event_class: str) -> str:
    mapping = {
        "demand_driver": "demand_to_capex_to_supply_chain",
        "supply_disruption": "constraint_to_pricing_to_margin",
        "capex_signal": "capex_to_capacity_to_future_supply",
        "tariff": "policy_to_cost_to_supply_chain_relocation",
        "oil_shock": "energy_to_freight_to_inflation_rates",
        "central_bank_decision": "rates_to_liquidity_to_valuation",
        "sanctions_change": "policy_to_market_access_to_supply",
    }
    return mapping.get(event_class, f"{event_class}_candidate_channel")


def _confidence_from_event(event: dict[str, Any]) -> Confidence:
    score = (
        float(event.get("materiality_score", 0.0) or 0.0)
        + float(event.get("reliability_score", 0.0) or 0.0)
    ) / 2
    if score >= 0.7:
        return Confidence.HIGH
    if score >= 0.45:
        return Confidence.MEDIUM
    return Confidence.LOW


def _require_as_of(value: str | None) -> str:
    parsed = parse_timezone_aware(value)
    if parsed is None:
        raise ValueError(
            "WorldModelEventLearningPacket requires timezone-aware as_of"
        )
    return parsed.isoformat()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "WORLD_MODEL_EVENT_LEARNING_CONTRACT",
    "WorldModelEventLearningPacket",
    "render_world_model_event_learning_markdown",
]
