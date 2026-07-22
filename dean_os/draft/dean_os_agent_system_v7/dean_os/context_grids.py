from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from dean_os.analysts.profiles import get_domain_profile
from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_metrics import PipelineMetricSnapshot
from dean_os.draft.dean_os_agent_system_v7.dean_os.structured_context_provenance import audit_market_context_structured

ContextLevel = Literal[
    "global",
    "regional",
    "country",
    "sector",
    "adjacent_sector",
    "company",
]
DimensionDirection = Literal["improving", "deteriorating", "stable", "mixed", "unknown"]
EvidenceStatus = Literal["point_in_time", "supporting_review_only", "missing", "excluded"]

GLOBAL_REGIME_DIMENSIONS = (
    "economic_phase",
    "market_phase",
    "credit_phase",
    "inflation_phase",
    "ai_cycle",
    "geopolitical_phase",
)


class ContextDimensionState(BaseModel):
    dimension: str
    state: str = "unknown"
    direction: DimensionDirection = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    as_of: str | None = None
    source: str = "missing"
    evidence_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ContextGridNode(BaseModel):
    node_id: str
    level: ContextLevel
    label: str
    parent_ids: list[str] = Field(default_factory=list)
    dimensions: dict[str, ContextDimensionState] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextGridEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    relation: str
    transmission_channels: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)


class ContextGrid(BaseModel):
    schema_version: str = "dean_context_grid_v1"
    domain_id: str
    as_of: str
    status: str
    nodes: list[ContextGridNode] = Field(default_factory=list)
    edges: list[ContextGridEdge] = Field(default_factory=list)
    required_global_dimensions: list[str] = Field(default_factory=lambda: list(GLOBAL_REGIME_DIMENSIONS))
    completeness: dict[str, Any] = Field(default_factory=dict)
    point_in_time: dict[str, Any] = Field(default_factory=dict)
    evidence_gaps: list[str] = Field(default_factory=list)


class IndicatorObservation(BaseModel):
    indicator_id: str
    family: str
    scope: str
    name: str
    value: Any = None
    unit: str | None = None
    period: str | None = None
    available_at: str | None = None
    as_of: str
    source: str
    evidence_status: EvidenceStatus
    direction: DimensionDirection = "unknown"
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class IndicatorStateGrid(BaseModel):
    schema_version: str = "dean_indicator_state_grid_v1"
    domain_id: str
    as_of: str
    status: str
    observations: list[IndicatorObservation] = Field(default_factory=list)
    family_counts: dict[str, int] = Field(default_factory=dict)
    missing_families: list[str] = Field(default_factory=list)
    point_in_time: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ContextIndicatorPacket(BaseModel):
    schema_version: str = "dean_context_indicator_packet_v1"
    domain_id: str
    as_of: str
    context_grid: ContextGrid
    indicator_state_grid: IndicatorStateGrid
    authority_boundary: dict[str, bool] = Field(
        default_factory=lambda: {
            "review_only": True,
            "can_trade": False,
            "can_write_production_config": False,
            "can_promote_model": False,
            "can_write_learning_memory": False,
        }
    )


class ContextIndicatorGridBuilder:
    """Build canonical qualitative and quantitative grids from one run.

    The builder is intentionally deterministic. It structures accepted evidence
    and agent outputs; it does not use an LLM and does not manufacture missing
    global, regional, or sector states.
    """

    def build(
        self,
        context: Any,
        *,
        domain_id: str,
        agent_reports: Iterable[dict[str, Any] | Any] | None = None,
        pipeline_metric_snapshot: PipelineMetricSnapshot | dict[str, Any] | None = None,
    ) -> ContextIndicatorPacket:
        as_of = str(getattr(context, "as_of", None) or "")
        if not as_of:
            raise ValueError("Context and indicator grids require a point-in-time as_of timestamp")

        reports = [_to_mapping(item) for item in (agent_reports or [])]
        snapshot = self._snapshot(pipeline_metric_snapshot)
        news_evidence = _news_evidence_inventory(getattr(context, "news", []) or [])
        structured_audit = audit_market_context_structured(context)

        context_grid = self._build_context_grid(
            context,
            domain_id=domain_id,
            as_of=as_of,
            reports=reports,
            news_evidence=news_evidence,
            structured_audit=structured_audit,
        )
        indicator_grid = self._build_indicator_grid(
            domain_id=domain_id,
            as_of=as_of,
            snapshot=snapshot,
            structured_audit=structured_audit,
        )
        return ContextIndicatorPacket(
            domain_id=domain_id,
            as_of=as_of,
            context_grid=context_grid,
            indicator_state_grid=indicator_grid,
        )

    def _build_context_grid(
        self,
        context: Any,
        *,
        domain_id: str,
        as_of: str,
        reports: list[dict[str, Any]],
        news_evidence: list[dict[str, Any]],
        structured_audit: dict[str, Any],
    ) -> ContextGrid:
        profile = get_domain_profile(domain_id)
        metadata = dict(getattr(context, "metadata", {}) or {})
        global_dimensions = self._global_dimensions(metadata, as_of)
        global_node = ContextGridNode(
            node_id="global",
            level="global",
            label="Global context",
            dimensions=global_dimensions,
            evidence_ids=_evidence_ids(news_evidence, directness={"macro", "geopolitical", "policy"}),
            evidence_gaps=[
                f"Missing explicit global regime dimension: {name}"
                for name, state in global_dimensions.items()
                if state.state == "unknown"
            ],
        )

        nodes: list[ContextGridNode] = [global_node]
        edges: list[ContextGridEdge] = []

        for region in _context_scope_items(metadata, "regions", "region_context"):
            node_id = f"region:{_slug(region['id'])}"
            nodes.append(
                ContextGridNode(
                    node_id=node_id,
                    level="regional",
                    label=region["label"],
                    parent_ids=["global"],
                    dimensions=_dimension_map(region.get("dimensions", {}), as_of, "metadata.region_context"),
                    evidence_ids=_as_string_list(region.get("evidence_ids")),
                    evidence_gaps=_as_string_list(region.get("evidence_gaps")),
                    metadata={key: value for key, value in region.items() if key not in {"dimensions", "evidence_ids", "evidence_gaps"}},
                )
            )
            edges.append(ContextGridEdge(source_node_id="global", target_node_id=node_id, relation="contains_context"))

        for country in _context_scope_items(metadata, "countries", "country_context"):
            parent = str(country.get("parent_id") or "global")
            node_id = f"country:{_slug(country['id'])}"
            nodes.append(
                ContextGridNode(
                    node_id=node_id,
                    level="country",
                    label=country["label"],
                    parent_ids=[parent],
                    dimensions=_dimension_map(country.get("dimensions", {}), as_of, "metadata.country_context"),
                    evidence_ids=_as_string_list(country.get("evidence_ids")),
                    evidence_gaps=_as_string_list(country.get("evidence_gaps")),
                    metadata={key: value for key, value in country.items() if key not in {"dimensions", "evidence_ids", "evidence_gaps"}},
                )
            )
            edges.append(ContextGridEdge(source_node_id=parent, target_node_id=node_id, relation="contains_context"))

        domain_report = _find_domain_report(reports, domain_id)
        sector_dimensions, sector_gaps = self._sector_dimensions(domain_report, as_of)
        sector_node_id = f"sector:{domain_id}"
        sector_evidence_ids = _evidence_ids(news_evidence, domain_id=domain_id)
        nodes.append(
            ContextGridNode(
                node_id=sector_node_id,
                level="sector",
                label=profile.display_name,
                parent_ids=["global"],
                dimensions=sector_dimensions,
                evidence_ids=sector_evidence_ids,
                evidence_gaps=sector_gaps,
                metadata={
                    "domain_profile_version": profile.version,
                    "required_evidence_types": profile.required_evidence_types,
                    "useful_evidence_types": profile.useful_evidence_types,
                },
            )
        )
        edges.append(
            ContextGridEdge(
                source_node_id="global",
                target_node_id=sector_node_id,
                relation="conditions",
                transmission_channels=["macro", "rates", "credit", "policy", "geopolitics", "market_expectations"],
                confidence=0.5 if any(item.state != "unknown" for item in global_dimensions.values()) else 0.0,
            )
        )

        for adjacent in profile.adjacent_sectors:
            adjacent_id = f"adjacent_sector:{_slug(adjacent)}"
            nodes.append(
                ContextGridNode(
                    node_id=adjacent_id,
                    level="adjacent_sector",
                    label=adjacent,
                    parent_ids=["global"],
                    dimensions={
                        "transmission_status": ContextDimensionState(
                            dimension="transmission_status",
                            state="candidate_channel",
                            confidence=0.0,
                            as_of=as_of,
                            source="domain_profile",
                            notes=["Requires event-specific evidence before activation."],
                        )
                    },
                    evidence_gaps=[f"No event-specific transmission evidence for {adjacent}."],
                )
            )
            edges.append(
                ContextGridEdge(
                    source_node_id=sector_node_id,
                    target_node_id=adjacent_id,
                    relation="potential_transmission",
                    transmission_channels=["demand", "supply", "capex", "pricing", "logistics", "power"],
                    confidence=0.0,
                )
            )

        known_dimensions = sum(
            state.state != "unknown"
            for node in nodes
            for state in node.dimensions.values()
        )
        total_dimensions = sum(len(node.dimensions) for node in nodes)
        gap_count = sum(len(node.evidence_gaps) for node in nodes)
        point_in_time_ready = bool(as_of) and structured_audit.get("status") not in {
            "blocked_no_point_in_time_structured_context"
        }
        global_complete = all(
            global_dimensions[name].state != "unknown"
            for name in GLOBAL_REGIME_DIMENSIONS
        )
        sector_stance_known = sector_dimensions.get("sector_stance", ContextDimensionState(dimension="sector_stance")).state != "unknown"
        status = (
            "ready"
            if global_complete and sector_stance_known and point_in_time_ready
            else "partial"
            if known_dimensions
            else "needs_evidence"
        )
        return ContextGrid(
            domain_id=domain_id,
            as_of=as_of,
            status=status,
            nodes=nodes,
            edges=edges,
            completeness={
                "known_dimensions": known_dimensions,
                "total_dimensions": total_dimensions,
                "coverage_ratio": round(known_dimensions / total_dimensions if total_dimensions else 0.0, 6),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "evidence_gap_count": gap_count,
            },
            point_in_time={
                "as_of_present": bool(as_of),
                "news_audit": metadata.get("news_point_in_time_audit", {}),
                "structured_context_audit": {
                    key: value
                    for key, value in structured_audit.items()
                    if key not in {"accepted_context", "accepted_observations", "exclusions"}
                },
            },
            evidence_gaps=_dedupe(
                gap for node in nodes for gap in node.evidence_gaps
            ),
        )

    def _global_dimensions(self, metadata: dict[str, Any], as_of: str) -> dict[str, ContextDimensionState]:
        explicit_sources = [
            metadata.get("regime_dimensions"),
            metadata.get("world_context"),
            metadata.get("regime_snapshot"),
            metadata.get("regime_context"),
        ]
        merged: dict[str, Any] = {}
        for source in explicit_sources:
            if isinstance(source, dict):
                merged.update(source)

        stage7 = metadata.get("stage7_regime_review")
        if isinstance(stage7, dict):
            contexts = stage7.get("contexts")
            if isinstance(contexts, list) and contexts:
                first = contexts[0] if isinstance(contexts[0], dict) else {}
                if first.get("regime") and "market_phase" not in merged:
                    merged["market_phase"] = {
                        "state": first.get("regime"),
                        "confidence": first.get("confidence", 0.0),
                        "as_of": first.get("as_of") or as_of,
                        "source": "stage7_regime_review",
                    }

        dimensions: dict[str, ContextDimensionState] = {}
        for name in GLOBAL_REGIME_DIMENSIONS:
            raw = merged.get(name)
            dimensions[name] = _dimension_state(
                name,
                raw,
                as_of=as_of,
                default_source="metadata.regime_dimensions",
            )
        return dimensions

    def _sector_dimensions(
        self,
        report: dict[str, Any] | None,
        as_of: str,
    ) -> tuple[dict[str, ContextDimensionState], list[str]]:
        if not report:
            return {
                "sector_stance": ContextDimensionState(
                    dimension="sector_stance",
                    state="unknown",
                    as_of=as_of,
                    source="missing",
                )
            }, ["Domain analytical report is missing."]

        payload = report.get("analysis_payload") or {}
        metrics = payload.get("domain_metrics") if isinstance(payload, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        confidence = _unit_interval(report.get("confidence"))
        dimensions = {
            "sector_stance": ContextDimensionState(
                dimension="sector_stance",
                state=str(metrics.get("stance") or report.get("verdict") or "unknown"),
                confidence=confidence,
                as_of=as_of,
                source="domain_analytical_agent",
            ),
            "expectation_gap": _dimension_state(
                "expectation_gap",
                metrics.get("expectation_gap"),
                as_of=as_of,
                default_source="domain_analytical_agent",
            ),
            "regime_context": _dimension_state(
                "regime_context",
                metrics.get("regime_context"),
                as_of=as_of,
                default_source="domain_analytical_agent",
            ),
        }
        gaps = _as_string_list(metrics.get("evidence_gaps"))
        if report.get("verdict") in {"needs_more_data", "blocked"}:
            gaps.extend(_as_string_list(report.get("reasons")))
        return dimensions, _dedupe(gaps)

    def _build_indicator_grid(
        self,
        *,
        domain_id: str,
        as_of: str,
        snapshot: PipelineMetricSnapshot | None,
        structured_audit: dict[str, Any],
    ) -> IndicatorStateGrid:
        observations: list[IndicatorObservation] = []
        warnings: list[str] = []

        if snapshot is not None:
            observations.extend(_pipeline_observations(snapshot, as_of))
            warnings.extend(snapshot.warnings)

        for item in structured_audit.get("accepted_observations", []) or []:
            if not isinstance(item, dict):
                continue
            observations.append(
                IndicatorObservation(
                    indicator_id=str(item.get("observation_sha256") or _stable_id(item)),
                    family=str(item.get("family") or "structured_context"),
                    scope=str(item.get("scope") or "global"),
                    name=str(item.get("name") or "unknown"),
                    value=item.get("value"),
                    unit=_string_or_none(item.get("unit")),
                    period=_string_or_none(item.get("period")),
                    available_at=_string_or_none(item.get("available_at")),
                    as_of=as_of,
                    source=str(item.get("source_locator") or "structured_context"),
                    evidence_status="point_in_time",
                    quality_score=1.0,
                    provenance={
                        key: value
                        for key, value in item.items()
                        if key not in {"value"}
                    },
                )
            )

        family_counts: dict[str, int] = {}
        for item in observations:
            family_counts[item.family] = family_counts.get(item.family, 0) + 1
        required_families = {"pipeline_profitability", "pipeline_risk", "pipeline_validation", "macro", "sector"}
        missing_families = sorted(required_families - set(family_counts))
        status = "ready" if observations and not missing_families else "partial" if observations else "needs_evidence"
        if missing_families:
            warnings.append(
                "Missing indicator families: " + ", ".join(missing_families)
            )
        return IndicatorStateGrid(
            domain_id=domain_id,
            as_of=as_of,
            status=status,
            observations=observations,
            family_counts=dict(sorted(family_counts.items())),
            missing_families=missing_families,
            point_in_time={
                "structured_context_status": structured_audit.get("status"),
                "accepted_structured_observations": structured_audit.get("accepted_count", 0),
                "excluded_structured_observations": structured_audit.get("excluded_count", 0),
                "pipeline_metric_as_of": snapshot.identity.as_of if snapshot else None,
            },
            warnings=_dedupe(warnings),
        )

    @staticmethod
    def _snapshot(value: PipelineMetricSnapshot | dict[str, Any] | None) -> PipelineMetricSnapshot | None:
        if value is None:
            return None
        if isinstance(value, PipelineMetricSnapshot):
            return value
        if isinstance(value, dict):
            return PipelineMetricSnapshot.model_validate(value)
        raise TypeError("pipeline_metric_snapshot must be a PipelineMetricSnapshot or mapping")


def _pipeline_observations(snapshot: PipelineMetricSnapshot, as_of: str) -> list[IndicatorObservation]:
    groups = {
        "pipeline_profitability": snapshot.profitability.model_dump(mode="json"),
        "pipeline_risk": snapshot.risk.model_dump(mode="json"),
        "pipeline_validation": snapshot.validation.model_dump(mode="json"),
        "pipeline_feature_stability": snapshot.feature_stability.model_dump(mode="json", exclude={"feature_importance", "unstable_features"}),
        "pipeline_data_quality": snapshot.data_quality.model_dump(mode="json", exclude={"warnings", "leakage_flags"}),
        "pipeline_replay": snapshot.replay.model_dump(mode="json"),
    }
    observations: list[IndicatorObservation] = []
    for family, values in groups.items():
        for name, value in values.items():
            if value is None:
                continue
            observations.append(
                IndicatorObservation(
                    indicator_id=_stable_id(
                        {
                            "run_id": snapshot.identity.run_id,
                            "family": family,
                            "name": name,
                            "value": value,
                        }
                    ),
                    family=family,
                    scope="|".join(snapshot.identity.tickers) or "pipeline_run",
                    name=name,
                    value=value,
                    unit=_pipeline_metric_unit(name),
                    available_at=snapshot.identity.as_of,
                    as_of=as_of,
                    source="dean_pipeline_metric_snapshot_v1",
                    evidence_status="supporting_review_only",
                    quality_score=snapshot.completeness.coverage_ratio,
                    provenance={
                        "pipeline_run_id": snapshot.identity.run_id,
                        "pipeline_status": snapshot.identity.pipeline_status,
                        "model_name": snapshot.identity.model_name,
                        "target_name": snapshot.identity.target_name,
                        "context_fingerprint": snapshot.identity.context_fingerprint,
                    },
                )
            )
    return observations


def _pipeline_metric_unit(name: str) -> str | None:
    if name in {
        "total_return",
        "win_rate",
        "max_drawdown",
        "volatility",
        "var_95",
        "expected_shortfall_95",
        "gross_exposure",
        "missing_ratio",
        "duplicate_ratio",
        "clear_hit_rate",
        "average_realized_return",
        "feature_concentration",
        "feature_stability_score",
        "train_score",
        "validation_score",
        "test_score",
        "train_test_gap",
    }:
        return "ratio"
    if name in {
        "total_trades",
        "sample_count",
        "train_sample_count",
        "validation_sample_count",
        "test_sample_count",
        "walk_forward_fold_count",
        "feature_count",
        "unstable_feature_count",
        "warning_count",
        "leakage_flag_count",
        "clear_evaluated_runs",
        "quality_blocked_runs",
        "replay_window_count",
    }:
        return "count"
    if name == "freshness_hours":
        return "hours"
    return None


def _find_domain_report(reports: list[dict[str, Any]], domain_id: str) -> dict[str, Any] | None:
    for report in reports:
        payload = report.get("analysis_payload")
        if isinstance(payload, dict) and payload.get("domain_id") == domain_id:
            return report
        if report.get("asset_or_sector") == domain_id:
            return report
        metrics = report.get("metrics_snapshot")
        if isinstance(metrics, dict) and metrics.get("domain_id") == domain_id:
            return report
    return None


def _dimension_map(raw: Any, as_of: str, source: str) -> dict[str, ContextDimensionState]:
    if not isinstance(raw, dict):
        return {}
    return {
        str(name): _dimension_state(str(name), value, as_of=as_of, default_source=source)
        for name, value in raw.items()
    }


def _dimension_state(
    name: str,
    raw: Any,
    *,
    as_of: str,
    default_source: str,
) -> ContextDimensionState:
    if isinstance(raw, dict):
        state = raw.get("state") or raw.get("status") or raw.get("regime") or raw.get("label") or raw.get("summary")
        if state is None and raw:
            state = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
        direction = str(raw.get("direction") or "unknown").lower()
        if direction not in {"improving", "deteriorating", "stable", "mixed", "unknown"}:
            direction = "unknown"
        return ContextDimensionState(
            dimension=name,
            state=str(state or "unknown"),
            direction=direction,  # type: ignore[arg-type]
            confidence=_unit_interval(raw.get("confidence")),
            as_of=str(raw.get("as_of") or as_of),
            source=str(raw.get("source") or default_source),
            evidence_ids=_as_string_list(raw.get("evidence_ids")),
            notes=_as_string_list(raw.get("notes")),
        )
    if raw not in (None, "", [], {}):
        return ContextDimensionState(
            dimension=name,
            state=str(raw),
            confidence=0.5,
            as_of=as_of,
            source=default_source,
        )
    return ContextDimensionState(
        dimension=name,
        state="unknown",
        confidence=0.0,
        as_of=as_of,
        source="missing",
    )


def _context_scope_items(metadata: dict[str, Any], list_key: str, mapping_key: str) -> list[dict[str, Any]]:
    raw = metadata.get(list_key)
    if raw is None:
        raw = metadata.get(mapping_key)
    items: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            payload = dict(value) if isinstance(value, dict) else {"dimensions": {"state": value}}
            payload.setdefault("id", str(key))
            payload.setdefault("label", str(key))
            items.append(payload)
    elif isinstance(raw, list):
        for value in raw:
            if isinstance(value, dict):
                payload = dict(value)
                identity = payload.get("id") or payload.get("code") or payload.get("name") or payload.get("label")
                if identity:
                    payload.setdefault("id", str(identity))
                    payload.setdefault("label", str(payload.get("name") or identity))
                    items.append(payload)
            elif str(value).strip():
                items.append({"id": str(value), "label": str(value)})
    return items


def _news_evidence_inventory(records: list[Any]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        semantic = item.get("_dean_semantic_evidence") if isinstance(item.get("_dean_semantic_evidence"), dict) else {}
        evidence_id = str(
            semantic.get("candidate_sha256")
            or item.get("evidence_id")
            or item.get("url")
            or _stable_id({"title": item.get("title"), "published_at": item.get("published_at")})
        )
        inventory.append(
            {
                "evidence_id": evidence_id,
                "domain_id": semantic.get("domain_id"),
                "evidence_type": semantic.get("evidence_type"),
                "directness": semantic.get("directness") or item.get("directness"),
                "tickers": _as_string_list(item.get("tickers")),
                "sectors": _as_string_list(item.get("sectors")),
            }
        )
    return inventory


def _evidence_ids(
    inventory: list[dict[str, Any]],
    *,
    domain_id: str | None = None,
    directness: set[str] | None = None,
) -> list[str]:
    output: list[str] = []
    for item in inventory:
        if domain_id and item.get("domain_id") not in {None, domain_id}:
            continue
        if directness and str(item.get("directness") or "") not in directness:
            continue
        output.append(str(item["evidence_id"]))
    return _dedupe(output)


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _unit_interval(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(number, 1.0))


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return _dedupe(str(item).strip() for item in values if str(item).strip())


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_id(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slug(value: Any) -> str:
    return "_".join(str(value).strip().lower().replace("/", " ").replace("-", " ").split())


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


__all__ = [
    "ContextDimensionState",
    "ContextGrid",
    "ContextGridEdge",
    "ContextGridNode",
    "ContextIndicatorGridBuilder",
    "ContextIndicatorPacket",
    "IndicatorObservation",
    "IndicatorStateGrid",
]
