from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.material_loaders import SUPPORTED_EXTENSIONS
from dean_os.schemas import MarketContext, PipelineReport


RESEARCH_CORPUS_TYPES = {"article", "book", "filing", "news", "report", "transcript"}
SPECIALIST_AGENT_ROUTES = {
    "financial_nlp": {"news", "article", "filing", "transcript", "report"},
    "specialist_research": {"article", "book", "filing", "news", "report", "transcript"},
    "evidence_synthesis": {"article", "book", "filing", "news", "report", "transcript"},
    "macro_policy": {"news", "report"},
    "news_catalyst": {"news", "article"},
    "sector_cycle": {"article", "report", "news"},
    "value_screening": {"filing", "report", "book"},
    "historical_analogies": {"book", "article", "report"},
}


class SourceRoutingAgent(BaseAgent):
    """Maps local data/research sources to pipeline or specialist-agent intake paths."""

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        routing = inspect_source_routing(
            materials_path=self.config.get("materials_path"),
            collector_inventory_path=self.config.get("collector_inventory_path"),
            collector_inventory=self.config.get("collector_inventory"),
        )
        context.metadata["source_routing"] = routing

        route_count = routing["summary"]["routable_source_count"]
        warning_count = len(routing["warnings"])
        if route_count == 0:
            verdict = "needs_more_data"
            reasons = ["No routable source inputs were found for pipeline or specialist agents."]
            quality_score = 0.25
            signal_strength = -0.2
        elif warning_count:
            verdict = "caution"
            reasons = [f"Source routing found {route_count} routable inputs with {warning_count} warning(s)."]
            quality_score = 0.6
            signal_strength = 0.0
        else:
            verdict = "clear"
            reasons = [f"Source routing found {route_count} routable inputs."]
            quality_score = 0.85
            signal_strength = 0.1

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.82,
            data_quality_score=quality_score,
            signal_strength=signal_strength,
            reasons=reasons,
            risks=[
                "Routing does not prove source quality; ingestion, schema checks, timestamps, and citations still need validation."
            ],
            blind_spots=[
                "This agent is local-only. It does not fetch missing sources, call APIs, or verify paid data availability."
            ],
            evidence=[
                self.evidence("metric", "source_routing", "summary", routing["summary"]),
                self.evidence("document", "source_routing.materials", "materials_by_type", routing["materials"]["by_source_type"]),
                self.evidence("config", "source_routing.collectors", "feeds_by_route", routing["collectors"]["by_route"]),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot=routing,
        )


def inspect_source_routing(
    materials_path: str | Path | None = None,
    collector_inventory_path: str | Path | None = None,
    collector_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    material_records, material_warnings = _inspect_materials(materials_path)
    inventory = _load_collector_inventory(collector_inventory_path, collector_inventory)
    collector_records = _inspect_collectors(inventory)
    analyst_inputs = _analyst_inputs(material_records, collector_records)
    warnings = [*material_warnings]
    if collector_inventory_path and inventory is None:
        warnings.append(f"Collector inventory JSON could not be read: {collector_inventory_path}")

    recommendations = _recommendations(material_records, collector_records, analyst_inputs, warnings)
    routable_count = len(material_records) + len(collector_records)
    return {
        "materials_path": str(materials_path) if materials_path else None,
        "collector_inventory_path": str(collector_inventory_path) if collector_inventory_path else None,
        "materials": {
            "document_count": len(material_records),
            "by_source_type": dict(sorted(Counter(item["source_type"] for item in material_records).items())),
            "records": material_records,
        },
        "collectors": {
            "collector_count": len(collector_records),
            "by_route": dict(sorted(Counter(item["route"] for item in collector_records).items())),
            "records": collector_records,
        },
        "analyst_inputs": analyst_inputs,
        "warnings": warnings,
        "recommendations": recommendations,
        "summary": {
            "routable_source_count": routable_count,
            "research_corpus_document_count": sum(1 for item in material_records if item["route"] == "research_corpus"),
            "pipeline_feed_count": sum(1 for item in collector_records if item["route"].startswith("pipeline_")),
            "research_specialist_feed_count": sum(1 for item in collector_records if item["route"] == "research_specialist_feed"),
            "warning_count": len(warnings),
        },
    }


def _inspect_materials(materials_path: str | Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    if not materials_path:
        return [], []
    root = Path(materials_path)
    if not root.exists():
        return [], [f"Materials path does not exist: {root}"]
    paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    records = []
    warnings = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            warnings.append(f"Unsupported material extension skipped: {path}")
            continue
        source_type = _infer_source_type(path)
        records.append(
            {
                "path": str(path),
                "source_type": source_type,
                "route": "research_corpus",
                "target": "ResearchCorpus",
                "suggested_agents": _agents_for_source_type(source_type),
                "tags_hint": _tags_hint(path, source_type),
            }
        )
    return records, warnings


def _load_collector_inventory(
    collector_inventory_path: str | Path | None,
    collector_inventory: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if collector_inventory:
        return collector_inventory
    if not collector_inventory_path:
        return None
    try:
        return json.loads(Path(collector_inventory_path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _inspect_collectors(inventory: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not inventory:
        return []
    configured = inventory.get("configured_collectors", [])
    records = []
    for item in configured:
        if not isinstance(item, dict):
            continue
        route = str(item.get("recommended_use") or "unknown")
        records.append(
            {
                "name": item.get("name"),
                "type": item.get("type"),
                "enabled": bool(item.get("enabled")),
                "data_type": item.get("data_type"),
                "route": route,
                "requires_api_key": bool(item.get("requires_api_key")),
                "class_found": bool(item.get("class_found")),
                "suggested_agents": _agents_for_collector_route(route, item),
                "next_check": _collector_next_check(route, item),
            }
        )
    return records


def _analyst_inputs(material_records: list[dict[str, Any]], collector_records: list[dict[str, Any]]) -> dict[str, Any]:
    inputs: dict[str, dict[str, Any]] = defaultdict(lambda: {"material_count": 0, "collector_count": 0, "source_types": [], "collector_types": []})
    for item in material_records:
        for agent in item["suggested_agents"]:
            inputs[agent]["material_count"] += 1
            inputs[agent]["source_types"].append(item["source_type"])
    for item in collector_records:
        for agent in item["suggested_agents"]:
            inputs[agent]["collector_count"] += 1
            inputs[agent]["collector_types"].append(item["type"])
    return {
        agent: {
            "material_count": payload["material_count"],
            "collector_count": payload["collector_count"],
            "source_types": sorted(set(payload["source_types"])),
            "collector_types": sorted(set(str(value) for value in payload["collector_types"] if value)),
        }
        for agent, payload in sorted(inputs.items())
    }


def _recommendations(
    material_records: list[dict[str, Any]],
    collector_records: list[dict[str, Any]],
    analyst_inputs: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    recommendations = []
    if material_records:
        recommendations.append("Run Agent Lab or research ingest on routed research_corpus materials before specialist review.")
    if any(item["route"] == "pipeline_news_feed" and item["enabled"] for item in collector_records):
        recommendations.append("Run isolated news collector health before using news feeds for candle/sentiment/event-study inputs.")
    if any(item["route"] == "research_specialist_feed" for item in collector_records):
        recommendations.append("Route filings/transcripts/insider-style feeds into ResearchCorpus before specialist synthesis.")
    if "value_screening" not in analyst_inputs:
        recommendations.append("Add filings, annual reports, or fundamental reports before expecting value-screening depth.")
    if "macro_policy" not in analyst_inputs:
        recommendations.append("Add macro reports/news or stable macro feeds before expecting macro-policy depth.")
    if warnings:
        recommendations.append("Resolve source-routing warnings before treating coverage as complete.")
    return recommendations or ["Source coverage looks sufficient for the current local routing snapshot."]


def _infer_source_type(path: Path) -> str:
    lower_name = path.name.lower()
    if "10-k" in lower_name or "10-q" in lower_name or "filing" in lower_name:
        return "filing"
    if "transcript" in lower_name or "earnings-call" in lower_name:
        return "transcript"
    if "book" in lower_name or "chapter" in lower_name:
        return "book"
    if "news" in lower_name:
        return "news"
    if "report" in lower_name or path.suffix.lower() in {".pdf", ".docx"}:
        return "report"
    return "article"


def _agents_for_source_type(source_type: str) -> list[str]:
    return sorted(agent for agent, source_types in SPECIALIST_AGENT_ROUTES.items() if source_type in source_types)


def _agents_for_collector_route(route: str, item: dict[str, Any]) -> list[str]:
    if route == "pipeline_news_feed":
        return ["news_catalyst", "financial_nlp", "macro_policy"]
    if route == "pipeline_macro_feed":
        return ["macro_policy", "sector_cycle"]
    if route == "pipeline_context_feed":
        return ["regime", "chief_review", "tuning"]
    if route == "pipeline_price_feed":
        return ["market_data_freshness", "regime", "model_performance", "paper_trading"]
    if route == "research_specialist_feed":
        collector_type = str(item.get("type") or "")
        if collector_type == "sec_filings":
            return ["value_screening", "specialist_research", "evidence_synthesis"]
        return ["specialist_research", "evidence_synthesis"]
    return []


def _collector_next_check(route: str, item: dict[str, Any]) -> str:
    if not item.get("enabled"):
        return "disabled_in_config"
    if item.get("requires_api_key"):
        return "check_api_key_and_rate_limits"
    if route.startswith("pipeline_"):
        return "run_isolated_health_check"
    if route == "research_specialist_feed":
        return "ingest_into_research_corpus_first"
    return "manual_review"


def _tags_hint(path: Path, source_type: str) -> list[str]:
    tags = [source_type]
    lower = path.name.lower()
    for token in ("ai", "energy", "defense", "ipo", "semiconductor", "macro", "fed", "rates", "earnings"):
        if token in lower:
            tags.append(token)
    return sorted(set(tags))
