from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport


PIPELINE_PRICE_TYPES = {"market_data", "yahoo_finance"}
PIPELINE_NEWS_TYPES = {"google_news", "newsapi", "rss"}
PIPELINE_MACRO_TYPES = {"fred", "economic_calendar"}
PIPELINE_CONTEXT_TYPES = {
    "aaii_sentiment",
    "alternative_me",
    "fear_greed",
    "free_google_trends",
    "put_call_ratio",
    "reddit_sentiment",
    "vix",
}
RESEARCH_SPECIALIST_TYPES = {"bigquery", "cftc", "custom_csv", "huggingface", "insider", "sec_filings"}


class CollectorInventoryAgent(BaseAgent):
    """Maps collector configs to local collector classes without running network calls."""

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        project_root = Path(self.config.get("project_root", ".")).resolve()
        config_path = _project_path(project_root, self.config.get("config_path", "src/config/collectors.yaml"))
        collectors_dir = _project_path(project_root, self.config.get("collectors_dir", "src/data/collectors"))

        inventory = inspect_collector_inventory(config_path=config_path, collectors_dir=collectors_dir)
        context.metadata["collector_inventory"] = inventory

        summary = inventory.get("summary", {})
        enabled_missing = summary.get("enabled_missing_classes", [])
        scan_error_count = len(inventory.get("scan_errors", []))

        if summary.get("status") == "unavailable":
            verdict = "blocked"
            reasons = [summary.get("reason", "Collector inventory is unavailable.")]
            risks = ["Collector health cannot be triaged until configs/classes are readable."]
            quality_score = 0.2
            signal_strength = -0.8
        elif enabled_missing:
            verdict = "blocked"
            reasons = [f"Enabled collectors have no discovered class: {', '.join(enabled_missing)}"]
            risks = ["Stage 1 may silently skip enabled data sources or fail during collector instantiation."]
            quality_score = 0.45
            signal_strength = -0.7
        elif scan_error_count:
            verdict = "caution"
            reasons = [f"Collector source scan had {scan_error_count} parse/read errors."]
            risks = ["Some collector files could not be inventoried safely."]
            quality_score = 0.65
            signal_strength = -0.2
        else:
            verdict = "clear"
            reasons = ["Collector inventory scan completed without enabled class gaps."]
            risks = []
            quality_score = 0.9
            signal_strength = 0.2

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.88,
            data_quality_score=quality_score,
            signal_strength=signal_strength,
            reasons=reasons,
            risks=risks,
            blind_spots=[
                "Inventory only reads local configs/classes; it does not test API keys, network reachability, schema quality, or rate limits."
            ],
            evidence=[
                self.evidence("config", str(config_path), "configured_count", summary.get("configured_count", 0)),
                self.evidence("file", str(collectors_dir), "discovered_class_count", summary.get("discovered_class_count", 0)),
                self.evidence("metric", "collector_inventory", "enabled_missing_classes", enabled_missing),
                self.evidence("metric", "collector_inventory", "rss_pipeline_status", inventory.get("rss_pipeline_status")),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot=inventory,
        )


def inspect_collector_inventory(config_path: str | Path, collectors_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    collectors_dir = Path(collectors_dir)
    class_map, scan_errors, duplicate_classes = discover_collector_classes(collectors_dir)

    config_result = _load_collectors_config(config_path)
    if config_result.get("status") == "unavailable":
        return {
            "config_path": str(config_path),
            "collectors_dir": str(collectors_dir),
            "configured_collectors": [],
            "unconfigured_classes": sorted(class_map),
            "scan_errors": scan_errors,
            "duplicate_classes": duplicate_classes,
            "rss_pipeline_status": None,
            "recommendations": config_result.get("recommendations", []),
            "summary": {
                "status": "unavailable",
                "reason": config_result.get("reason"),
                "configured_count": 0,
                "enabled_count": 0,
                "discovered_class_count": len(class_map),
                "enabled_missing_classes": [],
            },
        }

    collector_configs = config_result["collectors"]
    configured_records = [
        _build_collector_record(config_name, config, class_map)
        for config_name, config in sorted(collector_configs.items())
        if isinstance(config, dict)
    ]
    configured_types = {str(record["type"]) for record in configured_records}
    unconfigured_classes = sorted(collector_type for collector_type in class_map if collector_type not in configured_types)
    summary = _build_summary(configured_records, class_map)
    rss_pipeline_status = _rss_pipeline_status(configured_records)

    return {
        "config_path": str(config_path),
        "collectors_dir": str(collectors_dir),
        "configured_collectors": configured_records,
        "unconfigured_classes": unconfigured_classes,
        "scan_errors": scan_errors,
        "duplicate_classes": duplicate_classes,
        "rss_pipeline_status": rss_pipeline_status,
        "recommendations": _build_recommendations(configured_records, summary, rss_pipeline_status, scan_errors),
        "summary": summary,
    }


def discover_collector_classes(collectors_dir: str | Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    collectors_dir = Path(collectors_dir)
    class_map: dict[str, dict[str, Any]] = {}
    scan_errors: list[dict[str, Any]] = []
    duplicate_classes: list[dict[str, Any]] = []

    if not collectors_dir.exists():
        return class_map, [{"path": str(collectors_dir), "error": "Collectors directory does not exist."}], duplicate_classes

    for path in sorted(collectors_dir.glob("*.py")):
        if path.name.startswith("__"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            scan_errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            attrs = _class_literal_attrs(node)
            collector_type = attrs.get("collector_type")
            if not collector_type or collector_type == "default":
                continue
            info = {
                "collector_type": collector_type,
                "class_name": node.name,
                "module_path": str(path),
                "line": node.lineno,
                "data_type": attrs.get("data_type"),
            }
            if collector_type in class_map:
                duplicate_classes.append(
                    {
                        "collector_type": collector_type,
                        "previous": class_map[collector_type],
                        "replacement": info,
                    }
                )
            class_map[collector_type] = info
    return class_map, scan_errors, duplicate_classes


def _build_collector_record(config_name: str, config: dict[str, Any], class_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    collector_type = str(config.get("type") or config_name)
    class_info = class_map.get(collector_type)
    data_type = config.get("data_type") or (class_info or {}).get("data_type")
    enabled = bool(config.get("enabled", False))
    critical = bool(config.get("critical", False))
    requires_api_key = bool(config.get("requires_api_key", False))
    recommended_use = _recommended_use(collector_type, data_type, critical)

    record = {
        "name": config_name,
        "type": collector_type,
        "enabled": enabled,
        "critical": critical,
        "data_type": data_type,
        "table_name": config.get("table_name"),
        "cache_ttl": config.get("cache_ttl"),
        "cache_duration_minutes": config.get("cache_duration_minutes"),
        "requires_api_key": requires_api_key,
        "api_key_name": config.get("api_key_name") if requires_api_key else None,
        "class_found": class_info is not None,
        "class_name": (class_info or {}).get("class_name"),
        "module_path": (class_info or {}).get("module_path"),
        "recommended_use": recommended_use,
        "schedule_hint": _schedule_hint(collector_type, recommended_use),
        "repair_priority": _repair_priority(enabled, critical, requires_api_key, class_info is not None, recommended_use),
        "notes": _collector_notes(config_name, collector_type, enabled, requires_api_key, class_info is not None, recommended_use),
    }
    return record


def _load_collectors_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {
            "status": "unavailable",
            "reason": f"Collectors config does not exist: {config_path}",
            "recommendations": ["Create or point --config-path to src/config/collectors.yaml before collector triage."],
        }
    try:
        import yaml
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "PyYAML is not available, so collectors.yaml cannot be parsed.",
            "recommendations": ["Install PyYAML or provide a JSON-compatible config loader for collector inventory."],
        }
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"Could not read collectors config: {type(exc).__name__}: {exc}",
            "recommendations": ["Fix YAML syntax before running collector inventory."],
        }
    collectors = payload.get("collectors", payload)
    if not isinstance(collectors, dict):
        return {
            "status": "unavailable",
            "reason": "Collectors config did not resolve to a mapping.",
            "recommendations": ["Expected a mapping at top-level key 'collectors'."],
        }
    return {"status": "ok", "collectors": collectors}


def _class_literal_attrs(node: ast.ClassDef) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            value = _literal_str(statement.value)
            if value is None:
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    attrs[target.id] = value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            value = _literal_str(statement.value)
            if value is not None:
                attrs[statement.target.id] = value
    return attrs


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _recommended_use(collector_type: str, data_type: Any, critical: bool) -> str:
    data_type = str(data_type or "")
    if critical or collector_type in PIPELINE_PRICE_TYPES:
        return "pipeline_price_feed"
    if data_type == "news" or collector_type in PIPELINE_NEWS_TYPES:
        return "pipeline_news_feed"
    if data_type in {"macro", "macro_data"} or collector_type in PIPELINE_MACRO_TYPES:
        return "pipeline_macro_feed"
    if collector_type in RESEARCH_SPECIALIST_TYPES:
        return "research_specialist_feed"
    if collector_type in PIPELINE_CONTEXT_TYPES or data_type == "alternative":
        return "pipeline_context_feed"
    return "optional_manual_feed"


def _schedule_hint(collector_type: str, recommended_use: str) -> str:
    if collector_type == "sec_filings":
        return "research: on-demand by ticker/reporting calendar, or daily off-hours; not an intraday blocker"
    if collector_type == "insider":
        return "research: daily/off-hours review feed"
    if recommended_use == "pipeline_price_feed":
        return "pipeline: per approved market-data refresh and timeframe cadence"
    if recommended_use == "pipeline_news_feed":
        return "pipeline: hourly/intraday collection, then timestamp alignment to candles"
    if recommended_use == "pipeline_macro_feed":
        return "pipeline: daily or release-calendar cadence"
    if recommended_use == "pipeline_context_feed":
        return "pipeline/context: daily or pre-market, depending on source stability"
    if recommended_use == "research_specialist_feed":
        return "research: isolated test first, then on-demand or low-frequency scheduled ingestion"
    return "manual: run only when a downstream consumer is defined"


def _repair_priority(
    enabled: bool,
    critical: bool,
    requires_api_key: bool,
    class_found: bool,
    recommended_use: str,
) -> str:
    if enabled and not class_found:
        return "P0"
    if enabled and critical:
        return "P0"
    if enabled and recommended_use in {"pipeline_news_feed", "pipeline_macro_feed", "pipeline_context_feed"}:
        return "P1"
    if enabled and requires_api_key:
        return "P1"
    if recommended_use == "research_specialist_feed":
        return "P2" if enabled else "P3"
    return "P3"


def _collector_notes(
    config_name: str,
    collector_type: str,
    enabled: bool,
    requires_api_key: bool,
    class_found: bool,
    recommended_use: str,
) -> list[str]:
    notes: list[str] = []
    if not enabled:
        notes.append("Disabled in collectors config.")
    if not class_found:
        notes.append("No local collector class discovered for this type.")
    if requires_api_key:
        notes.append("Requires an API key; inventory does not inspect secret values.")
    if config_name == "rss" or collector_type == "rss":
        notes.append("RSS is a pipeline news feed when enabled because Stage 1 groups data_type=news sources for candle alignment.")
    if collector_type == "sec_filings":
        notes.append("SEC filings are best treated as research-specialist evidence first: filing -> ResearchDocument -> citations -> specialist patterns.")
    if recommended_use == "pipeline_news_feed":
        notes.append("Validate timestamp columns, deduplication, and publication-time integrity before trusting event studies.")
    return notes


def _build_summary(records: list[dict[str, Any]], class_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enabled_records = [record for record in records if record["enabled"]]
    role_counts: dict[str, int] = {}
    enabled_role_counts: dict[str, int] = {}
    for record in records:
        role_counts[record["recommended_use"]] = role_counts.get(record["recommended_use"], 0) + 1
        if record["enabled"]:
            enabled_role_counts[record["recommended_use"]] = enabled_role_counts.get(record["recommended_use"], 0) + 1

    return {
        "status": "ok",
        "configured_count": len(records),
        "enabled_count": len(enabled_records),
        "disabled_count": len(records) - len(enabled_records),
        "discovered_class_count": len(class_map),
        "class_found_count": sum(1 for record in records if record["class_found"]),
        "missing_class_count": sum(1 for record in records if not record["class_found"]),
        "enabled_missing_classes": [record["name"] for record in enabled_records if not record["class_found"]],
        "enabled_api_key_collectors": [record["name"] for record in enabled_records if record["requires_api_key"]],
        "role_counts": role_counts,
        "enabled_role_counts": enabled_role_counts,
        "pipeline_feeds_enabled": [
            record["name"] for record in enabled_records if str(record["recommended_use"]).startswith("pipeline_")
        ],
        "research_specialist_feeds": [record["name"] for record in records if record["recommended_use"] == "research_specialist_feed"],
        "priority_counts": _count_by(records, "repair_priority"),
    }


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _rss_pipeline_status(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        if record["name"] == "rss" or record["type"] == "rss":
            return {
                "enabled": record["enabled"],
                "class_found": record["class_found"],
                "recommended_use": record["recommended_use"],
                "table_name": record["table_name"],
                "schedule_hint": record["schedule_hint"],
            }
    return None


def _build_recommendations(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    rss_pipeline_status: dict[str, Any] | None,
    scan_errors: list[dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []
    missing = summary.get("enabled_missing_classes", [])
    if missing:
        recommendations.append(f"Fix enabled collector class mappings first: {', '.join(missing)}.")
    elif records:
        recommendations.append("Use this inventory as a safe preflight before isolated collector health tests.")

    if rss_pipeline_status:
        if rss_pipeline_status["enabled"] and rss_pipeline_status["class_found"]:
            recommendations.append("RSS is configured as an enabled pipeline_news_feed; test it next for timestamps, dedupe, and candle alignment.")
        elif not rss_pipeline_status["enabled"]:
            recommendations.append("RSS exists but is disabled; enable only when the news-event dataset path is ready.")
        else:
            recommendations.append("RSS is configured but no local RSS collector class was discovered.")

    newsapi = _find_record(records, "newsapi")
    if newsapi and newsapi["enabled"] and newsapi["requires_api_key"]:
        recommendations.append("NewsAPI is enabled and requires NEWS_API_KEY; validate key presence in an isolated health test, not in inventory output.")

    sec = _find_record(records, "sec_filings")
    if sec:
        recommendations.append("Keep SEC filings on the research-specialist path first: isolated SEC test, then ResearchCorpus ingestion; do not make it a daily pipeline blocker yet.")

    if scan_errors:
        recommendations.append("Resolve collector source parse/read errors before trusting missing-class counts.")
    return recommendations


def _find_record(records: list[dict[str, Any]], name_or_type: str) -> dict[str, Any] | None:
    for record in records:
        if record["name"] == name_or_type or record["type"] == name_or_type:
            return record
    return None


def _project_path(project_root: Path, relative_or_absolute: str | Path) -> Path:
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return project_root / path
