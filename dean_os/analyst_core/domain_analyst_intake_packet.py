from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.analyst_core.analyst_evidence_pack import documents_from_evidence_pack
from dean_os.analysts import AnalystEvidenceItem, BaseAnalystAgent, get_domain_profile
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import ResearchDocument, utc_now_iso
from dean_os.utils import clamp, json_ready

DEFAULT_EVIDENCE_PACK_JSON = "reports/dean_os/analyst_evidence_pack_cached_source_current/latest.json"
DEFAULT_SOURCE_GATE_JSON = "reports/dean_os/source_evidence_validation_gate_cached_source_current/latest.json"
DEFAULT_DOMAIN_ID = "semiconductor_ai_infrastructure"

EVIDENCE_TYPE_ALIASES: dict[str, list[str]] = {
    "sector_demand": [
        "demand",
        "ai infrastructure",
        "accelerator",
        "gpu",
        "data center",
        "semiconductor cycle",
        "orders",
        "backlog",
    ],
    "capex_cycle": [
        "capex",
        "capital expenditure",
        "capital spending",
        "ai spending",
        "spending on artificial intelligence",
        "hyperscaler",
        "hyperscale",
        "cloud spending",
        "cloud capex",
        "data center buildout",
        "data center build-out",
        "data center investment",
        "data center spending",
        "data centre investment",
        "infrastructure spending",
    ],
    "supply_chain": ["supply chain", "foundry", "capacity", "advanced packaging", "hbm", "wafer", "memory", "inventory"],
    "policy_or_geopolitical": [
        "export control",
        "sanction",
        "tariff",
        "china",
        "taiwan",
        "geopolitical",
        "regulation",
        "policy",
    ],
    "market_confirmation": [
        "market",
        "shares",
        "stock",
        "relative strength",
        "price action",
        "outperform",
        "underperform",
        "revision",
    ],
    "inflation": ["inflation", "cpi", "pce", "prices"],
    "rates_policy": ["rates", "fed", "central bank", "monetary policy", "yield"],
    "growth": ["growth", "gdp", "activity", "recession", "expansion"],
    "labor_market": ["labor", "jobs", "payroll", "unemployment", "wages"],
    "liquidity_conditions": ["liquidity", "reserves", "repo", "money market"],
    "credit_spreads": ["credit spread", "spreads", "high yield", "investment grade"],
    "funding_stress": ["funding", "stress", "banking", "repo"],
    "geopolitical_event": ["conflict", "war", "election", "sanction", "tariff"],
    "policy_or_sanctions": ["policy", "sanction", "tariff", "export control"],
    "exposure_mapping": ["exposure", "supply chain", "region", "country", "revenue exposure"],
    "supply": ["supply", "production", "opec", "rig", "inventory"],
    "demand": ["demand", "consumption", "travel", "industrial"],
    "inventories": ["inventory", "stockpile", "storage"],
    "data_traffic": ["data traffic", "bandwidth", "network usage", "streaming growth", "data consumption"],
    "telecom_capex": ["telecom capex", "network buildout", "5g investment", "fiber deployment", "spectrum auction"],
    "subscriber_additions": ["subscriber", "subscriber growth", "net adds", "churn rate", "arpu"],
    "consumer_spending": ["consumer spending", "retail sales", "discretionary spending", "consumer demand"],
    "auto_sales": ["auto sales", "vehicle sales", "car sales", "automotive demand"],
    "retail_inventories": ["retail inventory", "inventory levels", "destocking", "restocking"],
    "sales_volumes": ["sales volume", "unit sales", "volume growth"],
    "pricing_power": ["pricing power", "price increase", "price hikes", "margin expansion"],
    "input_costs": ["input costs", "commodity costs", "raw material costs", "cost inflation"],
    "trial_readouts": ["trial readout", "clinical trial", "phase 3", "phase 2", "trial results", "efficacy data"],
    "fda_decisions": ["fda", "fda approval", "fda decision", "pdufa", "regulatory approval"],
    "patent_cliffs": ["patent cliff", "patent expiration", "generic competition", "loss of exclusivity"],
    "pmi": ["pmi", "purchasing managers index", "manufacturing pmi", "ism index"],
    "new_orders": ["new orders", "order backlog", "bookings"],
    "industrial_production": ["industrial production", "factory output", "manufacturing output"],
    "demand_manufacturing": ["manufacturing demand", "industrial demand", "steel demand", "construction demand"],
    "supply_mining": ["mine supply", "mining output", "ore production", "mine disruption"],
    "electricity_load": ["electricity demand", "power demand", "load growth", "peak demand"],
    "generation_mix": ["generation mix", "renewable capacity", "coal retirement", "solar capacity", "wind capacity", "nuclear output"],
    "power_prices": ["power prices", "electricity prices", "wholesale power", "capacity prices"],
}
EVIDENCE_TYPE_PRIORITY: dict[str, int] = {
    "policy_or_geopolitical": 50,
    "policy_or_sanctions": 50,
    "geopolitical_event": 45,
    "capex_cycle": 40,
    "supply_chain": 30,
    "sector_demand": 20,
    "market_confirmation": 10,
}
STRONG_EVIDENCE_TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "capex_cycle": (
        "capex",
        "capital expenditure",
        "capital spending",
        "ai spending",
        "spending on artificial intelligence",
        "cloud spending",
        "cloud capex",
        "data center buildout",
        "data center build-out",
        "data center investment",
        "data center spending",
        "data centre investment",
        "infrastructure spending",
    ),
    "policy_or_geopolitical": (
        "export control",
        "sanction",
        "tariff",
        "geopolitical",
    ),
}

POSITIVE_TERMS = {
    "accelerate",
    "beat",
    "benefit",
    "constructive",
    "demand",
    "expansion",
    "growth",
    "increase",
    "outperform",
    "raise",
    "strong",
    "surge",
    "upgrade",
}
NEGATIVE_TERMS = {
    "cut",
    "decline",
    "delay",
    "deteriorate",
    "downgrade",
    "export control",
    "fall",
    "risk",
    "shortage",
    "slow",
    "underperform",
    "weak",
}


class DomainAnalystIntakePacket:
    """Normalize source/evidence packets into one domain analyst's evidence contract."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_intake_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        evidence_pack_json: str | Path = DEFAULT_EVIDENCE_PACK_JSON,
        *,
        source_gate_json: str | Path | None = DEFAULT_SOURCE_GATE_JSON,
        domain_id: str = DEFAULT_DOMAIN_ID,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        horizon_days: int | None = None,
        as_of: str | None = None,
        max_items: int = 200,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of = as_of or utc_now_iso()
        profile = get_domain_profile(domain_id)
        source_gate = _load_optional_json(source_gate_json)
        evidence_pack = _load_json(evidence_pack_json)
        documents = documents_from_evidence_pack(evidence_pack_json)
        resolved_tickers = _normalize_tickers(tickers or evidence_pack.get("coverage", {}).get("tickers", []) or profile.ticker_universe_hint)
        resolved_sectors = _normalize_strings(sectors or evidence_pack.get("coverage", {}).get("sectors", []))
        gate_context = _source_gate_context(source_gate)
        evidence_items, skipped_documents = _evidence_items_from_documents(
            documents=documents,
            domain_id=domain_id,
            profile_required_types=profile.required_evidence_types,
            profile_useful_types=profile.useful_evidence_types,
            tickers=resolved_tickers,
            sectors=resolved_sectors,
            as_of=as_of,
            max_items=max_items,
            source_gate_context=gate_context,
        )
        can_run_analyst = gate_context["can_enter_domain_research"] and bool(evidence_items)
        analyst_report = None
        if can_run_analyst:
            analyst = BaseAnalystAgent(domain_id=domain_id, agent_name=f"{domain_id}_working_analyst")
            analyst_report = analyst.run(
                evidence=evidence_items,
                tickers=resolved_tickers,
                horizon_days=horizon_days,
                as_of=as_of,
            ).model_dump(mode="json")

        checks = _review_checks(
            gate_context=gate_context,
            evidence_items=evidence_items,
            analyst_report=analyst_report,
            profile_required_types=profile.required_evidence_types,
        )
        status = _intake_status(checks, can_run_analyst, analyst_report)
        payload = {
            "run_id": _run_id("domain_analyst_intake_packet"),
            "created_at": as_of,
            "mode": "domain_analyst_intake_packet",
            "inputs": {
                "evidence_pack_json": str(evidence_pack_json),
                "source_gate_json": str(source_gate_json) if source_gate_json else None,
                "domain_id": domain_id,
                "tickers": resolved_tickers,
                "sectors": resolved_sectors,
                "max_items": max_items,
            },
            "summary": {
                "intake_status": status,
                "domain_id": domain_id,
                "document_count": len(documents),
                "evidence_item_count": len(evidence_items),
                "skipped_document_count": len(skipped_documents),
                "ticker_direct_count": sum(1 for item in evidence_items if item.directness == "ticker"),
                "sector_or_domain_count": sum(1 for item in evidence_items if item.directness in {"sector", "domain"}),
                "macro_policy_context_count": sum(1 for item in evidence_items if item.directness in {"macro", "policy", "geopolitical"}),
                "required_evidence_missing": (analyst_report or {}).get("quality_gates", {}).get("missing_required_evidence", profile.required_evidence_types),
                "analyst_report_created": analyst_report is not None,
                "can_run_domain_analyst": can_run_analyst,
                "can_create_direct_ticker_thesis_without_bridge": False,
                "can_write_learning_memory": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
            "source_gate_context": gate_context,
            "domain_profile_snapshot": {
                "domain_id": profile.domain_id,
                "display_name": profile.display_name,
                "required_evidence_types": profile.required_evidence_types,
                "useful_evidence_types": profile.useful_evidence_types,
                "ticker_universe_hint": profile.ticker_universe_hint,
                "direct_ticker_evidence_rules": profile.direct_ticker_evidence_rules,
                "source_registry_policy": profile.source_registry_policy,
                "ingestion_filter_policy": profile.ingestion_filter_policy,
                "evidence_scoring_policy": profile.evidence_scoring_policy,
                "review_output_policy": profile.review_output_policy,
                "feedback_label_policy": profile.feedback_label_policy,
            },
            "evidence_type_summary": dict(Counter(item.evidence_type for item in evidence_items)),
            "directness_summary": dict(Counter(item.directness for item in evidence_items)),
            "evidence_items": [item.model_dump(mode="json") for item in evidence_items],
            "skipped_documents": skipped_documents,
            "analyst_report": analyst_report,
            "review_checks": checks,
            "explicit_non_actions": _explicit_non_actions(),
            "recommendations": _recommendations(status, gate_context, analyst_report),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_intake_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_intake_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    report = payload.get("analyst_report") or {}
    thesis = report.get("thesis") or {}
    basket = report.get("ticker_basket") or {}
    lines = [
        "# DEAN-OS Domain Analyst Intake Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Intake status: `{summary.get('intake_status')}`",
        f"- Documents: {summary.get('document_count')}",
        f"- Evidence items: {summary.get('evidence_item_count')}",
        f"- Ticker-direct evidence: {summary.get('ticker_direct_count')}",
        f"- Sector/domain evidence: {summary.get('sector_or_domain_count')}",
        f"- Macro/policy/geopolitical context: {summary.get('macro_policy_context_count')}",
        f"- Analyst report created: {summary.get('analyst_report_created')}",
        f"- Can create direct ticker thesis without bridge: {summary.get('can_create_direct_ticker_thesis_without_bridge')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Analyst Thesis",
        "",
        f"- Recommendation: `{report.get('recommendation')}`",
        f"- Stance: `{thesis.get('stance')}`",
        f"- Expected direction: `{thesis.get('expected_direction')}`",
        f"- Confidence: `{thesis.get('confidence')}`",
        f"- Missing required evidence: `{', '.join(summary.get('required_evidence_missing') or []) or 'none'}`",
        "",
        str(thesis.get("thesis") or "No analyst thesis produced."),
        "",
        "## Ticker Bridge Guardrail",
        "",
        f"- Basket status: `{basket.get('basket_status')}`",
        f"- Direct-ready count: `{basket.get('direct_ready_count')}`",
        f"- Basket-candidate count: `{basket.get('basket_candidate_count')}`",
        f"- Blocked count: `{basket.get('blocked_count')}`",
        "",
        "## Evidence Shape",
        "",
        f"- Evidence types: `{payload.get('evidence_type_summary', {})}`",
        f"- Directness: `{payload.get('directness_summary', {})}`",
        "",
        "## Review Checks",
        "",
    ]
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _evidence_items_from_documents(
    *,
    documents: list[ResearchDocument],
    domain_id: str,
    profile_required_types: list[str],
    profile_useful_types: list[str],
    tickers: list[str],
    sectors: list[str],
    as_of: str,
    max_items: int,
    source_gate_context: dict[str, Any],
) -> tuple[list[AnalystEvidenceItem], list[dict[str, Any]]]:
    items: list[AnalystEvidenceItem] = []
    skipped: list[dict[str, Any]] = []
    for document in documents:
        if len(items) >= max_items:
            skipped.append({"document_id": document.document_id, "reason": "max_items_reached"})
            continue
        if document.quarantine_flags or document.quality_precheck in {"quarantined", "quarantine_detected"}:
            skipped.append(
                {
                    "document_id": document.document_id,
                    "title": document.title,
                    "reason": "document_quarantined",
                    "quarantine_flags": document.quarantine_flags,
                }
            )
            continue
        item = _document_to_evidence_item(
            document=document,
            domain_id=domain_id,
            profile_required_types=profile_required_types,
            profile_useful_types=profile_useful_types,
            requested_tickers=tickers,
            requested_sectors=sectors,
            as_of=as_of,
            source_gate_context=source_gate_context,
        )
        if item is None:
            skipped.append({"document_id": document.document_id, "title": document.title, "reason": "outside_domain_scope"})
            continue
        items.append(item)
    return items, skipped


def _document_to_evidence_item(
    *,
    document: ResearchDocument,
    domain_id: str,
    profile_required_types: list[str],
    profile_useful_types: list[str],
    requested_tickers: list[str],
    requested_sectors: list[str],
    as_of: str,
    source_gate_context: dict[str, Any],
) -> AnalystEvidenceItem | None:
    text = _search_text(document)
    evidence_type = _classify_evidence_type(text, document, profile_required_types, profile_useful_types)
    directness = _classify_directness(document, evidence_type, requested_tickers)
    if evidence_type == "domain_context" and directness == "domain" and not _domain_relevant(text, requested_sectors):
        return None
    limitations = _limitations(document, source_gate_context)
    blocked_windows = ["missing_published_at"] if not document.published_at else []
    tickers = _normalize_tickers([ticker for ticker in document.tickers if not requested_tickers or ticker.upper() in requested_tickers])
    if not tickers and directness == "ticker":
        directness = "sector" if document.sectors or requested_sectors else "domain"
    return AnalystEvidenceItem(
        evidence_id=f"domain_ev_{document.document_id[:20]}",
        source_type=document.source_type,
        source=document.uri or document.document_id,
        published_at=document.published_at,
        as_of=as_of,
        domain_id=domain_id,
        tickers=tickers,
        sectors=_normalize_strings(document.sectors or requested_sectors),
        evidence_type=evidence_type,
        summary=_summary(document),
        stance_hint=_stance_hint(text),
        strength=_strength(document, evidence_type, directness),
        freshness_score=_freshness_score(document.published_at, as_of),
        directness=directness,
        reliability_score=_reliability_score(document),
        limitations=limitations,
        blocked_windows=blocked_windows,
    )


def _classify_evidence_type(
    text: str,
    document: ResearchDocument,
    required_types: list[str],
    useful_types: list[str],
) -> str:
    candidates = [*required_types, *useful_types]
    tag_text = " ".join(document.tags).lower()
    for evidence_type, terms in STRONG_EVIDENCE_TYPE_TERMS.items():
        if evidence_type in candidates and any(term in text for term in terms):
            return evidence_type
    scored: list[tuple[int, int, str]] = []
    for evidence_type in candidates:
        aliases = [evidence_type.replace("_", " "), *EVIDENCE_TYPE_ALIASES.get(evidence_type, [])]
        score = sum(2 for alias in aliases if alias and alias in text)
        score += sum(1 for alias in aliases if alias and alias in tag_text)
        if score:
            scored.append((score, EVIDENCE_TYPE_PRIORITY.get(evidence_type, 0), evidence_type))
    if scored:
        return sorted(scored, reverse=True)[0][2]
    if document.source_type in {"filing", "transcript"}:
        return "earnings_guidance"
    if "macro" in document.tags:
        return "market_confirmation"
    if document.source_type in {"news", "article"}:
        return "sector_demand"
    return "domain_context"


def _classify_directness(document: ResearchDocument, evidence_type: str, requested_tickers: list[str]) -> str:
    doc_tickers = _normalize_tickers(document.tickers)
    if doc_tickers and (not requested_tickers or set(doc_tickers).intersection(requested_tickers)):
        return "ticker"
    if evidence_type in {"policy_or_geopolitical", "policy_or_sanctions"}:
        return "policy"
    if evidence_type in {"geopolitical_event", "exposure_mapping"}:
        return "geopolitical"
    if "macro" in document.tags:
        return "macro"
    if document.sectors:
        return "sector"
    return "domain"


def _source_gate_context(source_gate: dict[str, Any]) -> dict[str, Any]:
    if not source_gate.get("available"):
        return {
            "available": False,
            "path": source_gate.get("path"),
            "gate_status": "missing_source_gate",
            "can_enter_domain_research": False,
            "safe_downstream_boundary": False,
            "warnings": ["source_gate_not_attached"],
        }
    payload = source_gate["payload"]
    summary = payload.get("summary", {})
    safe = (
        summary.get("can_promote_to_evidence") is False
        and summary.get("can_extract_claims_events_entities") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )
    warnings = [check.get("code") for check in payload.get("validation_checks", []) if check.get("status") == "warn"]
    return {
        "available": True,
        "path": source_gate.get("path"),
        "gate_status": summary.get("gate_status"),
        "can_enter_domain_research": summary.get("can_enter_domain_research") is True and safe,
        "safe_downstream_boundary": safe,
        "warnings": sorted(str(item) for item in warnings if item),
    }


def _review_checks(
    *,
    gate_context: dict[str, Any],
    evidence_items: list[AnalystEvidenceItem],
    analyst_report: dict[str, Any] | None,
    profile_required_types: list[str],
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    checks.append(_check("pass" if gate_context["can_enter_domain_research"] else "fail", "source_gate_allows_domain_research", gate_context["gate_status"]))
    checks.append(_check("pass" if gate_context["safe_downstream_boundary"] else "fail", "source_gate_downstream_actions_disabled", "Source gate keeps extraction, learning writes, execution recommendations, and trading disabled."))
    checks.append(_check("pass" if evidence_items else "fail", "analyst_evidence_items_present", f"{len(evidence_items)} evidence items normalized."))
    available_types = {item.evidence_type for item in evidence_items}
    missing = [item for item in profile_required_types if item not in available_types]
    checks.append(_check("pass" if not missing else "warn", "required_domain_evidence_coverage", ", ".join(missing) if missing else "All required evidence types covered."))
    direct_ticker_count = sum(1 for item in evidence_items if item.directness == "ticker")
    checks.append(_check("pass" if direct_ticker_count else "warn", "direct_ticker_evidence_partitioned", f"{direct_ticker_count} direct ticker evidence items."))
    if analyst_report:
        checks.append(_check("pass", "base_domain_analyst_report_created", analyst_report.get("recommendation", "")))
        checks.append(_check("pass" if analyst_report.get("live_execution_allowed") is False else "fail", "analyst_report_no_live_execution", "live_execution_allowed=False"))
    else:
        checks.append(_check("warn", "base_domain_analyst_report_not_created", "Analyst report was not created because intake is blocked or empty."))
    return checks


def _intake_status(checks: list[dict[str, str]], can_run_analyst: bool, analyst_report: dict[str, Any] | None) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "blocked_domain_analyst_intake"
    if not can_run_analyst or analyst_report is None:
        return "domain_analyst_intake_waiting_for_source_gate"
    if any(check["status"] == "warn" for check in checks):
        return "domain_analyst_intake_ready_with_warnings"
    return "domain_analyst_intake_ready"


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _limitations(document: ResearchDocument, source_gate_context: dict[str, Any]) -> list[str]:
    limitations: list[str] = []
    if not document.published_at:
        limitations.append("missing_published_at")
    if document.quarantine_flags:
        limitations.append("document_quarantine_flags_present")
    if source_gate_context.get("warnings"):
        limitations.append("source_gate_has_warnings")
    limitations.append("normalized_for_review_only_not_evidence_promotion")
    return limitations


def _domain_relevant(text: str, requested_sectors: list[str]) -> bool:
    if any(sector.lower() in text for sector in requested_sectors):
        return True
    return any(term in text for terms in EVIDENCE_TYPE_ALIASES.values() for term in terms)


def _summary(document: ResearchDocument) -> str:
    text = re.sub(r"\s+", " ", document.text).strip()
    excerpt = text[:260]
    return f"{document.title}: {excerpt}" if excerpt else document.title


def _search_text(document: ResearchDocument) -> str:
    return " ".join([document.title, document.text, " ".join(document.tags), " ".join(document.sectors)]).lower()


def _stance_hint(text: str) -> str:
    positive = any(term in text for term in POSITIVE_TERMS)
    negative = any(term in text for term in NEGATIVE_TERMS)
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "unknown"


def _strength(document: ResearchDocument, evidence_type: str, directness: str) -> float:
    base = 0.45
    if evidence_type != "domain_context":
        base += 0.15
    if directness == "ticker":
        base += 0.1
    if document.source_type in {"filing", "transcript", "report"}:
        base += 0.1
    if document.published_at:
        base += 0.05
    return clamp(base, 0.0, 1.0)


def _reliability_score(document: ResearchDocument) -> float:
    if document.source_type in {"filing", "transcript"}:
        return 0.85
    if document.source_type == "report":
        return 0.75
    if document.source_type in {"news", "article"}:
        return 0.65
    if document.source_type == "book":
        return 0.6
    return 0.5


def _freshness_score(published_at: str | None, as_of: str) -> float:
    published = _parse_datetime(published_at)
    reference = _parse_datetime(as_of) or datetime.now(UTC)
    if not published:
        return 0.25
    age_days = max(0, (reference - published).days)
    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.8
    if age_days <= 180:
        return 0.6
    if age_days <= 365:
        return 0.4
    return 0.2


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False, "path": None, "error": "missing_artifact"}
    candidate = Path(path)
    if not candidate.exists():
        return {"available": False, "path": str(candidate), "error": "missing_artifact"}
    return {"available": True, "path": str(candidate), "payload": _load_json(candidate)}


def _normalize_tickers(values: list[str]) -> list[str]:
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _normalize_strings(values: list[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _explicit_non_actions() -> list[str]:
    return [
        "No live source retrieval or collector execution is performed.",
        "No claim/event/entity extraction is performed.",
        "No evidence promotion is performed.",
        "No learning memory write or analyst-weight change is performed.",
        "No execution, buy/sell/hold, allocation, price target, paper order, broker call, or live trade recommendation is generated.",
        "No new domain analyst profile is cloned or enabled.",
    ]


def _recommendations(status: str, gate_context: dict[str, Any], analyst_report: dict[str, Any] | None) -> list[str]:
    if status == "blocked_domain_analyst_intake":
        return ["Fix source gate or evidence coverage before running the analyst."]
    recommendations = [
        "Review normalized evidence items before accepting the analyst thesis.",
        "Use ticker candidates only through the ticker bridge; sector evidence remains basket-level context.",
    ]
    missing = (analyst_report or {}).get("quality_gates", {}).get("missing_required_evidence") or []
    if missing:
        recommendations.append("Add source coverage for missing required evidence: " + ", ".join(missing))
    if gate_context.get("warnings"):
        recommendations.append("Review source gate warnings before standardizing this analyst intake contract.")
    recommendations.append("After manual acceptance, reuse this intake pattern before cloning other domain analysts.")
    return recommendations


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
