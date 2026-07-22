from __future__ import annotations

from typing import Any

from dean_os.utils import clamp

PIPELINE_TAXONOMY_SOURCES = [
    "src/config/context.yaml",
    "src/config/news_impact_classification.yaml",
    "src/patterns/pattern_recognition_adjustment.py",
    "src/features/enrichers/context_map_enricher.py",
    "src/features/analysis/market_conditions_analyzer.py",
    "src/analytics/detectors/regime_detector.py",
]


NEWS_IMPACT_CLASSES: dict[str, dict[str, Any]] = {
    "market_wide": {
        "keywords": (
            "economy",
            "economic",
            "inflation",
            "fed",
            "federal reserve",
            "interest rates",
            "gdp",
            "unemployment",
            "recession",
            "stock market",
            "wall street",
            "bull market",
            "bear market",
            "volatility",
            "vix",
            "market sentiment",
        ),
        "impact_type": "market_wide",
        "affected_scope": "all_market",
        "timeframes": ("1d", "60m"),
        "impact_strength": "high",
    },
    "macro_economic": {
        "keywords": (
            "cpi",
            "consumer price index",
            "ppi",
            "producer price index",
            "jobless claims",
            "nonfarm payrolls",
            "retail sales",
            "manufacturing",
            "services pmi",
            "industrial production",
            "consumer confidence",
            "durable goods",
            "housing starts",
        ),
        "impact_type": "macro_economic",
        "affected_scope": "all_market",
        "timeframes": ("1d",),
        "impact_strength": "high",
    },
    "geopolitical": {
        "keywords": (
            "war",
            "conflict",
            "sanctions",
            "trade war",
            "tariff",
            "tariffs",
            "geopolitical",
            "china",
            "russia",
            "ukraine",
            "middle east",
            "iran",
            "israel",
            "election",
            "government",
            "export control",
        ),
        "impact_type": "geopolitical",
        "affected_scope": "all_market_or_sector",
        "timeframes": ("1d", "60m"),
        "impact_strength": "medium",
    },
    "technology": {
        "keywords": (
            "nvidia",
            "gpu",
            "ai",
            "artificial intelligence",
            "chatgpt",
            "semiconductor",
            "chip",
            "chips",
            "tsmc",
            "intel",
            "amd",
            "qualcomm",
            "data center",
            "cloud",
            "hbm",
            "advanced packaging",
        ),
        "impact_type": "sector_specific",
        "affected_scope": "technology_semiconductors",
        "timeframes": ("15m", "60m", "1d"),
        "impact_strength": "high",
    },
    "financial": {
        "keywords": (
            "bank",
            "banking",
            "jpmorgan",
            "bank of america",
            "goldman sachs",
            "wells fargo",
            "credit",
            "mortgage",
            "deposit",
            "loan",
            "loans",
            "liquidity",
        ),
        "impact_type": "sector_specific",
        "affected_scope": "financial_sector",
        "timeframes": ("60m", "1d"),
        "impact_strength": "high",
    },
    "energy": {
        "keywords": (
            "oil",
            "gas",
            "crude oil",
            "natural gas",
            "opec",
            "refinery",
            "pipeline",
            "energy",
            "power shortage",
            "renewable energy",
        ),
        "impact_type": "sector_specific",
        "affected_scope": "energy_sector",
        "timeframes": ("1d",),
        "impact_strength": "medium",
    },
    "company_specific": {
        "keywords": (
            "earnings report",
            "quarterly results",
            "guidance",
            "forecast",
            "outlook",
            "revenue",
            "profit",
            "eps",
            "beat estimates",
            "miss estimates",
            "dividend",
            "buyback",
            "acquisition",
            "merger",
            "bankruptcy",
            "lawsuit",
            "ceo",
            "cfo",
        ),
        "impact_type": "ticker_specific",
        "affected_scope": "dynamic_ticker_or_company",
        "timeframes": ("15m", "60m"),
        "impact_strength": "high",
    },
}


CRISIS_PATTERNS: dict[str, dict[str, Any]] = {
    "financial_crisis_2008": {
        "display_name": "Financial crisis 2008",
        "pattern_type": "crisis",
        "severity": 10,
        "indicators": ("bank failures", "credit freeze", "housing collapse", "liquidity crisis", "bailout"),
        "watch_metrics": ("credit_spreads", "bank_funding_stress", "liquidity", "default_risk"),
    },
    "covid_crash_2020": {
        "display_name": "COVID-19 crash 2020",
        "pattern_type": "crisis",
        "severity": 9,
        "indicators": ("pandemic", "lockdown", "supply chain disruption", "virus", "quarantine"),
        "watch_metrics": ("mobility", "supply_chain_delays", "policy_restrictions", "demand_shock"),
    },
    "dot_com_bubble": {
        "display_name": "Dot-com bubble",
        "pattern_type": "bubble_burst",
        "severity": 8,
        "indicators": ("tech overvaluation", "nasdaq crash", "speculation", "bubble", "growth stocks"),
        "watch_metrics": ("valuation_multiples", "earnings_revisions", "capex_quality", "retail_positioning"),
    },
    "inflation_spike_2022": {
        "display_name": "Inflation Spike 2022",
        "pattern_type": "inflation",
        "severity": 7,
        "indicators": ("cpi spike", "inflation", "wage pressure", "supply chain", "prices"),
        "watch_metrics": ("cpi_yoy", "ppi_yoy", "wage_growth", "input_costs"),
    },
    "rate_hike_cycle_2022": {
        "display_name": "Rate Hike Cycle 2022",
        "pattern_type": "monetary_policy",
        "severity": 6,
        "indicators": ("fed rate hikes", "tightening cycle", "yield curve inversion", "higher rates", "powell"),
        "watch_metrics": ("fed_funds", "real_rates", "yield_curve_slope", "credit_spreads"),
    },
    "banking_crisis_2023": {
        "display_name": "Banking Crisis 2023",
        "pattern_type": "banking_crisis",
        "severity": 7,
        "indicators": ("regional bank failures", "depositor runs", "svb", "credit suisse", "bank collapse"),
        "watch_metrics": ("deposit_outflows", "bank_funding_stress", "credit_conditions", "duration_losses"),
    },
    "geopolitical_crisis": {
        "display_name": "Geopolitical Crisis",
        "pattern_type": "geopolitical",
        "severity": 6,
        "indicators": ("war", "sanctions", "energy crisis", "invasion", "export control", "conflict"),
        "watch_metrics": ("sanctions", "export_controls", "energy_prices", "regional_revenue_exposure"),
    },
    "energy_crisis": {
        "display_name": "Energy Crisis",
        "pattern_type": "energy",
        "severity": 7,
        "indicators": ("oil prices", "gas prices", "energy shortage", "power shortage", "opec"),
        "watch_metrics": ("oil_price", "gas_price", "power_costs", "input_costs"),
    },
    "supply_chain_crisis": {
        "display_name": "Supply Chain Crisis",
        "pattern_type": "supply_chain",
        "severity": 6,
        "indicators": ("shipping costs", "inventory shortages", "production delays", "shortage", "bottleneck"),
        "watch_metrics": ("lead_times", "shipping_costs", "inventory", "capacity_utilization"),
    },
    "flash_crash_2010": {
        "display_name": "Flash crash 2010",
        "pattern_type": "flash_crash",
        "severity": 7,
        "indicators": ("flash crash", "liquidity evaporation", "market mechanism", "high frequency trading"),
        "watch_metrics": ("market_depth", "bid_ask_spread", "intraday_volatility", "liquidity"),
    },
}


PIPELINE_LEARNED_NEWS_PATTERNS: dict[str, dict[str, Any]] = {
    "banking_crisis": {
        "trigger_keywords": ("bank", "collapse", "bailout", "credit", "liquidity"),
        "sample_events": ("Lehman 2008", "SVB 2023", "Credit Suisse 2023"),
        "confidence": 0.85,
        "review_implication": "check funding stress, credit contraction, and contagion pathways",
    },
    "tech_breakthrough": {
        "trigger_keywords": ("breakthrough", "innovation", "launch", "ai", "revolutionary"),
        "sample_events": ("iPhone 2007", "ChatGPT 2022", "Internet 1995"),
        "confidence": 0.70,
        "review_implication": "separate durable adoption from hype and valuation pull-forward",
    },
    "geopolitical_crisis": {
        "trigger_keywords": ("war", "invasion", "sanctions", "conflict", "tension", "export control"),
        "sample_events": ("Ukraine 2022", "Gulf War 1991", "9/11 2001"),
        "confidence": 0.75,
        "review_implication": "map direct exposure, routing constraints, substitution, and policy duration",
    },
    "health_crisis": {
        "trigger_keywords": ("pandemic", "virus", "lockdown", "outbreak", "quarantine"),
        "sample_events": ("COVID 2020", "SARS 2003", "H1N1 2009"),
        "confidence": 0.80,
        "review_implication": "separate demand shock, supply shock, policy response, and behavioral shifts",
    },
    "monetary_policy_shift": {
        "trigger_keywords": ("fed", "interest", "rates", "monetary", "policy", "powell"),
        "sample_events": ("Volcker 1980", "Bernanke 2008", "Powell 2022"),
        "confidence": 0.90,
        "review_implication": "map discount-rate pressure, liquidity, credit, and duration sensitivity",
    },
}


HIGH_IMPACT_TERMS = (
    "crash",
    "collapse",
    "crisis",
    "bubble",
    "recession",
    "depression",
    "war",
    "sanctions",
    "default",
    "bankruptcy",
    "shutdown",
)

LOW_IMPACT_TERMS = (
    "rally",
    "recovery",
    "growth",
    "expansion",
    "bull market",
    "optimistic",
    "positive",
    "improving",
    "stabilizing",
    "support",
)


def classify_pipeline_news_context(text: str, *, sentiment_label: str | None = None) -> dict[str, Any]:
    lower = (text or "").lower()
    impact_classes = _impact_classes(lower)
    crisis_matches = _crisis_pattern_matches(lower, sentiment_label=sentiment_label)
    learned_matches = _learned_pattern_matches(lower, sentiment_label=sentiment_label)
    dominant_impact = impact_classes[0] if impact_classes else None
    dominant_crisis = crisis_matches[0] if crisis_matches else None
    context_tags = _context_tags(impact_classes, crisis_matches, learned_matches, lower)
    watch_metrics = _watch_metrics(impact_classes, crisis_matches, learned_matches)
    return {
        "adapter_id": "domain_analyst_pipeline_news_taxonomy_v1",
        "source_pipeline_modules": PIPELINE_TAXONOMY_SOURCES,
        "impact_classifications": impact_classes,
        "dominant_impact_classification": dominant_impact,
        "crisis_pattern_matches": crisis_matches,
        "dominant_crisis_pattern": dominant_crisis,
        "learned_pattern_matches": learned_matches,
        "context_tags": sorted(set(context_tags)),
        "watch_metrics": sorted(set(watch_metrics)),
        "review_flags": _review_flags(impact_classes, crisis_matches, learned_matches),
        "historical_analogy_rule": "Historical analogies are prompts for review, not statistical proof or trade signals.",
        "allowed_output": "pipeline_news_context_for_review",
        "forbidden_outputs": [
            "prediction_adjustment",
            "buy_sell_hold",
            "price_target",
            "trade_signal",
            "autonomous_portfolio_action",
            "broker_order",
            "paper_trade",
            "live_trade",
        ],
    }


def _impact_classes(lower: str) -> list[dict[str, Any]]:
    matches = []
    for class_id, config in NEWS_IMPACT_CLASSES.items():
        matched = sorted({keyword for keyword in config["keywords"] if keyword in lower})
        if not matched:
            continue
        strength_weight = {"high": 1.0, "medium": 0.7, "low": 0.5}.get(config["impact_strength"], 0.5)
        score = clamp((len(matched) / max(len(config["keywords"]), 1)) + 0.2 * strength_weight, 0.0, 1.0)
        matches.append(
            {
                "classification_id": class_id,
                "impact_type": config["impact_type"],
                "affected_scope": config["affected_scope"],
                "timeframes": list(config["timeframes"]),
                "impact_strength": config["impact_strength"],
                "matched_keywords": matched,
                "recognition_score": round(score, 3),
                "allowed_use": "review_priority_and_context_only",
            }
        )
    return sorted(matches, key=lambda item: (item["recognition_score"], len(item["matched_keywords"])), reverse=True)


def _crisis_pattern_matches(lower: str, *, sentiment_label: str | None) -> list[dict[str, Any]]:
    matches = []
    for pattern_id, config in CRISIS_PATTERNS.items():
        matched = sorted({indicator for indicator in config["indicators"] if indicator in lower})
        if not matched:
            continue
        base = len(matched) / max(len(config["indicators"]), 1)
        severity_boost = min(float(config["severity"]) / 20.0, 0.5)
        sentiment_boost = 0.1 if sentiment_label == "negative" else 0.0
        strength = clamp(base + severity_boost + sentiment_boost, 0.0, 1.0)
        matches.append(
            {
                "pattern_id": pattern_id,
                "display_name": config["display_name"],
                "pattern_type": config["pattern_type"],
                "severity": config["severity"],
                "matched_indicators": matched,
                "recognition_strength": round(strength, 3),
                "watch_metrics": list(config["watch_metrics"]),
                "review_implication": _crisis_review_implication(config["pattern_type"]),
            }
        )
    return sorted(matches, key=lambda item: (item["recognition_strength"], item["severity"]), reverse=True)


def _learned_pattern_matches(lower: str, *, sentiment_label: str | None) -> list[dict[str, Any]]:
    matches = []
    for pattern_id, config in PIPELINE_LEARNED_NEWS_PATTERNS.items():
        matched = sorted({keyword for keyword in config["trigger_keywords"] if keyword in lower})
        if not matched:
            continue
        base = len(matched) / max(len(config["trigger_keywords"]), 1)
        sentiment_boost = 0.1 if sentiment_label == "negative" and "crisis" in pattern_id else 0.0
        strength = clamp(base + sentiment_boost, 0.0, 1.0)
        matches.append(
            {
                "pattern_id": pattern_id,
                "matched_keywords": matched,
                "recognition_strength": round(strength, 3),
                "source_confidence": config["confidence"],
                "sample_events": list(config["sample_events"]),
                "review_implication": config["review_implication"],
                "allowed_use": "historical_analogy_for_review_only",
            }
        )
    return sorted(matches, key=lambda item: item["recognition_strength"], reverse=True)


def _context_tags(impact_classes: list[dict[str, Any]], crisis_matches: list[dict[str, Any]], learned_matches: list[dict[str, Any]], lower: str) -> list[str]:
    tags = []
    tags.extend(f"pipeline_news_{item['classification_id']}" for item in impact_classes)
    tags.extend(f"pipeline_crisis_{item['pattern_type']}" for item in crisis_matches)
    tags.extend(f"pipeline_learned_pattern_{item['pattern_id']}" for item in learned_matches)
    if any(term in lower for term in HIGH_IMPACT_TERMS):
        tags.append("pipeline_linguistic_high_impact")
    if any(term in lower for term in LOW_IMPACT_TERMS):
        tags.append("pipeline_linguistic_stabilizing_or_positive")
    if any(item["pattern_type"] in {"crisis", "banking_crisis", "flash_crash"} for item in crisis_matches):
        tags.append("pipeline_market_crisis_context")
    if any(item["classification_id"] == "technology" for item in impact_classes) and any("crisis" in item["pattern_type"] for item in crisis_matches):
        tags.append("pipeline_tech_news_inside_crisis_context")
    return tags


def _watch_metrics(impact_classes: list[dict[str, Any]], crisis_matches: list[dict[str, Any]], learned_matches: list[dict[str, Any]]) -> list[str]:
    metrics = set()
    for item in crisis_matches:
        metrics.update(item.get("watch_metrics", []))
    for item in impact_classes:
        if item["classification_id"] == "macro_economic":
            metrics.update({"cpi_yoy", "fed_funds", "yield_curve_slope", "credit_spreads"})
        if item["classification_id"] == "geopolitical":
            metrics.update({"sanctions", "export_controls", "regional_revenue_exposure"})
        if item["classification_id"] == "technology":
            metrics.update({"capex_guidance", "order_backlog", "supply_constraints", "valuation_multiples"})
        if item["classification_id"] == "financial":
            metrics.update({"credit_conditions", "deposit_outflows", "liquidity"})
        if item["classification_id"] == "energy":
            metrics.update({"oil_price", "gas_price", "power_costs"})
    for item in learned_matches:
        if item["pattern_id"] == "tech_breakthrough":
            metrics.update({"adoption_curve", "revenue_conversion", "hype_crowding"})
        if item["pattern_id"] == "monetary_policy_shift":
            metrics.update({"real_rates", "liquidity", "duration_sensitivity"})
    return sorted(metrics)


def _review_flags(impact_classes: list[dict[str, Any]], crisis_matches: list[dict[str, Any]], learned_matches: list[dict[str, Any]]) -> list[str]:
    flags = []
    if any(item["impact_strength"] == "high" for item in impact_classes):
        flags.append("pipeline_news_high_impact_requires_review")
    if any(item["severity"] >= 7 for item in crisis_matches):
        flags.append("pipeline_crisis_analogy_requires_human_review")
    if learned_matches:
        flags.append("pipeline_learned_pattern_is_analogy_not_prediction")
    if any(item["impact_type"] == "ticker_specific" for item in impact_classes):
        flags.append("ticker_specific_news_requires_direct_ticker_evidence")
    return sorted(set(flags))


def _crisis_review_implication(pattern_type: str) -> str:
    implications = {
        "banking_crisis": "separate liquidity, solvency, duration, and contagion channels",
        "geopolitical": "map policy duration, substitution path, routing constraints, and direct exposure",
        "inflation": "map pricing power, input costs, rate response, and demand destruction",
        "monetary_policy": "map discount-rate sensitivity, credit availability, and liquidity",
        "supply_chain": "map bottleneck node, duration, inventory buffers, and substitution",
        "energy": "map input-cost exposure, pass-through, logistics, and policy response",
        "bubble_burst": "separate real adoption from valuation compression and demand pull-forward",
        "flash_crash": "separate liquidity mechanics from fundamental information",
    }
    return implications.get(pattern_type, "treat as historical analogy requiring corroboration")
