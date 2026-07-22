"""EventClassifierLens — classifies raw event records into typed events.

This lens reads unstructured or semi-structured ``event_records`` from the
packet and enriches each with:
- ``event_class``: canonical taxonomy label (war_escalation, demand_driver, etc.)
- ``affected_sectors`` / ``affected_tickers``: extracted entities
- ``directness``: direct / indirect / contextual to the target domain
- ``sentiment``: positive / negative / mixed / unknown
- ``materiality_score``: 0..1 importance estimate

Deterministic keyword-based classification. No LLM, no network.
"""
from __future__ import annotations

import re
from typing import Any

from dean_os.analyst_core.lens_contract import AnalysisPacket, AnalystLens, ModuleDelta

# ──────────────────────────────────────────────────────────────────────────────
# Event class taxonomy (from design notes §6.3, §6.6)
# ──────────────────────────────────────────────────────────────────────────────

EVENT_CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    # ── Geopolitical / Security ──────────────────────────────────────────────
    "war_escalation": ("war", "invasion", "military", "attack", "missile", "occupation", "conflict"),
    "de_escalation": ("ceasefire", "de-escalation", "peace talks", "withdraw", "truce"),
    "sanctions_change": ("sanction", "export control", "blacklist", "restriction", "ban"),
    "political_transition": (
        "election", "inauguration", "coup", "government change", "regime change",
        "snap election", "prime minister", "president elect", "new government",
    ),
    # ── Macro / Monetary ─────────────────────────────────────────────────────
    "central_bank_decision": ("fed", "fomc", "rate decision", "interest rate", "monetary policy"),
    "inflation_release": ("cpi", "inflation", "pce", "price index", "deflation", "disinflation"),
    "recession_risk": ("recession", "downturn", "contraction", "gdp decline", "slowdown"),
    "expansion_signal": ("expansion", "gdp growth", "acceleration", "boom", "recovery"),
    "debt_crisis": (
        "sovereign default", "debt ceiling", "debt restructuring", "credit downgrade",
        "ratings cut", "imf bailout", "bond crisis", "fiscal crisis",
        "downgraded by moody", "downgraded by fitch", "downgraded by s&p",
        "sovereign debt downgrade", "debt downgraded", "rating downgrade",
        "credit rating cut", "junk status",
    ),
    # ── Energy / Commodities ─────────────────────────────────────────────────
    "commodity_supply_shock": ("supply shock", "opec", "production cut", "commodity shortage"),
    "oil_shock": ("oil spike", "oil price", "crude surge", "energy shock", "oil embargo"),
    # ── Technology / AI ──────────────────────────────────────────────────────
    "ai_capex_announcement": ("ai capex", "data center", "gpu order", "ai infrastructure", "compute investment"),
    "memory_supply_constraint": ("memory shortage", "hbm", "dram shortage", "supply constraint", "memory crunch"),
    "power_grid_constraint": ("power bottleneck", "grid constraint", "electricity shortage", "data center power"),
    # ── Physical / Climate Risk ──────────────────────────────────────────────
    "climate_disaster": (
        "hurricane", "typhoon", "cyclone", "flood", "wildfire", "drought",
        "freeze", "blizzard", "extreme weather", "texas freeze", "gulf coast storm",
        "natural disaster", "climate event", "heat wave",
    ),
    # ── Trade / Logistics ────────────────────────────────────────────────────
    "trade_route_disruption": (
        "suez canal", "panama canal", "strait of hormuz", "ormuz", "hormuz",
        "shipping lane", "blockade", "piracy", "port closure", "freight disruption",
        "maritime disruption", "red sea",
    ),
    "tariff": ("tariff", "duty", "customs", "trade barrier", "import tax"),
    # ── Corporate / Markets ──────────────────────────────────────────────────
    "risk_on_rotation": ("risk on", "rally", "surge", "breakout", "bull market"),
    "risk_off_rotation": ("risk off", "selloff", "plunge", "crash", "bear market"),
    "safe_haven_bid": ("safe haven", "flight to quality", "gold bid", "treasury bid"),
    "strategic_industrial_asset_mna": ("merger", "acquisition", "takeover", "buyout", "m&a"),
    "regulation": ("regulation", "regulator", "compliance", "new rule", "legislation"),
    "earnings_surprise": ("earnings beat", "earnings miss", "revenue surprise", "eps surprise"),
    "sector_rotation": ("sector rotation", "rotation into", "rotation out", "sector shift"),
    # ── Health / Pandemic ────────────────────────────────────────────────────
    "pandemic_health_shock": (
        "pandemic", "outbreak", "epidemic", "quarantine", "lockdown",
        "health emergency", "who alert", "virus spread",
    ),
}

# Controlled evidence types produced by the runtime carry more reliable
# semantics than a keyword miss. They are only used when text classification
# returns ``other``.
EVIDENCE_TYPE_TO_EVENT_CLASS: dict[str, str] = {
    "sector_demand": "demand_driver",
    "supply_chain": "supply_disruption",
    "capex_cycle": "capex_signal",
    "market_confirmation": "sector_rotation",
}
CANONICAL_SOURCE_EVENT_CLASSES = set(EVENT_CLASS_KEYWORDS).union(
    {"demand_driver", "supply_disruption", "capex_signal"}
)

# ──────────────────────────────────────────────────────────────────────────────
# Domain keyword maps for directness detection
# ──────────────────────────────────────────────────────────────────────────────

_DOMAIN_SECTOR_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    # ── Technology ───────────────────────────────────────────────────────────
    "semiconductor_ai_infrastructure": {
        "sector": (
            "semiconductor", "chip", "foundry", "nvidia", "amd", "intel", "tsmc",
            "hbm", "wafer", "lithography", "euv", "packaging", "gpu", "accelerator",
        ),
        "tickers": ("NVDA", "AMD", "INTC", "TSM", "TSMC", "ASML", "QCOM", "AVGO", "AMAT", "LRCX"),
    },
    # ── Energy ───────────────────────────────────────────────────────────────
    "energy": {
        "sector": (
            "oil", "gas", "crude", "opec", "pipeline", "lng", "refinery",
            "renewable", "natural gas", "brent", "wti", "upstream", "downstream",
        ),
        "tickers": ("XOM", "CVX", "COP", "EOG", "SLB", "OXY", "BP", "SHEL"),
    },
    # ── Macro / Policy ───────────────────────────────────────────────────────
    "macro_policy": {
        "sector": ("fed", "treasury", "fomc", "central bank", "monetary", "fiscal"),
        "tickers": (),
    },
    "geopolitics": {
        "sector": ("nato", "un security", "sanction", "tariff", "export control", "embargo"),
        "tickers": (),
    },
    # ── Logistics / Transport ────────────────────────────────────────────────
    "logistics": {
        "sector": (
            "shipping", "freight", "container", "port", "supply chain", "logistics",
            "suez", "panama canal", "hormuz", "strait", "trucking", "rail",
            "last mile", "warehouse", "3pl",
        ),
        "tickers": ("UPS", "FDX", "CHRW", "XPO", "JBHT"),
    },
    # ── Industrial / Manufacturing ───────────────────────────────────────────
    "industrial": {
        "sector": (
            "manufacturing", "factory", "plant", "capacity utilization", "industrial",
            "automaker", "aerospace", "defense", "boeing", "lockheed",
            "heavy equipment", "machinery",
        ),
        "tickers": ("GE", "CAT", "HON", "BA", "LMT", "RTX", "DE"),
    },
    # ── Consumer ─────────────────────────────────────────────────────────────
    "consumer": {
        "sector": (
            "retail", "consumer spending", "walmart", "amazon", "e-commerce",
            "discretionary", "staples", "household", "consumer confidence",
        ),
        "tickers": ("WMT", "AMZN", "COST", "TGT", "PG", "KO", "PEP"),
    },
    # ── Financials ───────────────────────────────────────────────────────────
    "financial": {
        "sector": (
            "bank", "credit", "lending", "mortgage", "jpmorgan", "goldman",
            "interest spread", "credit market", "default", "financial services",
        ),
        "tickers": ("JPM", "BAC", "GS", "MS", "WFC", "C", "BLK"),
    },
    # ── Healthcare ───────────────────────────────────────────────────────────
    "healthcare": {
        "sector": (
            "pharma", "biotech", "drug approval", "fda", "clinical trial",
            "healthcare", "insurance", "medicare", "hospital",
        ),
        "tickers": ("JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY"),
    },
    # ── Real Estate ──────────────────────────────────────────────────────────
    "real_estate": {
        "sector": (
            "housing", "mortgage rate", "home price", "reit", "construction",
            "commercial real estate", "office vacancy", "property",
        ),
        "tickers": ("AMT", "PLD", "EQIX", "SPG", "VNO"),
    },
    # ── Climate / Physical Risk ──────────────────────────────────────────────
    "climate_physical_risk": {
        "sector": (
            "hurricane", "flood", "drought", "wildfire", "storm", "climate disaster",
            "extreme weather", "texas freeze", "gulf coast", "natural disaster",
        ),
        "tickers": (),  # cross-sector impact — resolved via transmission channels
    },
}

POSITIVE_SIGNALS = ("growth", "increase", "expansion", "strong", "beat", "surge",
                    "upgrade", "accelerate", "breakout", "recovery", "boom")
NEGATIVE_SIGNALS = ("risk", "decline", "weak", "shortage", "delay", "restriction",
                    "sanction", "tariff", "cut", "crash", "selloff", "plunge")


class EventClassifierLens(AnalystLens):
    """Deterministic event classifier.

    Reads ``packet.event_records`` and enriches each with structured
    classification metadata. Returns a delta with the enriched records
    added to ``entity_links`` (structured event links).
    """

    lens_name = "event_classifier"
    lens_version = "0.1.0"
    event_classes_supported = ("*",)
    can_modify_existing = False

    def analyze(
        self, packet: AnalysisPacket, config: dict[str, Any] | None = None
    ) -> ModuleDelta:
        config = config or {}
        domain = config.get("domain_id", "semiconductor_ai_infrastructure")
        sector_kw = config.get("sector_keywords", [])
        ticker_uv = config.get("ticker_universe", [])
        classified_events: list[dict[str, Any]] = []

        for record in packet.event_records:
            classified = self._classify_event(record, domain, sector_kw, ticker_uv)
            classified_events.append(classified)

        review_notes: list[str] = []
        if not classified_events:
            review_notes.append("event_classifier: no events classified from input records")

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
            as_of=packet.as_of_date,
            classified_events_added=classified_events,
            fields_added=["classified_events"],
            evidence_ids=[
                e.get("event_id", "")
                for e in classified_events
                if e.get("event_id")
            ],
            confidence=self._overall_confidence(classified_events),
            reason_for_change=(
                f"Classified {len(classified_events)} events from "
                f"{len(packet.event_records)} evidence-backed event records."
            ),
            review_notes_added=review_notes,
        )

    # ── classification helpers ────────────────────────────────────────────

    def _classify_event(
        self,
        record: dict[str, Any],
        domain: str,
        sector_kw: list[str],
        ticker_uv: list[str],
    ) -> dict[str, Any]:
        """Classify a single event record."""
        text = self._extract_text(record)
        lower = text.lower()

        source_evidence_type = str(
            record.get("evidence_type") or record.get("event_class") or ""
        ).strip()
        event_class = ""
        if source_evidence_type == "macro_context":
            event_class = self._classify_macro_observation(record)
        if source_evidence_type == "policy_or_geopolitical":
            event_class = self._classify_policy_observation(lower)
        if not event_class:
            event_class = EVIDENCE_TYPE_TO_EVENT_CLASS.get(
                source_evidence_type,
                "",
            )
        if not event_class and source_evidence_type in CANONICAL_SOURCE_EVENT_CLASSES:
            event_class = source_evidence_type
        if not event_class:
            event_class = self._detect_event_class(lower)
        source_directness = str(record.get("directness", "contextual")).strip().lower()
        directness = self._normalize_directness(
            source_directness,
            tickers=record.get("tickers", []),
        )
        sentiment = str(record.get("stance_hint") or "").strip().lower()
        if sentiment not in {"positive", "negative", "neutral", "mixed", "unknown"}:
            sentiment = self._detect_sentiment(lower)
        affected_sectors = self._detect_sectors(lower, domain, sector_kw)
        affected_sectors = sorted(
            set(affected_sectors).union(str(item) for item in record.get("sectors", []))
        )
        affected_tickers = sorted(
            {
                str(item).upper().strip()
                for item in record.get("tickers", [])
                if str(item).strip()
            }
        )
        materiality = self._compute_materiality(
            event_class,
            directness,
            lower,
            strength=float(record.get("strength", 0.0) or 0.0),
            reliability=float(record.get("reliability_score", 0.0) or 0.0),
        )

        return {
            "event_id": record.get("event_id", record.get("id", "")),
            "evidence_id": record.get("evidence_id", record.get("event_id", "")),
            "source_id": record.get("source_id", record.get("source", "")),
            "source_type": record.get("source_type", ""),
            "title": record.get("title", text[:120]),
            "event_class": event_class,
            "directness": directness,
            "source_directness": source_directness,
            "sentiment": sentiment,
            "affected_sectors": affected_sectors,
            "affected_tickers": affected_tickers,
            "materiality_score": round(materiality, 3),
            "source_evidence_type": record.get(
                "evidence_type", record.get("event_class", "")
            ),
            "strength": record.get("strength", 0.0),
            "reliability_score": record.get("reliability_score", 0.0),
            "freshness_score": record.get("freshness_score", 0.0),
            "required_lane_eligible": record.get("required_lane_eligible", False),
            "provenance": record.get("provenance", {}),
            "point_in_time": record.get("point_in_time", {}),
            "published_at": record.get("published_at"),
            "as_of": record.get("as_of"),
            "text_preview": text[:300],
            "classified_by": "event_classifier_v0.2",
        }

    def _detect_event_class(self, lower: str) -> str:
        best_class = "other"
        best_hits = 0
        for event_class, keywords in EVENT_CLASS_KEYWORDS.items():
            hits = sum(1 for kw in keywords if self._contains_keyword(lower, kw))
            if hits > best_hits:
                best_hits = hits
                best_class = event_class
        return best_class

    @staticmethod
    def _contains_keyword(text: str, keyword: str) -> bool:
        """Match complete words/phrases, never arbitrary substrings.

        In particular, ``war`` must not classify ``warehouse``, ``awarded``,
        ``ransomware`` or ``warns`` as a war escalation.
        """
        normalized = " ".join(str(keyword).casefold().split())
        if not normalized:
            return False
        pattern = r"(?<![a-z0-9])" + re.escape(normalized).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        return re.search(pattern, text.casefold()) is not None

    @staticmethod
    def _classify_macro_observation(record: dict[str, Any]) -> str:
        name = str(record.get("provenance", {}).get("name") or "").lower()
        if any(term in name for term in ("cpi", "pce", "inflation")):
            return "inflation_observation"
        if any(
            term in name
            for term in (
                "fed_funds",
                "federal_reserve",
                "central_bank",
                "policy_rate",
            )
        ):
            return "liquidity_observation"
        return "other"

    @staticmethod
    def _classify_policy_observation(lower: str) -> str:
        """Keep the broad policy lane from becoming an untyped event.

        The producer lane can contain sanctions/export controls, tariffs, or
        general regulation.  We therefore use the verified lane only as a
        routing hint and still require narrow text evidence for the subtype.
        """
        if any(
            term in lower
            for term in (
                "sanction",
                "export control",
                "export restriction",
                "license requirement",
                "advanced computing items",
                "blacklist",
            )
        ):
            return "sanctions_change"
        if "tariff" in lower or "import duty" in lower:
            return "tariff"
        if any(term in lower for term in ("regulation", "new rule", "legislation")):
            return "regulation"
        return "other"

    def _detect_directness(
        self,
        lower: str,
        domain: str,
        sector_kw: list[str],
        ticker_uv: list[str],
    ) -> str:
        domain_kw = _DOMAIN_SECTOR_KEYWORDS.get(domain)
        if domain_kw is not None:
            tickers = domain_kw.get("tickers", ())
            sector = domain_kw.get("sector", ())
        else:
            tickers = tuple(t.lower() for t in ticker_uv)
            sector = tuple(k.lower() for k in sector_kw)

        for ticker in tickers:
            if ticker.lower() in lower:
                return "direct"
        for kw in sector:
            if kw in lower:
                return "indirect"
        return "contextual"

    def _detect_sentiment(self, lower: str) -> str:
        pos = sum(1 for s in POSITIVE_SIGNALS if s in lower)
        neg = sum(1 for s in NEGATIVE_SIGNALS if s in lower)
        if pos > 0 and neg > 0:
            return "mixed"
        if pos > 0:
            return "positive"
        if neg > 0:
            return "negative"
        return "unknown"

    def _detect_sectors(self, lower: str, domain: str, sector_kw: list[str]) -> list[str]:
        sectors: list[str] = []
        domain_kw = _DOMAIN_SECTOR_KEYWORDS.get(domain)
        if domain_kw is not None:
            keywords = domain_kw.get("sector", ())
        else:
            keywords = tuple(k.lower() for k in sector_kw)
        for kw in keywords:
            if kw in lower:
                sectors.append(domain)
                break
        # Generic cross-domain sector detection (runs for all domains)
        generic = {
            "technology": ("tech", "software", "saas", "cloud"),
            "semiconductor_ai_infrastructure": ("chip", "semiconductor", "gpu", "nvidia", "tsmc"),
            "healthcare": ("pharma", "biotech", "fda", "drug", "hospital"),
            "financial": ("bank", "credit", "loan", "insurance", "mortgage"),
            "consumer": ("retail", "consumer", "brand", "ecommerce", "walmart"),
            "industrial": ("manufacturing", "industrial", "factory", "aerospace"),
            "energy": ("oil", "gas", "crude", "opec", "refinery", "lng"),
            "logistics": ("shipping", "freight", "container", "port", "suez", "hormuz"),
            "real_estate": ("housing", "mortgage", "reit", "property"),
            "climate_physical_risk": ("hurricane", "flood", "wildfire", "drought", "extreme weather"),
        }
        for sector_name, kws in generic.items():
            if any(kw in lower for kw in kws):
                sectors.append(sector_name)
        return sorted(set(sectors))

    @staticmethod
    def _normalize_directness(source_directness: str, *, tickers: list[Any]) -> str:
        if source_directness in {"ticker", "direct"} and tickers:
            return "direct"
        if source_directness in {"sector", "indirect"}:
            return "indirect"
        return "contextual"

    def _compute_materiality(
        self,
        event_class: str,
        directness: str,
        lower: str,
        *,
        strength: float = 0.0,
        reliability: float = 0.0,
    ) -> float:
        score = 0.15 + 0.30 * strength + 0.25 * reliability
        if event_class != "other":
            score += 0.12
        if directness == "direct":
            score += 0.10
        elif directness == "indirect":
            score += 0.05
        # Boost for high-impact event classes
        high_impact = {
            "war_escalation", "sanctions_change", "oil_shock",
            "commodity_supply_shock", "tariff", "climate_disaster",
            "trade_route_disruption", "debt_crisis", "pandemic_health_shock",
        }
        if event_class in high_impact:
            score += 0.08
        # Boost for specific keywords
        key_terms = (
            "earnings", "guidance", "ban", "export control", "fab", "capacity",
            "hurricane", "blockade", "default", "lockdown", "suez", "hormuz",
        )
        hits = sum(1 for t in key_terms if t in lower)
        score += min(hits * 0.02, 0.06)
        return max(0.0, min(1.0, score))

    def _extract_text(self, record: dict[str, Any]) -> str:
        parts = []
        for key in ("text", "title", "summary", "description", "content", "body", "headline"):
            val = record.get(key)
            if val and isinstance(val, str):
                parts.append(val)
        return " ".join(parts) if parts else str(record)

    def _overall_confidence(self, events: list[dict[str, Any]]) -> float:
        if not events:
            return 0.2
        classified = sum(1 for e in events if e.get("event_class") != "other")
        ratio = classified / len(events) if events else 0
        return 0.3 + ratio * 0.5  # 0.3..0.8


__all__ = [
    "EventClassifierLens",
    "EVENT_CLASS_KEYWORDS",
    "EVIDENCE_TYPE_TO_EVENT_CLASS",
    "CANONICAL_SOURCE_EVENT_CLASSES",
]
