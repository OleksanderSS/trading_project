from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from dean_os.base import AnalyticalAgent
from dean_os.schemas import AnalyticalReport, MarketContext

# ── Event Taxonomy ────────────────────────────────────────────────────────

EVENT_TAXONOMY: dict[str, list[str]] = {
    "macro": [
        "cpi", "inflation", "gdp", "employment", "payroll", "unemployment",
        "consumer spending", "retail sales", "industrial production",
        "manufacturing", "pmi", "services pmi", "durable goods",
    ],
    "monetary_policy": [
        "fed", "federal reserve", "interest rate", "rate hike", "rate cut",
        "quantitative easing", "tightening", "easing", "central bank",
        "balance sheet", "liquidity",
    ],
    "fiscal_policy": [
        "stimulus", "infrastructure bill", "inflation reduction act",
        "tax", "tariff", "government spending", "fiscal",
        "deficit", "debt ceiling", "budget",
    ],
    "geopolitical": [
        "war", "sanctions", "conflict", "military", "defense",
        "export control", "trade war", "tariffs", "chips act",
        "blockade", "embargo", "treaty", "alliance",
    ],
    "corporate": [
        "earnings", "guidance", "revenue", "profit margin", "buyback",
        "dividend", "layoff", "restructuring", "ceo", "management change",
        "merger", "acquisition", "ipo", "spinoff",
    ],
    "supply_chain": [
        "shortage", "bottleneck", "backlog", "inventory", "lead time",
        "logistics", "freight", "shipping", "supply chain", "capacity",
        "allocation", "constraint", "chip shortage",
    ],
    "technology": [
        "ai", "artificial intelligence", "gpu", "machine learning",
        "data center", "cloud", "quantum", "software", "semiconductor",
        "breakthrough", "patent", "r&d",
    ],
    "energy": [
        "oil", "gas", "natural gas", "lng", "opec", "brent", "wti",
        "refinery", "electricity", "renewable", "solar", "wind",
        "nuclear", "power grid", "energy",
    ],
    "commodity": [
        "copper", "steel", "aluminum", "lithium", "rare earth",
        "cobalt", "nickel", "wheat", "corn", "soybean",
        "iron ore", "gold", "silver",
    ],
    "regulatory": [
        "regulation", "compliance", "antitrust", "lawsuit", "fine",
        "fcc", "sec", "ftc", "doj", "investigation", "sanctions",
        "license", "approval", "rejection",
    ],
    "natural_disaster": [
        "earthquake", "flood", "hurricane", "drought", "wildfire",
        "pandemic", "epidemic", "storm", "extreme weather",
    ],
    "credit_financial": [
        "credit spread", "default", "bankruptcy", "bank", "lending",
        "credit", "bond", "yield", "spread", "liquidity crisis",
        "financial stability", "contagion",
    ],
}

SHOCK_SIGNALS: dict[str, tuple[list[str], list[str]]] = {
    "positive": (
        ["surge", "soar", "beat", "exceed", "bullish", "upside", "growth",
         "expansion", "strong demand", "record", "breakthrough", "optimistic"],
        ["raise guidance", "buyback", "dividend increase", "capacity expansion",
         "new contract", "partnership"],
    ),
    "negative": (
        ["plunge", "slump", "miss", "decline", "loss", "bearish", "downside",
         "contraction", "weak demand", "layoff", "restructuring", "cut"],
        ["lower guidance", "default", "bankruptcy", "investigation",
         "downgrade", "shutdown"],
    ),
}


def classify_event_type(text: str) -> str:
    """Classify news text into one event type from the taxonomy.

    Uses word-boundary matching for short keywords (< 4 chars) to prevent
    substring traps (e.g. 'ai' matching inside 'Taiwan').
    """
    import re
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for event_type, keywords in EVENT_TAXONOMY.items():
        score = 0
        for kw in keywords:
            if len(kw) < 4:
                # Word-boundary match to avoid 'ai' in 'Taiwan', 'pmi' in 'company' etc.
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    score += 1
            else:
                if kw in text_lower:
                    score += 1
        if score > 0:
            scores[event_type] = score
    if not scores:
        return "uncategorized"
    return max(scores, key=scores.get)


def classify_shock(text: str) -> tuple[str, float]:
    """Classify as positive, negative, neutral with confidence."""
    text_lower = text.lower()
    pos_score = 0
    neg_score = 0

    for word in SHOCK_SIGNALS["positive"][0] + SHOCK_SIGNALS["positive"][1]:
        if word in text_lower:
            pos_score += 1
    for word in SHOCK_SIGNALS["negative"][0] + SHOCK_SIGNALS["negative"][1]:
        if word in text_lower:
            neg_score += 1

    total = pos_score + neg_score
    if total == 0:
        return "neutral", 0.5
    if pos_score > neg_score:
        confidence = pos_score / (pos_score + neg_score)
        return "positive", min(confidence + 0.1, 0.95)
    elif neg_score > pos_score:
        confidence = neg_score / (neg_score + pos_score)
        return "negative", min(confidence + 0.1, 0.95)
    else:
        return "neutral", 0.5


def estimate_impact(text: str, shock: str) -> float:
    """Estimate impact magnitude -1 to 1 based on intensity words."""
    intensity_high = [
        "crisis", "collapse", "catastrophic", "historic", "unprecedented",
        "emergency", "severe", "massive", "worst",
    ]
    intensity_medium = [
        "significant", "major", "substantial", "sharp", "steep",
        "accelerate", "deteriorate", "transform",
    ]
    text_lower = text.lower()
    base = 0.3 if shock == "positive" else -0.3 if shock == "negative" else 0.0
    high_hits = sum(1 for w in intensity_high if w in text_lower)
    med_hits = sum(1 for w in intensity_medium if w in text_lower)
    boost = (high_hits * 0.2) + (med_hits * 0.1)
    result = base + (boost if shock == "positive" else -boost if shock == "negative" else boost * 0.05)
    return max(-1.0, min(1.0, result))


def estimate_predictability(text: str) -> float:
    """Estimate how predictable the event was (0=surprise, 1=expected)."""
    expected_signals = [
        "as expected", "in line", "expected", "forecast", "anticipated",
        "scheduled", "routine", "regular",
    ]
    surprise_signals = [
        "unexpected", "surprise", "shock", "unprecedented", "sudden",
        "unforeseen", "unpredictable",
    ]
    text_lower = text.lower()
    exp = sum(1 for s in expected_signals if s in text_lower)
    sur = sum(1 for s in surprise_signals if s in text_lower)
    if exp > sur:
        return 0.7 + min(exp * 0.1, 0.25)
    elif sur > exp:
        return max(0.1, 0.5 - sur * 0.1)
    return 0.5


def estimate_time_to_impact(text: str) -> str:
    """Estimate when impact materializes."""
    immediate = ["effective immediately", "today", "tonight", "now"]
    days = ["this week", "next week", "in the coming days"]
    weeks = ["this month", "next month", "in the coming weeks"]
    quarters = ["this quarter", "next quarter", "this year", "fiscal year"]
    text_lower = text.lower()
    if any(s in text_lower for s in immediate):
        return "immediate"
    if any(s in text_lower for s in days):
        return "days"
    if any(s in text_lower for s in weeks):
        return "weeks"
    if any(s in text_lower for s in quarters):
        return "quarters"
    return "months"


# ── News Event Schema ─────────────────────────────────────────────────────


class NewsEvent:
    """Structured news event with classification metadata."""

    def __init__(
        self,
        headline: str,
        source: str = "",
        published_at: str = "",
        event_type: str | None = None,
        shock: str | None = None,
        shock_confidence: float | None = None,
        impact: float | None = None,
        predictability: float | None = None,
        time_to_impact: str | None = None,
        affected_sectors: list[str] | None = None,
    ):
        # Every override below defaults to text classification of `headline`;
        # callers with a known-good classification (e.g. a quantitative
        # indicator like a VIX spike, where keyword classification of a
        # synthetic headline would be unreliable) can supply it directly
        # instead.
        self.headline = headline
        self.source = source
        self.published_at = published_at
        self.event_type = event_type if event_type is not None else classify_event_type(headline)
        if shock is not None and shock_confidence is not None:
            self.shock, self.shock_confidence = shock, shock_confidence
        else:
            self.shock, self.shock_confidence = classify_shock(headline)
        self.impact = impact if impact is not None else estimate_impact(headline, self.shock)
        self.predictability = predictability if predictability is not None else estimate_predictability(headline)
        self.time_to_impact = time_to_impact if time_to_impact is not None else estimate_time_to_impact(headline)
        self.affected_sectors = affected_sectors if affected_sectors is not None else self._detect_sectors(headline)

    def _detect_sectors(self, text: str) -> list[str]:
        text_lower = text.lower()
        sectors: list[str] = []
        mapping: dict[str, list[str]] = {
            "semiconductor": ["semiconductor", "chip", "gpu", "foundry", "tsmc", "nvidia", "amd", "hbm", "memory"],
            "energy": ["oil", "gas", "opec", "brent", "wti", "lng", "refinery", "renewable"],
            "technology": ["ai", "software", "cloud", "data center", "saas", "cyber"],
            "consumer": ["retail", "consumer", "e-commerce", "amazon", "walmart"],
            "finance": ["bank", "fed", "interest rate", "credit", "bond", "yield"],
            "healthcare": ["pharma", "biotech", "healthcare", "fda", "drug"],
            "logistics": ["shipping", "freight", "port", "logistics", "supply chain", "warehouse"],
            "agriculture": ["crop", "wheat", "corn", "soybean", "fertilizer", "farm"],
            "real_estate": ["reit", "real estate", "property", "mortgage", "housing"],
            "defense": ["defense", "military", "aerospace", "lockheed", "raytheon"],
        }
        for sector, keywords in mapping.items():
            if any(kw in text_lower for kw in keywords):
                sectors.append(sector)
        return sectors

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "source": self.source,
            "published_at": self.published_at,
            "event_type": self.event_type,
            "shock": self.shock,
            "shock_confidence": round(self.shock_confidence, 3),
            "impact": round(self.impact, 3),
            "predictability": round(self.predictability, 3),
            "time_to_impact": self.time_to_impact,
            "affected_sectors": self.affected_sectors,
        }


# ── News Event Analyzer Agent ─────────────────────────────────────────────


SIGNIFICANT_EVENT_TYPES: set[str] = {
    "geopolitical", "natural_disaster", "credit_financial",
}


class NewsEventAnalyzerAgent(AnalyticalAgent):
    """Classifies news into structured events with impact analysis.

    Replaces simple keyword counting with event type taxonomy,
    shock classification, impact estimation, and sector mapping.
    """

    version = "0.2.0"
    branch = "analytical"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        from dean_os.event_causal_graph import EventCausalGraphBuilder, graphs_to_context_metadata
        
        news_items = self._get_news_items(context)
        if not news_items:
            return self._empty_report()

        # NewsEvent's constructor only accepts headline/source/published_at
        # (plus optional classification overrides, used by the VIX injection
        # below) -- news_items are raw dataframe records (src/features/nlp's
        # news collectors use a "title" column, not "headline"), so splatting
        # the whole dict as **item would raise TypeError on any unrecognized
        # key the collector schema happens to include.
        events = [
            NewsEvent(
                headline=item.get("headline") or item.get("title") or "",
                source=item.get("source", ""),
                published_at=item.get("published_at") or item.get("date") or item.get("timestamp") or "",
            )
            for item in news_items
        ]
        
        # Inject macro events from quantitative dataframes (Collector Integration)
        if "vix_data" in context.dataframes:
            try:
                vix_df = context.dataframes["vix_data"]
                if not vix_df.empty:
                    # Depending on collector schema, the value might be 'vix_current' or just 'close'
                    val_col = "vix_current" if "vix_current" in vix_df.columns else "Close"
                    if val_col in vix_df.columns:
                        latest_vix = float(vix_df.iloc[-1][val_col])
                        if latest_vix > 25.0:
                            events.append(NewsEvent(
                                headline=f"VIX Spike to {latest_vix:.1f}",
                                source="MacroCollector",
                                published_at=datetime.utcnow().isoformat(),
                                event_type="credit_financial",
                                shock="negative",
                                shock_confidence=0.95,
                                impact=min(0.9, latest_vix / 50.0),
                                predictability=0.4,
                                time_to_impact="1w",
                                affected_sectors=["technology", "consumer_discretionary", "financials"],
                            ))
            except Exception:
                pass

        # Build causal graphs
        graph_builder = EventCausalGraphBuilder(context_tickers=context.tickers)
        graphs = graph_builder.build_multi(events)
        if graphs:
            if not isinstance(context.metadata, dict):
                context.metadata = {}
            context.metadata.update(graphs_to_context_metadata(graphs))

        # ── Register significant events in OutcomeTracker ─────────────────
        if bool(self.config.get("register_outcomes", False)):
            self._register_significant_events(events)

        shock_counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        total_impact = 0.0
        sector_hits: dict[str, int] = {}
        event_type_hits: dict[str, int] = {}

        for ev in events:
            shock_counts[ev.shock] = shock_counts.get(ev.shock, 0) + 1
            total_impact += ev.impact
            for sector in ev.affected_sectors:
                sector_hits[sector] = sector_hits.get(sector, 0) + 1
            event_type_hits[ev.event_type] = event_type_hits.get(ev.event_type, 0) + 1

        total = len(events)
        net_sentiment = (shock_counts.get("positive", 0) - shock_counts.get("negative", 0)) / max(total, 1)
        avg_impact = total_impact / max(total, 1)

        # Determine overall verdict
        if net_sentiment > 0.3:
            verdict = "bullish"
        elif net_sentiment < -0.3:
            verdict = "bearish"
        else:
            verdict = "neutral"

        top_sectors = sorted(sector_hits, key=sector_hits.get, reverse=True)[:3]
        top_types = sorted(event_type_hits, key=event_type_hits.get, reverse=True)[:3]

        reasons = [
            f"Analyzed {total} news items",
            f"Net sentiment: {net_sentiment:+.2f} (pos={shock_counts.get('positive', 0)}, neg={shock_counts.get('negative', 0)}, neu={shock_counts.get('neutral', 0)})",
            f"Average impact: {avg_impact:+.3f}",
            f"Top sectors: {', '.join(top_sectors) if top_sectors else 'none detected'}",
            f"Top event types: {', '.join(top_types) if top_types else 'none detected'}",
        ]

        evidence = [
            self.evidence("news_analysis", self.name, "total_events", total),
            self.evidence("news_analysis", self.name, "net_sentiment", round(net_sentiment, 3)),
            self.evidence("news_analysis", self.name, "avg_impact", round(avg_impact, 3)),
            self.evidence("news_analysis", self.name, "sector_hits", sector_hits),
            self.evidence("news_analysis", self.name, "event_type_breakdown", event_type_hits),
            self.evidence("news_analysis", self.name, "shock_counts", shock_counts),
        ]

        # Add tracker calibration as evidence
        tracker_stats = self._get_tracker_stats()
        if tracker_stats:
            evidence.append(
                self.evidence("outcome_tracker", self.name, "tracker", tracker_stats),
            )

        # Add top events as evidence
        for i, ev in enumerate(events[:5]):
            evidence.append(
                self.evidence(
                    "news_event",
                    ev.source,
                    f"event_{i}",
                    ev.to_dict(),
                )
            )

        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=min(abs(net_sentiment) * 0.8 + 0.2, 0.95),
            data_quality_score=min(total / 20.0, 1.0),
            signal_strength=net_sentiment,
            reasons=reasons,
            risks=self._extract_risks(events),
            evidence=evidence,
            ticker="MULTI",
            asset_or_sector="global",
            horizon_years=0.5,
            thesis=f"News event analysis: {verdict} across {total} events",
            data_quality="strong" if total >= 20 else "partial" if total >= 5 else "weak",
            position_bias=verdict,
            catalysts=top_types,
            tailwinds=top_sectors,
            watchlist_score=min(abs(net_sentiment) * 0.5, 1.0),
        )

    def _register_significant_events(self, events: list[NewsEvent]) -> None:
        try:
            from dean_os.outcome_tracker import OutcomeTracker
            tracker = OutcomeTracker()
            for ev in events:
                is_significant = (
                    ev.shock_confidence > 0.6 and abs(ev.impact) > 0.3
                    or ev.event_type in SIGNIFICANT_EVENT_TYPES
                )
                if is_significant:
                    direction_map = {
                        "positive": "bullish",
                        "negative": "bearish",
                    }
                    base_dir = direction_map.get(ev.shock, "neutral")
                    intervals = {d: base_dir for d in [1, 5, 30, 60, 120]}
                    tracker.register(
                        headline=ev.headline,
                        event_type=ev.event_type,
                        shock=ev.shock,
                        impact_estimate=ev.impact,
                        confidence=ev.shock_confidence,
                        sectors=ev.affected_sectors,
                        source=ev.source,
                        directions=intervals,
                    )
        except Exception:
            pass

    def _get_tracker_stats(self) -> dict | None:
        try:
            from dean_os.outcome_tracker import OutcomeTracker
            tracker = OutcomeTracker()
            cal = tracker.calibrate()
            return {
                "total_events": cal.total_events,
                "total_outcomes": cal.total_outcomes,
                "brier_score": cal.brier_score,
                "accuracy_rate": cal.accuracy_rate,
            }
        except Exception:
            return None

    def _get_news_items(self, context: MarketContext) -> list[dict]:
        items: list[dict] = []
        if context.news:
            if isinstance(context.news, list):
                items.extend(context.news)
            elif hasattr(context.news, "__iter__"):
                try:
                    for item in context.news:
                        if isinstance(item, dict):
                            items.append(item)
                except TypeError:
                    pass
        if not items:
            items.append({"headline": "No news data available", "source": "system", "published_at": ""})
        return items

    def _extract_risks(self, events: list[NewsEvent]) -> list[str]:
        risks = []
        neg_events = [e for e in events if e.shock == "negative" and e.impact < -0.3]
        if neg_events:
            risks.append(f"{len(neg_events)} significant negative events detected")
        low_pred = [e for e in events if e.predictability < 0.3]
        if len(low_pred) > len(events) * 0.3:
            risks.append(f"High proportion ({len(low_pred)}/{len(events)}) of low-predictability events")
        return risks[:5]

    def _empty_report(self) -> AnalyticalReport:
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict="neutral",
            confidence=0.5,
            data_quality_score=0.0,
            signal_strength=0.0,
            reasons=["No news items to analyze"],
            evidence=[],
            ticker=None,
            asset_or_sector="global",
            horizon_years=0.5,
            thesis="News event analysis skipped",
            data_quality="weak",
            position_bias="neutral",
            watchlist_score=0.0,
        )


__all__ = ["NewsEvent", "NewsEventAnalyzerAgent"]
