from __future__ import annotations

from typing import Any

from dean_os.base import AnalyticalAgent
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import (
    audit_market_context_news,
)
from dean_os.schemas import AnalyticalReport, MarketContext
from dean_os.draft.dean_os_agent_system_v7.dean_os.structured_context_provenance import (
    apply_market_context_structured_boundary,
)
from dean_os.utils import clamp


class KeywordDomainAgent(AnalyticalAgent):
    version = "0.1.0"
    keywords: tuple[str, ...] = ()
    bullish_terms: tuple[str, ...] = ()
    bearish_terms: tuple[str, ...] = ()
    default_horizon_years = 1.0
    thesis_template = "Domain context is present, but not yet strong enough for a thesis."
    asset_or_sector = "market"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        structured_audit = apply_market_context_structured_boundary(
            context
        )
        news_audit = audit_market_context_news(context)
        context.metadata["news_point_in_time_audit"] = {
            key: value
            for key, value in news_audit.items()
            if key != "accepted"
        }
        texts = self._news_texts(news_audit["accepted"])
        keyword_hits = self._count_terms(texts, self.keywords)
        bullish_hits = self._count_terms(texts, self.bullish_terms)
        bearish_hits = self._count_terms(texts, self.bearish_terms)
        has_structured_data = self._has_structured_data(context)
        data_quality = self._data_quality(keyword_hits, has_structured_data)
        position_bias = self._position_bias(bullish_hits, bearish_hits, data_quality)
        confidence = self._confidence(keyword_hits, bullish_hits, bearish_hits, data_quality)
        watchlist_score = self._watchlist_score(keyword_hits, bullish_hits, bearish_hits, data_quality)

        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=self._verdict(position_bias, data_quality),
            confidence=confidence,
            data_quality_score={"strong": 0.9, "partial": 0.6, "weak": 0.2}[data_quality],
            signal_strength=self._signal_strength(position_bias, confidence),
            ticker=",".join(context.tickers) if context.tickers else None,
            asset_or_sector=self.asset_or_sector,
            horizon_years=float(self.config.get("horizon_years", self.default_horizon_years)),
            thesis=self._thesis(keyword_hits, bullish_hits, bearish_hits, data_quality),
            data_quality=data_quality,
            position_bias=position_bias,
            catalysts=self._catalysts(context, keyword_hits),
            tailwinds=self._tailwinds(position_bias, bullish_hits),
            headwinds=self._headwinds(position_bias, bearish_hits),
            watchlist_score=watchlist_score,
            reasons=self._reasons(keyword_hits, bullish_hits, bearish_hits, data_quality),
            risks=self._risks(data_quality, bearish_hits),
            blind_spots=self._blind_spots(),
            evidence=[
                self.evidence("news", "context.news", "keyword_hits", keyword_hits),
                self.evidence("news", "context.news", "bullish_hits", bullish_hits),
                self.evidence("news", "context.news", "bearish_hits", bearish_hits),
                self.evidence(
                    "news",
                    "context.news",
                    "point_in_time_accepted_count",
                    news_audit["accepted_count"],
                ),
                self.evidence(
                    "news",
                    "context.news",
                    "point_in_time_excluded_count",
                    news_audit["excluded_count"],
                ),
                self.evidence(
                    "audit_finding",
                    "context.metadata.structured_context_point_in_time_audit",
                    "point_in_time_accepted_count",
                    structured_audit["accepted_count"],
                ),
                self.evidence(
                    "audit_finding",
                    "context.metadata.structured_context_point_in_time_audit",
                    "point_in_time_excluded_count",
                    structured_audit["excluded_count"],
                ),
            ],
            input_hash=self.context_hash(context),
        )

    def _news_texts(self, news: list[Any]) -> list[str]:
        texts: list[str] = []
        for item in news:
            if isinstance(item, str):
                texts.append(item.lower())
            elif isinstance(item, dict):
                parts = [
                    str(item.get(key, ""))
                    for key in ("title", "headline", "summary", "description", "content", "text")
                ]
                texts.append(" ".join(parts).lower())
        return texts

    def _count_terms(self, texts: list[str], terms: tuple[str, ...]) -> int:
        return sum(text.count(term.lower()) for text in texts for term in terms)

    def _has_structured_data(self, context: MarketContext) -> bool:
        return bool(context.macro or context.sector_data or context.fundamentals)

    def _data_quality(self, keyword_hits: int, has_structured_data: bool) -> str:
        if keyword_hits >= 3 and has_structured_data:
            return "strong"
        if keyword_hits > 0 or has_structured_data:
            return "partial"
        return "weak"

    def _position_bias(self, bullish_hits: int, bearish_hits: int, data_quality: str) -> str:
        if data_quality == "weak":
            return "insufficient_data"
        if bullish_hits > bearish_hits:
            return "bullish"
        if bearish_hits > bullish_hits:
            return "bearish"
        return "neutral"

    def _confidence(self, keyword_hits: int, bullish_hits: int, bearish_hits: int, data_quality: str) -> float:
        base = {"strong": 0.55, "partial": 0.35, "weak": 0.15}[data_quality]
        return clamp(base + min(keyword_hits, 8) * 0.04 + abs(bullish_hits - bearish_hits) * 0.03, 0.0, 0.9)

    def _watchlist_score(self, keyword_hits: int, bullish_hits: int, bearish_hits: int, data_quality: str) -> float:
        if data_quality == "weak":
            return 0.0
        return clamp(0.25 + keyword_hits * 0.08 + max(bullish_hits, bearish_hits) * 0.05, 0.0, 1.0)

    def _signal_strength(self, position_bias: str, confidence: float) -> float | None:
        if position_bias == "bullish":
            return confidence
        if position_bias == "bearish":
            return -confidence
        return 0.0

    def _verdict(self, position_bias: str, data_quality: str) -> str:
        if data_quality == "weak":
            return "needs_more_data"
        if position_bias == "bullish":
            return "bullish"
        if position_bias == "bearish":
            return "bearish"
        return "neutral"

    def _thesis(self, keyword_hits: int, bullish_hits: int, bearish_hits: int, data_quality: str) -> str:
        if data_quality == "weak":
            return "Insufficient verified data for a domain thesis."
        if bullish_hits > bearish_hits:
            return self.thesis_template
        if bearish_hits > bullish_hits:
            return "Domain context points to a risk-heavy setup rather than a long thesis."
        return "Domain context is mixed; keep as watchlist evidence, not a trade signal."

    def _catalysts(self, context: MarketContext, keyword_hits: int) -> list[str]:
        if keyword_hits == 0:
            return []
        return [f"{self.name} detected {keyword_hits} relevant keyword hits in supplied news/context"]

    def _tailwinds(self, position_bias: str, bullish_hits: int) -> list[str]:
        return [f"{bullish_hits} bullish domain references"] if position_bias == "bullish" else []

    def _headwinds(self, position_bias: str, bearish_hits: int) -> list[str]:
        return [f"{bearish_hits} bearish domain references"] if position_bias == "bearish" else []

    def _reasons(self, keyword_hits: int, bullish_hits: int, bearish_hits: int, data_quality: str) -> list[str]:
        if data_quality == "weak":
            return ["No sufficient domain evidence supplied"]
        return [f"Detected {keyword_hits} domain keyword hits, bullish={bullish_hits}, bearish={bearish_hits}"]

    def _risks(self, data_quality: str, bearish_hits: int) -> list[str]:
        risks = ["Domain agent uses supplied context only; no network retrieval in MVP"]
        if bearish_hits:
            risks.append(f"Detected {bearish_hits} bearish references")
        if data_quality != "strong":
            risks.append("Domain evidence is partial")
        return risks

    def _blind_spots(self) -> list[str]:
        return ["No LLM reasoning or external browsing is used by this MVP agent"]


class MacroPolicyAgent(KeywordDomainAgent):
    keywords = ("rate", "inflation", "fed", "fiscal", "stimulus", "tariff", "sanction", "yield", "cpi")
    bullish_terms = ("rate cut", "easing", "stimulus", "disinflation", "soft landing")
    bearish_terms = ("rate hike", "recession", "sticky inflation", "sanction", "tariff")
    default_horizon_years = 1.5
    thesis_template = "Macro policy context may create a valuation tailwind before it is fully priced."
    asset_or_sector = "macro"

    def _has_structured_data(self, context: MarketContext) -> bool:
        return bool(context.macro)


class GeoPoliticalAgent(KeywordDomainAgent):
    keywords = ("war", "defense", "sanction", "supply chain", "energy security", "export control", "budget")
    bullish_terms = ("defense budget", "rearmament", "energy security", "reshoring", "infrastructure")
    bearish_terms = ("sanction", "export control", "escalation", "blockade", "shortage")
    default_horizon_years = 2.0
    thesis_template = "Geopolitical pressure may redirect capital toward strategic industries."
    asset_or_sector = "geopolitics"


class SectorCycleAgent(KeywordDomainAgent):
    keywords = ("sector", "industry", "relative strength", "cycle", "capex", "orders", "inventory")
    bullish_terms = ("upgrade", "capex boom", "orders growth", "recovery", "relative strength")
    bearish_terms = ("downgrade", "inventory glut", "margin pressure", "slowdown")
    default_horizon_years = 1.0
    thesis_template = "Sector cycle evidence suggests improving relative strength."
    asset_or_sector = "sector"

    def _has_structured_data(self, context: MarketContext) -> bool:
        return bool(context.sector_data)


class IndustryMapAgent(KeywordDomainAgent):
    keywords = ("semiconductor", "software", "industrial", "defense", "energy", "healthcare", "financial")
    bullish_terms = ("backlog", "pricing power", "market share", "new contract", "demand growth")
    bearish_terms = ("competition", "margin compression", "demand decline", "regulatory pressure")
    default_horizon_years = 2.0
    thesis_template = "Industry-level structure may favor companies with pricing power and durable demand."
    asset_or_sector = "industry"


class NewsCatalystAgent(KeywordDomainAgent):
    keywords = ("contract", "earnings", "guidance", "approval", "merger", "buyback", "partnership", "probe")
    bullish_terms = ("beat", "raise guidance", "approval", "buyback", "contract win", "partnership")
    bearish_terms = ("miss", "cut guidance", "probe", "lawsuit", "delay", "recall")
    default_horizon_years = 0.5
    thesis_template = "A recent catalyst may force near-term repricing."
    asset_or_sector = "news_catalyst"


class ContrarianThesisAgent(KeywordDomainAgent):
    keywords = ("undervalued", "ignored", "underappreciated", "selloff", "discount", "unloved", "mispriced")
    bullish_terms = ("undervalued", "underappreciated", "discount", "mispriced", "margin of safety")
    bearish_terms = ("value trap", "structural decline", "debt concern", "terminal decline")
    default_horizon_years = 2.0
    thesis_template = "The market may be underpricing a recoverable business or sector setup."
    asset_or_sector = "contrarian"


class ValueScreeningAgent(KeywordDomainAgent):
    default_horizon_years = 3.0
    asset_or_sector = "value"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        structured_audit = apply_market_context_structured_boundary(
            context
        )
        gate = _fundamental_gate(context)
        if (
            structured_audit["input_count"] > 0
            and not context.fundamentals
        ):
            return self._blocked_by_structured_boundary(
                context,
                structured_audit,
            )
        if (
            context.fundamentals
            and gate.get(
                "can_feed_value_screening_after_manual_review"
            )
            is not True
        ):
            return self._blocked_by_fundamental_gate(context, gate)
        gate_fingerprint = (
            gate.get("gate_structured_accepted_fingerprint")
            or gate.get("structured_accepted_fingerprint")
            or (gate.get("structured_context_audit") or {}).get("accepted_fingerprint")
        )
        scores = self._score_fundamentals(context.fundamentals)
        if not scores:
            return self._blocked_missing_screening_metrics(
                context,
                gate,
            )
        average_score = sum(scores.values()) / len(scores)
        best_ticker = max(scores, key=scores.get)
        best_score = scores[best_ticker]
        data_quality = "partial"
        position_bias = "bullish" if best_score >= 0.6 else "neutral"
        verdict = "undervalued" if best_score >= 0.7 else "neutral"
        confidence = clamp(0.35 + best_score * 0.45, 0.0, 0.9)
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=confidence,
            data_quality_score=0.65,
            signal_strength=confidence if position_bias == "bullish" else 0.0,
            ticker=best_ticker,
            asset_or_sector="value",
            horizon_years=float(self.config.get("horizon_years", self.default_horizon_years)),
            thesis=f"{best_ticker} has the strongest supplied value screen score.",
            data_quality=data_quality,
            position_bias=position_bias,
            valuation_gap=f"best_value_score={best_score:.2f}; average_value_score={average_score:.2f}",
            watchlist_score=clamp(best_score, 0.0, 1.0),
            catalysts=["Fundamental value screen passed threshold"] if best_score >= 0.6 else [],
            tailwinds=["Margin of safety indicators are present"] if best_score >= 0.6 else [],
            headwinds=[],
            reasons=[
                f"Best supplied value score: {best_ticker}={best_score:.2f}",
                f"Average value score across supplied fundamentals: {average_score:.2f}",
            ],
            risks=_fundamental_gate_risks(gate),
            blind_spots=[
                "No SEC filing parser or full financial statement normalization is active yet",
                "No ratio computation, ratio interpretation, valuation, recommendation, allocation, or trading is authorized.",
            ],
            evidence=[
                self.evidence("fundamental", "context.fundamentals", "value_scores", scores),
                self.evidence("fundamental", "context.metadata.fundamental_input_readiness_gate", "gate", gate),
            ],
            input_hash=self.context_hash(context),
        )

    def _blocked_missing_screening_metrics(
        self,
        context: MarketContext,
        gate: dict[str, Any],
    ) -> AnalyticalReport:
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict="needs_more_data",
            confidence=0.2,
            data_quality_score=0.4,
            signal_strength=0.0,
            ticker=(
                ",".join(context.tickers)
                if context.tickers
                else None
            ),
            asset_or_sector="value",
            horizon_years=float(
                self.config.get(
                    "horizon_years",
                    self.default_horizon_years,
                )
            ),
            thesis=(
                "Source-bound statement facts are available, but the "
                "required value-screening ratios are not."
            ),
            data_quality="partial",
            position_bias="insufficient_data",
            valuation_gap=None,
            watchlist_score=0.0,
            catalysts=[],
            tailwinds=[],
            headwinds=[],
            reasons=[
                (
                    "Value screening requires at least one of pe, pb, "
                    "debt_to_equity, fcf_yield, or roe."
                ),
                (
                    "Raw revenue, income, assets, liabilities, equity, "
                    "cash, or capex are evidence inputs, not valuation "
                    "scores."
                ),
            ],
            risks=_fundamental_gate_risks(gate),
            blind_spots=[
                (
                    "No reviewed ratio computation or market-price "
                    "alignment is active."
                ),
                (
                    "No valuation, recommendation, allocation, or "
                    "trading action is authorized."
                ),
            ],
            evidence=[
                self.evidence(
                    "fundamental",
                    "context.fundamentals",
                    "available_metric_names",
                    {
                        ticker: sorted(
                            key
                            for key in metrics
                            if not str(key).startswith("_")
                        )
                        for ticker, metrics in sorted(
                            context.fundamentals.items()
                        )
                    },
                ),
                self.evidence(
                    "fundamental",
                    (
                        "context.metadata."
                        "fundamental_input_readiness_gate"
                    ),
                    "gate",
                    gate,
                ),
            ],
            input_hash=self.context_hash(context),
        )

    def _blocked_by_structured_boundary(
        self,
        context: MarketContext,
        audit: dict[str, Any],
    ) -> AnalyticalReport:
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict="needs_more_data",
            confidence=0.1,
            data_quality_score=0.1,
            signal_strength=0.0,
            ticker=",".join(context.tickers) if context.tickers else None,
            asset_or_sector="value",
            horizon_years=float(
                self.config.get(
                    "horizon_years",
                    self.default_horizon_years,
                )
            ),
            thesis=(
                "Fundamental value screen is blocked because no supplied "
                "metric satisfies the point-in-time semantic contract."
            ),
            data_quality="weak",
            position_bias="insufficient_data",
            valuation_gap=None,
            watchlist_score=0.0,
            catalysts=[],
            tailwinds=[],
            headwinds=[],
            reasons=[
                f"Structured context status: {audit.get('status')}",
                (
                    "Every metric needs value, unit, period, availability "
                    "timestamp, and source locator."
                ),
            ],
            risks=[
                "Rejected fundamental values were not used for scoring."
            ],
            blind_spots=[
                "No ratio computation, valuation, recommendation, allocation, or trading is authorized."
            ],
            evidence=[
                self.evidence(
                    "fundamental",
                    "context.metadata.structured_context_point_in_time_audit",
                    "audit",
                    context.metadata.get(
                        "structured_context_point_in_time_audit",
                        {},
                    ),
                )
            ],
            input_hash=self.context_hash(context),
        )

    def _blocked_by_fundamental_gate(self, context: MarketContext, gate: dict[str, Any]) -> AnalyticalReport:
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict="needs_more_data",
            confidence=0.2,
            data_quality_score=0.2,
            signal_strength=0.0,
            ticker=",".join(context.tickers) if context.tickers else None,
            asset_or_sector="value",
            horizon_years=float(self.config.get("horizon_years", self.default_horizon_years)),
            thesis="Fundamental value screen is blocked until the fundamental input readiness gate is clean.",
            data_quality="weak",
            position_bias="insufficient_data",
            valuation_gap=None,
            watchlist_score=0.0,
            catalysts=[],
            tailwinds=[],
            headwinds=[],
            reasons=[
                f"Fundamental input gate status: {gate.get('readiness_status')}",
                "Value screening requires a clean reviewed fundamental input gate when one is attached.",
            ],
            risks=_fundamental_gate_risks(gate),
            blind_spots=[
                "Fundamental values were not used for scoring.",
                "No ratio computation, ratio interpretation, valuation, recommendation, allocation, or trading is authorized.",
            ],
            evidence=[self.evidence("fundamental", "context.metadata.fundamental_input_readiness_gate", "gate", gate)],
            input_hash=self.context_hash(context),
        )

    def _extract_value(self, raw: Any) -> float | None:
        if isinstance(raw, dict):
            raw = raw.get("value")
        return self._as_float(raw)

    def _score_fundamentals(self, fundamentals: dict[str, dict[str, Any]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ticker, metrics in fundamentals.items():
            points = 0.0
            checks = 0
            pe = self._extract_value(metrics.get("pe") or metrics.get("price_to_earnings"))
            pb = self._extract_value(metrics.get("pb") or metrics.get("price_to_book"))
            debt_to_equity = self._extract_value(metrics.get("debt_to_equity"))
            fcf_yield = self._extract_value(metrics.get("fcf_yield"))
            roe = self._extract_value(metrics.get("roe"))
            if pe is not None:
                checks += 1
                points += 1.0 if pe > 0 and pe <= 15 else 0.0
            if pb is not None:
                checks += 1
                points += 1.0 if pb > 0 and pb <= 1.5 else 0.0
            if debt_to_equity is not None:
                checks += 1
                points += 1.0 if debt_to_equity <= 1.0 else 0.0
            if fcf_yield is not None:
                checks += 1
                points += 1.0 if fcf_yield >= 0.05 else 0.0
            if roe is not None:
                checks += 1
                points += 1.0 if roe >= 0.12 else 0.0
            if checks:
                scores[ticker] = points / checks
        return scores

    def _as_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _fundamental_gate(context: MarketContext) -> dict[str, Any]:
    gate = context.metadata.get("fundamental_input_readiness_gate")
    if not isinstance(gate, dict):
        return {"gate_attached": False, "readiness_status": "not_attached"}
    summary = gate.get("summary") or {}
    gate_enhanced = {
        "gate_attached": True,
        "can_feed_value_screening_after_manual_review": (
            summary.get("can_feed_value_screening_after_manual_review") or False
        ),
        "structured_accepted_fingerprint": (
            summary.get("structured_accepted_fingerprint")
            or (gate.get("structured_context_audit") or {}).get("accepted_fingerprint")
        ),
        "readiness_status": summary.get("readiness_status", "not_attached"),
    }
    return gate_enhanced


def _fundamental_gate_risks(gate: dict[str, Any]) -> list[str]:
    risks = ["Fundamental data feed is caller-supplied and not independently verified"]
    if gate.get("gate_attached") is not True:
        risks.append("No FundamentalInputReadinessGate artifact is attached.")
    elif gate.get("can_feed_value_screening_after_manual_review") is not True:
        risks.append("Attached FundamentalInputReadinessGate is not clean enough for value screening.")
    else:
        risks.append("FundamentalInputReadinessGate is attached and clean, but still review-only.")
    return risks
