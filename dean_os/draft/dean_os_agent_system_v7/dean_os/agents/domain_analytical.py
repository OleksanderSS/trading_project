from __future__ import annotations

from typing import Any

from dean_os.agents.domain_analyst import DomainAnalystAgent
from dean_os.base import AnalyticalAgent
from dean_os.schemas import AnalyticalReport, EvidenceItem, MarketContext
from dean_os.utils import sha256_json


class DomainAnalyticalAgent(AnalyticalAgent):
    """Compatibility adapter that places domain analysis on the analytical branch.

    The existing :class:`DomainAnalystAgent` remains available for old review
    packets that expect ``PipelineReport``. This adapter reuses its evidence and
    sector-analysis implementation but returns an ``AnalyticalReport`` and has
    no direct trade authority.
    """

    version = "1.0.0"
    branch = "analytical"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        legacy = DomainAnalystAgent(name=self.name, config=self.config)
        source_report = await legacy.run(context)
        payload = dict(source_report.metrics_snapshot or {})
        domain_id = str(payload.get("domain_id") or self.config.get("domain_id") or "unknown")
        horizon_days = int(self.config.get("horizon_days") or 180)

        thesis = str(payload.get("thesis") or "").strip()
        if not thesis:
            thesis = source_report.reasons[0] if source_report.reasons else f"{domain_id} analysis requires review."

        report = AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=source_report.verdict,
            confidence=source_report.confidence,
            data_quality_score=source_report.data_quality_score,
            # Domain analysis updates the world model and review queue. It does
            # not directly create a trading signal for consensus.
            signal_strength=0.0,
            reasons=list(source_report.reasons),
            risks=list(source_report.risks),
            blind_spots=list(source_report.blind_spots),
            evidence=[
                *source_report.evidence,
                EvidenceItem(
                    source_type="report",
                    source=self.name,
                    key="domain_analysis_payload",
                    value=payload,
                    timestamp=context.as_of,
                ),
            ],
            input_hash=source_report.input_hash or self.context_hash(context),
            config_hash=source_report.config_hash or sha256_json(self.config),
            ticker=None,
            asset_or_sector=domain_id,
            horizon_years=max(horizon_days / 365.0, 0.0),
            thesis=thesis,
            data_quality=_quality_label(source_report.data_quality_score),
            position_bias=_position_bias(payload, source_report.verdict),
            catalysts=_hypothesis_texts(payload),
            tailwinds=_tailwinds(payload),
            headwinds=list(source_report.risks[:8]),
            valuation_gap=_expectation_gap(payload),
            watchlist_score=_watchlist_score(source_report.verdict, source_report.confidence),
            analysis_payload={
                "domain_id": domain_id,
                "source_report": source_report.model_dump(mode="json"),
                "domain_metrics": payload,
                "authority_boundary": {
                    "review_only": True,
                    "decision_influence": False,
                    "can_create_ticker_forecast": False,
                    "can_trade": False,
                    "can_write_pipeline_config": False,
                },
            },
        )
        return report


def _quality_label(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.45:
        return "partial"
    return "weak"


def _position_bias(payload: dict[str, Any], verdict: str) -> str:
    stance = str(payload.get("stance") or "").lower()
    if stance in {"constructive", "bullish", "positive"} or verdict in {"bullish", "undervalued"}:
        return "bullish"
    if stance in {"risk_heavy", "bearish", "negative"} or verdict in {"bearish", "overvalued"}:
        return "bearish"
    if verdict == "needs_more_data" or stance == "insufficient_data":
        return "insufficient_data"
    return "neutral"


def _hypothesis_texts(payload: dict[str, Any]) -> list[str]:
    results: list[str] = []
    for item in payload.get("hypotheses", []) or []:
        if isinstance(item, dict):
            text = item.get("hypothesis") or item.get("statement") or item.get("summary")
        else:
            text = str(item)
        if text:
            results.append(str(text))
    return results[:8]


def _tailwinds(payload: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    for item in payload.get("watch_signals", []) or []:
        if isinstance(item, dict):
            text = item.get("summary") or item.get("signal") or item.get("label")
        else:
            text = str(item)
        if text:
            signals.append(str(text))
    return signals[:8]


def _expectation_gap(payload: dict[str, Any]) -> str | None:
    value = payload.get("expectation_gap")
    if not value:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("summary") or value.get("status") or value)
    return str(value)


def _watchlist_score(verdict: str, confidence: float) -> float:
    if verdict in {"blocked", "needs_more_data"}:
        return 0.0
    if verdict == "caution":
        return min(confidence, 0.5)
    return min(max(confidence, 0.0), 1.0)


__all__ = ["DomainAnalyticalAgent"]
