from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dean_os.base import AnalyticalAgent
from dean_os.schemas import AnalyticalReport, MarketContext


def _parse_ts(val: Any) -> datetime | None:
    if not val:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=UTC)
    raw = str(val).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


FRESHNESS_THRESHOLDS: dict[str, int] = {
    "news": 7,        # 7 days
    "macro": 30,      # 30 days
    "prices": 5,      # 5 days
    "fundamentals": 90,  # 90 days (quarterly)
}


class FreshnessAuditAgent(AnalyticalAgent):
    """Audits MarketContext field timestamps vs as_of.

    Flags stale data (older than threshold) and missing timestamps.
    """

    version = "0.1.0"
    branch = "analytical"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        flags: list[dict[str, Any]] = []
        healthy = 0
        stale = 0
        missing_ts = 0

        as_of = _parse_ts(context.as_of) or datetime.now(UTC)

        # News freshness
        news_items = context.news or []
        if isinstance(news_items, list):
            for item in news_items[:200]:
                ts = _parse_ts(item.get("published_at") or item.get("timestamp") or item.get("date"))
                if ts:
                    age = (as_of - ts).days
                    if age > FRESHNESS_THRESHOLDS["news"]:
                        stale += 1
                        if stale <= 3:
                            flags.append({"field": "news", "age_days": age, "severity": "stale"})
                    else:
                        healthy += 1
                else:
                    missing_ts += 1

        # Macro freshness
        macro = context.macro or {}
        for key, obs in (macro.items() if isinstance(macro, dict) else []):
            ts = _parse_ts(obs.get("available_at") if isinstance(obs, dict) else None)
            if ts:
                age = (as_of - ts).days
                if age > FRESHNESS_THRESHOLDS["macro"]:
                    stale += 1
                    flags.append({"field": f"macro.{key}", "age_days": age, "severity": "stale"})
                else:
                    healthy += 1
            else:
                missing_ts += 1

        # Fundamentals freshness
        for ticker, data in (context.fundamentals or {}).items():
            for metric, val in (data.items() if isinstance(data, dict) else []):
                ts = _parse_ts(val.get("timestamp") or val.get("period") if isinstance(val, dict) else None)
                if ts:
                    age = (as_of - ts).days
                    if age > FRESHNESS_THRESHOLDS["fundamentals"]:
                        stale += 1
                        flags.append({"field": f"fundamentals.{ticker}.{metric}", "age_days": age, "severity": "stale"})
                    else:
                        healthy += 1
                else:
                    missing_ts += 1

        total = healthy + stale + missing_ts
        freshness_score = healthy / max(total, 1)
        verdict = "caution" if freshness_score < 0.7 or stale > 0 else "neutral"
        confidence = max(0.5, freshness_score)

        reasons = [
            f"Checked {total} data points across news/macro/fundamentals",
            f"Fresh: {healthy}, Stale: {stale}, Missing timestamps: {missing_ts}",
            f"Freshness score: {freshness_score:.0%}",
        ]
        if stale > 0:
            reasons.append(f"Stale items: {flags[:3]}")

        evidence = [
            self.evidence("report", self.name, "freshness_score", round(freshness_score, 3)),
            self.evidence("report", self.name, "stale_count", stale),
            self.evidence("report", self.name, "healthy_count", healthy),
            self.evidence("report", self.name, "missing_timestamps", missing_ts),
        ]
        for f in flags[:5]:
            evidence.append(self.evidence("report", self.name, f"stale:{f['field']}", f))

        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=confidence,
            data_quality_score=freshness_score,
            signal_strength=-0.2 if stale > 3 else 0.0,
            ticker="MULTI",
            asset_or_sector="global",
            horizon_years=0.25,
            thesis=f"Data freshness: {healthy}/{total} fresh, {stale} stale, {missing_ts} missing timestamps",
            data_quality="strong" if freshness_score > 0.8 else "partial" if freshness_score > 0.5 else "weak",
            position_bias="insufficient_data" if stale > 5 else "neutral",
            catalysts=[],
            headwinds=[f["field"] for f in flags[:3]],
            watchlist_score=freshness_score,
            reasons=reasons,
            risks=["Stale data may cause incorrect conclusions.", "Missing timestamps cannot be validated."],
            blind_spots=["Only checks top 200 news items.", "Thresholds are static defaults."],
            evidence=evidence,
            input_hash=self.context_hash(context),
        )


__all__ = ["FreshnessAuditAgent"]
