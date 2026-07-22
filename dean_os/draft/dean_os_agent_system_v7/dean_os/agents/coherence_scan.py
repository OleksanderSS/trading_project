from __future__ import annotations

from typing import Any

from dean_os.base import AnalyticalAgent
from dean_os.draft.dean_os_agent_system_v7.dean_os.outcome_tracker import OutcomeTracker
from dean_os.schemas import AnalyticalReport, MarketContext

# Domain ↔ ticker keywords for overlap detection
DOMAIN_TICKER_HINTS: dict[str, list[str]] = {
    "semiconductor_ai_infrastructure": ["nvda", "amd", "intc", "tsm", "mu", "asml", "amat", "klac", "qcom", "mrvl", "mchi", "stm"],
    "energy": ["xle", "uso", "xom", "cvx", "shell", "bp", "cop", "oih", "eqnr"],
    "macro_policy": ["spy", "qqq", "iwm", "tlt", "shy", "gld", "dxy"],
    "agriculture": ["adm", "bg", "de", "mos", "cf", "mon"],
    "logistics": ["fdx", "ups", "jbht", "xpo", "csx", "unp"],
    "real_estate": ["pld", "amt", "eqix", "well", "spg", "o"],
    "geopolitics": ["spy", "gld", "usd", "oil", "gdx"],
    "liquidity_credit": ["tlt", "hyg", "shy", "lqd", "bklc"],
}

# Agents that produce directional verdicts by domain
AGENT_DOMAIN_MAP: dict[str, str] = {
    "semiconductor_analyst": "semiconductor_ai_infrastructure",
    "energy_analyst": "energy",
    "macro_analyst": "macro_policy",
    "agriculture_analyst": "agriculture",
    "logistics_analyst": "logistics",
    "real_estate_analyst": "real_estate",
    "macro_policy": "macro_policy",
    "geopolitical": "geopolitics",
    "news_catalyst": "global",
    "sector_cycle": "global",
    "historical_analogies": "global",
    "value_screening": "global",
    "contrarian_thesis": "global",
    "industry_map": "global",
}

# Pairs of domains that often overlap (shared tickers, shared news keywords)
OVERLAP_PAIRS: list[tuple[str, str]] = [
    ("semiconductor_ai_infrastructure", "energy"),  # NVDA power usage, data center energy
    ("semiconductor_ai_infrastructure", "geopolitics"),  # Taiwan/chip sanctions
    ("semiconductor_ai_infrastructure", "macro_policy"),  # rates affect semicon capex
    ("energy", "geopolitics"),  # oil sanctions
    ("energy", "macro_policy"),  # rates vs energy demand
    ("energy", "logistics"),  # fuel costs
    ("agriculture", "logistics"),  # grain shipping
    ("agriculture", "energy"),  # fertilizer = natgas
    ("real_estate", "macro_policy"),  # rates affect REITs
    ("liquidity_credit", "macro_policy"),  # fed policy → credit
    ("liquidity_credit", "real_estate"),  # credit → REITs
    ("geopolitics", "energy"),
    ("geopolitics", "semiconductor_ai_infrastructure"),
]


class CoherenceScanAgent(AnalyticalAgent):
    """Cross-references agent verdicts for domain-level contradictions.

    Flags pairs of agents covering overlapping domains whose verdicts
    conflict (e.g. semiconductor bullish vs energy bearish when both
    share the NVDA→power narrative).
    """

    version = "0.1.0"
    branch = "analytical"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        reports = getattr(context, "_agent_reports", None) or context.metadata.get("agent_reports", [])
        if not reports:
            return self._empty("No agent reports to scan.")

        verdicts: dict[str, dict[str, Any]] = {}
        for r in reports:
            name = r.get("agent_name", r.get("name", ""))
            verdict = r.get("verdict", "neutral")
            conf = r.get("confidence", 0.5)
            domain = AGENT_DOMAIN_MAP.get(name, "")
            if domain:
                verdicts[name] = {"domain": domain, "verdict": verdict, "confidence": conf}

        if not verdicts:
            return self._empty("No recognized domain agents in reports.")

        contradictions: list[dict[str, Any]] = []
        checked_pairs: set[tuple[str, str]] = set()

        for name_a, info_a in verdicts.items():
            for name_b, info_b in verdicts.items():
                if name_a >= name_b:
                    continue
                pair = tuple(sorted([name_a, name_b]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                dom_a, dom_b = info_a["domain"], info_b["domain"]
                if dom_a == dom_b or (dom_a, dom_b) in OVERLAP_PAIRS or (dom_b, dom_a) in OVERLAP_PAIRS:
                    va, vb = info_a["verdict"], info_b["verdict"]
                    if self._is_contradiction(va, vb):
                        contradictions.append({
                            "agent_a": name_a, "verdict_a": va, "conf_a": info_a["confidence"],
                            "agent_b": name_b, "verdict_b": vb, "conf_b": info_b["confidence"],
                            "domains": f"{dom_a} ↔ {dom_b}",
                            "severity": "high" if (info_a["confidence"] > 0.6 and info_b["confidence"] > 0.6) else "medium",
                        })

        contradiction_count = len(contradictions)
        total_compared = len(checked_pairs)
        coherence = 1.0 - (contradiction_count / max(total_compared, 1))

        # Also load tracker calibration
        cal = self._load_calibration()

        reasons = [
            f"Scanned {len(verdicts)} domain agents across {total_compared} overlap pairs",
            f"Found {contradiction_count} contradictions (coherence={coherence:.2f})",
        ]
        if contradictions:
            top = contradictions[0]
            reasons.append(f"Top: {top['agent_a']}({top['verdict_a']}) vs {top['agent_b']}({top['verdict_b']}) on {top['domains']}")
        if cal:
            reasons.append(f"Tracker calibration: accuracy={cal.get('accuracy_rate', 0):.0%}, {cal.get('total_outcomes', 0)} outcomes")

        evidence = [
            self.evidence("report", self.name, "coherence_score", round(coherence, 3)),
            self.evidence("report", self.name, "contradictions", contradiction_count),
            self.evidence("report", self.name, "agents_scanned", len(verdicts)),
        ]
        for c in contradictions[:3]:
            evidence.append(self.evidence("report", self.name, f"contradiction:{c['domains']}", c))

        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict="caution" if contradiction_count > 0 else "neutral",
            confidence=max(0.5, coherence),
            data_quality_score=min(total_compared / 20.0, 1.0),
            signal_strength=0.5 - (contradiction_count * 0.1),
            ticker="MULTI",
            asset_or_sector="global",
            horizon_years=0.5,
            thesis=f"Coherence scan: {contradiction_count} contradictions across {len(verdicts)} agents (coherence={coherence:.2f})",
            data_quality="strong" if total_compared >= 10 else "partial",
            position_bias="insufficient_data" if coherence < 0.3 else "neutral",
            catalysts=[f"{c['agent_a']} vs {c['agent_b']}" for c in contradictions[:3]],
            headwinds=[f"{c['domains']}: {c['verdict_a']} vs {c['verdict_b']}" for c in contradictions[:3]],
            watchlist_score=max(0.0, 1.0 - contradiction_count * 0.2),
            reasons=reasons,
            risks=["Contradictions may indicate regime uncertainty, not error."],
            blind_spots=[
                "Only checks pairwise verdicts, not underlying reasoning.",
                "Domain overlap map is static; real overlap varies by market conditions.",
            ],
            evidence=evidence,
            input_hash=self.context_hash(context),
        )

    @staticmethod
    def _is_contradiction(v1: str, v2: str) -> bool:
        bulls = {"bullish", "clear"}
        bears = {"bearish", "blocked"}
        return (v1 in bulls and v2 in bears) or (v1 in bears and v2 in bulls)

    @staticmethod
    def _load_calibration() -> dict | None:
        try:
            cal = OutcomeTracker().calibrate()
            if cal.total_outcomes > 0:
                return {
                    "total_outcomes": cal.total_outcomes,
                    "accuracy_rate": cal.accuracy_rate,
                    "brier_score": cal.brier_score,
                }
        except Exception:
            pass
        return None

    def _empty(self, reason: str) -> AnalyticalReport:
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict="needs_more_data",
            confidence=0.0,
            data_quality_score=0.0,
            signal_strength=0.0,
            ticker="MULTI",
            asset_or_sector="global",
            horizon_years=0.5,
            thesis=f"Blocked: {reason}",
            data_quality="weak",
            position_bias="insufficient_data",
            watchlist_score=0.0,
            reasons=[reason],
            risks=[],
            blind_spots=[],
            evidence=[],
            input_hash=None,
        )


__all__ = ["CoherenceScanAgent"]
