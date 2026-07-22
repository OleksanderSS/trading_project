from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import yaml

from dean_os.base import AnalyticalAgent
from dean_os.schemas import AnalyticalReport, BaseAgentReport, MarketContext
from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state import WorldStateBuilder, WorldStateSnapshot

logger = logging.getLogger(__name__)

# Mapping from current regime/sector verdicts to historical regime tags
VERDICT_TO_TAG: dict[str, list[str]] = {
    "bullish": ["industrial_expansion", "asset_inflation", "productivity", "tech_cycle"],
    "bearish": ["recession", "depression", "banking_crisis", "credit_freeze", "deflation"],
    "caution": ["late_cycle", "pre_crisis", "recession_risk", "valuation_reset"],
    "neutral": ["globalization", "low_rates", "qe"],
    "CRISIS": ["banking_crisis", "credit_freeze", "emergency_policy", "crash"],
    "RECOVERY": ["reconstruction", "fiscal_stimulus", "industrial_expansion"],
    "GROWTH": ["industrial_expansion", "globalization", "tech_cycle"],
    "mixed": ["late_cycle", "multipolarity", "supply_constraints"],
    "insufficient_data": ["uncertainty", "policy_experimentation"],
}

SECTOR_TO_TAG: dict[str, list[str]] = {
    "semiconductor_ai_infrastructure": ["tech_cycle", "ai_capex", "supply_chain_relocalization"],
    "energy": ["oil_shock", "energy_security", "energy_shock", "energy_transition"],
    "agriculture": ["food_shock", "supply_chain_disruption", "commodity_cycle"],
    "logistics": ["supply_chain_disruption", "supply_constraints", "trade_disruption"],
    "real_estate": ["housing_crash", "asset_inflation", "credit_pressure"],
    "macro_policy": ["inflation", "rate_hikes", "monetary_tightening", "fiscal_stimulus", "qe"],
}


class HistoricalAnalogiesAgent(AnalyticalAgent):
    """Compares current World State against historical periods.

    Uses structured world state dimensions (regime, sector stances,
    news events) instead of raw keyword matching against news text.
    """

    version = "0.3.0"
    default_horizon_years = 3.0

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.periods = self._load_historical_periods()
        self.ws_builder = WorldStateBuilder()

    def _load_historical_periods(self) -> list[dict[str, Any]]:
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "draft",
            "dean_os_after_385_macro_regime_historical_hypothesis_kit",
            "HISTORICAL_PERIODS_SEED_LIST.yaml",
        )
        if not os.path.exists(yaml_path):
            logger.warning(f"Historical periods not found at {yaml_path}")
            return []
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data.get("historical_periods_seed_list", [])
                if isinstance(data, list):
                    return data
                return []
        except Exception as e:
            logger.error(f"Failed to load historical periods: {e}")
            return []

    async def run(self, context: MarketContext) -> AnalyticalReport:
        if not self.periods:
            return self._blocked_report("No historical periods seed list available.")

        # Build current tags from structured world state.
        # Historical analogies must see preflight/pipeline reports in context.
        reports = getattr(context, "_agent_reports", []) or []
        if not reports and isinstance(context.metadata, dict):
           raw_reports = context.metadata.get("agent_reports", [])
           if isinstance(raw_reports, list):
               parsed_reports: list[BaseAgentReport] = []
               for raw in raw_reports:
                   if isinstance(raw, dict):
                       try:
                           parsed_reports.append(BaseAgentReport.model_validate(raw))
                       except Exception:
                           continue
               reports = parsed_reports

        decision = getattr(context, "_decision", None)
        if reports:
           world_state = self.ws_builder.build(
               reports=reports, decision=decision, as_of=context.as_of,
           )
        else:
           world_state = None

        current_tags = self._extract_current_tags(world_state, context)

        if not current_tags:
            return self._blocked_report("No current regime/sector dimensions to compare.")

        # Score each historical period against current tags
        scored: list[dict[str, Any]] = []
        for period in self.periods:
            period_tags = [str(t).lower().replace("_", " ") for t in period.get("regime_tags", [])]
            current_clean = [t.lower().replace("_", " ") for t in current_tags]

            matches = sum(1 for ct in current_clean if any(ct in pt or pt in ct for pt in period_tags))
            total = max(len(period_tags), 1)
            score = matches / total

            scored.append({
                "period_id": period["period_id"],
                "years": period.get("years", ""),
                "tags": period.get("regime_tags", []),
                "score": round(score, 3),
                "matched_tags": [
                    t for t in period.get("regime_tags", [])
                    if any(ct in str(t).lower().replace("_", " ") or str(t).lower().replace("_", " ") in ct for ct in current_clean)
                ],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:3] if scored[0]["score"] > 0 else []

        if not top:
            return self._blocked_report("No historical periods matched current world state.")

        best = top[0]
        confidence = min(0.15 + (best["score"] * 0.8), 0.9)
        bearish_periods = [
            "great_depression", "volcker_disinflation", "global_financial_crisis",
            "covid_shock", "second_oil_shock_late_1970s",
        ]
        is_bearish = any(bp in best["period_id"] for bp in bearish_periods)
        verdict = "bearish" if is_bearish else "bullish" if best["score"] > 0.4 else "neutral"
        position_bias = verdict

        thesis_parts = [
            f"Current World State matches '{best['period_id']}' ({best['years']})",
            f"with {best['score']:.0%} similarity",
        ]
        if top[1]["score"] > 0.2:
            thesis_parts.append(f"secondary analog: '{top[1]['period_id']}' ({top[1]['score']:.0%})")
        if top[2]["score"] > 0.2:
            thesis_parts.append(f"tertiary analog: '{top[2]['period_id']}' ({top[2]['score']:.0%})")
        thesis = "; ".join(thesis_parts)

        matched_tags_str = ", ".join(str(t) for t in best["matched_tags"])

        calibration_adj = self._apply_calibration(confidence, verdict)
        adjusted_confidence = calibration_adj["adjusted_confidence"]

        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=adjusted_confidence,
            data_quality_score=0.8 if best["score"] > 0.4 else 0.5,
            signal_strength=adjusted_confidence if verdict == "bullish" else -adjusted_confidence if verdict == "bearish" else 0.0,
            ticker=None,
            asset_or_sector="macro",
            horizon_years=float(self.config.get("horizon_years", self.default_horizon_years)),
            thesis=thesis,
            data_quality="strong" if best["score"] > 0.4 else "partial",
            position_bias=position_bias,
            catalysts=[f"Regime alignment with {best['period_id']}"],
            tailwinds=[f"Matched precedent: {best['period_id']}"] if verdict == "bullish" else [],
            headwinds=[f"Matched precedent: {best['period_id']}"] if verdict == "bearish" else [],
            watchlist_score=adjusted_confidence,
            reasons=[
                f"Primary analog: {best['period_id']} (score={best['score']:.2f}, tags=[{matched_tags_str}])",
                f"Secondary: {top[1]['period_id']} ({top[1]['score']:.2f})" if len(top) > 1 else "",
                f"World state active tags: {', '.join(current_tags[:8])}",
            ],
            risks=[
                "Historical analogies are not deterministic predictions.",
                "Structural changes since the matched period may invalidate the analogy.",
            ],
            blind_spots=[
                "Only regime/sector level dimensions compared, not company-level fundamentals.",
                "Limited to {len(self.periods)} seed periods in library.",
            ],
            evidence=[
                self.evidence("historical_match", "HISTORICAL_PERIODS_SEED_LIST", "scored_periods", scored[:5]),
                self.evidence("historical_match", self.name, "current_tags", current_tags),
                self.evidence("historical_match", self.name, "best_period", best["period_id"]),
                self.evidence("historical_match", self.name, "best_score", best["score"]),
                self.evidence("historical_match", self.name, "matched_tags", best["matched_tags"]),
                self.evidence("calibration", "outcome_tracker", "adjustment", calibration_adj),
            ],
            input_hash=self.context_hash(context),
        )

    def _extract_current_tags(
        self,
        world_state: WorldStateSnapshot | None,
        context: MarketContext,
    ) -> list[str]:
        tags: list[str] = []

        if world_state:
            regime = world_state.global_state.regime
            if regime:
                tags.extend(VERDICT_TO_TAG.get(regime.upper(), []))
                tags.append(f"regime:{regime}")

            for sector_id, sector_state in world_state.sectors.items():
                tags.append(f"sector:{sector_id}:{sector_state.stance}")
                tags.extend(VERDICT_TO_TAG.get(sector_state.stance, []))
                tags.extend(SECTOR_TO_TAG.get(sector_id, []))

            macro_stance = world_state.global_state.macro_stance
            if macro_stance:
                tags.extend(VERDICT_TO_TAG.get(macro_stance, []))

        # From news context
        if context.news:
            tags.append("news:available")

        # From fundamentals
        if context.fundamentals:
            tags.append("fundamentals:available")

        # From macro data
        if context.macro:
            tags.append("macro:available")

        return list(set(tags))

    def _apply_calibration(self, base_confidence: float, verdict: str) -> dict:
        """Load tracker calibration and adjust confidence."""
        try:
            from dean_os.draft.dean_os_agent_system_v7.dean_os.outcome_tracker import OutcomeTracker
            tracker = OutcomeTracker()
            cal = tracker.calibrate()
        except Exception:
            return {"base_confidence": base_confidence, "adjusted_confidence": base_confidence, "adjustment": 0.0}

        if cal.total_outcomes < 5:
            return {"base_confidence": base_confidence, "adjusted_confidence": base_confidence, "adjustment": 0.0}

        accuracy = cal.accuracy_rate
        brier = cal.brier_score
        penalty = max(0.0, 0.5 - accuracy) * 0.5
        if brier > 0.25:
            penalty += (brier - 0.25) * 0.3
        adjusted = max(0.05, base_confidence - penalty)
        return {
            "base_confidence": round(base_confidence, 3),
            "adjusted_confidence": round(adjusted, 3),
            "adjustment": round(adjusted - base_confidence, 3),
            "tracker_accuracy": accuracy,
            "tracker_brier": brier,
            "total_outcomes": cal.total_outcomes,
        }

    def _blocked_report(self, reason: str) -> AnalyticalReport:
        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict="needs_more_data",
            confidence=0.0,
            data_quality_score=0.0,
            signal_strength=0.0,
            ticker=None,
            asset_or_sector="historical_analogy",
            horizon_years=float(self.config.get("horizon_years", self.default_horizon_years)),
            thesis=f"Blocked: {reason}",
            data_quality="weak",
            position_bias="insufficient_data",
            watchlist_score=0.0,
            catalysts=[],
            tailwinds=[],
            headwinds=[],
            reasons=[reason],
            risks=["No historical analogies could be drawn."],
            blind_spots=[],
            evidence=[],
            input_hash=None,
        )
