"""Expectation-gap lens with a strict measured-vs-qualitative boundary.

Keyword novelty, crowding, and staleness remain review context. They never
become probabilities or measured actual-minus-expected surprises.
"""
from __future__ import annotations

from typing import Any

from dean_os.analyst_core.lens_contract import AnalysisPacket, AnalystLens, ModuleDelta
from dean_os.draft.dean_os_agent_system_v7.dean_os.expectation_evidence import validate_expectation_evidence

CROWDING_INDICATORS = (
    "crowded", "consensus", "everyone expects", "priced in",
    "market expects", "unanimous", "overwhelming consensus",
)
NOVELTY_INDICATORS = (
    "surprise", "unexpected", "shock", "first time", "unprecedented",
    "sudden", "abrupt", "unanticipated", "out of nowhere",
)
STALENESS_INDICATORS = (
    "months ago", "weeks ago", "already known", "previously announced",
    "leaked", "rumored", "widely reported", "market has had time",
)


class ExpectationGapLens(AnalystLens):
    """Measure expectation gaps only from sourced actual/expected pairs."""

    lens_name = "expectation_gap"
    lens_version = "0.3.0"
    event_classes_supported = ("*",)
    can_modify_existing = False

    def analyze(
        self, packet: AnalysisPacket, config: dict[str, Any] | None = None
    ) -> ModuleDelta:
        event_gaps: list[dict[str, Any]] = []
        for event in [*packet.entity_links, *packet.event_records]:
            if isinstance(event, dict):
                gap = self._estimate_gap(event, as_of=packet.as_of_date)
                if gap:
                    event_gaps.append(gap)

        summary = self._build_summary(event_gaps)
        watch_signals = []
        for gap in event_gaps:
            magnitude = gap["surprise_magnitude"]
            if magnitude is None:
                magnitude = gap["qualitative_signal_strength"]
            if magnitude <= 0.5:
                continue
            watch_signals.append(
                {
                    "signal_type": (
                        "expectation_gap"
                        if gap["quantitative_gap_available"]
                        else "expectation_context_review"
                    ),
                    "event_id": gap["event_id"],
                    "gap_direction": gap["surprise_direction"],
                    "magnitude": magnitude,
                    "quantitative_gap_available": gap[
                        "quantitative_gap_available"
                    ],
                }
            )

        return ModuleDelta(
            module_name=self.lens_name,
            module_version=self.lens_version,
            expectation_gap=summary,
            fields_added=["expectation_gap"],
            watch_signals_added=watch_signals,
            confidence=self._overall_confidence(event_gaps),
            reason_for_change=(
                f"Assessed expectation context for {len(event_gaps)} events; "
                f"quantified {summary.get('quantified_event_count', 0)}."
            ),
        )

    def _estimate_gap(self, event: dict[str, Any], *, as_of: str) -> dict[str, Any] | None:
        event_id = event.get("event_id", event.get("id", ""))
        if not event_id:
            return None
        lower = self._event_text(event).lower()
        crowding = self._keyword_score(lower, CROWDING_INDICATORS, 0.3)
        novelty = self._keyword_score(lower, NOVELTY_INDICATORS, 0.35)
        staleness = self._keyword_score(lower, STALENESS_INDICATORS, 0.3)

        validation = validate_expectation_evidence(
            event.get("expectation_evidence"), as_of=as_of
        )
        accepted = validation["accepted"][0] if validation["accepted"] else {}
        actual_observation = accepted.get("actual") or {}
        expected_observation = accepted.get("expected") or {}
        actual = actual_observation.get("value")
        expected = expected_observation.get("value")
        actual_source = actual_observation.get("source_locator")
        expectation_source = expected_observation.get("source_locator")
        quantitative = validation["quantitative_gap_allowed"]
        surprise_value = actual - expected if quantitative else None
        surprise_pct = (
            surprise_value / abs(expected)
            if quantitative and expected not in {None, 0.0}
            else None
        )
        expectation_std = accepted.get("expectation_std")
        standardized = (
            surprise_value / expectation_std
            if quantitative and expectation_std not in {None, 0.0}
            else None
        )
        magnitude_basis = (
            standardized
            if standardized is not None
            else surprise_pct
            if surprise_pct is not None
            else surprise_value
        )
        magnitude = (
            round(min(1.0, abs(magnitude_basis or 0.0)), 3)
            if quantitative
            else None
        )
        direction = self._direction(event, lower, surprise_value, quantitative)
        qualitative_strength = round(
            min(1.0, novelty * 0.6 + crowding * 0.25 + staleness * 0.15),
            3,
        )
        explicit_priced = self._numeric(event.get("already_priced_likelihood"))

        return {
            "event_id": str(event_id),
            "event_class": str(event.get("event_class", "")),
            "status": (
                "quantitative_expectation_gap"
                if quantitative
                else "qualitative_expectation_context_only"
            ),
            "quantitative_gap_available": quantitative,
            "actual_value": actual,
            "expected_value": expected,
            "unit": actual_observation.get("unit"),
            "expectation_type": accepted.get("expectation_type"),
            "expectation_evidence_sha256": accepted.get("evidence_sha256"),
            "expectation_validation_status": validation["status"],
            "expectation_validation_reasons": validation["reasons"],
            "actual_source": actual_source or None,
            "expectation_source": expectation_source or None,
            "surprise_value": surprise_value,
            "surprise_pct": surprise_pct,
            "standardized_surprise": standardized,
            "surprise_magnitude": magnitude,
            "surprise_direction": direction,
            "qualitative_crowding_signal": round(crowding, 3),
            "qualitative_novelty_signal": round(novelty, 3),
            "qualitative_staleness_signal": round(staleness, 3),
            "qualitative_signal_strength": qualitative_strength,
            "already_priced_likelihood": explicit_priced,
            "positioning_crowdedness": self._crowding_label(crowding),
            "market_implied_probability": self._numeric(
                event.get("market_implied_probability")
            ),
            "options_implied_volatility": self._numeric(
                event.get("options_implied_volatility")
            ),
            "credit_spread_signal": self._numeric(event.get("credit_spread_signal")),
            "gap_note": self._gap_note(magnitude, direction, quantitative),
            "limitations": (
                []
                if quantitative
                else [
                    "No point-in-time validated actual-versus-expected pair",
                    "Keyword novelty/crowding is qualitative context, not probability",
                    *validation["reasons"],
                ]
            ),
        }

    def _build_summary(self, gaps: list[dict[str, Any]]) -> dict[str, Any]:
        if not gaps:
            return {
                "status": "no_events_to_assess",
                "event_count": 0,
                "quantified_event_count": 0,
                "qualitative_only_event_count": 0,
                "average_surprise_magnitude": None,
                "high_surprise_events": 0,
                "already_priced_events": 0,
                "event_assessments": [],
            }
        magnitudes = [
            gap["surprise_magnitude"]
            for gap in gaps
            if gap["surprise_magnitude"] is not None
        ]
        already_priced = sum(
            1
            for gap in gaps
            if gap["already_priced_likelihood"] is not None
            and gap["already_priced_likelihood"] > 0.6
        )
        return {
            "status": (
                "expectation_gap_quantified"
                if magnitudes
                else "expectation_context_qualitative_only"
            ),
            "event_count": len(gaps),
            "quantified_event_count": len(magnitudes),
            "qualitative_only_event_count": len(gaps) - len(magnitudes),
            "average_surprise_magnitude": (
                round(sum(magnitudes) / len(magnitudes), 3)
                if magnitudes
                else None
            ),
            "high_surprise_events": sum(value > 0.5 for value in magnitudes),
            "already_priced_events": already_priced,
            "crowded_context_signal_events": sum(
                gap["positioning_crowdedness"] == "highly_crowded"
                for gap in gaps
            ),
            "dominant_direction": self._dominant_direction(gaps),
            "risk_summary": self._risk_summary(gaps),
            "event_assessments": gaps,
        }

    def _gap_note(
        self, magnitude: float | None, direction: str, quantitative: bool
    ) -> str:
        if not quantitative or magnitude is None:
            return (
                "Qualitative expectation context only; actual-minus-expected "
                "is not available."
            )
        if magnitude > 0.7:
            return (
                f"Large measured surprise ({direction}); market interpretation "
                "requires event study."
            )
        if magnitude > 0.4:
            return f"Moderate measured surprise ({direction}); priced-in status remains unproven."
        return f"Small measured surprise ({direction}); priced-in status remains unproven."

    def _direction(
        self,
        event: dict[str, Any],
        lower: str,
        surprise_value: float | None,
        quantitative: bool,
    ) -> str:
        if quantitative:
            if surprise_value and surprise_value > 0:
                return "positive_surprise"
            if surprise_value and surprise_value < 0:
                return "negative_surprise"
            return "neutral_surprise"
        sentiment = str(event.get("sentiment", "")).lower()
        if sentiment in {"positive", "negative"}:
            return f"qualitative_{sentiment}_context"
        pos = sum(term in lower for term in ("beat", "strong", "surge", "upgrade"))
        neg = sum(term in lower for term in ("miss", "weak", "crash", "downgrade"))
        return (
            "qualitative_positive_context"
            if pos > neg
            else "qualitative_negative_context"
            if neg > pos
            else "qualitative_mixed_context"
        )

    def _dominant_direction(self, gaps: list[dict[str, Any]]) -> str:
        quantified = [
            gap["surprise_direction"]
            for gap in gaps
            if gap["quantitative_gap_available"]
        ]
        if not quantified:
            return "not_quantified"
        positive = sum("positive" in item for item in quantified)
        negative = sum("negative" in item for item in quantified)
        return (
            "net_positive_surprise"
            if positive > negative
            else "net_negative_surprise"
            if negative > positive
            else "balanced_or_mixed"
        )

    def _risk_summary(self, gaps: list[dict[str, Any]]) -> str:
        high = [
            gap
            for gap in gaps
            if gap["surprise_magnitude"] is not None
            and gap["surprise_magnitude"] > 0.5
        ]
        if not high:
            return "No sourced high-surprise actual-versus-expected events detected."
        classes = sorted({gap["event_class"] for gap in high})
        return f"{len(high)} sourced high-surprise events in: {', '.join(classes)}."

    def _overall_confidence(self, gaps: list[dict[str, Any]]) -> float:
        if not gaps:
            return 0.2
        quantified = sum(gap["quantitative_gap_available"] for gap in gaps)
        return 0.25 + (quantified / len(gaps)) * 0.5

    def _event_text(self, event: dict[str, Any]) -> str:
        values = [
            str(event[key])
            for key in ("text", "title", "summary", "description", "text_preview")
            if event.get(key) and isinstance(event.get(key), str)
        ]
        return " ".join(values) if values else str(event)

    @staticmethod
    def _keyword_score(text: str, indicators: tuple[str, ...], step: float) -> float:
        return min(1.0, sum(indicator in text for indicator in indicators) * step)

    @staticmethod
    def _crowding_label(score: float) -> str:
        if score >= 0.6:
            return "highly_crowded"
        if score >= 0.3:
            return "moderately_crowded"
        return "not_crowded"

    @staticmethod
    def _numeric(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


__all__ = ["ExpectationGapLens"]
