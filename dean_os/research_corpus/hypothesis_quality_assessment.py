from __future__ import annotations

from typing import Any


HYPOTHESIS_QUALITY_ASSESSMENT_CONTRACT = (
    "dean_hypothesis_quality_assessment_v1"
)
HYPOTHESIS_OUTCOME_ASSESSMENT_CONTRACT = (
    "dean_hypothesis_outcome_assessment_v1"
)

_DIMENSION_WEIGHTS = {
    "evidence_quality": 1.2,
    "evidence_independence": 1.0,
    "expectation_surprise_context": 1.2,
    "causal_mechanism": 1.1,
    "exposure_definition": 1.2,
    "falsifiability_observability": 1.4,
    "time_definition": 1.0,
    "confounder_control": 0.9,
}

_OUTCOME_LABELS = [
    "confirmed",
    "partially_confirmed",
    "falsified",
    "inconclusive",
    "unobservable",
    "right_thesis_wrong_market_reaction",
    "right_market_reaction_wrong_causal_explanation",
    "confounder_dominated",
]


def assessment_policy() -> dict[str, Any]:
    """Return the stable pre-outcome and post-outcome evaluation policy."""

    return {
        "quality_contract": HYPOTHESIS_QUALITY_ASSESSMENT_CONTRACT,
        "outcome_contract": HYPOTHESIS_OUTCOME_ASSESSMENT_CONTRACT,
        "pre_outcome_rule": (
            "Assess whether the claim is evidence-backed, scoped, falsifiable and "
            "observable. This is not a forecast result and not permission to trade."
        ),
        "score_interpretation": (
            "The 0-100 score is structural/evidential readiness, not a calibrated "
            "probability of truth and not a directional trading signal."
        ),
        "dimension_scale": {
            "minimum": 0,
            "maximum": 4,
            "labels": {
                "0": "missing",
                "1": "weak",
                "2": "limited",
                "3": "adequate",
                "4": "strong",
            },
        },
        "quality_bands": {
            "weak": [0, 39],
            "limited": [40, 59],
            "moderate": [60, 74],
            "strong": [75, 100],
        },
        "bottleneck_rule": (
            "Critical missing dimensions cap the total score; a weighted average "
            "cannot hide missing observability, timing, exposure or context."
        ),
        "outcome_labels": list(_OUTCOME_LABELS),
        "post_outcome_dimensions": [
            "direction",
            "magnitude",
            "timing",
            "causal_mechanism",
            "relative_market_reaction",
            "confounder_attribution",
            "confidence_calibration",
        ],
        "learning_boundary": (
            "One outcome may diagnose a case but cannot by itself promote a rule, "
            "change agent weights or write production learning memory."
        ),
    }


def assess_hypothesis_quality(
    hypothesis: dict[str, Any],
    *,
    trigger_event: dict[str, Any] | None,
    evidence_events: list[dict[str, Any]],
    packet_summary: dict[str, Any],
    alignment: dict[str, Any],
    replay_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic review card without claiming forecast accuracy."""

    measurement = dict(hypothesis.get("measurement_spec") or {})
    measurement_context = dict(measurement.get("measurement_context") or {})
    expected = list(hypothesis.get("expected_observations") or [])
    invalidation = list(hypothesis.get("invalidation_signals") or [])
    horizons = _positive_ints(hypothesis.get("horizons_to_check") or [])
    blockers = [str(item) for item in hypothesis.get("registration_blockers") or []]
    trigger_ids = [str(item) for item in hypothesis.get("trigger_evidence_ids") or []]
    supporting_ids = [
        str(item) for item in hypothesis.get("supporting_evidence_ids") or []
    ]
    evidence_by_id = {
        str(item.get("evidence_id") or item.get("event_id")): item
        for item in evidence_events
        if isinstance(item, dict)
    }
    linked_events = [
        evidence_by_id[item]
        for item in trigger_ids + supporting_ids
        if item in evidence_by_id
    ]

    dimensions = {
        "evidence_quality": _dimension(
            _evidence_quality(trigger_event),
            "Point-in-time source tier and provenance of the trigger evidence.",
        ),
        "evidence_independence": _dimension(
            _evidence_independence(linked_events, supporting_ids),
            "Independent supporting sources beyond the trigger event.",
        ),
        "expectation_surprise_context": _dimension(
            _expectation_context(packet_summary, measurement, measurement_context),
            "Pre-event baseline, consensus or other expectation needed to measure surprise.",
        ),
        "causal_mechanism": _dimension(
            _causal_mechanism(hypothesis, alignment, expected, measurement),
            "Explicit transmission path from trigger to observable consequence.",
        ),
        "exposure_definition": _dimension(
            _exposure_definition(hypothesis, measurement, measurement_context, blockers),
            "Named affected population, basket, issuer, product or policy scope.",
        ),
        "falsifiability_observability": _dimension(
            _observability(expected, invalidation, measurement, horizons),
            "Measurable targets, assessment rule and explicit invalidation conditions.",
        ),
        "time_definition": _dimension(
            _time_definition(hypothesis, replay_tasks, measurement, horizons),
            "Dated event anchor, fixed horizons and checkpoint definition.",
        ),
        "confounder_control": _dimension(
            _confounder_control(hypothesis, measurement),
            "Benchmark, contradictory metric or explicit alternative-explanation control.",
        ),
    }

    raw_score = _weighted_score(dimensions)
    score_caps: list[dict[str, Any]] = []
    if not supporting_ids:
        score_caps.append(
            {
                "cap": 69,
                "reason": "trigger_only_without_independent_support",
            }
        )
    if dimensions["expectation_surprise_context"]["score"] == 0:
        score_caps.append({"cap": 49, "reason": "expectation_context_missing"})
    if blockers:
        score_caps.append({"cap": 39, "reason": "registration_blockers_present"})
    if dimensions["exposure_definition"]["score"] <= 1:
        score_caps.append({"cap": 39, "reason": "affected_exposure_not_defined"})
    if dimensions["falsifiability_observability"]["score"] <= 2:
        score_caps.append({"cap": 39, "reason": "measurement_floor_not_met"})
    if dimensions["time_definition"]["score"] <= 2:
        score_caps.append({"cap": 39, "reason": "time_anchor_floor_not_met"})
    applied_cap = min([item["cap"] for item in score_caps], default=100)
    score = min(raw_score, applied_cap)

    hard_blockers: list[str] = []
    if dimensions["evidence_quality"]["score"] < 2:
        hard_blockers.append("credible_point_in_time_trigger_missing")
    if dimensions["exposure_definition"]["score"] < 2:
        hard_blockers.append("affected_exposure_not_defined")
    if dimensions["falsifiability_observability"]["score"] < 3:
        hard_blockers.append("falsifiability_or_measurement_floor_not_met")
    if dimensions["time_definition"]["score"] < 3:
        hard_blockers.append("event_anchor_or_horizon_floor_not_met")
    hard_blockers.extend(blockers)
    hard_blockers = list(dict.fromkeys(hard_blockers))
    replay_eligible = not hard_blockers

    weaknesses = [
        name
        for name, value in dimensions.items()
        if int(value["score"]) <= 1
    ]
    missing_evidence = _missing_evidence(
        dimensions,
        supporting_ids=supporting_ids,
        registration_blockers=blockers,
    )
    return {
        "contract": HYPOTHESIS_QUALITY_ASSESSMENT_CONTRACT,
        "assessment_stage": "pre_outcome",
        "hypothesis_quality_score": score,
        "uncapped_weighted_score": raw_score,
        "quality_band": _quality_band(score),
        "bottleneck_score": min(
            int(value["score"]) for value in dimensions.values()
        ),
        "dimensions": dimensions,
        "score_caps_applied": score_caps,
        "critical_weaknesses": weaknesses,
        "missing_evidence": missing_evidence,
        "replay_eligible": replay_eligible,
        "replay_eligibility_blockers": hard_blockers,
        "max_allowed_use": (
            "replay_observation_only"
            if replay_eligible
            else "defer_until_quality_floor_met"
        ),
        "reported_generator_confidence": hypothesis.get("confidence"),
        "confidence_probability": None,
        "confidence_status": "uncalibrated_no_matured_reviewed_outcomes",
        "directional_trading_signal": None,
        "human_review_questions": _review_questions(dimensions, blockers),
        "outcome_assessment_plan": _outcome_assessment_plan(
            hypothesis, measurement, replay_tasks
        ),
        "safety": {
            "review_only": True,
            "automatic_disposition_allowed": False,
            "automatic_outcome_scoring_allowed": False,
            "can_register_replay_tasks": False,
            "can_write_learning_memory": False,
            "can_trade": False,
        },
    }


def _dimension(score: int, meaning: str) -> dict[str, Any]:
    labels = {0: "missing", 1: "weak", 2: "limited", 3: "adequate", 4: "strong"}
    return {"score": score, "level": labels[score], "meaning": meaning}


def _evidence_quality(trigger: dict[str, Any] | None) -> int:
    if not trigger:
        return 0
    provenance = dict(trigger.get("provenance") or {})
    tier = str(provenance.get("source_tier") or trigger.get("source_tier") or "")
    score = {
        "tier_1_core_evidence": 4,
        "tier_1_primary": 4,
        "tier_1_official": 4,
        "tier_2_strong_context": 3,
        "tier_3_context": 2,
        "tier_3_event_context": 2,
        "tier_4_weak_or_unverified": 1,
    }.get(tier, 1)
    has_locator = bool(trigger.get("source_id"))
    has_time = bool(provenance.get("published_at") or trigger.get("published_at"))
    return score if has_locator and has_time else min(score, 1)


def _evidence_independence(
    linked_events: list[dict[str, Any]], supporting_ids: list[str]
) -> int:
    sources = {
        str((item.get("provenance") or {}).get("source_identity") or item.get("source_id"))
        for item in linked_events
        if (item.get("provenance") or {}).get("source_identity") or item.get("source_id")
    }
    if len(sources) >= 3 and len(supporting_ids) >= 2:
        return 4
    if len(sources) >= 2 and supporting_ids:
        return 3
    if supporting_ids:
        return 2
    return 1 if linked_events else 0


def _expectation_context(
    packet_summary: dict[str, Any],
    measurement: dict[str, Any],
    context: dict[str, Any],
) -> int:
    if packet_summary.get("expectation_context_available") is True:
        return 4
    context_text = _searchable_text(context)
    if context and any(term in context_text for term in ("baseline", "consensus", "expectation")):
        return 3
    measurement_text = _searchable_text(measurement)
    if any(term in measurement_text for term in ("pre_event", "baseline", "consensus", "relative")):
        return 2
    return 0


def _causal_mechanism(
    hypothesis: dict[str, Any],
    alignment: dict[str, Any],
    expected: list[Any],
    measurement: dict[str, Any],
) -> int:
    explicit = bool(hypothesis.get("mechanism") or alignment.get("mechanism"))
    has_chain = bool(expected and measurement.get("target_metrics"))
    if explicit and has_chain:
        return 4
    if has_chain:
        return 3
    if expected:
        return 2
    return 1 if hypothesis.get("hypothesis") else 0


def _exposure_definition(
    hypothesis: dict[str, Any],
    measurement: dict[str, Any],
    context: dict[str, Any],
    blockers: list[str],
) -> int:
    blocker_text = " ".join(blockers).lower()
    if any(term in blocker_text for term in ("exposure", "scope", "affected_set", "basket")):
        return 0
    context_text = _searchable_text(context)
    if context and any(term in context_text for term in ("basket", "members", "ticker", "issuer")):
        return 4
    if measurement.get("target_metrics"):
        return 3
    return 2 if hypothesis.get("hypothesis") else 0


def _observability(
    expected: list[Any],
    invalidation: list[Any],
    measurement: dict[str, Any],
    horizons: list[int],
) -> int:
    metrics = list(measurement.get("target_metrics") or [])
    if metrics and measurement.get("assessment_rule") and invalidation and horizons:
        return 4
    if expected and invalidation and horizons:
        return 3
    if expected and horizons:
        return 2
    return 1 if horizons else 0


def _time_definition(
    hypothesis: dict[str, Any],
    replay_tasks: list[dict[str, Any]],
    measurement: dict[str, Any],
    horizons: list[int],
) -> int:
    anchors = [task.get("trigger_event_at") for task in replay_tasks if task.get("trigger_event_at")]
    primary = measurement.get("primary_horizon_days")
    if anchors and horizons and primary in horizons:
        return 4
    if anchors and horizons:
        return 3
    if hypothesis.get("as_of") and horizons:
        return 3
    return 2 if horizons else 0


def _confounder_control(
    hypothesis: dict[str, Any], measurement: dict[str, Any]
) -> int:
    text = _searchable_text(
        {
            "measurement": measurement,
            "contradicting": hypothesis.get("contradicting_evidence_ids") or [],
            "alternatives": hypothesis.get("alternative_explanations") or [],
        }
    )
    if "confound" in text or "alternative_explanation" in text:
        return 4
    if any(term in text for term in ("benchmark", "relative", "contradict")):
        return 3
    if measurement:
        return 2
    return 1


def _weighted_score(dimensions: dict[str, dict[str, Any]]) -> int:
    numerator = sum(
        int(dimensions[name]["score"]) * weight
        for name, weight in _DIMENSION_WEIGHTS.items()
    )
    denominator = 4 * sum(_DIMENSION_WEIGHTS.values())
    return round(100 * numerator / denominator)


def _quality_band(score: int) -> str:
    if score < 40:
        return "weak"
    if score < 60:
        return "limited"
    if score < 75:
        return "moderate"
    return "strong"


def _missing_evidence(
    dimensions: dict[str, dict[str, Any]],
    *,
    supporting_ids: list[str],
    registration_blockers: list[str],
) -> list[str]:
    missing: list[str] = []
    if not supporting_ids:
        missing.append("independent_supporting_evidence")
    mapping = {
        "expectation_surprise_context": "pre_event_expectation_or_surprise_context",
        "exposure_definition": "affected_exposure_definition",
        "falsifiability_observability": "complete_measurement_and_invalidation_spec",
        "confounder_control": "alternative_explanation_or_benchmark_control",
    }
    for dimension, label in mapping.items():
        if int(dimensions[dimension]["score"]) <= 1:
            missing.append(label)
    missing.extend(registration_blockers)
    return list(dict.fromkeys(missing))


def _review_questions(
    dimensions: dict[str, dict[str, Any]], blockers: list[str]
) -> list[str]:
    questions = [
        "Is the trigger only a reason to test the claim, rather than proof of it?",
        "Would the stated invalidation rule genuinely change the conclusion?",
        "Are the event-response and slower thesis horizons kept separate?",
    ]
    if int(dimensions["expectation_surprise_context"]["score"]) <= 1:
        questions.append("What was known or priced in before the trigger?")
    if int(dimensions["confounder_control"]["score"]) <= 1:
        questions.append("Which competing event could produce the same observed outcome?")
    if blockers:
        questions.append("Have all named registration blockers been resolved point-in-time?")
    return questions


def _outcome_assessment_plan(
    hypothesis: dict[str, Any],
    measurement: dict[str, Any],
    replay_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "contract": HYPOTHESIS_OUTCOME_ASSESSMENT_CONTRACT,
        "status": "pending_matured_verified_outcomes",
        "allowed_result_labels": list(_OUTCOME_LABELS),
        "dimensions_to_review": {
            "direction": "pending",
            "magnitude": "pending",
            "timing": "pending",
            "causal_mechanism": "pending",
            "relative_market_reaction": "pending",
            "confounder_attribution": "pending",
            "confidence_calibration": "pending",
        },
        "target_metrics": list(measurement.get("target_metrics") or []),
        "assessment_rule": measurement.get("assessment_rule"),
        "checkpoint_count": len(replay_tasks),
        "invalidation_signals": list(hypothesis.get("invalidation_signals") or []),
        "automatic_outcome_scoring_allowed": False,
        "human_causal_attribution_required": True,
        "single_case_rule_promotion_allowed": False,
    }


def _positive_ints(values: list[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            result.append(parsed)
    return result


def _searchable_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(
            f"{key} {_searchable_text(item)}" for key, item in value.items()
        ).lower()
    if isinstance(value, (list, tuple, set)):
        return " ".join(_searchable_text(item) for item in value).lower()
    return str(value or "").lower()


__all__ = [
    "HYPOTHESIS_OUTCOME_ASSESSMENT_CONTRACT",
    "HYPOTHESIS_QUALITY_ASSESSMENT_CONTRACT",
    "assess_hypothesis_quality",
    "assessment_policy",
]
