from __future__ import annotations

from typing import Any


HYPOTHESIS_REVERSE_ANALYSIS_CONTRACT = "dean_hypothesis_reverse_analysis_v1"

_FAILED_RESULTS = {"falsified", "weakened", "miss"}

_SOURCE_DIAGNOSES = {
    "credible_context_source_but_trigger_polarity_conflicts_with_generated_claim": [
        "trigger_polarity_mismatch",
        "expectation_context_missing",
    ],
    "tier_1_primary_source_with_precise_rule_but_generated_claim_is_too_broad": [
        "event_novelty_misread",
        "claim_scope_overreach",
        "exposure_mapping_missing",
        "expectation_context_missing",
    ],
    "credible_context_source_but_contingent_project_risk_was_generalized_into_persistent_sector_constraint": [
        "contingent_risk_generalized",
        "claim_scope_overreach",
        "expectation_context_missing",
    ],
}

_ERROR_LAYER = {
    "trigger_polarity_mismatch": "claim_framing",
    "claim_scope_overreach": "claim_framing",
    "event_novelty_misread": "event_classification",
    "contingent_risk_generalized": "claim_framing",
    "expectation_context_missing": "expectation_context",
    "exposure_mapping_missing": "exposure_mapping",
    "time_anchor_error": "time_definition",
    "horizon_mismatch": "time_definition",
    "unsupported_causal_leap": "causal_mechanism",
    "wrong_transmission_channel": "causal_mechanism",
    "false_analogy": "historical_analogy",
    "priced_in_blindness": "market_translation",
    "data_quality_failure": "data_quality",
    "outcome_not_observable": "measurement",
    "true_hypothesis_wrong_market_reaction": "market_translation",
    "confounder_dominated": "causal_attribution",
    "inconclusive": "outcome_evidence",
    "unknown_falsification_cause": "root_cause_unknown",
}

_COUNTERFACTUAL_TESTS = {
    "trigger_polarity_mismatch": "Would the claim keep the same direction if it repeated only the trigger's explicit meaning?",
    "claim_scope_overreach": "Would the claim survive if restricted to the named entities, products and geography?",
    "event_novelty_misread": "What dated legal or operational delta exists versus the prior rule?",
    "contingent_risk_generalized": "Does the result persist after removing the contingent project or condition?",
    "expectation_context_missing": "Would the result still be surprising relative to a point-in-time pre-event baseline?",
    "exposure_mapping_missing": "Is the observed effect stronger in the predeclared exposed set than in the control set?",
    "time_anchor_error": "Can every input be proven available before the event anchor?",
    "horizon_mismatch": "Does the conclusion change when evaluated only on its declared horizon family?",
    "unsupported_causal_leap": "Did the declared intermediate state move before the final outcome?",
    "wrong_transmission_channel": "Which competing channel better explains the intermediate observations?",
    "false_analogy": "Does the analog still match after conditioning on regime, mechanism and expectations?",
    "priced_in_blindness": "Was the fundamental result already embedded in prices or consensus before the event?",
    "data_quality_failure": "Does the diagnosis survive reacquisition from verified point-in-time sources?",
    "outcome_not_observable": "Can the same checkpoint be reconstructed from the declared metric, universe and coverage rule?",
    "true_hypothesis_wrong_market_reaction": "Does the fundamental leg pass when scored separately from price reaction?",
    "confounder_dominated": "Does the attributed effect remain after comparing with a benchmark exposed to the confounder?",
    "unknown_falsification_cause": "Which evidence would distinguish a false mechanism from bad timing, bad measurement or a confounder?",
}


def build_hypothesis_reverse_analysis(
    *,
    hypothesis: dict[str, Any],
    review: dict[str, Any],
    outcome: dict[str, Any] | None,
    allowed_error_codes: set[str] | None = None,
) -> dict[str, Any]:
    """Build a machine-generated, review-only post-mortem card.

    The function can diagnose structured evidence and propose what to inspect next.
    It deliberately cannot promote a rule, change agent configuration or trade.
    """

    outcome = dict(outcome or {})
    hypothesis_id = str(
        review.get("hypothesis_id") or hypothesis.get("hypothesis_id") or "unknown"
    )
    result_label = str(outcome.get("result_label") or "").strip() or None
    dimensions = _outcome_dimensions(outcome)
    quality = dict(review.get("quality_assessment") or {})
    diagnosis: dict[str, dict[str, Any]] = {}

    def add(
        error_code: str,
        *,
        strength: str,
        basis: str,
        evidence: list[str],
    ) -> None:
        if allowed_error_codes is not None and error_code not in allowed_error_codes:
            return
        rank = {"candidate": 1, "supported": 2, "confirmed": 3}
        previous = diagnosis.get(error_code)
        if previous and rank[previous["diagnostic_strength"]] > rank[strength]:
            return
        diagnosis[error_code] = {
            "error_code": error_code,
            "failure_layer": _ERROR_LAYER.get(error_code, "unclassified"),
            "diagnostic_strength": strength,
            "basis": basis,
            "evidence_trace": evidence,
            "counterfactual_test": _COUNTERFACTUAL_TESTS.get(error_code),
            "proposal_eligible": strength in {"supported", "confirmed"},
        }

    for code in outcome.get("error_labels") or []:
        add(
            str(code),
            strength="confirmed",
            basis="explicit_reviewed_outcome_label",
            evidence=[f"outcome.error_labels contains {code}"],
        )

    source_assessment = str(review.get("source_assessment") or "")
    for code in _SOURCE_DIAGNOSES.get(source_assessment, []):
        add(
            code,
            strength="supported",
            basis="review_gate_diagnosis",
            evidence=[f"review.source_assessment={source_assessment}"],
        )

    _add_structured_outcome_diagnoses(
        add=add,
        hypothesis=hypothesis,
        review=review,
        outcome=outcome,
        dimensions=dimensions,
        result_label=result_label,
    )
    _add_quality_risk_diagnoses(
        add=add,
        quality=quality,
        result_label=result_label,
        dimensions=dimensions,
    )

    failed = result_label in _FAILED_RESULTS
    if failed and not diagnosis:
        add(
            "unknown_falsification_cause",
            strength="candidate",
            basis="failed_outcome_without_causal_attribution",
            evidence=[f"outcome.result_label={result_label}", "no structured root-cause evidence"],
        )
    if result_label in {"inconclusive", "unobservable"} and not diagnosis:
        code = "outcome_not_observable" if result_label == "unobservable" else "inconclusive"
        add(
            code,
            strength="supported",
            basis="outcome_resolution_status",
            evidence=[f"outcome.result_label={result_label}"],
        )

    candidates = sorted(
        diagnosis.values(),
        key=lambda item: (
            {"confirmed": 0, "supported": 1, "candidate": 2}[item["diagnostic_strength"]],
            item["error_code"],
        ),
    )
    eligible_codes = [
        item["error_code"] for item in candidates if item["proposal_eligible"]
    ]
    overlooked = _overlooked_signals(quality, review)
    alternatives = _alternative_explanations(hypothesis, outcome, dimensions)
    next_action = _next_action(
        result_label=result_label,
        candidates=candidates,
        outcome=outcome,
    )
    return {
        "contract": HYPOTHESIS_REVERSE_ANALYSIS_CONTRACT,
        "hypothesis_id": hypothesis_id,
        "analysis_stage": "post_outcome" if outcome else "pre_outcome_review_diagnosis",
        "machine_analysis_status": _analysis_status(
            outcome=outcome,
            result_label=result_label,
            eligible_codes=eligible_codes,
            candidates=candidates,
        ),
        "claim_before": review.get("hypothesis") or hypothesis.get("hypothesis"),
        "claim_after_review": review.get("proposed_hypothesis"),
        "review_disposition": review.get("disposition"),
        "outcome_id": outcome.get("outcome_id"),
        "result_label": result_label,
        "outcome_decomposition": {
            "dimensions": dimensions,
            "observations": outcome.get("observations")
            or outcome.get("observed_values")
            or [],
            "fundamental_result": outcome.get("fundamental_result"),
            "market_reaction_result": outcome.get("market_reaction_result"),
        },
        "machine_diagnosis_candidates": candidates,
        "proposal_eligible_error_codes": eligible_codes,
        "overlooked_or_underweighted_signals": overlooked,
        "alternative_explanations": alternatives,
        "counterfactual_tests": [
            item["counterfactual_test"]
            for item in candidates
            if item.get("counterfactual_test")
        ],
        "recommended_next_action": next_action,
        "agent_change_proposal": {
            "status": (
                "candidate_rules_may_be_prepared_for_human_review"
                if eligible_codes
                else "insufficient_root_cause_evidence_for_rule_proposal"
            ),
            "target_error_codes": eligible_codes,
            "automatic_application_allowed": False,
            "required_before_promotion": [
                "independent reviewed case threshold",
                "regression test",
                "explicit human promotion decision",
            ],
        },
        "safety": {
            "review_only": True,
            "machine_may_analyze": True,
            "machine_may_propose": True,
            "machine_may_apply_rule": False,
            "learning_memory_write_performed": False,
            "production_rule_update_performed": False,
            "broker_access_performed": False,
            "can_trade": False,
        },
    }


def _add_structured_outcome_diagnoses(
    *,
    add: Any,
    hypothesis: dict[str, Any],
    review: dict[str, Any],
    outcome: dict[str, Any],
    dimensions: dict[str, Any],
    result_label: str | None,
) -> None:
    if not outcome:
        return
    data_status = _value(outcome.get("data_quality_status") or dimensions.get("data_quality"))
    if data_status in {"failed", "invalid", "stale", "incomplete", "corrupted"}:
        add(
            "data_quality_failure",
            strength="confirmed",
            basis="structured_data_quality_failure",
            evidence=[f"data_quality_status={data_status}"],
        )
    observable = outcome.get("observable")
    coverage = _value(outcome.get("coverage_status") or dimensions.get("observability"))
    if observable is False or coverage in {"failed", "insufficient", "unobservable"}:
        add(
            "outcome_not_observable",
            strength="confirmed",
            basis="structured_observability_failure",
            evidence=[f"observable={observable}", f"coverage_status={coverage}"],
        )

    hypothesis_family = str(
        hypothesis.get("horizon_family") or review.get("horizon_family") or ""
    )
    outcome_family = str(outcome.get("horizon_family") or "")
    if hypothesis_family and outcome_family and hypothesis_family != outcome_family:
        add(
            "horizon_mismatch",
            strength="confirmed",
            basis="declared_horizon_family_mismatch",
            evidence=[
                f"hypothesis.horizon_family={hypothesis_family}",
                f"outcome.horizon_family={outcome_family}",
            ],
        )

    causal = _value(dimensions.get("causal_mechanism"))
    intermediate = _value(outcome.get("intermediate_state_result"))
    if causal in {"failed", "wrong", "not_supported"}:
        code = (
            "unsupported_causal_leap"
            if intermediate in {"missing", "not_observed", "undefined", ""}
            else "wrong_transmission_channel"
        )
        add(
            code,
            strength="supported",
            basis="structured_causal_mechanism_failure",
            evidence=[f"causal_mechanism={causal}", f"intermediate_state_result={intermediate}"],
        )

    confounder = _value(
        outcome.get("confounder_attribution") or dimensions.get("confounder_attribution")
    )
    if confounder in {"dominated", "material", "high", "confounder_dominated"}:
        add(
            "confounder_dominated",
            strength="supported",
            basis="structured_confounder_attribution",
            evidence=[f"confounder_attribution={confounder}"],
        )

    fundamental = _value(outcome.get("fundamental_result"))
    market = _value(
        outcome.get("market_reaction_result")
        or dimensions.get("relative_market_reaction")
    )
    if fundamental in {"confirmed", "hit", "supported", "true"} and market in {
        "failed",
        "miss",
        "wrong",
        "opposite",
    }:
        add(
            "true_hypothesis_wrong_market_reaction",
            strength="confirmed",
            basis="fundamental_and_market_legs_diverge",
            evidence=[f"fundamental_result={fundamental}", f"market_reaction_result={market}"],
        )
    expectation_available = review.get("expectation_context_available")
    if (
        result_label in _FAILED_RESULTS
        and market in {"failed", "miss", "wrong", "opposite"}
        and expectation_available is False
    ):
        add(
            "priced_in_blindness",
            strength="supported",
            basis="market_miss_without_pre_event_expectation_context",
            evidence=["expectation_context_available=false", f"market_reaction_result={market}"],
        )


def _add_quality_risk_diagnoses(
    *,
    add: Any,
    quality: dict[str, Any],
    result_label: str | None,
    dimensions: dict[str, Any],
) -> None:
    if result_label not in _FAILED_RESULTS:
        return
    quality_dimensions = quality.get("dimensions") or {}
    mapping = {
        "expectation_surprise_context": "expectation_context_missing",
        "exposure_definition": "exposure_mapping_missing",
        "falsifiability_observability": "outcome_not_observable",
        "causal_mechanism": "unsupported_causal_leap",
    }
    for dimension, error_code in mapping.items():
        item = quality_dimensions.get(dimension) or {}
        try:
            score = int(item.get("score"))
        except (TypeError, ValueError):
            continue
        if score <= 1:
            add(
                error_code,
                strength="candidate",
                basis="pre_outcome_weakness_correlated_with_later_miss",
                evidence=[f"quality.{dimension}.score={score}", f"outcome.result_label={result_label}"],
            )


def _outcome_dimensions(outcome: dict[str, Any]) -> dict[str, Any]:
    raw = outcome.get("dimensions") or outcome.get("assessment_dimensions") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _overlooked_signals(
    quality: dict[str, Any], review: dict[str, Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, value in (quality.get("dimensions") or {}).items():
        try:
            score = int((value or {}).get("score"))
        except (TypeError, ValueError):
            continue
        if score <= 1:
            results.append(
                {
                    "signal": name,
                    "pre_outcome_score": score,
                    "interpretation": "known weakness that should be tested as a cause, not assumed to be the cause",
                }
            )
    if review.get("expectation_context_available") is False and not any(
        item["signal"] == "expectation_surprise_context" for item in results
    ):
        results.append(
            {
                "signal": "expectation_context",
                "pre_outcome_score": None,
                "interpretation": "pre-event expectations were unavailable or explicitly incomplete",
            }
        )
    return results


def _alternative_explanations(
    hypothesis: dict[str, Any], outcome: dict[str, Any], dimensions: dict[str, Any]
) -> list[Any]:
    alternatives: list[Any] = []
    alternatives.extend(hypothesis.get("alternative_explanations") or [])
    alternatives.extend(outcome.get("alternative_explanations") or [])
    alternatives.extend(outcome.get("confounders") or [])
    attributed = dimensions.get("confounder_attribution")
    if isinstance(attributed, dict):
        alternatives.extend(attributed.get("candidates") or [])
    unique: list[Any] = []
    seen: set[str] = set()
    for item in alternatives:
        key = str(item)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _next_action(
    *, result_label: str | None, candidates: list[dict[str, Any]], outcome: dict[str, Any]
) -> dict[str, Any]:
    codes = {item["error_code"] for item in candidates}
    if not outcome:
        action = "retain_as_review_diagnosis_and_wait_for_matured_outcome"
    elif "data_quality_failure" in codes or "outcome_not_observable" in codes:
        action = "repair_or_reconstruct_outcome_evidence_before_judging_the_hypothesis"
    elif "confounder_dominated" in codes:
        action = "keep_attribution_unresolved_and_prepare_a_confounder_controlled_reformulation"
    elif "true_hypothesis_wrong_market_reaction" in codes:
        action = "retain_the_fundamental_mechanism_and_reformulate_the_market_reaction_leg"
    elif result_label in _FAILED_RESULTS and any(item["proposal_eligible"] for item in candidates):
        action = "prepare_a_bounded_rule_or_template_change_for_human_review"
    elif result_label in _FAILED_RESULTS:
        action = "collect_root_cause_evidence_before_proposing_a_rule_change"
    elif result_label in {"confirmed", "partially_confirmed"}:
        action = "record_as_positive_case_without_generalizing_from_one_outcome"
    else:
        action = "continue_observation_until_the_outcome_is_resolved"
    return {
        "action": action,
        "automatic_execution_allowed": False,
        "requires_human_rule_promotion": True,
    }


def _analysis_status(
    *,
    outcome: dict[str, Any],
    result_label: str | None,
    eligible_codes: list[str],
    candidates: list[dict[str, Any]],
) -> str:
    if not outcome:
        return "pre_outcome_diagnostic_only"
    if eligible_codes:
        return "machine_root_cause_candidates_ready_for_review"
    if result_label in _FAILED_RESULTS and candidates:
        return "failed_outcome_root_cause_evidence_insufficient"
    if result_label in {"confirmed", "partially_confirmed"}:
        return "positive_outcome_recorded_no_rule_generalization"
    return "outcome_recorded_continue_analysis"


def _value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("status", "result", "label", "assessment"):
            if value.get(key) is not None:
                return str(value[key]).strip().lower()
        return ""
    return str(value or "").strip().lower()


__all__ = [
    "HYPOTHESIS_REVERSE_ANALYSIS_CONTRACT",
    "build_hypothesis_reverse_analysis",
]
