from __future__ import annotations

from typing import Any


def build_review_feedback_taxonomy(
    *,
    case_family: str,
    outcome_labels: list[str] | None = None,
    profile_feedback_issue_types: list[str] | None = None,
) -> dict[str, Any]:
    """Return one taxonomy with explicit case-family semantics.

    Domain forecasts and pipeline evaluations share review/process labels, but
    they do not share outcome semantics. Callers select the applicable family
    and must not use labels from the other family as substitutes.
    """

    labels = {
        "analysis_quality": [
            "correct",
            "partially_correct",
            "wrong_archetype",
            "missed_counterforce",
            "overclaim",
            "too_low_confidence",
            "too_high_confidence",
            "unsupported_by_sources",
        ],
        "data_quality": [
            "weak_source",
            "stale_source",
            "missing_primary_source",
            "bad_date_metadata",
            "unit_period_error",
            "dedupe_error",
            "retrieval_miss",
            "data_issue_suspected",
        ],
        "causal_quality": [
            "causal_graph_good",
            "causal_graph_partial",
            "causal_graph_wrong",
            "wrong_directness_label",
            "correct_for_stated_reasons",
            "correct_but_lucky_or_wrong_reason",
        ],
        "domain_outcome_review": _unique(outcome_labels or []),
        "model_evaluation_review": [
            "evaluation_block_valid",
            "evaluation_caution_valid",
            "evaluation_clear_valid",
            "evidence_binding_issue",
            "constraint_contract_issue",
            "generalization_gap_confirmed",
            "feature_instability_confirmed",
            "implementation_issue_suspected",
            "data_issue_suspected",
            "needs_new_forward_data",
            "insufficient_evaluation_evidence",
        ],
        "profile_feedback_issue": _unique(
            profile_feedback_issue_types or []
        ),
        "process_review": [
            "approved",
            "corrected",
            "rejected",
            "needs_more_evidence",
            "missing_source",
        ],
        "learning_action": [
            "create_eval_case",
            "create_eval_test_candidate",
            "create_incident",
            "create_incident_candidate",
            "update_prompt_candidate",
            "update_pattern_candidate",
            "update_source_registry",
            "request_more_evidence",
            "propose_pipeline_fix",
            "propose_model_iteration_after_new_data",
            "no_learning_update",
        ],
        "agent_error_taxonomy": [
            "wrong_event_type",
            "missed_affected_sector",
            "unsupported_inference",
            "bad_historical_analogue",
            "failed_expectation_gap",
            "overconfidence",
            "tool_misuse",
            "schema_violation",
            "source_grounding_failure",
            "loop_detected",
            "unsafe_action_attempt",
        ],
    }
    applicable_groups = {
        "domain_analyst": [
            "analysis_quality",
            "data_quality",
            "causal_quality",
            "domain_outcome_review",
            "profile_feedback_issue",
            "process_review",
            "learning_action",
            "agent_error_taxonomy",
        ],
        "pipeline_model": [
            "data_quality",
            "model_evaluation_review",
            "process_review",
            "learning_action",
            "agent_error_taxonomy",
        ],
    }
    if case_family not in applicable_groups:
        raise ValueError(f"Unsupported feedback case family: {case_family}")
    return {
        "taxonomy_id": "dean_review_feedback_taxonomy_v1",
        "case_family": case_family,
        "source_templates": [
            "REVIEW_LABEL_TAXONOMY.json",
            "HUMAN_AGENT_PARALLEL_ANALYSIS_SCHEMA.json",
            "FEEDBACK_TO_LEARNING_PIPELINE_TEMPLATE.yaml",
            "OUTCOME_REVIEW_TEMPLATE.yaml",
        ],
        "labels": labels,
        "applicable_label_groups": applicable_groups[case_family],
        "required_distinctions": [
            "evaluation_contract_result_vs_realized_forecast_outcome",
            "correct_for_stated_reasons_vs_lucky_or_wrong_reason",
            "directional_outcome_vs_causal_reasoning",
            "domain_sector_scope_vs_direct_ticker_scope",
            "feedback_candidate_vs_approved_learning_update",
        ],
        "family_rules": {
            "domain_analyst": (
                "Domain outcome labels require a frozen expectation and "
                "matured outcome window."
            ),
            "pipeline_model": (
                "Model evaluation labels describe evidence and constraint "
                "results; they are never hit/miss market outcomes."
            ),
        },
    }


def applicable_labels(taxonomy: dict[str, Any]) -> set[str]:
    labels = taxonomy.get("labels", {})
    groups = taxonomy.get("applicable_label_groups", [])
    return {
        str(label)
        for group in groups
        for label in labels.get(group, [])
        if label
    }


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
