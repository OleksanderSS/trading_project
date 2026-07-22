from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.research_corpus.hypothesis_reverse_analysis import (
    HYPOTHESIS_REVERSE_ANALYSIS_CONTRACT,
    build_hypothesis_reverse_analysis,
)
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready

HYPOTHESIS_LEARNING_REVIEW_CONTRACT = "dean_hypothesis_learning_review_v1"

OUTCOME_ERROR_CODES = {
    "trigger_polarity_mismatch",
    "claim_scope_overreach",
    "event_novelty_misread",
    "contingent_risk_generalized",
    "expectation_context_missing",
    "exposure_mapping_missing",
    "time_anchor_error",
    "horizon_mismatch",
    "unsupported_causal_leap",
    "wrong_transmission_channel",
    "false_analogy",
    "priced_in_blindness",
    "data_quality_failure",
    "outcome_not_observable",
    "true_hypothesis_wrong_market_reaction",
    "confounder_dominated",
    "inconclusive",
    "unknown_falsification_cause",
}

_ASSESSMENT_ERRORS = {
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

_ASSESSMENT_PRIMARY_ERROR = {
    "credible_context_source_but_trigger_polarity_conflicts_with_generated_claim": "trigger_polarity_mismatch",
    "tier_1_primary_source_with_precise_rule_but_generated_claim_is_too_broad": "event_novelty_misread",
    "credible_context_source_but_contingent_project_risk_was_generalized_into_persistent_sector_constraint": "contingent_risk_generalized",
}

_RULES = {
    "trigger_polarity_mismatch": {
        "proposal_type": "template_guard",
        "target_component": "event_to_hypothesis_template",
        "proposed_rule": (
            "A warning, doubt, funding concern, downside revision or question about a "
            "mechanism must not instantiate a positive persistence or acceleration claim. "
            "The candidate claim must preserve the trigger's direction or be deferred."
        ),
        "activation_conditions": [
            "trigger contains warning, doubt, concern, question, cut or downside language",
            "generated claim asserts positive persistence, growth or acceleration",
        ],
    },
    "claim_scope_overreach": {
        "proposal_type": "template_guard",
        "target_component": "claim_scope_validator",
        "proposed_rule": (
            "The hypothesis scope must not exceed the entities, products, geographies and "
            "mechanism explicitly supported by the trigger and exposure map."
        ),
        "activation_conditions": [
            "trigger names a bounded entity, product, geography or project",
            "generated claim generalizes to the whole sector or market",
        ],
    },
    "event_novelty_misread": {
        "proposal_type": "classification_guard",
        "target_component": "policy_event_classifier",
        "proposed_rule": (
            "A clarification, enforcement reminder or continuation of an existing policy "
            "must not be classified as a new policy or sanctions change unless a dated "
            "delta from the previous rule is identified."
        ),
        "activation_conditions": [
            "official source describes guidance, clarification, enforcement or continuation",
            "no explicit new legal or operational delta is recorded",
        ],
    },
    "contingent_risk_generalized": {
        "proposal_type": "template_guard",
        "target_component": "event_to_hypothesis_template",
        "proposed_rule": (
            "Possible, prospective, avoidable or project-contingent constraints must remain "
            "conditional and project-scoped; they cannot become an observed persistent "
            "sector constraint without confirming evidence."
        ),
        "activation_conditions": [
            "trigger describes possible, future, avoidable or project-specific risk",
            "generated claim asserts an existing or persistent sector-wide condition",
        ],
    },
    "expectation_context_missing": {
        "proposal_type": "evidence_requirement",
        "target_component": "expectation_context_gate",
        "proposed_rule": (
            "When a claim depends on surprise, acceleration, weakening or repricing, record "
            "the pre-event expectation baseline or explicitly defer confidence calibration."
        ),
        "activation_conditions": [
            "claim uses relative expectation or repricing language",
            "structured pre-event expectation context is absent",
        ],
    },
    "exposure_mapping_missing": {
        "proposal_type": "evidence_requirement",
        "target_component": "exposure_mapping_gate",
        "proposed_rule": (
            "Entity-level policy or supply hypotheses require an explicit affected-entity, "
            "customer, product or revenue exposure map before broad transmission claims."
        ),
        "activation_conditions": [
            "trigger applies only to named entities, products, customers or jurisdictions",
            "affected issuer exposure is not mapped",
        ],
    },
}

_RULES.update(
    {
        "time_anchor_error": {
            "proposal_type": "hard_invariant",
            "target_component": "point_in_time_gate",
            "proposed_rule": (
                "A hypothesis, baseline or outcome with a missing, naive or post-event "
                "as-of timestamp must be rejected from replay until point-in-time lineage "
                "is reconstructed."
            ),
            "activation_conditions": [
                "an evidence or measurement timestamp is missing or not timezone-aware",
                "a baseline source became available after the event anchor",
            ],
        },
        "horizon_mismatch": {
            "proposal_type": "hard_invariant",
            "target_component": "horizon_scope_gate",
            "proposed_rule": (
                "Sector-thesis horizons and event-response horizons must remain separate; "
                "an outcome from one family cannot confirm or falsify the other."
            ),
            "activation_conditions": [
                "a sector thesis is scored on an event-response checkpoint",
                "an event response is substituted with a 30/90/180-day sector horizon",
            ],
        },
        "unsupported_causal_leap": {
            "proposal_type": "reasoning_guard",
            "target_component": "transmission_chain_validator",
            "proposed_rule": (
                "Do not move from trigger to outcome without naming the intervening "
                "mechanism, measurable state change and invalidation condition."
            ),
            "activation_conditions": [
                "the claim skips one or more required transmission steps",
                "no observable intermediate state connects trigger and outcome",
            ],
        },
        "wrong_transmission_channel": {
            "proposal_type": "reasoning_guard",
            "target_component": "mechanism_router",
            "proposed_rule": (
                "Route the event through the mechanism supported by its evidence; do not "
                "reuse a demand, capex, supply or policy template solely because its words match."
            ),
            "activation_conditions": [
                "the evidence mechanism differs from the generated transmission channel",
                "the selected channel lacks a measurable intermediate state",
            ],
        },
        "false_analogy": {
            "proposal_type": "analogy_guard",
            "target_component": "historical_analog_selector",
            "proposed_rule": (
                "A historical analog must match the relevant regime, mechanism and "
                "expectation state; lexical similarity alone cannot support an analogy."
            ),
            "activation_conditions": [
                "the analog matches event wording but not the world-state variables",
                "a structural break or materially different expectation state is present",
            ],
        },
        "priced_in_blindness": {
            "proposal_type": "evidence_requirement",
            "target_component": "expectation_context_gate",
            "proposed_rule": (
                "Separate fundamental correctness from market surprise and require a "
                "pre-event expectation or positioning baseline before predicting price reaction."
            ),
            "activation_conditions": [
                "fundamental follow-through occurs but the predicted market reaction does not",
                "the packet lacks consensus, positioning or market-implied expectation context",
            ],
        },
        "data_quality_failure": {
            "proposal_type": "data_guard",
            "target_component": "evidence_quality_gate",
            "proposed_rule": (
                "Quarantine a failed observation and rerun from verified lineage; do not "
                "learn a market rule from missing, stale, duplicated or corrupted data."
            ),
            "activation_conditions": [
                "the outcome depends on stale, incomplete, duplicated or corrupted data",
                "the required source lineage or coverage threshold fails",
            ],
        },
        "outcome_not_observable": {
            "proposal_type": "measurement_guard",
            "target_component": "replay_registration_gate",
            "proposed_rule": (
                "Do not register a replay whose target universe, metric, baseline, due time "
                "or minimum coverage is undefined."
            ),
            "activation_conditions": [
                "the outcome metric cannot be reconstructed point-in-time",
                "the measurement context has no explicit coverage threshold or checkpoint rule",
            ],
        },
        "true_hypothesis_wrong_market_reaction": {
            "proposal_type": "outcome_decomposition",
            "target_component": "hypothesis_outcome_scorer",
            "proposed_rule": (
                "Score fundamental follow-through and market reaction as separate legs; a "
                "true mechanism with a wrong price reaction must not rewrite the mechanism rule."
            ),
            "activation_conditions": [
                "the fundamental target is met but the market-reaction target is missed",
                "the claim contains both fundamental and price-response observations",
            ],
        },
        "confounder_dominated": {
            "proposal_type": "causal_review",
            "target_component": "outcome_attribution_gate",
            "proposed_rule": (
                "When a stronger contemporaneous driver dominates the checkpoint, mark the "
                "case unresolved or confounded rather than falsifying the original mechanism."
            ),
            "activation_conditions": [
                "a dated macro, policy, earnings or market shock overlaps the outcome window",
                "the confounder plausibly dominates the target metric",
            ],
        },
    }
)

_ACTION_PLAYBOOK = {
    "trigger_polarity_mismatch": (
        "reformulate the claim to preserve trigger direction",
        "defer when direction cannot be stated without adding unsupported evidence",
        ["polarity test passes", "new claim has explicit invalidation signals"],
    ),
    "claim_scope_overreach": (
        "shrink the claim to named entities, products, geographies and mechanism",
        "defer until an exposure map supports broader scope",
        ["claim entities are a subset of mapped exposure", "sector generalization is absent"],
    ),
    "event_novelty_misread": (
        "compare the official event with the prior rule and record the dated delta",
        "classify as clarification or continuation when no delta is proven",
        ["prior rule is cited", "new legal or operational delta is explicit"],
    ),
    "contingent_risk_generalized": (
        "keep the claim conditional and project-specific",
        "defer until a project state or confirming primary source is observable",
        ["project scope is predeclared", "persistent sector language is removed"],
    ),
    "expectation_context_missing": (
        "attach a pre-event expectation baseline and checkpoint comparison rule",
        "defer calibration while keeping the event as trigger-only evidence",
        ["baseline predates trigger", "target universe and minimum coverage are explicit"],
    ),
    "exposure_mapping_missing": (
        "build an issuer/customer/product/jurisdiction exposure map",
        "defer entity-level transmission claims",
        ["affected set is point-in-time", "each mapped exposure has a source locator"],
    ),
    "time_anchor_error": (
        "reject the affected packet and reconstruct point-in-time lineage",
        "quarantine the case when timestamps cannot be recovered",
        ["all timestamps are timezone-aware", "baseline availability is not after trigger"],
    ),
    "horizon_mismatch": (
        "route the claim to the correct horizon family and rebuild its checkpoints",
        "reject scoring when the original horizon cannot be recovered",
        ["horizon family is explicit", "no cross-family substitution is present"],
    ),
    "unsupported_causal_leap": (
        "insert measurable intermediate states into the transmission chain",
        "downgrade to a conditional scenario or defer",
        ["each causal edge has evidence", "at least one intermediate state is observable"],
    ),
    "wrong_transmission_channel": (
        "reroute through the evidence-supported mechanism",
        "reject the claim when no supported channel exists",
        ["mechanism matches event class", "channel-specific metric is declared"],
    ),
    "false_analogy": (
        "remove the analog and search by world-state similarity",
        "continue without an analog rather than force a match",
        ["regime and expectation states match", "structural breaks are recorded"],
    ),
    "priced_in_blindness": (
        "split fundamental and market-reaction legs and attach expectation context",
        "score only the fundamental leg when market expectations are unavailable",
        ["pre-event expectation baseline exists", "price reaction is not treated as mechanism truth"],
    ),
    "data_quality_failure": (
        "quarantine the observation and reacquire verified data",
        "mark outcome unresolved if repair is impossible",
        ["lineage and coverage gates pass", "duplicate/stale checks pass"],
    ),
    "outcome_not_observable": (
        "define metric, universe, baseline, due time and coverage before registration",
        "defer the replay candidate",
        ["measurement context validates", "checkpoint can be reconstructed point-in-time"],
    ),
    "true_hypothesis_wrong_market_reaction": (
        "record separate fundamental and price-leg results",
        "open priced-in or confounder review before any rule change",
        ["leg-level scores are present", "mechanism rule is unchanged automatically"],
    ),
    "confounder_dominated": (
        "attach the confounder and mark attribution unresolved",
        "extend observation only through a new reviewed hypothesis version",
        ["confounder is dated", "relative contribution is reviewed"],
    ),
}

_HARD_INVARIANT_ERRORS = {"time_anchor_error", "horizon_mismatch"}
_EMPIRICAL_MIN_CASES = 3


class HypothesisLearningReview:
    """Diagnose hypothesis failures and create review-only learning proposals."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/hypothesis_learning_review_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        packet_json: str | Path | dict[str, Any],
        review_gate_json: str | Path | dict[str, Any],
        *,
        outcome_json: str | Path | dict[str, Any] | None = None,
        journal_path: str | Path = "data/dean_os/system_journal.jsonl",
        save: bool = True,
    ) -> dict[str, Any]:
        packet, packet_path = _load_object(packet_json)
        gate, gate_path = _load_object(review_gate_json)
        outcomes, outcome_path = (
            _load_object(outcome_json) if outcome_json is not None else ({}, None)
        )
        _verify_gate_binding(packet, packet_path, gate)

        packet_hypotheses = {
            str(item.get("hypothesis_id")): item
            for item in packet.get("hypotheses", []) or []
            if item.get("hypothesis_id")
        }
        outcome_by_hypothesis = _outcome_index(outcomes)
        existing_cases = _existing_pattern_cases(journal_path)
        cases: list[dict[str, Any]] = []
        proposal_basis: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for review in gate.get("hypothesis_review", []) or []:
            hypothesis_id = str(review.get("hypothesis_id") or "").strip()
            if not hypothesis_id:
                continue
            hypothesis = packet_hypotheses.get(hypothesis_id, {})
            outcome = outcome_by_hypothesis.get(hypothesis_id)
            reverse_analysis = build_hypothesis_reverse_analysis(
                hypothesis=hypothesis,
                review=review,
                outcome=outcome,
                allowed_error_codes=OUTCOME_ERROR_CODES,
            )
            error_codes = [
                str(item.get("error_code"))
                for item in reverse_analysis.get("machine_diagnosis_candidates", [])
                if item.get("error_code")
            ]
            primary_error_code = _ASSESSMENT_PRIMARY_ERROR.get(
                str(review.get("source_assessment"))
            )
            result_label = str((outcome or {}).get("result_label") or "").strip() or None
            disposition = review.get("disposition")
            learning_candidate = bool(
                disposition in {"reformulate", "reject"}
                or result_label
                in {"falsified", "weakened", "miss", "unobservable"}
            )
            case_id = "hypothesis_case_" + _digest(
                {
                    "hypothesis_id": hypothesis_id,
                    "gate_run_id": gate.get("run_id"),
                    "outcome_id": (outcome or {}).get("outcome_id"),
                    "disposition": disposition,
                    "error_codes": error_codes,
                }
            )[:24]
            case = {
                "case_id": case_id,
                "hypothesis_id": hypothesis_id,
                "hypothesis": review.get("hypothesis") or hypothesis.get("hypothesis"),
                "trigger_event": review.get("trigger_event"),
                "disposition": disposition,
                "rationale": review.get("rationale"),
                "proposed_hypothesis": review.get("proposed_hypothesis"),
                "source_assessment": review.get("source_assessment"),
                "expectation_context_available": review.get(
                    "expectation_context_available"
                ),
                "outcome": outcome,
                "result_label": result_label,
                "error_codes": error_codes,
                "primary_error_code": primary_error_code,
                "reverse_analysis": reverse_analysis,
                "learning_candidate": learning_candidate,
                "diagnosis_status": (
                    "machine_root_cause_candidates_ready_for_review"
                    if reverse_analysis.get("proposal_eligible_error_codes")
                    else "root_cause_review_required"
                ),
                "positive_example": disposition == "accept_for_replay" and outcome is None,
            }
            cases.append(case)
            if learning_candidate:
                proposal_error_codes = set(
                    str(item) for item in (outcome or {}).get("error_labels") or []
                )
                if primary_error_code:
                    proposal_error_codes.add(primary_error_code)
                for item in reverse_analysis.get("machine_diagnosis_candidates", []) or []:
                    if (
                        item.get("proposal_eligible") is True
                        and str(item.get("basis") or "").startswith(
                            (
                                "structured_",
                                "declared_",
                                "fundamental_",
                                "market_",
                            )
                        )
                    ):
                        proposal_error_codes.add(str(item.get("error_code")))
                for error_code in sorted(proposal_error_codes):
                    rule = _RULES.get(error_code)
                    if rule is not None:
                        proposal_basis[(error_code, rule["target_component"])].append(case)

        proposals = _build_proposals(
            proposal_basis,
            existing_cases=existing_cases,
            gate_run_id=str(gate.get("run_id") or "unknown"),
        )
        unresolved = [
            case
            for case in cases
            if case["learning_candidate"]
            and case["diagnosis_status"] == "root_cause_review_required"
        ]
        payload: dict[str, Any] = {
            "run_id": _run_id("hypothesis_learning_review"),
            "created_at": utc_now_iso(),
            "mode": "hypothesis_learning_review",
            "contract": HYPOTHESIS_LEARNING_REVIEW_CONTRACT,
            "inputs": {
                "world_model_packet": _binding(packet_path, packet),
                "review_gate": _binding(gate_path, gate),
                "outcome_artifact": _binding(outcome_path, outcomes),
                "journal_path": str(journal_path),
            },
            "summary": {
                "hypothesis_case_count": len(cases),
                "accepted_positive_example_count": sum(
                    bool(case["positive_example"]) for case in cases
                ),
                "learning_candidate_count": sum(
                    bool(case["learning_candidate"]) for case in cases
                ),
                "diagnosed_error_count": sum(len(case["error_codes"]) for case in cases),
                "learning_proposal_count": len(proposals),
                "promotion_ready_proposal_count": sum(
                    proposal["promotion_status"] == "eligible_for_human_promotion_review"
                    for proposal in proposals
                ),
                "unresolved_root_cause_count": len(unresolved),
                "reverse_analysis_card_count": len(cases),
                "machine_root_cause_ready_count": sum(
                    bool(
                        case.get("reverse_analysis", {}).get(
                            "proposal_eligible_error_codes"
                        )
                    )
                    for case in cases
                ),
                "automatic_rule_update_allowed": False,
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "can_trade": False,
            },
            "error_taxonomy": sorted(OUTCOME_ERROR_CODES),
            "reverse_analysis_contract": HYPOTHESIS_REVERSE_ANALYSIS_CONTRACT,
            "hypothesis_cases": cases,
            "reverse_analysis_cards": [
                case["reverse_analysis"] for case in cases
            ],
            "learning_proposals": proposals,
            "action_playbook": {
                error_code: _playbook_entry(error_code, rule)
                for error_code, rule in sorted(_RULES.items())
            },
            "unresolved_root_cause_cases": unresolved,
            "promotion_policy": {
                "empirical_pattern_minimum_independent_reviewed_cases": _EMPIRICAL_MIN_CASES,
                "hard_invariant_minimum_reproduced_regression_cases": 1,
                "human_review_required": True,
                "regression_test_required": True,
                "automatic_prompt_or_template_update_allowed": False,
                "single_failure_can_rewrite_production_rules": False,
            },
            "safety": _safety(),
        }
        if save:
            reverse_payload = {
                "run_id": _run_id("hypothesis_reverse_analysis"),
                "created_at": payload["created_at"],
                "mode": "hypothesis_reverse_analysis",
                "contract": HYPOTHESIS_REVERSE_ANALYSIS_CONTRACT,
                "inputs": payload["inputs"],
                "summary": {
                    "card_count": len(cases),
                    "post_outcome_card_count": sum(
                        case["reverse_analysis"].get("analysis_stage") == "post_outcome"
                        for case in cases
                    ),
                    "machine_root_cause_ready_count": payload["summary"][
                        "machine_root_cause_ready_count"
                    ],
                    "automatic_rule_update_allowed": False,
                    "can_trade": False,
                },
                "reverse_analysis_cards": payload["reverse_analysis_cards"],
                "safety": _safety(),
            }
            reverse_dir = self.output_dir.parent / "hypothesis_reverse_analysis_current"
            reverse_payload["saved_paths"] = ReviewArtifactWriter(reverse_dir).write(
                payload=reverse_payload,
                markdown=render_hypothesis_reverse_analysis_markdown(reverse_payload),
                run_id=reverse_payload["run_id"],
            )
            payload["reverse_analysis_report"] = {
                "contract": HYPOTHESIS_REVERSE_ANALYSIS_CONTRACT,
                "run_id": reverse_payload["run_id"],
                "saved_paths": reverse_payload["saved_paths"],
            }
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_hypothesis_learning_review_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_hypothesis_learning_review_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Hypothesis Failure & Learning Review",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Hypotheses reviewed: {summary.get('hypothesis_case_count')}",
        f"- Learning candidates: {summary.get('learning_candidate_count')}",
        f"- Learning proposals: {summary.get('learning_proposal_count')}",
        f"- Reverse-analysis cards: {summary.get('reverse_analysis_card_count')}",
        f"- Machine root-cause candidates ready for review: {summary.get('machine_root_cause_ready_count')}",
        f"- Ready for human promotion review: {summary.get('promotion_ready_proposal_count')}",
        f"- Automatic rule update allowed: {summary.get('automatic_rule_update_allowed')}",
        f"- Learning memory write performed: {summary.get('learning_memory_write_performed')}",
        "",
        "## Hypothesis Cases",
        "",
    ]
    for case in payload.get("hypothesis_cases", []) or []:
        trigger = case.get("trigger_event") or {}
        lines.extend(
            [
                f"### `{case.get('hypothesis_id')}`",
                "",
                f"- Claim: {case.get('hypothesis')}",
                f"- Decision: `{case.get('disposition')}`",
                f"- Trigger: {trigger.get('title') or 'missing'}",
                f"- Error codes: {', '.join(case.get('error_codes') or []) or 'none'}",
                f"- Diagnosis: `{case.get('diagnosis_status')}`",
                f"- Rationale: {case.get('rationale') or 'none'}",
            ]
        )
        if case.get("proposed_hypothesis"):
            lines.append(f"- Reformulation: {case.get('proposed_hypothesis')}")
        reverse = case.get("reverse_analysis") or {}
        lines.extend(
            [
                f"- Reverse-analysis status: `{reverse.get('machine_analysis_status')}`",
                f"- Recommended next action: `{(reverse.get('recommended_next_action') or {}).get('action')}`",
            ]
        )
        for diagnosis in reverse.get("machine_diagnosis_candidates", []) or []:
            lines.append(
                f"  - `{diagnosis.get('error_code')}` / `{diagnosis.get('diagnostic_strength')}` / "
                f"{diagnosis.get('failure_layer')}: {diagnosis.get('basis')}"
            )
        lines.append("")
    lines.extend(["## Learning Proposals", ""])
    if not payload.get("learning_proposals"):
        lines.append("- none")
    for proposal in payload.get("learning_proposals", []) or []:
        lines.extend(
            [
                f"### `{proposal.get('proposal_id')}`",
                "",
                f"- Error: `{proposal.get('error_code')}`",
                f"- Target: `{proposal.get('target_component')}`",
                f"- Rule: {proposal.get('proposed_rule')}",
                "- When:",
                *[
                    f"  - {condition}"
                    for condition in proposal.get("activation_conditions") or []
                ],
                f"- Recommended action: {proposal.get('recommended_action')}",
                f"- Fallback: {proposal.get('fallback_action')}",
                "- Verify:",
                *[
                    f"  - {item}"
                    for item in proposal.get("verification_requirements") or []
                ],
                f"- Independent reviewed cases: {proposal.get('current_independent_case_count')}/{proposal.get('minimum_independent_case_count')}",
                f"- Status: `{proposal.get('promotion_status')}`",
                "- Production update performed: false",
                "",
            ]
        )
    lines.extend(
        [
            "## Governance",
            "",
            "A proposal is diagnostic evidence, not an active rule. Empirical patterns need at least three independent reviewed cases, a regression test and explicit human promotion review.",
            "",
        ]
    )
    return "\n".join(lines)


def render_hypothesis_reverse_analysis_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# DEAN-OS Hypothesis Reverse Analysis",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Cards: {summary.get('card_count')}",
        f"- Post-outcome cards: {summary.get('post_outcome_card_count')}",
        f"- Machine root-cause candidates ready for review: {summary.get('machine_root_cause_ready_count')}",
        "- Automatic rule update allowed: false",
        "- Trading allowed: false",
        "",
    ]
    for card in payload.get("reverse_analysis_cards", []) or []:
        lines.extend(
            [
                f"## `{card.get('hypothesis_id')}`",
                "",
                f"- Stage: `{card.get('analysis_stage')}`",
                f"- Status: `{card.get('machine_analysis_status')}`",
                f"- Claim before: {card.get('claim_before') or 'missing'}",
                f"- Claim after review: {card.get('claim_after_review') or 'unchanged'}",
                f"- Outcome: `{card.get('result_label') or 'pending'}`",
                "- Machine diagnoses:",
            ]
        )
        diagnoses = card.get("machine_diagnosis_candidates") or []
        if not diagnoses:
            lines.append("  - none; do not invent a failure cause")
        for diagnosis in diagnoses:
            lines.extend(
                [
                    f"  - `{diagnosis.get('error_code')}` — `{diagnosis.get('diagnostic_strength')}` / `{diagnosis.get('failure_layer')}`",
                    f"    - basis: {diagnosis.get('basis')}",
                    f"    - evidence: {', '.join(diagnosis.get('evidence_trace') or []) or 'none'}",
                    f"    - counterfactual: {diagnosis.get('counterfactual_test') or 'not defined'}",
                ]
            )
        lines.append("- Overlooked or underweighted signals:")
        overlooked = card.get("overlooked_or_underweighted_signals") or []
        if not overlooked:
            lines.append("  - none identified")
        for item in overlooked:
            lines.append(
                f"  - `{item.get('signal')}`: {item.get('interpretation')}"
            )
        alternatives = card.get("alternative_explanations") or []
        lines.append(
            "- Alternative explanations: "
            + ("; ".join(str(item) for item in alternatives) if alternatives else "none recorded")
        )
        lines.extend(
            [
                f"- Recommended next action: `{(card.get('recommended_next_action') or {}).get('action')}`",
                f"- Rule proposal status: `{(card.get('agent_change_proposal') or {}).get('status')}`",
                "- Automatic application: false",
                "",
            ]
        )
    lines.extend(
        [
            "## Governance",
            "",
            "The machine may diagnose, compare, prepare counterfactuals and propose bounded changes. It may not promote a rule, rewrite a production prompt, change a model or trade. A failed result without causal evidence remains unresolved rather than becoming a fabricated lesson.",
            "",
            "Forecast quality is direction-neutral: a correctly predicted decline is a successful forecast. Positive or negative realized market return describes market performance only; confirmation or falsification must compare that realization with the predeclared claim direction, threshold and horizon.",
            "",
        ]
    )
    return "\n".join(lines)


def _diagnose_errors(
    review: dict[str, Any], outcome: dict[str, Any] | None
) -> list[str]:
    errors = list(_ASSESSMENT_ERRORS.get(str(review.get("source_assessment")), []))
    explicit = list((outcome or {}).get("error_labels") or [])
    invalid = sorted({str(item) for item in explicit if str(item) not in OUTCOME_ERROR_CODES})
    if invalid:
        raise ValueError(f"unsupported hypothesis outcome error labels: {invalid}")
    errors.extend(str(item) for item in explicit)
    result_label = str((outcome or {}).get("result_label") or "").strip()
    if result_label in {"falsified", "weakened", "miss"} and not errors:
        errors.append("unknown_falsification_cause")
    return sorted(set(errors))


def _build_proposals(
    proposal_basis: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    existing_cases: dict[str, set[str]],
    gate_run_id: str,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for (error_code, target_component), cases in sorted(proposal_basis.items()):
        rule = _RULES[error_code]
        pattern_key = _digest(
            {
                "error_code": error_code,
                "target_component": target_component,
                "proposed_rule": rule["proposed_rule"],
            }
        )
        current_case_ids = {
            f"{gate_run_id}:{case['hypothesis_id']}" for case in cases
        }
        independent_case_ids = existing_cases.get(pattern_key, set()) | current_case_ids
        minimum = 1 if error_code in _HARD_INVARIANT_ERRORS else _EMPIRICAL_MIN_CASES
        ready = len(independent_case_ids) >= minimum
        proposal_id = "learning_proposal_" + _digest(
            {"pattern_key": pattern_key, "gate_run_id": gate_run_id}
        )[:24]
        proposals.append(
            {
                "proposal_id": proposal_id,
                "pattern_key": pattern_key,
                "error_code": error_code,
                "proposal_type": rule["proposal_type"],
                "target_component": target_component,
                "proposed_rule": rule["proposed_rule"],
                "activation_conditions": rule["activation_conditions"],
                "recommended_action": _playbook_entry(error_code, rule)[
                    "recommended_action"
                ],
                "fallback_action": _playbook_entry(error_code, rule)[
                    "fallback_action"
                ],
                "verification_requirements": _playbook_entry(error_code, rule)[
                    "verification_requirements"
                ],
                "current_case_ids": sorted(current_case_ids),
                "current_independent_case_count": len(independent_case_ids),
                "minimum_independent_case_count": minimum,
                "promotion_status": (
                    "eligible_for_human_promotion_review"
                    if ready
                    else "collect_more_independent_reviewed_cases"
                ),
                "human_review_required": True,
                "regression_test_required": True,
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            }
        )
    return proposals


def _playbook_entry(error_code: str, rule: dict[str, Any]) -> dict[str, Any]:
    action, fallback, verification = _ACTION_PLAYBOOK.get(
        error_code,
        (
            "route the case to manual root-cause review",
            "leave the hypothesis unresolved and do not update rules",
            ["root cause is explicitly labeled", "evidence lineage is complete"],
        ),
    )
    return {
        "target_component": rule["target_component"],
        "activation_conditions": list(rule["activation_conditions"]),
        "recommended_action": action,
        "fallback_action": fallback,
        "verification_requirements": list(verification),
    }


def _existing_pattern_cases(journal_path: str | Path) -> dict[str, set[str]]:
    journal = SystemJournal(journal_path)
    cases: dict[str, set[str]] = defaultdict(set)
    for record in journal.read_verified():
        if record.get("event_type") != "learning_proposal_created":
            continue
        payload = record.get("payload") or {}
        pattern_key = str(payload.get("pattern_key") or "").strip()
        if not pattern_key:
            continue
        for case_id in payload.get("current_case_ids") or []:
            cases[pattern_key].add(str(case_id))
    return cases


def _outcome_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = payload.get("hypothesis_outcomes") or payload.get("outcomes") or []
    if not isinstance(items, list):
        raise ValueError("hypothesis outcome artifact must contain an outcomes list")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not item.get("hypothesis_id"):
            continue
        hypothesis_id = str(item["hypothesis_id"])
        if hypothesis_id in result:
            raise ValueError(f"duplicate outcome for hypothesis: {hypothesis_id}")
        result[hypothesis_id] = item
    return result


def _verify_gate_binding(
    packet: dict[str, Any], packet_path: Path | None, gate: dict[str, Any]
) -> None:
    source = gate.get("source_packet") or {}
    if source.get("run_id") != packet.get("run_id"):
        raise ValueError("review gate is not bound to the supplied world-model packet run")
    expected_sha = source.get("sha256")
    if expected_sha and packet_path is None:
        raise ValueError("hash-bound review gate requires the packet file path")
    if expected_sha and artifact_binding(packet_path, packet)["sha256"] != expected_sha:
        raise ValueError("world-model packet changed after manual review")


def _load_object(
    value: str | Path | dict[str, Any] | None,
) -> tuple[dict[str, Any], Path | None]:
    if value is None:
        return {}, None
    if isinstance(value, dict):
        return value, None
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload, path


def _binding(path: Path | None, payload: dict[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    return artifact_binding(path, payload)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "automatic_learning_disabled": True,
        "learning_memory_write_performed": False,
        "production_rule_update_performed": False,
        "model_promotion_performed": False,
        "broker_access_performed": False,
        "can_trade": False,
    }


__all__ = [
    "HYPOTHESIS_LEARNING_REVIEW_CONTRACT",
    "OUTCOME_ERROR_CODES",
    "HypothesisLearningReview",
    "render_hypothesis_learning_review_markdown",
    "render_hypothesis_reverse_analysis_markdown",
]
