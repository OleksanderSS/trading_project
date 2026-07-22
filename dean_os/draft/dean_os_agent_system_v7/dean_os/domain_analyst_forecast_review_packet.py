from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_DOMAIN_THESIS_REVIEW_JSON = "reports/dean_os/domain_analyst_thesis_review_packet_current/latest.json"
DEFAULT_VERTICAL_SLICE_JSON = "reports/dean_os/domain_analyst_vertical_slice_current/latest.json"
DEFAULT_REGIME_SCENARIO_JSON = None


class DomainAnalystForecastReviewPacket:
    """Review-only expectation ledger for a domain analyst thesis.

    This packet turns a domain thesis into explicit forecast candidates and an
    outcome-review protocol. It deliberately does not call the statement a trade
    recommendation, does not evaluate outcomes early, and does not write
    learning memory or analyst configuration.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_forecast_review_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_thesis_review_json: str | Path = DEFAULT_DOMAIN_THESIS_REVIEW_JSON,
        vertical_slice_json: str | Path | None = DEFAULT_VERTICAL_SLICE_JSON,
        regime_scenario_json: str | Path | None = DEFAULT_REGIME_SCENARIO_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        thesis_review = _load_json(domain_thesis_review_json)
        vertical = _load_optional_json(vertical_slice_json)
        regime_scenario = _load_optional_json(regime_scenario_json)
        regime_context = _forecast_regime_context(thesis_review, regime_scenario)
        candidates = _forecast_candidates(thesis_review, regime_context)
        planes = _analyst_control_planes()
        taxonomy = _outcome_taxonomy()
        checks = _review_checks(thesis_review=thesis_review, vertical=vertical, regime_context=regime_context, candidates=candidates)
        status = _packet_status(checks)
        payload = {
            "run_id": _run_id("domain_analyst_forecast_review_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_forecast_review_packet",
            "inputs": {
                "domain_thesis_review_json": str(domain_thesis_review_json),
                "domain_thesis_review_run_id": thesis_review.get("run_id"),
                "vertical_slice_json": str(vertical_slice_json) if vertical_slice_json else None,
                "vertical_slice_run_id": vertical.get("run_id") if vertical else None,
                "regime_scenario_json": str(regime_scenario_json) if regime_scenario_json else None,
                "regime_scenario_run_id": regime_scenario.get("run_id") if regime_scenario else None,
            },
            "summary": _summary(status, thesis_review, candidates, planes, taxonomy, regime_context),
            "naming_contract": _naming_contract(),
            "regime_scenario_context": regime_context,
            "forecast_candidates": candidates,
            "analyst_control_planes": planes,
            "outcome_taxonomy": taxonomy,
            "outcome_review_protocol": _outcome_review_protocol(),
            "self_improvement_boundary": _self_improvement_boundary(),
            "review_checks": checks,
            "decision_guidance": _decision_guidance(status, checks),
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(domain_thesis_review_json),
            "operator_next_steps": _operator_next_steps(status, checks),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_forecast_review_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_forecast_review_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Domain Analyst Forecast Review Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Forecast candidates: {summary.get('forecast_candidate_count')}",
        f"- Regime scenario context: {summary.get('regime_scenario_context_available')} gaps={summary.get('scenario_evidence_gap_count')}",
        f"- Analyst control planes: {summary.get('analyst_control_plane_count')}",
        f"- Manual review required: {summary.get('manual_review_required')}",
        f"- Can register case after review: {summary.get('can_register_case_after_manual_review')}",
        f"- Can promote learning now: {summary.get('can_promote_learning_now')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can create analyst research recommendation: {summary.get('can_create_analyst_research_recommendation')}",
        f"- Can create execution recommendation: {summary.get('can_create_execution_recommendation')}",
        f"- Legacy can create recommendation: {summary.get('can_create_recommendation')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Naming Contract",
        "",
        f"- Preferred term: `{payload.get('naming_contract', {}).get('preferred_term')}`",
        f"- Avoided term: `{payload.get('naming_contract', {}).get('avoided_term')}`",
        f"- Boundary: {payload.get('naming_contract', {}).get('boundary')}",
        "",
        "## Forecast Candidates",
        "",
    ]
    for item in payload.get("forecast_candidates", []):
        lines.extend(
            [
                f"- `{item.get('expectation_id')}`",
                f"  - direction: `{item.get('expected_direction')}`",
                f"  - horizon days: {item.get('horizon_days')}",
                f"  - confidence: {item.get('confidence')}",
                f"  - outcome status: `{item.get('outcome_status')}`",
            ]
        )
    if not payload.get("forecast_candidates"):
        lines.append("- none")
    lines.extend(["", "## Regime Scenario Context", ""])
    regime = payload.get("regime_scenario_context", {})
    lines.append(f"- Available: {regime.get('available')}")
    lines.append(f"- Packet status: `{regime.get('packet_status')}`")
    lines.append(f"- Probability mass valid: {regime.get('probability_mass_valid')}")
    lines.append(f"- Scenario probabilities: `{regime.get('scenario_probabilities')}`")
    lines.append(f"- Self-check horizons: `{', '.join(regime.get('self_check_horizons') or []) or 'none'}`")
    lines.extend(["", "## Analyst Control Planes", ""])
    for plane in payload.get("analyst_control_planes", []):
        lines.append(f"- `{plane.get('plane_id')}`: {plane.get('purpose')}")
    lines.extend(["", "## Outcome Taxonomy", ""])
    for item in payload.get("outcome_taxonomy", []):
        lines.append(f"- `{item.get('bucket_id')}`: {item.get('meaning')}")
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")
    lines.extend(["", "## Decision Guidance", ""])
    lines.extend(f"- {item}" for item in guidance.get("reasons", []))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _summary(
    status: str,
    thesis_review: dict[str, Any],
    candidates: list[dict[str, Any]],
    planes: list[dict[str, Any]],
    taxonomy: list[dict[str, Any]],
    regime_context: dict[str, Any],
) -> dict[str, Any]:
    thesis_summary = thesis_review.get("summary", {})
    return {
        "packet_status": status,
        "domain_id": thesis_summary.get("domain_id"),
        "source_thesis_review_status": thesis_summary.get("packet_status"),
        "forecast_candidate_count": len(candidates),
        "regime_scenario_context_available": regime_context.get("available"),
        "regime_scenario_status": regime_context.get("packet_status"),
        "scenario_evidence_gap_count": len(regime_context.get("top_evidence_gaps") or []),
        "scenario_probability_mass_valid": regime_context.get("probability_mass_valid"),
        "self_check_horizon_count": len(regime_context.get("self_check_horizons") or []),
        "analyst_control_plane_count": len(planes),
        "outcome_taxonomy_count": len(taxonomy),
        "manual_review_required": True,
        "requires_future_outcome_observation": True,
        "can_register_case_after_manual_review": status in {
            "forecast_review_ready_pending_outcomes",
            "forecast_review_ready_with_cautions_pending_outcomes",
        },
        "can_promote_learning_now": False,
        "can_create_analyst_research_recommendation": True,
        "can_create_analyst_self_improvement_proposal": True,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _naming_contract() -> dict[str, Any]:
    return {
        "preferred_term": "thesis_expectation_or_forecast_candidate",
        "allowed_recommendation_term": "review_only_analyst_recommendation",
        "avoided_term": "execution_or_investment_recommendation",
        "why": (
            "The analyst is allowed to state reviewable expectations, research recommendations, evidence requests, "
            "scenario priorities, and improvement proposals. It is not allowed to tell the system or operator to buy, "
            "sell, hold, size, or route orders."
        ),
        "boundary": "An analyst recommendation must be evidence-bound, review-only, non-executing, and tied to a future outcome review protocol when it makes a forecast-like claim.",
    }


def _forecast_regime_context(
    thesis_review: dict[str, Any],
    regime_scenario: dict[str, Any] | None,
) -> dict[str, Any]:
    if regime_scenario:
        summary = regime_scenario.get("summary", {})
        graph = regime_scenario.get("scenario_outcome_graph", {})
        return {
            "available": True,
            "source": "regime_scenario_json",
            "packet_status": summary.get("packet_status"),
            "source_run_id": regime_scenario.get("run_id"),
            "active_regime_fields": thesis_review.get("regime_scenario_context", {}).get("active_regime_fields", []),
            "scenario_probabilities": graph.get("scenario_probabilities", {}),
            "probability_mass_valid": graph.get("probability_mass_check", {}).get("valid"),
            "top_evidence_gaps": regime_scenario.get("evidence_gap_priorities", [])[:8],
            "self_check_horizons": graph.get("horizons", []),
            "review_note": "Scenario context is frozen for future outcome review; it is not an execution signal.",
        }
    embedded = thesis_review.get("regime_scenario_context")
    if isinstance(embedded, dict) and embedded.get("available"):
        context = dict(embedded)
        context["source"] = "domain_thesis_review_embedded_context"
        return context
    return {
        "available": False,
        "source": None,
        "packet_status": None,
        "active_regime_fields": [],
        "scenario_probabilities": {},
        "probability_mass_valid": None,
        "top_evidence_gaps": [],
        "self_check_horizons": [],
        "review_note": "No regime/scenario context supplied; outcome review must rely on thesis evidence and future observations.",
    }


def _forecast_candidates(thesis_review: dict[str, Any], regime_context: dict[str, Any]) -> list[dict[str, Any]]:
    summary = thesis_review.get("summary", {})
    thesis = thesis_review.get("thesis_snapshot", {})
    thesis_id = thesis.get("thesis_id") or summary.get("thesis_id") or thesis_review.get("run_id")
    thesis_text = str(thesis.get("thesis") or "").strip()
    if not thesis_text:
        return []
    supporting = thesis_review.get("supporting_evidence_examples", [])
    contradicting = thesis_review.get("contradicting_evidence_examples", [])
    expected_direction = thesis.get("expected_direction") or summary.get("expected_direction")
    direct_ticker_count = int(summary.get("ticker_direct_count") or 0)
    return [
        {
            "expectation_id": f"domain_expectation:{thesis_id}",
            "expectation_type": "domain_or_sector_thesis_expectation",
            "source_artifact_mode": thesis_review.get("mode"),
            "source_run_id": thesis_review.get("run_id"),
            "domain_id": summary.get("domain_id") or thesis.get("domain_id"),
            "thesis_id": thesis_id,
            "as_of": thesis.get("as_of") or thesis_review.get("created_at"),
            "horizon_days": thesis.get("horizon_days"),
            "stance": thesis.get("stance") or summary.get("thesis_stance"),
            "expected_direction": expected_direction,
            "confidence": thesis.get("confidence") or summary.get("confidence"),
            "thesis": thesis_text,
            "key_drivers": thesis.get("key_drivers", []),
            "assumptions": thesis.get("assumptions", []),
            "supporting_evidence_ids": [item.get("evidence_id") for item in supporting if item.get("evidence_id")],
            "contradicting_evidence_ids": [item.get("evidence_id") for item in contradicting if item.get("evidence_id")],
            "evidence_balance": {
                "supporting_count": len(supporting),
                "contradicting_count": len(contradicting),
                "required_evidence_missing": summary.get("required_evidence_missing", []),
                "ticker_direct_count": direct_ticker_count,
            },
            "evaluation_scope": "domain_or_sector_level" if direct_ticker_count == 0 else "domain_or_sector_plus_direct_ticker_bridge_required",
            "required_outcome_observations": _required_outcome_observations(expected_direction, direct_ticker_count),
            "invalidation_triggers": _invalidation_triggers(thesis_review),
            "regime_scenario_context": {
                "available": regime_context.get("available"),
                "source": regime_context.get("source"),
                "packet_status": regime_context.get("packet_status"),
                "scenario_probabilities": regime_context.get("scenario_probabilities", {}),
                "top_evidence_gaps": regime_context.get("top_evidence_gaps", []),
                "self_check_horizons": regime_context.get("self_check_horizons", []),
                "probability_mass_valid": regime_context.get("probability_mass_valid"),
            },
            "outcome_status": "pending_future_observation",
            "allowed_future_labels": [item["bucket_id"] for item in _outcome_taxonomy()],
            "allowed_review_outputs": [
                "analyst_summary",
                "scenario_priority",
                "evidence_request",
                "research_recommendation",
                "self_improvement_proposal",
            ],
            "is_investment_recommendation": False,
            "is_execution_recommendation": False,
            "is_review_only_analyst_recommendation": True,
            "can_create_trade": False,
            "learning_use": "candidate_case_only_until_outcome_and_human_causal_review",
        }
    ]


def _required_outcome_observations(expected_direction: Any, direct_ticker_count: int) -> list[str]:
    observations = [
        "Observe whether the domain/sector outcome moved in the expected direction over the stated horizon.",
        "Check whether the stated key drivers actually materialized during the horizon.",
        "Check whether contradicting evidence or listed risks materialized enough to explain the result.",
        "Compare confidence to outcome quality; do not reward overconfident mixed or underspecified statements.",
    ]
    if expected_direction == "mixed":
        observations.append("Define mixed-outcome criteria before scoring; otherwise label as inconclusive_or_not_mature.")
    if direct_ticker_count == 0:
        observations.append("Do not score as a ticker-specific forecast unless a later ticker bridge supplies direct ticker evidence.")
    else:
        observations.append("Keep ticker-specific attribution separate from the domain expectation.")
    return observations


def _invalidation_triggers(thesis_review: dict[str, Any]) -> list[str]:
    risks = thesis_review.get("risk_and_blind_spot_review", {}).get("risks", []) or []
    blind_spots = thesis_review.get("risk_and_blind_spot_review", {}).get("blind_spots", []) or []
    triggers = [
        "Required evidence lane is later found stale, wrong, synthetic, or uncited.",
        "Outcome appears correct but the named causal drivers did not materialize.",
        "Contradicting evidence dominates the support during the review horizon.",
    ]
    for item in risks[:6]:
        triggers.append(f"Risk materializes: {item}")
    for item in blind_spots[:6]:
        triggers.append(f"Blind spot remains unresolved: {item}")
    return triggers


def _analyst_control_planes() -> list[dict[str, Any]]:
    return [
        _plane("evidence_coverage", "Required evidence lanes are present and source-backed."),
        _plane("source_quality_timestamp", "Sources have usable timestamps, provenance, and local artifact lineage."),
        _plane("regime_scenario_context", "Regime vector, scenario probabilities, evidence gaps, and self-check horizons are frozen beside the expectation."),
        _plane("thesis_falsifiability", "The thesis can be checked later against observable outcomes and drivers."),
        _plane("horizon_maturity", "The forecast horizon is explicit and outcomes are not judged before maturity unless labeled diagnostic."),
        _plane("confidence_calibration", "Confidence is compared with evidence quality and eventual outcome quality."),
        _plane("contradiction_handling", "Contradicting evidence, risks, and blind spots stay visible beside supporting evidence."),
        _plane("causal_attribution", "A hit is separated from correct reasoning; right outcome with wrong reason is not promoted as skill."),
        _plane("luck_vs_skill", "Correct, lucky, wrong-reason, miss, inconclusive, and unavailable outcomes use separate buckets."),
        _plane("ticker_directness_boundary", "Sector/domain expectations are not scored as direct ticker theses without a bridge."),
        _plane("learning_promotion_readiness", "Lessons become proposals only after outcome review and human approval."),
    ]


def _plane(plane_id: str, purpose: str) -> dict[str, str]:
    return {
        "plane_id": plane_id,
        "purpose": purpose,
        "decision_authority": "can_block_or_request_human_review_only",
    }


def _outcome_taxonomy() -> list[dict[str, str]]:
    return [
        _bucket("correct_for_stated_reasons", "Outcome matched the expectation and the stated drivers materially explain it."),
        _bucket("correct_but_lucky_or_wrong_reason", "Outcome matched direction, but causal review does not support the analyst's stated reasoning."),
        _bucket("incorrect_forecast", "Outcome moved against the expectation or key assumptions failed."),
        _bucket("inconclusive_or_not_mature", "The horizon has not matured or the result is mixed/too small to judge."),
        _bucket("unfalsifiable_or_underspecified", "The statement lacked enough measurable criteria to score fairly."),
        _bucket("data_unavailable", "Required outcome, price, sector, or source data is unavailable or invalid."),
    ]


def _bucket(bucket_id: str, meaning: str) -> dict[str, str]:
    return {"bucket_id": bucket_id, "meaning": meaning, "learning_use": "review_only_until_human_approved"}


def _outcome_review_protocol() -> list[dict[str, str]]:
    return [
        {
            "step_id": "freeze_expectation",
            "rule": "Save the expectation before the outcome is known; do not rewrite thesis text after the fact.",
        },
        {
            "step_id": "wait_or_mark_diagnostic",
            "rule": "Do not judge a forecast before horizon maturity unless the run is explicitly marked as early diagnostic.",
        },
        {
            "step_id": "score_direction_and_reason",
            "rule": "Score both directional outcome and causal reasoning; correct direction alone is not enough for learning promotion.",
        },
        {
            "step_id": "retain_all_buckets",
            "rule": "Keep hits, misses, lucky hits, inconclusive cases, underspecified cases, and unavailable-data cases together.",
        },
        {
            "step_id": "propose_do_not_apply",
            "rule": "The analyst may propose changes to prompts/profiles/evidence rules, but cannot apply them without a human-approved learning loop.",
        },
    ]


def _self_improvement_boundary() -> dict[str, Any]:
    return {
        "allowed": [
            "create detailed review-only analyst recommendations",
            "summarize why a thesis was right or wrong",
            "separate correct outcome from correct reasoning",
            "request missing evidence or outcome data",
            "propose profile, prompt, evidence-lane, or weighting changes for review",
            "rerun review-only local artifact packets when inputs are supplied",
        ],
        "blocked": [
            "write learning memory without review",
            "change analyst weights or profiles directly",
            "write production config",
            "create execution, buy/sell/hold, sizing, allocation, or order recommendations",
            "route paper or live orders",
            "publish an unreviewed dashboard",
        ],
    }


def _review_checks(
    *,
    thesis_review: dict[str, Any],
    vertical: dict[str, Any] | None,
    regime_context: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    summary = thesis_review.get("summary", {})
    thesis = thesis_review.get("thesis_snapshot", {})
    checks = [
        _check(
            "pass" if thesis_review.get("mode") == "domain_analyst_thesis_review_packet" else "fail",
            "domain_thesis_review_artifact_type",
            str(thesis_review.get("mode")),
        ),
        _check("pass" if candidates else "fail", "forecast_candidates_present", f"{len(candidates)} candidates."),
        _check("pass" if thesis.get("thesis") else "fail", "thesis_text_present", str(thesis.get("thesis_id"))),
        _check("pass" if _positive_int(thesis.get("horizon_days")) else "fail", "horizon_days_present", str(thesis.get("horizon_days"))),
        _check("pass" if thesis.get("expected_direction") or summary.get("expected_direction") else "fail", "expected_direction_present", str(thesis.get("expected_direction") or summary.get("expected_direction"))),
        _check("pass" if _confidence_present(thesis.get("confidence") or summary.get("confidence")) else "fail", "confidence_present", str(thesis.get("confidence") or summary.get("confidence"))),
        _check("pass" if summary.get("manual_review_required") is True else "fail", "manual_review_required", str(summary.get("manual_review_required"))),
        _must_be_false(summary, "can_create_direct_ticker_thesis_without_bridge", "no_direct_ticker_thesis_without_bridge"),
        _must_be_false(summary, "can_write_learning_memory", "no_learning_write"),
        _must_be_false(summary, "can_change_analyst_weights", "no_weight_change"),
        _must_be_false(summary, "can_write_config", "no_config_write"),
        _check(
            "pass" if summary.get("can_create_analyst_research_recommendation") is not False else "fail",
            "analyst_research_recommendation_not_blocked",
            "Review-only analyst recommendations remain allowed unless an upstream artifact explicitly blocks them.",
        ),
        _must_not_be_true(summary, "can_create_execution_recommendation", "no_execution_recommendation"),
        _must_not_be_true(summary, "can_create_buy_sell_hold_recommendation", "no_buy_sell_hold_recommendation"),
        _must_be_false(summary, "can_create_recommendation", "legacy_no_execution_recommendation"),
        _must_be_false(summary, "can_trade", "no_trading"),
    ]
    if summary.get("required_evidence_missing"):
        checks.append(_check("warn", "required_evidence_missing", ", ".join(summary.get("required_evidence_missing") or [])))
    else:
        checks.append(_check("pass", "required_evidence_covered_or_reviewed", "No required evidence missing in thesis summary."))
    if (thesis.get("expected_direction") or summary.get("expected_direction")) == "mixed":
        checks.append(_check("warn", "mixed_direction_needs_explicit_outcome_definition", "Mixed expectations need multi-outcome criteria."))
    if int(summary.get("ticker_direct_count") or 0) == 0:
        checks.append(_check("warn", "no_ticker_direct_evidence_for_ticker_scoring", "Keep evaluation at domain/sector level unless a later bridge adds direct ticker evidence."))
    else:
        checks.append(_check("pass", "ticker_direct_evidence_requires_separate_bridge", f"{summary.get('ticker_direct_count')} ticker-direct items present."))
    if vertical:
        vertical_summary = vertical.get("summary", {})
        audit = vertical.get("synthetic_fixture_audit", {})
        checks.extend(
            [
                _check(
                    "pass" if vertical.get("mode") == "domain_analyst_vertical_slice_run" else "fail",
                    "vertical_slice_artifact_type",
                    str(vertical.get("mode")),
                ),
                _check("pass" if audit.get("has_synthetic_marker") is False else "fail", "vertical_no_synthetic_marker", str(audit.get("has_synthetic_marker"))),
                _check("pass" if audit.get("has_fixture_marker") is False else "fail", "vertical_no_fixture_marker", str(audit.get("has_fixture_marker"))),
                _must_not_be_true(vertical_summary, "can_create_execution_recommendation", "vertical_no_execution_recommendation"),
                _must_not_be_true(vertical_summary, "can_create_buy_sell_hold_recommendation", "vertical_no_buy_sell_hold_recommendation"),
                _must_not_be_true(vertical_summary, "can_create_recommendation", "vertical_legacy_no_execution_recommendation"),
                _check("pass" if vertical_summary.get("can_trade") is False else "fail", "vertical_no_trading", "Vertical slice has no trading authority."),
            ]
        )
    if regime_context.get("available"):
        checks.append(
            _check(
                "pass" if str(regime_context.get("packet_status", "")).startswith("domain_analyst_regime_scenario_ready") else "warn",
                "regime_scenario_context_review_ready",
                str(regime_context.get("packet_status")),
            )
        )
        checks.append(
            _check(
                "pass" if regime_context.get("probability_mass_valid") is True else "fail",
                "regime_scenario_probability_mass_valid",
                str(regime_context.get("probability_mass_valid")),
            )
        )
    else:
        checks.append(_check("warn", "regime_scenario_context_not_supplied", "Forecast remains valid, but future causal review has less context."))
    return checks


def _packet_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "blocked_forecast_review"
    if any(check["status"] == "warn" for check in checks):
        return "forecast_review_ready_with_cautions_pending_outcomes"
    return "forecast_review_ready_pending_outcomes"


def _decision_guidance(status: str, checks: list[dict[str, str]]) -> dict[str, Any]:
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    failures = [check["code"] for check in checks if check["status"] == "fail"]
    if failures:
        action = "fix_failed_forecast_review_checks"
    elif warnings:
        action = "manual_review_expectation_cautions_before_case_registry"
    else:
        action = "register_expectations_as_pending_cases_after_manual_review"
    reasons = [
        f"Packet status is {status}.",
        "The analyst output is treated as review-only thesis expectations, not investment recommendations.",
        "Future learning must compare outcome direction and causal reasoning, including lucky or wrong-reason hits.",
    ]
    if warnings:
        reasons.append("Warnings: " + ", ".join(warnings) + ".")
    if failures:
        reasons.append("Failures: " + ", ".join(failures) + ".")
    return {
        "recommended_review_action": action,
        "pass_count": sum(1 for check in checks if check["status"] == "pass"),
        "warning_count": len(warnings),
        "fail_count": len(failures),
        "reasons": reasons,
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No live source fetch or external API call is made.",
        "No GPT or FinBERT adapter is invoked.",
        "No outcome is scored before future observation data is supplied.",
        "No learning memory, analyst weight, prompt, profile, model, or production config is changed.",
        "No buy/sell/hold recommendation, allocation, price target, paper order, broker route, or live trade is generated.",
        "No unreviewed dashboard publication is authorized.",
    ]


def _commands(domain_thesis_review_json: str | Path) -> dict[str, str]:
    return {
        "rerun_forecast_review_packet": (
            "python run_agent_domain_analyst_forecast_review_packet.py "
            f"--domain-thesis-review-json {domain_thesis_review_json} "
            "--output-dir reports\\dean_os\\domain_analyst_forecast_review_packet_current"
        ),
        "future_case_registry": (
            "python run_agent_domain_analyst_case_registry_packet.py "
            f"--domain-thesis-review-json {domain_thesis_review_json} "
            "--domain-template-standardization-json reports\\dean_os\\domain_analyst_template_standardization_packet_current\\latest.json "
            "--output-dir reports\\dean_os\\domain_analyst_case_registry_packet_current"
        ),
    }


def _operator_next_steps(status: str, checks: list[dict[str, str]]) -> list[str]:
    if status == "blocked_forecast_review":
        return ["Fix failed forecast-review checks before treating the thesis as a future learning case."]
    steps = [
        "Use this packet as the expectation ledger for the semiconductor analyst template review.",
        "If the template is manually accepted, register this expectation as a pending case; do not promote learning yet.",
        "When the horizon matures, evaluate outcome direction and causal reasoning separately.",
        "Keep misses, lucky hits, inconclusive cases, and unavailable-data cases visible beside correct-for-reasons cases.",
    ]
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    if warnings:
        steps.append("Resolve or explicitly accept forecast cautions before learning review: " + ", ".join(warnings) + ".")
    return steps


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _confidence_present(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= number <= 1.0


def _must_be_false(summary: dict[str, Any], key: str, code: str) -> dict[str, str]:
    return _check(
        "pass" if summary.get(key) is False else "fail",
        code,
        f"{key}={summary.get(key)!r}.",
    )


def _must_not_be_true(summary: dict[str, Any], key: str, code: str) -> dict[str, str]:
    return _check(
        "pass" if summary.get(key) is not True else "fail",
        code,
        f"{key} must not be True; got {summary.get(key)!r}.",
    )


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return _load_json(resolved)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
