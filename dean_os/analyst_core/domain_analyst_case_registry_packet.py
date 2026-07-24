from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_DOMAIN_THESIS_REVIEW_JSON = "reports/dean_os/domain_analyst_thesis_review_packet_current/latest.json"
DEFAULT_DOMAIN_TEMPLATE_STANDARDIZATION_JSON = "reports/dean_os/domain_analyst_template_standardization_packet_current/latest.json"
DEFAULT_DOMAIN_FORECAST_REVIEW_JSON = None
DEFAULT_OUTCOME_EVALUATION_JSON = None


class DomainAnalystCaseRegistryPacket:
    """Review-only case registry for domain analyst observations and outcomes.

    The registry is deliberately not a learning-memory writer. It preserves
    pending, hit, miss, inconclusive, and invalid/unresolved cases so future
    learning promotion cannot quietly train only on correct-looking examples.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_case_registry_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_thesis_review_json: str | Path = DEFAULT_DOMAIN_THESIS_REVIEW_JSON,
        domain_template_standardization_json: str | Path | None = DEFAULT_DOMAIN_TEMPLATE_STANDARDIZATION_JSON,
        domain_forecast_review_json: str | Path | None = DEFAULT_DOMAIN_FORECAST_REVIEW_JSON,
        outcome_evaluation_json: str | Path | None = DEFAULT_OUTCOME_EVALUATION_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        thesis_review = _load_json(domain_thesis_review_json)
        thesis_review_sha256 = _file_sha256(
            Path(domain_thesis_review_json)
        )
        template_standardization = _load_optional_json(domain_template_standardization_json)
        forecast_review = _load_optional_json(domain_forecast_review_json)
        outcome_evaluation = _load_optional_json(outcome_evaluation_json)
        case_entries = _case_entries(
            thesis_review,
            forecast_review,
            outcome_evaluation,
            thesis_review_sha256=thesis_review_sha256,
        )
        observation_entries = _source_observation_entries(thesis_review)
        checks = _review_checks(
            thesis_review=thesis_review,
            template_standardization=template_standardization,
            forecast_review=forecast_review,
            outcome_evaluation=outcome_evaluation,
            case_entries=case_entries,
        )
        status = _registry_status(checks, case_entries)
        payload = {
            "run_id": _run_id("domain_analyst_case_registry_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_case_registry_packet",
            "inputs": {
                "domain_thesis_review_json": str(domain_thesis_review_json),
                "domain_thesis_review_run_id": thesis_review.get("run_id"),
                "domain_thesis_review_sha256": thesis_review_sha256,
                "domain_template_standardization_json": str(domain_template_standardization_json)
                if domain_template_standardization_json
                else None,
                "domain_template_standardization_run_id": template_standardization.get("run_id")
                if template_standardization
                else None,
                "domain_forecast_review_json": str(domain_forecast_review_json)
                if domain_forecast_review_json
                else None,
                "domain_forecast_review_run_id": forecast_review.get("run_id") if forecast_review else None,
                "outcome_evaluation_json": str(outcome_evaluation_json) if outcome_evaluation_json else None,
                "outcome_evaluation_run_id": outcome_evaluation.get("run_id") if outcome_evaluation else None,
            },
            "summary": _summary(status, thesis_review, case_entries, observation_entries),
            "registry_policy": _registry_policy(),
            "prospective_registration_contract": _prospective_registration_contract(),
            "case_entries": case_entries,
            "source_observation_entries": observation_entries,
            "comparison_axes": _comparison_axes(),
            "review_checks": checks,
            "decision_guidance": _decision_guidance(status, checks, case_entries),
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(domain_thesis_review_json, domain_template_standardization_json, domain_forecast_review_json),
            "operator_next_steps": _operator_next_steps(status, checks, case_entries),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_case_registry_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_case_registry_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Domain Analyst Case Registry Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Registry status: `{summary.get('registry_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Cases: {summary.get('case_count')}",
        f"- Expectation cases: expectations={summary.get('expectation_case_count')}",
        f"- Observations: {summary.get('source_observation_count')}",
        f"- Outcome buckets: `{summary.get('outcome_bucket_counts')}`",
        f"- Manual review required: {summary.get('manual_review_required')}",
        f"- Can write case registry artifact: {summary.get('can_write_case_registry_artifact')}",
        f"- Can create analyst learning recommendation: {summary.get('can_create_analyst_learning_recommendation')}",
        f"- Can train from hits only: {summary.get('can_train_from_hits_only')}",
        f"- Can drop miss cases: {summary.get('can_drop_miss_cases')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can create execution recommendation: {summary.get('can_create_execution_recommendation')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Registry Policy",
        "",
    ]
    for item in payload.get("registry_policy", {}).get("rules", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Case Entries", ""])
    for case in payload.get("case_entries", []):
        lines.append(
            f"- `{case.get('case_id')}` bucket=`{case.get('outcome_bucket')}` "
            f"direction=`{case.get('expected_direction')}` confidence={case.get('confidence')} "
            f"prospective=`{case.get('prospective_registration_status')}`"
        )
        if case.get("monitoring_horizons_days"):
            lines.append(
                "  Monitoring horizons: "
                + ", ".join(
                    f"{value}d"
                    for value in case.get(
                        "monitoring_horizons_days", []
                    )
                )
            )
        reasoning = case.get("verified_reasoning_baseline", {})
        if reasoning:
            lines.append(
                "  Reasoning baseline: "
                f"status=`{reasoning.get('status')}` "
                f"hash_bound={reasoning.get('hash_bound')} "
                f"channels={reasoning.get('transmission_channel_count')} "
                f"ticker_directional={reasoning.get('directional_ticker_reasoning_event_count')}"
            )
    lines.extend(["", "## Source Observation Entries", ""])
    for item in payload.get("source_observation_entries", [])[:12]:
        lines.append(
            f"- `{item.get('observation_id')}` {item.get('evidence_type')} "
            f"stance={item.get('stance_hint')} season={item.get('seasonality_context', {}).get('season_tag')}"
        )
    if not payload.get("source_observation_entries"):
        lines.append("- none")
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
    case_entries: list[dict[str, Any]],
    observation_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    thesis_summary = thesis_review.get("summary", {})
    bucket_counts = Counter(item.get("outcome_bucket", "unknown") for item in case_entries)
    return {
        "registry_status": status,
        "domain_id": thesis_summary.get("domain_id"),
        "case_count": len(case_entries),
        "expectation_case_count": sum(1 for item in case_entries if item.get("case_type") == "domain_thesis_expectation"),
        "source_observation_count": len(observation_entries),
        "outcome_bucket_counts": dict(sorted(bucket_counts.items())),
        "hit_count": bucket_counts.get("hit", 0),
        "miss_count": bucket_counts.get("miss", 0),
        "inconclusive_count": bucket_counts.get("inconclusive", 0),
        "pending_count": sum(count for bucket, count in bucket_counts.items() if str(bucket).startswith("pending")),
        "invalid_or_unresolved_count": bucket_counts.get("invalid_or_unresolved", 0),
        "manual_review_required": True,
        "can_write_case_registry_artifact": True,
        "can_create_analyst_learning_recommendation": True,
        "can_create_analyst_self_improvement_proposal": True,
        "can_promote_learning_now": False,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_train_from_hits_only": False,
        "can_drop_miss_cases": False,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _registry_policy() -> dict[str, Any]:
    return {
        "purpose": "Preserve analyst observations, hypotheses, forecasts, outcomes, and misses before any learning promotion.",
        "rules": [
            "Register a case before the outcome is known whenever possible.",
            "Keep hit, miss, inconclusive, pending, and invalid/unresolved buckets visible.",
            "Do not promote only correct cases; misses and inconclusive cases must remain comparable.",
            "Treat a correct outcome as a candidate lesson, not proof that the causal thesis was right.",
            "Treat a miss as a useful negative case, not garbage to delete.",
            "Allow detailed analyst learning recommendations and postmortems, but keep them proposal-only until human review.",
            "Separate source observations, domain theses, ticker theses, outcome evaluation, learning promotion, and trading.",
            "Attach seasonal, macro, policy, and regime context so future comparisons are made against similar cases.",
        ],
    }


def _prospective_registration_contract() -> dict[str, Any]:
    return {
        "contract": "dean_domain_prospective_case_v1",
        "purpose": (
            "Freeze the thesis, evidence balance, source hash, baseline "
            "market snapshot, and evaluation schedule before outcomes are "
            "known."
        ),
        "default_monitoring_horizons_days": [30, 90, 180],
        "required_evaluation_dimensions": [
            "equal_weight_sector_basket_return",
            "excess_return_versus_qqq",
            "sector_positive_breadth",
            "required_evidence_lane_status_changes",
            "hyperscaler_capex_guidance_changes",
            "official_policy_rule_changes",
            "comparable_fundamental_margin_changes",
            "causal_driver_materialization",
        ],
        "outcome_labels": [
            "correct_for_stated_reasons",
            "correct_but_lucky_or_wrong_reason",
            "incorrect_forecast",
            "inconclusive_or_not_mature",
            "data_unavailable",
        ],
        "manual_review_required": True,
        "automatic_learning_promotion_allowed": False,
        "automatic_ticker_forecast_allowed": False,
        "automatic_trading_allowed": False,
    }


def _case_entries(
    thesis_review: dict[str, Any],
    forecast_review: dict[str, Any] | None,
    outcome_evaluation: dict[str, Any] | None,
    *,
    thesis_review_sha256: str | None = None,
) -> list[dict[str, Any]]:
    entries = _forecast_expectation_cases(forecast_review) if forecast_review else []
    if not entries:
        entries = [
            _domain_thesis_case(
                thesis_review,
                source_sha256=thesis_review_sha256,
            )
        ]
    if outcome_evaluation:
        entries.extend(_outcome_cases(outcome_evaluation))
    return entries


def _domain_thesis_case(
    thesis_review: dict[str, Any],
    *,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    summary = thesis_review.get("summary", {})
    thesis = thesis_review.get("thesis_snapshot", {})
    analytical = thesis_review.get("analytical_review", {})
    reasoning = thesis_review.get("reasoning_snapshot_context", {})
    as_of = thesis.get("as_of") or thesis_review.get("created_at")
    expected_direction = thesis.get("expected_direction") or summary.get("expected_direction")
    primary_horizon = int(thesis.get("horizon_days") or 180)
    monitoring_horizons = sorted(
        set([30, 90, primary_horizon])
    )
    return {
        "case_id": f"domain_thesis:{thesis.get('thesis_id') or thesis_review.get('run_id')}",
        "case_type": "domain_thesis",
        "source_artifact_mode": thesis_review.get("mode"),
        "source_run_id": thesis_review.get("run_id"),
        "source_artifact_sha256": source_sha256,
        "domain_id": summary.get("domain_id") or thesis.get("domain_id"),
        "thesis_level": "domain_or_sector_thesis",
        "created_at": as_of,
        "horizon_days": primary_horizon,
        "monitoring_horizons_days": monitoring_horizons,
        "evaluation_schedule": _evaluation_schedule(
            as_of,
            monitoring_horizons,
        ),
        "prospective_registration_status": (
            "registered_before_outcome"
        ),
        "manual_review_status": analytical.get(
            "prospective_case_status",
            "pending_manual_review",
        ),
        "thesis_text": thesis.get("thesis"),
        "stance": thesis.get("stance"),
        "expected_direction": expected_direction,
        "confidence": thesis.get("confidence") or summary.get("confidence"),
        "confidence_interpretation": analytical.get(
            "confidence_interpretation"
        ),
        "outcome_bucket": "pending_domain_outcome",
        "outcome_label": None,
        "realized_return": None,
        "evidence_balance": {
            "supporting_count": summary.get("supporting_evidence_count"),
            "contradicting_count": summary.get("contradicting_evidence_count"),
            "ticker_direct_count": summary.get("ticker_direct_count"),
            "required_evidence_missing": summary.get("required_evidence_missing", []),
        },
        "baseline_market_snapshot": analytical.get(
            "market_snapshot", {}
        ),
        "scenario_framework": analytical.get(
            "scenario_framework", []
        ),
        "verified_reasoning_baseline": {
            "reasoning_snapshot_run_id": reasoning.get("run_id"),
            "reasoning_snapshot_sha256": reasoning.get("snapshot_sha256"),
            "runtime_sha256": reasoning.get("runtime_sha256_actual"),
            "status": reasoning.get("status"),
            "hash_bound": reasoning.get("hash_bound"),
            "classified_event_count": reasoning.get(
                "classified_event_count"
            ),
            "transmission_channel_count": reasoning.get(
                "transmission_channel_count"
            ),
            "transmission_channel_counts": reasoning.get(
                "transmission_channel_counts", {}
            ),
            "evidence_touched_regime_dimension_count": reasoning.get(
                "evidence_touched_regime_dimension_count"
            ),
            "regime_context": reasoning.get("regime_context"),
            "scenario_graph_status": reasoning.get(
                "scenario_graph_status"
            ),
            "expectation_gap_status": reasoning.get(
                "expectation_gap_status"
            ),
            "directional_ticker_reasoning_event_count": reasoning.get(
                "directional_ticker_reasoning_event_count"
            ),
        },
        "candidate_hypotheses": reasoning.get("hypothesis_ledger", []),
        "reasoning_evidence_gaps": reasoning.get("evidence_gaps", []),
        "quality_cautions": analytical.get(
            "quality_cautions", []
        ),
        "required_evaluation_dimensions": (
            _prospective_registration_contract()[
                "required_evaluation_dimensions"
            ]
        ),
        "ticker_scope": {
            "status": analytical.get("ticker_decision"),
            "direct_ticker_thesis_allowed": False,
        },
        "linked_artifact_verification_status": (
            thesis_review.get("linked_artifact_verification") or {}
        ).get("status"),
        "seasonality_context": _seasonality_context(as_of),
        "context_tags": _context_tags_from_thesis(thesis),
        "quality_notes": _quality_notes_for_domain_thesis(thesis_review),
        "learning_use": "candidate_case_only_until_outcome_and_review",
    }


def _forecast_expectation_cases(forecast_review: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not forecast_review:
        return []
    entries = []
    taxonomy = forecast_review.get("outcome_taxonomy", [])
    for candidate in forecast_review.get("forecast_candidates", []):
        as_of = candidate.get("as_of") or forecast_review.get("created_at")
        entries.append(
            {
                "case_id": candidate.get("expectation_id") or f"forecast_expectation:{candidate.get('thesis_id')}",
                "case_type": "domain_thesis_expectation",
                "source_artifact_mode": forecast_review.get("mode"),
                "source_run_id": forecast_review.get("run_id"),
                "domain_id": candidate.get("domain_id"),
                "thesis_id": candidate.get("thesis_id"),
                "thesis_level": candidate.get("expectation_type") or "domain_or_sector_thesis_expectation",
                "created_at": as_of,
                "horizon_days": candidate.get("horizon_days"),
                "expected_direction": candidate.get("expected_direction"),
                "confidence": candidate.get("confidence"),
                "outcome_bucket": "pending_expectation_outcome",
                "outcome_label": None,
                "realized_return": None,
                "evidence_balance": candidate.get("evidence_balance", {}),
                "required_outcome_observations": candidate.get("required_outcome_observations", []),
                "invalidation_triggers": candidate.get("invalidation_triggers", []),
                "allowed_future_labels": candidate.get("allowed_future_labels", []),
                "allowed_review_outputs": candidate.get(
                    "allowed_review_outputs",
                    [
                        "analyst_summary",
                        "causal_postmortem",
                        "evidence_request",
                        "learning_recommendation",
                        "self_improvement_proposal",
                    ],
                ),
                "outcome_taxonomy": taxonomy,
                "evaluation_scope": candidate.get("evaluation_scope"),
                "seasonality_context": _seasonality_context(as_of),
                "context_tags": _context_tags_from_forecast(candidate),
                "quality_notes": _quality_notes_for_forecast_expectation(candidate),
                "learning_use": "frozen_expectation_case_only_until_outcome_and_human_causal_review",
            }
        )
    return entries


def _outcome_cases(outcome_evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    result = outcome_evaluation.get("outcome_evaluation", outcome_evaluation)
    entries = []
    for item in result.get("evaluations", []):
        bucket = _outcome_bucket(item)
        entries.append(
            {
                "case_id": f"outcome:{item.get('record_id')}",
                "case_type": "evaluated_learning_record",
                "source_artifact_mode": outcome_evaluation.get("mode"),
                "source_run_id": outcome_evaluation.get("run_id"),
                "record_id": item.get("record_id"),
                "agent_name": item.get("agent_name"),
                "topic": item.get("topic"),
                "tickers": item.get("tickers", []),
                "expected_direction": item.get("expected_direction"),
                "confidence": None,
                "horizon_days": item.get("horizon_days"),
                "created_at": item.get("created_at"),
                "due_at": item.get("due_at"),
                "target_at": item.get("target_at"),
                "outcome_bucket": bucket,
                "outcome_label": item.get("outcome_label"),
                "realized_return": item.get("realized_return"),
                "evaluation_status": item.get("status"),
                "reason": item.get("reason"),
                "context_tags": item.get("context_tags", []),
                "regime_tags": item.get("regime_tags", []),
                "seasonality_context": _seasonality_context(item.get("created_at")),
                "quality_notes": _quality_notes_for_outcome(item),
                "learning_use": "comparison_case_only_until_balanced_review",
            }
        )
    return entries


def _outcome_bucket(item: dict[str, Any]) -> str:
    label = item.get("outcome_label")
    status = item.get("status")
    if label in {"hit", "miss", "inconclusive"}:
        return str(label)
    if status == "not_due":
        return "pending_not_due"
    if status in {"evaluable", "updated"}:
        return "pending_outcome_label"
    if status in {"missing_tickers", "missing_price_window", "no_price_after_created_at"}:
        return "invalid_or_unresolved"
    return "pending_unclassified"


def _source_observation_entries(thesis_review: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for group_name, items in (
        ("supporting", thesis_review.get("supporting_evidence_examples", [])),
        ("contradicting", thesis_review.get("contradicting_evidence_examples", [])),
    ):
        for item in items:
            observation_id = item.get("evidence_id") or f"{group_name}:{len(entries) + 1}"
            entries.append(
                {
                    "observation_id": observation_id,
                    "observation_group": group_name,
                    "source_type": item.get("source_type"),
                    "published_at": item.get("published_at"),
                    "evidence_type": item.get("evidence_type"),
                    "directness": item.get("directness"),
                    "stance_hint": item.get("stance_hint"),
                    "summary": item.get("summary"),
                    "limitations": item.get("limitations", []),
                    "seasonality_context": _seasonality_context(item.get("published_at")),
                    "learning_use": "source_observation_not_forecast",
                }
            )
    return entries


def _comparison_axes() -> list[dict[str, str]]:
    return [
        {"axis": "outcome_bucket", "rule": "Compare hits, misses, inconclusive, pending, and invalid cases side by side."},
        {"axis": "evidence_type", "rule": "Compare cases by evidence lane such as capex_cycle, sector_demand, policy, and supply_chain."},
        {"axis": "seasonality_context", "rule": "Compare summer, earnings season, year-end, and other calendar contexts separately."},
        {"axis": "directness", "rule": "Do not compare sector-only observations as if they were ticker-direct evidence."},
        {"axis": "regime_or_macro_context", "rule": "Separate market, policy, rates, liquidity, and geopolitical regimes before learning."},
        {"axis": "causal_plausibility", "rule": "A hit is not automatically a valid causal thesis; check if the stated driver was present."},
    ]


def _review_checks(
    *,
    thesis_review: dict[str, Any],
    template_standardization: dict[str, Any] | None,
    forecast_review: dict[str, Any] | None,
    outcome_evaluation: dict[str, Any] | None,
    case_entries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    checks = [
        _check(
            "pass" if thesis_review.get("mode") == "domain_analyst_thesis_review_packet" else "fail",
            "domain_thesis_review_artifact_type",
            str(thesis_review.get("mode")),
        ),
        _check("pass" if case_entries else "fail", "case_entries_present", f"{len(case_entries)} cases."),
        _check("pass", "balanced_buckets_defined", "Hit, miss, inconclusive, pending, and invalid buckets are explicit."),
        _check("pass", "hits_only_training_blocked", "Correct cases cannot be promoted without comparable misses and pending cases."),
        _check("pass", "miss_cases_retained", "Miss cases are retained for negative-case learning review."),
        _check("pass", "pending_cases_retained", "Pending cases remain visible until due/outcome windows mature."),
    ]
    thesis_summary = thesis_review.get("summary", {})
    checks.extend(
        [
            _must_be_false(thesis_summary, "can_write_learning_memory", "thesis_review_no_learning_write"),
            _must_be_false(thesis_summary, "can_change_analyst_weights", "thesis_review_no_weight_change"),
            _must_not_be_true(thesis_summary, "can_create_execution_recommendation", "thesis_review_no_execution_recommendation"),
            _must_be_false(thesis_summary, "can_create_recommendation", "thesis_review_legacy_no_execution_recommendation"),
            _must_be_false(thesis_summary, "can_trade", "thesis_review_no_trading"),
        ]
    )
    domain_case = next(
        (
            item
            for item in case_entries
            if item.get("case_type") == "domain_thesis"
        ),
        None,
    )
    if domain_case:
        reasoning = domain_case.get("verified_reasoning_baseline", {})
        if reasoning.get("reasoning_snapshot_run_id"):
            checks.extend(
                [
                    _check(
                        "pass"
                        if reasoning.get("hash_bound") is True
                        and reasoning.get("reasoning_snapshot_sha256")
                        else "fail",
                        "prospective_case_freezes_reasoning_snapshot",
                        (
                            f"run_id={reasoning.get('reasoning_snapshot_run_id')!r} "
                            f"sha256={reasoning.get('reasoning_snapshot_sha256')!r}"
                        ),
                    ),
                    _check(
                        "pass"
                        if int(
                            reasoning.get(
                                "directional_ticker_reasoning_event_count"
                            )
                            or 0
                        )
                        == 0
                        else "fail",
                        "prospective_case_no_directional_ticker_leakage",
                        str(
                            reasoning.get(
                                "directional_ticker_reasoning_event_count"
                            )
                        ),
                    ),
                ]
            )
    if template_standardization:
        template_summary = template_standardization.get("summary", {})
        linked_thesis_run_id = (
            template_standardization.get("inputs") or {}
        ).get("domain_thesis_review_run_id")
        checks.extend(
            [
                _must_be_false(template_summary, "can_mark_template_accepted_now", "template_no_auto_acceptance"),
                _must_be_false(template_summary, "can_write_learning_memory", "template_no_learning_write"),
                _must_not_be_true(template_summary, "can_create_execution_recommendation", "template_no_execution_recommendation"),
                _must_be_false(template_summary, "can_trade", "template_no_trading"),
                _check(
                    (
                        "pass"
                        if linked_thesis_run_id
                        == thesis_review.get("run_id")
                        else (
                            "warn"
                            if linked_thesis_run_id is None
                            else "fail"
                        )
                    ),
                    "template_bound_to_current_thesis_review",
                    (
                        f"template_thesis_run_id="
                        f"{linked_thesis_run_id!r}; "
                        f"current_thesis_run_id="
                        f"{thesis_review.get('run_id')!r}."
                    ),
                ),
            ]
        )
    if forecast_review:
        forecast_summary = forecast_review.get("summary", {})
        checks.extend(
            [
                _check(
                    "pass" if forecast_review.get("mode") == "domain_analyst_forecast_review_packet" else "fail",
                    "forecast_review_artifact_type",
                    str(forecast_review.get("mode")),
                ),
                _check(
                    "pass" if any(item.get("case_type") == "domain_thesis_expectation" for item in case_entries) else "fail",
                    "forecast_expectation_cases_loaded",
                    f"{sum(1 for item in case_entries if item.get('case_type') == 'domain_thesis_expectation')} expectation cases.",
                ),
                _check(
                    "pass" if _taxonomy_has_luck_vs_skill(forecast_review) else "fail",
                    "forecast_taxonomy_separates_luck_vs_skill",
                    "correct_but_lucky_or_wrong_reason must remain explicit.",
                ),
                _must_be_false(forecast_summary, "can_write_learning_memory", "forecast_no_learning_write"),
                _check(
                    "pass" if forecast_summary.get("can_create_analyst_research_recommendation") is True else "fail",
                    "forecast_allows_review_only_analyst_recommendations",
                    "Forecast packet should allow evidence-bound analyst recommendations.",
                ),
                _must_be_false(forecast_summary, "can_create_execution_recommendation", "forecast_no_execution_recommendation"),
                _must_be_false(forecast_summary, "can_create_recommendation", "forecast_legacy_no_execution_recommendation"),
                _must_be_false(forecast_summary, "can_trade", "forecast_no_trading"),
            ]
        )
    else:
        checks.append(_check("warn", "forecast_review_not_attached", "Registry falls back to a basic pending domain thesis case."))
    if outcome_evaluation:
        result = outcome_evaluation.get("outcome_evaluation", outcome_evaluation)
        counts = result.get("status_counts", {})
        checks.append(_check("pass" if result.get("evaluations") else "warn", "outcome_cases_loaded", str(counts)))
    else:
        checks.append(_check("warn", "outcome_evaluation_not_attached", "Registry starts with pending thesis/source cases only."))
    return checks


def _registry_status(checks: list[dict[str, str]], case_entries: list[dict[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "blocked_case_registry"
    buckets = Counter(item.get("outcome_bucket") for item in case_entries)
    if buckets.get("hit") or buckets.get("miss") or buckets.get("inconclusive"):
        return "case_registry_ready_with_outcome_buckets"
    if any(check["status"] == "warn" for check in checks):
        return "case_registry_ready_pending_outcomes"
    return "case_registry_ready"


def _decision_guidance(
    status: str,
    checks: list[dict[str, str]],
    case_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    failures = [check["code"] for check in checks if check["status"] == "fail"]
    buckets = Counter(item.get("outcome_bucket") for item in case_entries)
    if failures:
        action = "fix_case_registry_before_learning_review"
    elif buckets.get("hit") and not buckets.get("miss"):
        action = "collect_or_compare_negative_cases_before_learning_promotion"
    elif warnings:
        action = "keep_case_registry_pending_until_outcomes_arrive"
    else:
        action = "case_registry_can_support_balanced_learning_review"
    reasons = [
        f"Registry status is {status}.",
        "The registry preserves correct, incorrect, inconclusive, pending, and invalid cases.",
        "This packet can support future learning review but cannot write learning memory.",
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
        "outcome_bucket_counts": dict(sorted(buckets.items())),
        "reasons": reasons,
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No learning memory is written.",
        "No analyst weights or model parameters are changed.",
        "No correct-only training set is created.",
        "No miss, inconclusive, pending, or invalid case is dropped.",
        "Review-only analyst recommendations and learning proposals may be preserved.",
        "No execution recommendation, allocation, price target, paper order, broker call, or live trade is generated.",
        "No sector-to-ticker bridge or domain scaling is executed.",
    ]


def _commands(
    domain_thesis_review_json: str | Path,
    domain_template_standardization_json: str | Path | None,
    domain_forecast_review_json: str | Path | None,
) -> dict[str, str]:
    template_arg = (
        f"--domain-template-standardization-json {domain_template_standardization_json} "
        if domain_template_standardization_json
        else ""
    )
    forecast_arg = (
        f"--domain-forecast-review-json {domain_forecast_review_json} "
        if domain_forecast_review_json
        else ""
    )
    return {
        "rerun_case_registry": (
            "python run_agent_domain_analyst_case_registry_packet.py "
            f"--domain-thesis-review-json {domain_thesis_review_json} "
            f"{template_arg}"
            f"{forecast_arg}"
            "--output-dir reports\\dean_os\\domain_analyst_case_registry_packet_current"
        ),
        "future_with_outcomes": (
            "Attach --outcome-evaluation-json reports\\dean_os\\analyst_outcome_evaluation\\latest.json "
            "after outcome windows mature."
        ),
    }


def _operator_next_steps(
    status: str,
    checks: list[dict[str, str]],
    case_entries: list[dict[str, Any]],
) -> list[str]:
    if status == "blocked_case_registry":
        return ["Fix failed case-registry checks before using it for any learning review."]
    buckets = Counter(item.get("outcome_bucket") for item in case_entries)
    steps = ["Use this registry as the neutral casebook for domain thesis and source observations."]
    if not (buckets.get("hit") or buckets.get("miss") or buckets.get("inconclusive")):
        steps.append("Wait for or attach outcome evaluation before promoting lessons.")
    if buckets.get("hit") and not buckets.get("miss"):
        steps.append("Find comparable miss/neutral cases before trusting the successful pattern.")
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    if warnings:
        steps.append("Review warnings before learning promotion: " + ", ".join(warnings) + ".")
    steps.append("Future learning promotion should consume balanced case summaries, not raw hits only.")
    return steps


def _quality_notes_for_domain_thesis(thesis_review: dict[str, Any]) -> list[str]:
    summary = thesis_review.get("summary", {})
    notes = ["Domain thesis is a pre-outcome case; do not treat it as a proven pattern."]
    if summary.get("ticker_direct_count") == 0:
        notes.append("No ticker-direct evidence; keep outcome comparisons at domain/sector level unless a bridge later supplies direct ticker evidence.")
    if summary.get("required_evidence_missing"):
        notes.append("Required evidence lanes were missing at thesis time.")
    return notes


def _quality_notes_for_forecast_expectation(candidate: dict[str, Any]) -> list[str]:
    notes = [
        "Forecast expectation is frozen before outcome review; do not rewrite it after the result is known.",
        "Evaluate direction and stated causal reasoning separately.",
        "Correct direction alone is not enough for learning promotion.",
    ]
    if candidate.get("expected_direction") == "mixed":
        notes.append("Mixed direction requires explicit outcome criteria before scoring.")
    if candidate.get("evidence_balance", {}).get("ticker_direct_count") == 0:
        notes.append("No ticker-direct evidence; keep scoring at domain/sector level unless a later bridge supplies direct ticker evidence.")
    return notes


def _quality_notes_for_outcome(item: dict[str, Any]) -> list[str]:
    notes = []
    if item.get("outcome_label") == "hit":
        notes.append("Hit requires causal review before becoming a positive lesson.")
    if item.get("outcome_label") == "miss":
        notes.append("Miss should be retained as a negative/comparison case.")
    if item.get("status") in {"missing_tickers", "missing_price_window", "no_price_after_created_at"}:
        notes.append("Outcome quality is unresolved because market data or ticker mapping was insufficient.")
    return notes or ["Outcome case requires review before learning promotion."]


def _context_tags_from_thesis(thesis: dict[str, Any]) -> list[str]:
    tags = []
    for key in ("key_drivers", "assumptions"):
        for value in thesis.get(key, []) or []:
            tag = _slug(value)
            if tag:
                tags.append(tag)
    return sorted(set(tags))


def _context_tags_from_forecast(candidate: dict[str, Any]) -> list[str]:
    tags = []
    for key in ("key_drivers", "assumptions", "required_outcome_observations"):
        for value in candidate.get(key, []) or []:
            tag = _slug(value)
            if tag:
                tags.append(tag)
    return sorted(set(tags))


def _taxonomy_has_luck_vs_skill(forecast_review: dict[str, Any]) -> bool:
    buckets = {item.get("bucket_id") for item in forecast_review.get("outcome_taxonomy", [])}
    return {"correct_for_stated_reasons", "correct_but_lucky_or_wrong_reason"}.issubset(buckets)


def _seasonality_context(value: str | None) -> dict[str, Any]:
    dt = _parse_datetime(value)
    if dt is None:
        return {"season_tag": "unknown", "month": None, "quarter": None}
    month = dt.month
    if month in {12, 1, 2}:
        season = "winter"
    elif month in {3, 4, 5}:
        season = "spring"
    elif month in {6, 7, 8}:
        season = "summer"
    else:
        season = "autumn"
    tags = [season, f"q{((month - 1) // 3) + 1}", dt.strftime("%B").lower()]
    if month in {7, 8}:
        tags.append("summer_vacation_liquidity_context")
    if month in {1, 4, 7, 10}:
        tags.append("earnings_season_context")
    return {
        "season_tag": season,
        "month": month,
        "quarter": ((month - 1) // 3) + 1,
        "tags": tags,
    }


def _evaluation_schedule(
    as_of: str | None,
    horizons_days: list[int],
) -> list[dict[str, Any]]:
    anchor = _parse_datetime(as_of)
    return [
        {
            "horizon_days": int(horizon),
            "due_at": (
                (anchor + timedelta(days=int(horizon))).isoformat()
                if anchor
                else None
            ),
            "status": "pending_not_due",
        }
        for horizon in horizons_days
    ]


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    chars = [ch if ch.isalnum() else "_" for ch in text]
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug[:80]


def _must_be_false(summary: dict[str, Any], field: str, code: str) -> dict[str, str]:
    if summary.get(field) is False:
        return _check("pass", code, f"{field}=False.")
    return _check("fail", code, f"{field} must stay False, got {summary.get(field)!r}.")


def _must_not_be_true(summary: dict[str, Any], field: str, code: str) -> dict[str, str]:
    if summary.get(field) is not True:
        return _check("pass", code, f"{field} is not True.")
    return _check("fail", code, f"{field} must not be True.")


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
