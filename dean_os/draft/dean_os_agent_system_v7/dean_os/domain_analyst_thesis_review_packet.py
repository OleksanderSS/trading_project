from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_DOMAIN_INTAKE_JSON = (
    "reports/dean_os/semiconductor_analyst_runtime_current/latest.json"
)
DEFAULT_DOMAIN_INSTANCE_CONTRACT_JSON = "reports/dean_os/domain_analyst_instance_contract_current/latest.json"
DEFAULT_REGIME_SCENARIO_JSON = None
DEFAULT_ARCHITECTURE_MAP_JSON = "reports/dean_os/current_architecture_map_current/latest.json"
DEFAULT_REASONING_SNAPSHOT_JSON = (
    "reports/dean_os/analyst_core_reasoning_snapshot_current/latest.json"
)


class DomainAnalystThesisReviewPacket:
    """Review-only packet for the domain/sector thesis before ticker mapping.

    This is the missing layer between DomainAnalystIntakePacket and any
    sector-to-ticker bridge. It reviews the analyst's domain thesis as a domain
    thesis only. It never creates ticker recommendations, learning promotions,
    config writes, paper decisions, or trades.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_thesis_review_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_intake_json: str | Path = DEFAULT_DOMAIN_INTAKE_JSON,
        domain_instance_contract_json: str | Path | None = DEFAULT_DOMAIN_INSTANCE_CONTRACT_JSON,
        regime_scenario_json: str | Path | None = DEFAULT_REGIME_SCENARIO_JSON,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        reasoning_snapshot_json: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        review_source = _load_json(domain_intake_json)
        linked_artifact_verification, linked_artifacts = (
            _load_linked_runtime_artifacts(review_source)
        )
        domain_intake = _normalize_review_source(review_source)
        domain_instance = _load_optional_json(domain_instance_contract_json)
        regime_scenario = _load_optional_json(regime_scenario_json)
        architecture_map = _load_optional_json(architecture_map_json)
        reasoning_snapshot = _load_optional_json(reasoning_snapshot_json)
        reasoning_context = _reasoning_snapshot_context(
            reasoning_snapshot=reasoning_snapshot,
            reasoning_snapshot_json=reasoning_snapshot_json,
            domain_intake_json=domain_intake_json,
            domain_intake=domain_intake,
        )
        checks = _review_checks(
            domain_intake=domain_intake,
            domain_instance=domain_instance,
            regime_scenario=regime_scenario,
            architecture_map=architecture_map,
            linked_artifact_verification=linked_artifact_verification,
        )
        if reasoning_snapshot is not None:
            checks.extend(_reasoning_snapshot_checks(reasoning_context))
        status = _packet_status(domain_intake, domain_instance, checks)
        summary = _summary(status, domain_intake, domain_instance, regime_scenario)
        summary.update(
            {
                "reasoning_snapshot_status": reasoning_context.get("status"),
                "reasoning_snapshot_hash_bound": reasoning_context.get("hash_bound"),
                "classified_event_count": reasoning_context.get(
                    "classified_event_count"
                ),
                "transmission_channel_count": reasoning_context.get(
                    "transmission_channel_count"
                ),
                "reasoning_hypothesis_count": reasoning_context.get(
                    "hypothesis_count"
                ),
                "reasoning_evidence_gap_count": reasoning_context.get(
                    "evidence_gap_count"
                ),
                "directional_ticker_reasoning_event_count": reasoning_context.get(
                    "directional_ticker_reasoning_event_count"
                ),
            }
        )
        payload = {
            "run_id": _run_id("domain_analyst_thesis_review_packet"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_thesis_review_packet",
            "inputs": {
                "domain_intake_json": str(domain_intake_json),
                "domain_intake_run_id": domain_intake.get("run_id"),
                "review_source_mode": review_source.get("mode"),
                "domain_instance_contract_json": str(domain_instance_contract_json) if domain_instance_contract_json else None,
                "domain_instance_contract_run_id": domain_instance.get("run_id") if domain_instance else None,
                "regime_scenario_json": str(regime_scenario_json) if regime_scenario_json else None,
                "regime_scenario_run_id": regime_scenario.get("run_id") if regime_scenario else None,
                "architecture_map_json": str(architecture_map_json) if architecture_map_json else None,
                "reasoning_snapshot_json": str(reasoning_snapshot_json)
                if reasoning_snapshot_json
                else None,
                "reasoning_snapshot_run_id": reasoning_snapshot.get("run_id")
                if reasoning_snapshot
                else None,
            },
            "summary": summary,
            "domain_thesis_contract": _domain_thesis_contract(),
            "thesis_snapshot": _thesis_snapshot(domain_intake),
            "analytical_review": _analytical_review(
                domain_intake=domain_intake,
                review_source=review_source,
                linked_artifacts=linked_artifacts,
            ),
            "linked_artifact_verification": linked_artifact_verification,
            "regime_scenario_context": _regime_scenario_context(regime_scenario),
            "reasoning_snapshot_context": reasoning_context,
            "evidence_lane_coverage": _evidence_lane_coverage(domain_intake),
            "supporting_evidence_examples": _evidence_examples(domain_intake, "supporting"),
            "contradicting_evidence_examples": _evidence_examples(domain_intake, "contradicting"),
            "risk_and_blind_spot_review": _risk_and_blind_spot_review(domain_intake),
            "ticker_bridge_boundary": _ticker_bridge_boundary(domain_intake),
            "review_checks": checks,
            "decision_guidance": _decision_guidance(status, checks),
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(domain_intake_json, reasoning_snapshot_json),
            "operator_next_steps": _operator_next_steps(status, checks),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_thesis_review_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_domain_analyst_thesis_review_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    thesis = payload.get("thesis_snapshot", {})
    analytical = payload.get("analytical_review", {})
    coverage = payload.get("evidence_lane_coverage", {})
    boundary = payload.get("ticker_bridge_boundary", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Domain Analyst Thesis Review Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Sectors: {', '.join(summary.get('sectors', [])) or 'none'}",
        f"- Thesis stance: `{summary.get('thesis_stance')}`",
        f"- Expected direction: `{summary.get('expected_direction')}`",
        f"- Confidence: {summary.get('confidence')}",
        f"- Analyst recommendation: `{summary.get('analyst_recommendation')}`",
        f"- Regime scenario: `{summary.get('regime_scenario_status')}` active_fields={summary.get('active_regime_field_count')} gaps={summary.get('scenario_evidence_gap_count')}",
        f"- Verified reasoning: `{summary.get('reasoning_snapshot_status')}` hash_bound={summary.get('reasoning_snapshot_hash_bound')} classified={summary.get('classified_event_count')} channels={summary.get('transmission_channel_count')}",
        f"- Can create analyst research recommendation: {summary.get('can_create_analyst_research_recommendation')}",
        f"- Can create execution recommendation: {summary.get('can_create_execution_recommendation')}",
        f"- Evidence items: {summary.get('evidence_item_count')}",
        f"- Required evidence missing: `{', '.join(summary.get('required_evidence_missing') or []) or 'none'}`",
        f"- Manual review required: {summary.get('manual_review_required')}",
        f"- Can standardize after manual review: {summary.get('can_standardize_domain_template_after_manual_review')}",
        f"- Can create direct ticker thesis without bridge: {summary.get('can_create_direct_ticker_thesis_without_bridge')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Thesis Snapshot",
        "",
        str(thesis.get("thesis") or "No thesis text found."),
        "",
        "## Analytical Assessment",
        "",
        f"- Assessment: `{analytical.get('assessment_status')}`",
        f"- Scope decision: `{analytical.get('scope_decision')}`",
        f"- Ticker decision: `{analytical.get('ticker_decision')}`",
        f"- Prospective case status: `{analytical.get('prospective_case_status')}`",
        f"- Confidence interpretation: {analytical.get('confidence_interpretation')}",
        "",
        str(analytical.get("executive_conclusion") or "No analytical conclusion available."),
        "",
        "### Evidence Balance",
        "",
    ]
    evidence_balance = analytical.get("evidence_balance", {})
    lines.extend(
        [
            f"- Required lanes: {evidence_balance.get('satisfied_required_lane_count')}/{evidence_balance.get('required_lane_count')}",
            f"- Raw context items: {evidence_balance.get('raw_context_item_count')}",
            f"- Required-lane eligible items: {evidence_balance.get('required_lane_eligible_item_count')}",
            f"- Supporting items selected by analyst: {evidence_balance.get('supporting_evidence_count')}",
            f"- Contradicting items selected by analyst: {evidence_balance.get('contradicting_evidence_count')}",
            "",
            "### Market Snapshot",
            "",
        ]
    )
    market = analytical.get("market_snapshot", {})
    lines.extend(
        [
            f"- Window: {market.get('common_session_count')} common sessions; return lookback={market.get('lookback_sessions')} sessions.",
            f"- Sector median return: {_display_percent(market.get('sector_median_return_percent'))}",
            f"- Positive breadth: {_display_ratio_percent(market.get('positive_breadth_ratio'))}",
            f"- Median excess return vs QQQ: {_display_percentage_points(market.get('median_excess_return_vs_qqq_percentage_points'))}",
            f"- Cross-ticker dispersion: {_display_percent(market.get('return_dispersion_percent'))}",
        ]
    )
    ticker_returns = market.get("ticker_returns_percent", {})
    if ticker_returns:
        rendered_returns = ", ".join(
            f"{ticker}={_display_percent(value)}"
            for ticker, value in sorted(ticker_returns.items())
        )
        lines.append(f"- Ticker returns: {rendered_returns}")
    lines.extend(["", "### News and Policy Coverage", ""])
    for lane in analytical.get("news_policy_coverage", {}).get("news_lanes", []):
        lines.append(
            f"- `{lane.get('evidence_type')}`: `{lane.get('status')}`, "
            f"strong sources={lane.get('independent_strong_source_count')} "
            f"({', '.join(lane.get('independent_strong_sources', [])) or 'none'})"
        )
    policy = analytical.get("news_policy_coverage", {}).get("official_policy", {})
    lines.append(
        f"- `policy_or_geopolitical`: official corroboration ready={policy.get('policy_lane_eligible')}, "
        f"sources={', '.join(policy.get('combined_independent_sources', [])) or 'none'}"
    )
    lines.extend(["", "### Comparable Fundamental Ratios", ""])
    ratio_review = analytical.get("fundamental_ratio_review", {})
    for item in ratio_review.get("quarterly_comparable_ratios", []):
        lines.append(
            f"- {item.get('ticker')} `{item.get('ratio_name')}`: "
            f"{_display_percent(item.get('value_percent'))} "
            f"({item.get('comparison_period_class')})"
        )
    if ratio_review.get("separate_annual_ratios"):
        lines.extend(
            [
                "",
                "TSM annual observations are kept separate from US quarterly comparisons:",
                "",
            ]
        )
        for item in ratio_review.get("separate_annual_ratios", []):
            lines.append(
                f"- {item.get('ticker')} `{item.get('ratio_name')}`: "
                f"{_display_percent(item.get('value_percent'))} "
                f"({item.get('comparison_period_class')}, {item.get('source_currency')})"
            )
    lines.extend(["", "### Scenario Framework (not forecast probabilities)", ""])
    for scenario in analytical.get("scenario_framework", []):
        lines.append(f"- **{scenario.get('name')}**: {scenario.get('description')}")
        lines.append(
            "  Watch: "
            + "; ".join(scenario.get("watch_conditions", []))
        )
    lines.extend(["", "### Quality Cautions", ""])
    lines.extend(
        f"- {item}" for item in analytical.get("quality_cautions", [])
    )
    lines.extend(
        [
            "",
        "## Regime Scenario Context",
        "",
        ]
    )
    regime = payload.get("regime_scenario_context", {})
    lines.append(f"- Available: {regime.get('available')}")
    lines.append(f"- Packet status: `{regime.get('packet_status')}`")
    lines.append(f"- Probability mass valid: {regime.get('probability_mass_valid')}")
    lines.append(f"- Scenario probabilities: `{regime.get('scenario_probabilities')}`")
    for gap in regime.get("top_evidence_gaps", []):
        lines.append(f"- Gap `{gap.get('gap_id')}` priority=`{gap.get('priority')}`: {gap.get('description')}")
    lines.extend(
        [
            "",
        "## Verified Reasoning Snapshot",
        "",
        ]
    )
    reasoning = payload.get("reasoning_snapshot_context", {})
    lines.extend(
        [
            f"- Available: {reasoning.get('available')}",
            f"- Status: `{reasoning.get('status')}`",
            f"- Runtime hash bound: {reasoning.get('hash_bound')}",
            f"- Classified events: {reasoning.get('classified_event_count')}",
            f"- Transmission channels: {reasoning.get('transmission_channel_count')}",
            f"- Candidate hypotheses: {reasoning.get('hypothesis_count')}",
            f"- Evidence gaps: {reasoning.get('evidence_gap_count')}",
            f"- Directional ticker reasoning events: {reasoning.get('directional_ticker_reasoning_event_count')}",
            f"- Scenario graph: `{reasoning.get('scenario_graph_status')}`",
            f"- Expectation gap: `{reasoning.get('expectation_gap_status')}`",
            "",
            "## Evidence Lane Coverage",
            "",
        ]
    )
    for lane in coverage.get("required_lanes", []):
        lines.append(f"- `{lane.get('evidence_type')}`: count={lane.get('count')} status=`{lane.get('status')}`")
    lines.extend(
        [
            "",
            "## Ticker Bridge Boundary",
            "",
            f"- Boundary: {boundary.get('boundary')}",
            f"- Ticker-direct evidence: {boundary.get('ticker_direct_count')}",
            f"- Basket status: `{boundary.get('basket_status')}`",
            f"- Next separate step allowed after manual review: {boundary.get('can_prepare_separate_ticker_bridge_after_manual_review')}",
            "",
            "## Supporting Evidence Examples",
            "",
        ]
    )
    for item in payload.get("supporting_evidence_examples", []):
        lines.append(f"- `{item.get('evidence_id')}` {item.get('evidence_type')} {item.get('directness')}: {item.get('summary')}")
    if not payload.get("supporting_evidence_examples"):
        lines.append("- none")
    lines.extend(["", "## Contradicting Evidence Examples", ""])
    for item in payload.get("contradicting_evidence_examples", []):
        lines.append(f"- `{item.get('evidence_id')}` {item.get('evidence_type')} {item.get('directness')}: {item.get('summary')}")
    if not payload.get("contradicting_evidence_examples"):
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


def _normalize_review_source(source: dict[str, Any]) -> dict[str, Any]:
    """Map the live semiconductor runtime onto the existing review template."""
    if source.get("mode") != "semiconductor_analyst_runtime":
        return source

    report = source.get("analyst_report") or {}
    evidence_items = report.get("evidence") or []
    runtime_summary = source.get("summary") or {}
    lane_coverage = source.get("evidence_lane_coverage") or {}
    required_lanes = lane_coverage.get("required_lanes") or []
    required_types = [
        str(item.get("evidence_type"))
        for item in required_lanes
        if item.get("evidence_type")
    ]
    eligible_counts = {
        str(key): int(value or 0)
        for key, value in (
            lane_coverage.get("eligible_evidence_type_counts") or {}
        ).items()
    }
    all_counts = {
        str(key): int(value or 0)
        for key, value in (
            lane_coverage.get("all_evidence_type_counts") or {}
        ).items()
    }
    directness_counts = Counter(
        str(item.get("directness") or "unknown")
        for item in evidence_items
        if isinstance(item, dict)
    )
    ticker_direct_ids = {
        str(item.get("evidence_id"))
        for item in evidence_items
        if isinstance(item, dict)
        and item.get("directness") == "ticker"
        and (item.get("provenance") or {}).get(
            "ticker_thesis_eligible"
        )
        is True
        and item.get("evidence_id")
    }
    missing = list(
        runtime_summary.get("missing_required_evidence") or []
    )
    integration = source.get("integration_boundary") or {}
    safety = source.get("safety") or {}
    sector_ready = (
        runtime_summary.get("sector_thesis_ready") is True
        and not missing
    )
    safe_boundary = (
        safety.get("review_only") is True
        and integration.get("training_allowed") is False
        and integration.get("tuning_allowed") is False
        and integration.get("automatic_trading_allowed") is False
        and runtime_summary.get("can_trade") is False
    )
    normalized_summary = dict(runtime_summary)
    normalized_summary.update(
        {
            "intake_status": source.get("status"),
            "domain_id": source.get("domain_id")
            or report.get("domain_id"),
            "document_count": None,
            "evidence_item_count": len(evidence_items),
            "ticker_direct_count": len(ticker_direct_ids),
            "sector_or_domain_count": sum(
                count
                for directness, count in directness_counts.items()
                if directness in {"sector", "domain"}
            ),
            "macro_policy_context_count": (
                all_counts.get("macro_context", 0)
                + all_counts.get("policy_or_geopolitical", 0)
            ),
            "required_evidence_missing": missing,
            "analyst_report_created": bool(report),
            "can_run_domain_analyst": sector_ready,
            "can_create_direct_ticker_thesis_without_bridge": False,
            "can_write_learning_memory": False,
            "can_create_recommendation": False,
            "can_trade": False,
        }
    )
    return {
        **source,
        "inputs": {
            **(source.get("inputs") or {}),
            "domain_id": source.get("domain_id")
            or report.get("domain_id"),
            "sectors": ["semiconductor"],
        },
        "summary": normalized_summary,
        "source_gate_context": {
            "available": True,
            "gate_status": (
                "source_evidence_ready_for_domain_research"
                if sector_ready
                else "source_evidence_incomplete"
            ),
            "can_enter_domain_research": sector_ready,
            "safe_downstream_boundary": safe_boundary,
            "warnings": [],
        },
        "domain_profile_snapshot": {
            "domain_id": source.get("domain_id")
            or report.get("domain_id"),
            "required_evidence_types": required_types,
            "useful_evidence_types": sorted(
                set(all_counts).difference(required_types)
            ),
            "ticker_universe_hint": [
                item.get("ticker")
                for item in (
                    (report.get("ticker_basket") or {}).get(
                        "candidates"
                    )
                    or []
                )
                if item.get("ticker")
            ],
        },
        "evidence_type_summary": eligible_counts,
        "all_evidence_type_summary": all_counts,
        "directness_summary": dict(sorted(directness_counts.items())),
        "evidence_items": evidence_items,
        "analyst_report": report,
    }


def _load_linked_runtime_artifacts(
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if source.get("mode") != "semiconductor_analyst_runtime":
        return (
            {
                "status": "not_applicable",
                "artifact_count": 0,
                "verified_artifact_count": 0,
                "all_hashes_match": None,
                "artifacts": [],
            },
            {},
        )

    verification_rows: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any]] = {}
    for artifact_name, descriptor in (
        source.get("source_artifacts") or {}
    ).items():
        descriptor = descriptor if isinstance(descriptor, dict) else {}
        raw_path = descriptor.get("path")
        path = Path(str(raw_path)) if raw_path else None
        exists = bool(path and path.is_file())
        actual_sha256 = _file_sha256(path) if exists and path else None
        expected_sha256 = descriptor.get("sha256")
        hash_matches = bool(
            expected_sha256
            and actual_sha256
            and actual_sha256 == expected_sha256
        )
        row = {
            "artifact_name": artifact_name,
            "path": str(raw_path) if raw_path else None,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "exists": exists,
            "hash_matches": hash_matches,
            "excluded_from_sector_evidence": (
                artifact_name == "excluded_pipeline_case"
            ),
        }
        verification_rows.append(row)
        if hash_matches and path:
            try:
                payload = _load_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                row["content_load_status"] = "invalid_json"
            else:
                row["content_load_status"] = "loaded"
                loaded[str(artifact_name)] = payload
        else:
            row["content_load_status"] = "not_loaded"

    all_hashes_match = bool(verification_rows) and all(
        row["hash_matches"] for row in verification_rows
    )
    return (
        {
            "status": (
                "verified"
                if all_hashes_match
                else "missing_or_changed_linked_artifact"
            ),
            "artifact_count": len(verification_rows),
            "verified_artifact_count": sum(
                1 for row in verification_rows if row["hash_matches"]
            ),
            "all_hashes_match": all_hashes_match,
            "artifacts": verification_rows,
        },
        loaded,
    )


def _analytical_review(
    *,
    domain_intake: dict[str, Any],
    review_source: dict[str, Any],
    linked_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = domain_intake.get("summary") or {}
    report = domain_intake.get("analyst_report") or {}
    thesis = report.get("thesis") or {}
    lane_coverage = review_source.get("evidence_lane_coverage") or {}
    required_lanes = lane_coverage.get("required_lanes") or []
    eligible_count = sum(
        int(item.get("eligible_evidence_count") or 0)
        for item in required_lanes
    )

    market_artifact = linked_artifacts.get("sector_market") or {}
    market_summary = market_artifact.get("summary") or {}
    market_metrics = {
        item.get("name"): item.get("value")
        for item in market_artifact.get("metrics") or []
        if isinstance(item, dict) and item.get("name")
    }
    ticker_returns = {
        ticker: market_metrics.get(
            f"{ticker.lower()}_return_20_session"
        )
        for ticker in ("AMD", "INTC", "NVDA", "TSM")
        if market_metrics.get(
            f"{ticker.lower()}_return_20_session"
        )
        is not None
    }

    ratios_artifact = linked_artifacts.get("derived_ratios") or {}
    ratio_summary = ratios_artifact.get("summary") or {}
    selected_ratio_names = {
        "operating_margin",
        "net_margin",
        "capex_to_revenue",
        "cash_to_assets",
        "equity_to_assets",
    }
    selected_ratios = []
    for item in ratios_artifact.get("ratios") or []:
        if (
            not isinstance(item, dict)
            or item.get("ratio_name") not in selected_ratio_names
        ):
            continue
        value = item.get("value")
        selected_ratios.append(
            {
                "ticker": item.get("ticker"),
                "ratio_name": item.get("ratio_name"),
                "value": value,
                "value_percent": (
                    round(float(value) * 100.0, 2)
                    if value is not None
                    else None
                ),
                "comparison_period_class": item.get(
                    "comparison_period_class"
                ),
                "source_currency": item.get("source_currency"),
                "available_at": item.get("available_at"),
            }
        )
    ratio_order = {
        "operating_margin": 0,
        "net_margin": 1,
        "capex_to_revenue": 2,
        "cash_to_assets": 3,
        "equity_to_assets": 4,
    }
    selected_ratios.sort(
        key=lambda item: (
            ratio_order.get(str(item.get("ratio_name")), 99),
            str(item.get("ticker")),
        )
    )
    quarterly_ratios = [
        item
        for item in selected_ratios
        if item.get("comparison_period_class") == "quarterly_Q1"
    ]
    annual_ratios = [
        item
        for item in selected_ratios
        if item.get("comparison_period_class") == "annual"
    ]

    news_artifact = (
        linked_artifacts.get("semiconductor_news") or {}
    )
    news_lanes = []
    for item in news_artifact.get("lane_review") or []:
        if not isinstance(item, dict):
            continue
        news_lanes.append(
            {
                "evidence_type": item.get("evidence_type"),
                "status": item.get("status"),
                "candidate_count": item.get("candidate_count"),
                "strong_candidate_count": item.get(
                    "strong_candidate_count"
                ),
                "independent_strong_source_count": item.get(
                    "independent_strong_source_count"
                ),
                "independent_strong_sources": item.get(
                    "independent_strong_sources", []
                ),
            }
        )
    policy_artifact = linked_artifacts.get("official_policy") or {}
    policy_corroboration = policy_artifact.get("corroboration") or {}

    raw_context_count = int(
        summary.get("evidence_item_count")
        or len(report.get("evidence") or [])
    )
    required_lane_count = int(
        summary.get("required_lane_count")
        or len(required_lanes)
    )
    satisfied_lane_count = int(
        summary.get("satisfied_required_lane_count")
        or sum(
            1
            for item in required_lanes
            if item.get("status") in {"satisfied", "covered"}
        )
    )
    confidence = thesis.get("confidence")
    market_window = market_summary.get("common_session_count")
    cautions = [
        (
            f"The thesis confidence ({confidence}) is an evidence-quality "
            "aggregation, not a calibrated probability of return."
        ),
        (
            f"The runtime contains {raw_context_count} context items, but "
            f"only {eligible_count} explicitly eligible items close required "
            "lanes; raw item count is not independent-source count."
        ),
        (
            "News rows are source-tiered and exact duplicates are screened, "
            "but semantically overlapping headlines can still overweight one "
            "underlying event."
        ),
        (
            f"Market confirmation uses only {market_window} common sessions; "
            "it is a short confirmation window, not a long-horizon forecast."
        ),
        (
            "Fundamental source coverage is four of four tickers, but the "
            "artifact explicitly blocks a full-cohort comparability claim."
        ),
        (
            "TSM annual TWD observations remain separate from US quarterly "
            "USD comparison lanes; no currency translation is performed."
        ),
        (
            "No company currently has eligible directional ticker evidence; "
            "AMD, INTC, NVDA, and TSM remain basket candidates only."
        ),
        (
            "The AMD single-ticker model case is hash-bound but explicitly "
            "excluded from sector evidence and cannot close a sector lane."
        ),
    ]
    market_median = market_metrics.get(
        "sector_median_return_20_session"
    )
    breadth = market_metrics.get("sector_positive_breadth")
    dispersion = market_metrics.get(
        "sector_return_dispersion_20_session"
    )
    nvda_return = ticker_returns.get("NVDA")
    conclusion = (
        f"All {satisfied_lane_count}/{required_lane_count} required sector "
        "lanes are source-backed. Demand, capex, supply-chain, official-policy, "
        "and market evidence support a reviewable semiconductor-sector case. "
        f"The market snapshot is constructive in aggregate "
        f"(median {market_median:.2f}% and breadth {breadth:.0%})"
        if market_median is not None and breadth is not None
        else (
            f"All {satisfied_lane_count}/{required_lane_count} required "
            "sector lanes are represented."
        )
    )
    if (
        dispersion is not None
        and nvda_return is not None
    ):
        conclusion += (
            f", but dispersion is {dispersion:.2f}% and NVDA is "
            f"{nvda_return:.2f}% over the same window. "
        )
    else:
        conclusion += " "
    conclusion += (
        "The defensible conclusion is therefore mixed at sector level: "
        "eligible as a prospective review case, not as a direct ticker "
        "forecast or trading signal."
    )
    return {
        "assessment_status": (
            "sector_thesis_reviewable_with_cautions"
            if satisfied_lane_count == required_lane_count
            and required_lane_count > 0
            else "sector_thesis_needs_more_evidence"
        ),
        "scope_decision": "sector_thesis_only",
        "ticker_decision": "no_direct_ticker_thesis",
        "prospective_case_status": "candidate_pending_manual_review",
        "template_decision": "not_accepted_by_this_packet",
        "confidence_interpretation": (
            "Evidence-quality heuristic only; not a calibrated return "
            "probability."
        ),
        "executive_conclusion": conclusion,
        "evidence_balance": {
            "required_lane_count": required_lane_count,
            "satisfied_required_lane_count": satisfied_lane_count,
            "raw_context_item_count": raw_context_count,
            "required_lane_eligible_item_count": eligible_count,
            "supporting_evidence_count": len(
                thesis.get("supporting_evidence_ids") or []
            ),
            "contradicting_evidence_count": len(
                thesis.get("contradicting_evidence_ids") or []
            ),
        },
        "market_snapshot": {
            "common_session_count": market_window,
            "lookback_sessions": market_summary.get(
                "lookback_sessions"
            ),
            "sector_median_return_percent": market_median,
            "positive_breadth_ratio": breadth,
            "median_excess_return_vs_qqq_percentage_points": (
                market_metrics.get(
                    "sector_median_excess_return_vs_qqq"
                )
            ),
            "return_dispersion_percent": dispersion,
            "ticker_returns_percent": ticker_returns,
            "qqq_return_percent": market_metrics.get(
                "qqq_return_20_session"
            ),
        },
        "news_policy_coverage": {
            "news_lanes": news_lanes,
            "official_policy": {
                "policy_lane_eligible": policy_corroboration.get(
                    "policy_lane_eligible"
                ),
                "combined_independent_source_count": (
                    policy_corroboration.get(
                        "combined_independent_source_count"
                    )
                ),
                "combined_independent_sources": (
                    policy_corroboration.get(
                        "combined_independent_sources", []
                    )
                ),
            },
        },
        "fundamental_ratio_review": {
            "source_fact_count": (
                (
                    linked_artifacts.get("fundamental") or {}
                ).get("summary")
                or {}
            ).get("source_fact_count"),
            "derived_ratio_count": ratio_summary.get(
                "derived_ratio_count"
            ),
            "multi_ticker_comparison_lane_count": ratio_summary.get(
                "multi_ticker_comparison_lane_count"
            ),
            "full_cohort_comparison_lane_count": ratio_summary.get(
                "full_cohort_comparison_lane_count"
            ),
            "can_claim_full_cohort_comparability": ratio_summary.get(
                "can_claim_full_cohort_comparability"
            ),
            "quarterly_comparable_ratios": quarterly_ratios,
            "separate_annual_ratios": annual_ratios,
        },
        "scenario_framework": [
            {
                "name": "Base / mixed continuation",
                "description": (
                    "AI demand and capex remain supportive, while policy "
                    "constraints, supply bottlenecks, and company dispersion "
                    "prevent a uniform sector or ticker call."
                ),
                "watch_conditions": [
                    "sector breadth and excess return versus QQQ",
                    "next hyperscaler capex guidance",
                    "next filing-cycle margin changes",
                    "official export-control changes",
                ],
            },
            {
                "name": "Upside broadening",
                "description": (
                    "Demand and capex evidence persist, supply constraints "
                    "support pricing, and market strength broadens beyond the "
                    "current winners without policy tightening."
                ),
                "watch_conditions": [
                    "broader positive ticker participation",
                    "stable or improving comparable margins",
                    "continued multi-source demand corroboration",
                ],
            },
            {
                "name": "Downside constraint",
                "description": (
                    "Capex or demand slows, restrictions intensify, or margin "
                    "pressure spreads while market breadth deteriorates."
                ),
                "watch_conditions": [
                    "capex guidance cuts",
                    "new official license restrictions",
                    "negative breadth and excess return",
                    "margin deterioration across comparable filings",
                ],
            },
        ],
        "quality_cautions": cautions,
    }


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def _display_ratio_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def _display_percentage_points(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f} pp"


def _summary(
    status: str,
    domain_intake: dict[str, Any],
    domain_instance: dict[str, Any] | None,
    regime_scenario: dict[str, Any] | None,
) -> dict[str, Any]:
    intake_summary = domain_intake.get("summary", {})
    report = domain_intake.get("analyst_report") or {}
    thesis = report.get("thesis") or {}
    instance_summary = domain_instance.get("summary", {}) if domain_instance else {}
    regime_summary = regime_scenario.get("summary", {}) if regime_scenario else {}
    required_missing = intake_summary.get("required_evidence_missing") or []
    return {
        "packet_status": status,
        "domain_id": intake_summary.get("domain_id") or report.get("domain_id"),
        "sectors": domain_intake.get("inputs", {}).get("sectors", []),
        "domain_instance_status": instance_summary.get("instance_status"),
        "domain_intake_status": intake_summary.get("intake_status"),
        "regime_scenario_status": regime_summary.get("packet_status"),
        "active_regime_field_count": _active_regime_field_count(regime_scenario),
        "scenario_node_count": regime_summary.get("scenario_node_count"),
        "scenario_evidence_gap_count": regime_summary.get("evidence_gap_count"),
        "scenario_probability_mass_valid": regime_summary.get("probability_mass_valid"),
        "thesis_id": thesis.get("thesis_id"),
        "thesis_stance": thesis.get("stance"),
        "expected_direction": thesis.get("expected_direction"),
        "confidence": thesis.get("confidence"),
        "data_quality": thesis.get("data_quality"),
        "analyst_recommendation": report.get("recommendation"),
        "evidence_item_count": intake_summary.get("evidence_item_count"),
        "ticker_direct_count": intake_summary.get("ticker_direct_count"),
        "sector_or_domain_count": intake_summary.get("sector_or_domain_count"),
        "macro_policy_context_count": intake_summary.get("macro_policy_context_count"),
        "required_evidence_missing": required_missing,
        "supporting_evidence_count": len(thesis.get("supporting_evidence_ids") or []),
        "contradicting_evidence_count": len(thesis.get("contradicting_evidence_ids") or []),
        "manual_review_required": True,
        "can_enter_manual_thesis_review": status != "blocked_domain_thesis_review",
        "can_standardize_domain_template_after_manual_review": status in {
            "domain_thesis_review_ready",
            "domain_thesis_review_ready_with_cautions",
        },
        "can_prepare_separate_ticker_bridge_after_manual_review": status in {
            "domain_thesis_review_ready",
            "domain_thesis_review_ready_with_cautions",
        },
        "can_create_direct_ticker_thesis_without_bridge": False,
        "can_create_analyst_research_recommendation": True,
        "can_use_regime_scenario_context_for_review": regime_scenario is not None,
        "can_create_analyst_self_improvement_proposal": True,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _domain_thesis_contract() -> dict[str, Any]:
    return {
        "packet_type": "domain_analyst_thesis_review_packet",
        "thesis_level": "domain_or_sector_thesis",
        "required_inputs": [
            "DomainAnalystIntakePacket or SemiconductorAnalystRuntime",
            "DomainAnalystInstanceContract optional but preferred",
            "DomainAnalystRegimeScenarioPacket optional but preferred for context-aware thesis review",
        ],
        "required_outputs": [
            "thesis_snapshot",
            "regime_scenario_context",
            "evidence_lane_coverage",
            "supporting_evidence_examples",
            "contradicting_evidence_examples",
            "ticker_bridge_boundary",
            "operator_next_steps",
        ],
        "boundary": "This packet reviews the domain thesis only; a runtime source is normalized into the same review contract. Ticker mapping, learning promotion, execution recommendations, and trading remain separate gates.",
    }


def _regime_scenario_context(regime_scenario: dict[str, Any] | None) -> dict[str, Any]:
    if not regime_scenario:
        return {
            "available": False,
            "packet_status": None,
            "active_regime_fields": [],
            "scenario_probabilities": {},
            "top_evidence_gaps": [],
            "self_check_horizons": [],
            "review_note": "No regime/scenario packet supplied; thesis review remains source/evidence based.",
        }
    summary = regime_scenario.get("summary", {})
    vector = regime_scenario.get("regime_context_vector", {}).get("fields", {})
    active_fields = [
        {
            "field": field,
            "state": item.get("state"),
            "intensity": item.get("intensity"),
            "trend": item.get("trend"),
            "confidence": item.get("confidence"),
            "evidence_ids": item.get("evidence_ids", []),
        }
        for field, item in vector.items()
        if isinstance(item, dict) and float(item.get("intensity") or 0.0) > 0
    ]
    graph = regime_scenario.get("scenario_outcome_graph", {})
    return {
        "available": True,
        "packet_status": summary.get("packet_status"),
        "source_run_id": regime_scenario.get("run_id"),
        "active_regime_fields": active_fields,
        "scenario_probabilities": graph.get("scenario_probabilities", {}),
        "probability_mass_valid": graph.get("probability_mass_check", {}).get("valid"),
        "top_evidence_gaps": regime_scenario.get("evidence_gap_priorities", [])[:8],
        "self_check_horizons": graph.get("horizons", []),
        "report_extension": regime_scenario.get("domain_analyst_report_extension", {}),
        "review_note": "Regime/scenario context is review-only and cannot override source evidence or create execution recommendations.",
    }


def _thesis_snapshot(domain_intake: dict[str, Any]) -> dict[str, Any]:
    report = domain_intake.get("analyst_report") or {}
    thesis = report.get("thesis") or {}
    return {
        "thesis_id": thesis.get("thesis_id"),
        "domain_id": thesis.get("domain_id"),
        "as_of": thesis.get("as_of"),
        "horizon_days": thesis.get("horizon_days"),
        "stance": thesis.get("stance"),
        "expected_direction": thesis.get("expected_direction"),
        "confidence": thesis.get("confidence"),
        "data_quality": thesis.get("data_quality"),
        "review_required": thesis.get("review_required"),
        "thesis": thesis.get("thesis"),
        "key_drivers": thesis.get("key_drivers", []),
        "assumptions": thesis.get("assumptions", []),
    }


def _evidence_lane_coverage(domain_intake: dict[str, Any]) -> dict[str, Any]:
    profile = domain_intake.get("domain_profile_snapshot", {})
    evidence_type_summary = domain_intake.get("evidence_type_summary", {})
    required = profile.get("required_evidence_types", [])
    missing = set(domain_intake.get("summary", {}).get("required_evidence_missing") or [])
    required_lanes = []
    for evidence_type in required:
        count = int(evidence_type_summary.get(evidence_type) or 0)
        required_lanes.append(
            {
                "evidence_type": evidence_type,
                "count": count,
                "status": "missing" if evidence_type in missing or count == 0 else "covered",
            }
        )
    return {
        "required_lanes": required_lanes,
        "useful_evidence_types": profile.get("useful_evidence_types", []),
        "evidence_type_summary": evidence_type_summary,
        "directness_summary": domain_intake.get("directness_summary", {}),
        "required_evidence_missing": sorted(missing),
    }


def _evidence_examples(domain_intake: dict[str, Any], kind: str, limit: int = 8) -> list[dict[str, Any]]:
    report = domain_intake.get("analyst_report") or {}
    thesis = report.get("thesis") or {}
    ids = thesis.get("supporting_evidence_ids" if kind == "supporting" else "contradicting_evidence_ids") or []
    by_id = {item.get("evidence_id"): item for item in domain_intake.get("evidence_items", [])}
    examples = []
    for evidence_id in ids[:limit]:
        item = by_id.get(evidence_id)
        if not item:
            examples.append({"evidence_id": evidence_id, "missing_from_intake": True})
            continue
        examples.append(
            {
                "evidence_id": item.get("evidence_id"),
                "source_type": item.get("source_type"),
                "published_at": item.get("published_at"),
                "evidence_type": item.get("evidence_type"),
                "directness": item.get("directness"),
                "stance_hint": item.get("stance_hint"),
                "summary": item.get("summary"),
                "limitations": item.get("limitations", []),
            }
        )
    return examples


def _risk_and_blind_spot_review(domain_intake: dict[str, Any]) -> dict[str, Any]:
    thesis = (domain_intake.get("analyst_report") or {}).get("thesis") or {}
    return {
        "risks": thesis.get("risks", []),
        "blind_spots": thesis.get("blind_spots", []),
        "assumptions": thesis.get("assumptions", []),
        "review_note": "Risks and blind spots belong to thesis review; they do not authorize ticker selection or learning promotion.",
    }


def _ticker_bridge_boundary(domain_intake: dict[str, Any]) -> dict[str, Any]:
    intake_summary = domain_intake.get("summary", {})
    report = domain_intake.get("analyst_report") or {}
    basket = report.get("ticker_basket") or {}
    ticker_candidates = [
        {
            "ticker": item.get("ticker"),
            "candidate_status": item.get("candidate_status"),
            "expected_direction": item.get("expected_direction"),
            "confidence": item.get("confidence"),
            "ticker_specific_evidence_count": len(
                item.get("ticker_specific_evidence_ids") or []
            ),
            "required_missing_evidence": item.get(
                "required_missing_evidence", []
            ),
            "blocked_reasons": item.get("blocked_reasons", []),
        }
        for item in basket.get("candidates") or []
        if isinstance(item, dict)
    ]
    can_prepare = (
        bool(report.get("thesis", {}).get("thesis"))
        and not intake_summary.get("required_evidence_missing")
        and report.get("recommendation") not in {"needs_more_data", "blocked"}
        and intake_summary.get("can_create_direct_ticker_thesis_without_bridge") is False
        and intake_summary.get("can_trade") is False
    )
    return {
        "boundary": "Ticker thesis requires a separate sector-to-ticker bridge and direct ticker evidence review.",
        "basket_status": basket.get("basket_status"),
        "direct_ready_count": basket.get("direct_ready_count"),
        "basket_candidate_count": basket.get("basket_candidate_count"),
        "blocked_count": basket.get("blocked_count"),
        "ticker_candidates": ticker_candidates,
        "ticker_direct_count": intake_summary.get("ticker_direct_count"),
        "can_create_direct_ticker_thesis_without_bridge": False,
        "can_prepare_separate_ticker_bridge_after_manual_review": can_prepare,
        "allowed_next_artifact": "SectorThesisToTickerBasketBridge or SectorToTickerReviewPacket after manual domain-thesis review.",
    }


def _reasoning_snapshot_context(
    *,
    reasoning_snapshot: dict[str, Any] | None,
    reasoning_snapshot_json: str | Path | None,
    domain_intake_json: str | Path,
    domain_intake: dict[str, Any],
) -> dict[str, Any]:
    if not reasoning_snapshot:
        return {
            "available": False,
            "status": "not_supplied",
            "hash_bound": None,
            "classified_event_count": 0,
            "transmission_channel_count": 0,
            "hypothesis_count": 0,
            "evidence_gap_count": 0,
            "directional_ticker_reasoning_event_count": 0,
            "scenario_graph_status": "not_supplied",
            "expectation_gap_status": "not_supplied",
        }

    snapshot_inputs = reasoning_snapshot.get("inputs", {})
    snapshot_summary = reasoning_snapshot.get("summary", {})
    snapshot_path = Path(reasoning_snapshot_json) if reasoning_snapshot_json else None
    snapshot_sha256 = (
        _file_sha256(snapshot_path)
        if snapshot_path is not None and snapshot_path.is_file()
        else None
    )
    intake_path = Path(domain_intake_json)
    actual_runtime_sha256 = (
        _file_sha256(intake_path) if intake_path.is_file() else None
    )
    expected_runtime_sha256 = snapshot_inputs.get("runtime_sha256")
    report = domain_intake.get("analyst_report") or {}
    domain_matches = snapshot_inputs.get("domain_id") == domain_intake.get(
        "domain_id"
    )
    as_of_matches = snapshot_inputs.get("as_of") == report.get("as_of")
    hash_bound = bool(
        actual_runtime_sha256
        and expected_runtime_sha256
        and actual_runtime_sha256 == expected_runtime_sha256
    )
    return {
        "available": True,
        "run_id": reasoning_snapshot.get("run_id"),
        "snapshot_sha256": snapshot_sha256,
        "contract": reasoning_snapshot.get("contract"),
        "mode": reasoning_snapshot.get("mode"),
        "status": reasoning_snapshot.get("status"),
        "runtime_sha256_expected": expected_runtime_sha256,
        "runtime_sha256_actual": actual_runtime_sha256,
        "hash_bound": hash_bound,
        "domain_matches": domain_matches,
        "as_of_matches": as_of_matches,
        "classified_event_count": snapshot_summary.get("classified_event_count"),
        "transmission_channel_count": snapshot_summary.get(
            "transmission_channel_count"
        ),
        "hypothesis_count": snapshot_summary.get("hypothesis_count"),
        "evidence_gap_count": snapshot_summary.get("evidence_gap_count"),
        "evidence_touched_regime_dimension_count": snapshot_summary.get(
            "evidence_touched_regime_dimension_count"
        ),
        "directional_ticker_reasoning_event_count": snapshot_summary.get(
            "directional_ticker_reasoning_event_count"
        ),
        "scenario_graph_status": snapshot_summary.get("scenario_graph_status"),
        "expectation_gap_status": snapshot_summary.get("expectation_gap_status"),
        "module_policy": reasoning_snapshot.get("module_policy", {}),
        "regime_context": reasoning_snapshot.get("regime_context"),
        "transmission_channel_counts": dict(
            sorted(
                Counter(
                    str(item.get("channel_name") or "unknown")
                    for item in reasoning_snapshot.get(
                        "transmission_channels", []
                    )
                    if isinstance(item, dict)
                ).items()
            )
        ),
        "hypothesis_ledger": reasoning_snapshot.get("hypothesis_ledger", []),
        "evidence_gaps": reasoning_snapshot.get("evidence_gaps", []),
        "review_checks": reasoning_snapshot.get("review_checks", []),
    }


def _reasoning_snapshot_checks(
    context: dict[str, Any],
) -> list[dict[str, str]]:
    nested_checks = context.get("review_checks", [])
    nested_failures = [
        item
        for item in nested_checks
        if isinstance(item, dict) and item.get("status") == "fail"
    ]
    return [
        _check(
            "pass"
            if context.get("contract")
            == "dean_analyst_core_reasoning_snapshot_v1"
            and context.get("mode") == "analyst_core_reasoning_snapshot"
            else "fail",
            "reasoning_snapshot_artifact_type",
            f"{context.get('contract')} / {context.get('mode')}",
        ),
        _check(
            "pass" if context.get("hash_bound") is True else "fail",
            "reasoning_snapshot_runtime_hash_bound",
            (
                f"expected={context.get('runtime_sha256_expected')} "
                f"actual={context.get('runtime_sha256_actual')}"
            ),
        ),
        _check(
            "pass"
            if context.get("domain_matches") is True
            and context.get("as_of_matches") is True
            else "fail",
            "reasoning_snapshot_scope_matches_runtime",
            (
                f"domain_matches={context.get('domain_matches')} "
                f"as_of_matches={context.get('as_of_matches')}"
            ),
        ),
        _check(
            "pass" if not nested_failures else "fail",
            "reasoning_snapshot_checks_have_no_failures",
            f"nested_failures={len(nested_failures)}",
        ),
        _check(
            "pass"
            if int(
                context.get("directional_ticker_reasoning_event_count") or 0
            )
            == 0
            else "fail",
            "reasoning_snapshot_no_directional_ticker_leakage",
            str(context.get("directional_ticker_reasoning_event_count")),
        ),
        _check(
            "warn"
            if context.get("scenario_graph_status") == "not_generated"
            else "fail",
            "reasoning_snapshot_scenario_graph_boundary",
            str(context.get("scenario_graph_status")),
        ),
    ]


def _review_checks(
    *,
    domain_intake: dict[str, Any],
    domain_instance: dict[str, Any] | None,
    regime_scenario: dict[str, Any] | None,
    architecture_map: dict[str, Any] | None,
    linked_artifact_verification: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    intake_summary = domain_intake.get("summary", {})
    report = domain_intake.get("analyst_report") or {}
    thesis = report.get("thesis") or {}
    source_gate = domain_intake.get("source_gate_context", {})
    source_mode = domain_intake.get("mode")
    checks = [
        _check(
            "pass"
            if source_mode
            in {
                "domain_analyst_intake_packet",
                "semiconductor_analyst_runtime",
            }
            else "fail",
            "domain_intake_artifact_type",
            str(source_mode),
        ),
        _check("pass" if intake_summary.get("analyst_report_created") is True else "fail", "analyst_report_created", str(intake_summary.get("intake_status"))),
        _check("pass" if thesis.get("thesis") else "fail", "domain_thesis_text_present", str(thesis.get("thesis_id"))),
        _check("pass" if thesis.get("review_required") is True else "fail", "domain_thesis_requires_review", f"review_required={thesis.get('review_required')!r}."),
        _check("pass" if source_gate.get("can_enter_domain_research") is True else "fail", "source_gate_allows_domain_research", str(source_gate.get("gate_status"))),
        _check("pass" if source_gate.get("safe_downstream_boundary") is True else "fail", "source_gate_downstream_boundary_safe", "Source gate downstream actions remain disabled."),
        _check("pass" if int(intake_summary.get("evidence_item_count") or 0) > 0 else "fail", "evidence_items_present", str(intake_summary.get("evidence_item_count"))),
        _check("pass" if not intake_summary.get("required_evidence_missing") else "warn", "required_evidence_lanes_covered", ", ".join(intake_summary.get("required_evidence_missing") or []) or "All required lanes covered."),
        _check("pass" if intake_summary.get("can_create_direct_ticker_thesis_without_bridge") is False else "fail", "ticker_thesis_requires_bridge", "Ticker thesis remains bridge-gated."),
        _check("pass" if intake_summary.get("can_write_learning_memory") is False else "fail", "no_learning_write", "No learning memory write."),
        _check("pass" if intake_summary.get("can_create_recommendation") is False else "fail", "legacy_no_execution_recommendation", "No execution recommendation."),
        _check("pass" if intake_summary.get("can_trade") is False else "fail", "no_trading", "No trading."),
    ]
    if int(intake_summary.get("ticker_direct_count") or 0) == 0:
        checks.append(_check("pass", "sector_domain_thesis_not_ticker_forced", "No direct ticker evidence; thesis remains sector/domain-only."))
    else:
        checks.append(_check("warn", "ticker_direct_evidence_present", f"{intake_summary.get('ticker_direct_count')} direct ticker evidence items require bridge review."))
    if report.get("recommendation") in {"needs_more_data", "blocked"}:
        checks.append(_check("warn", "analyst_recommendation_not_standardizable_yet", str(report.get("recommendation"))))
    else:
        checks.append(_check("pass", "analyst_recommendation_reviewable", str(report.get("recommendation"))))
    if source_mode == "semiconductor_analyst_runtime":
        checks.extend(
            [
                _check(
                    "pass"
                    if intake_summary.get("sector_thesis_ready") is True
                    else "fail",
                    "runtime_sector_thesis_ready",
                    str(intake_summary.get("sector_thesis_ready")),
                ),
                _check(
                    "pass"
                    if intake_summary.get(
                        "can_create_ticker_forecast"
                    )
                    is False
                    else "fail",
                    "runtime_no_direct_ticker_forecast",
                    (
                        "The sector runtime cannot create a direct ticker "
                        "forecast."
                    ),
                ),
                _check(
                    "pass"
                    if (
                        domain_intake.get("integration_boundary") or {}
                    ).get("ticker_model_case_is_sector_evidence")
                    is False
                    else "fail",
                    "runtime_ticker_model_case_excluded",
                    (
                        "The AMD model case remains outside sector "
                        "evidence."
                    ),
                ),
                _check(
                    "pass"
                    if (
                        linked_artifact_verification or {}
                    ).get("all_hashes_match")
                    is True
                    else "fail",
                    "runtime_linked_artifacts_hash_bound",
                    str(
                        (
                            linked_artifact_verification or {}
                        ).get("status")
                    ),
                ),
                _check(
                    "warn",
                    "runtime_confidence_is_not_forecast_probability",
                    (
                        "Confidence is an evidence-quality heuristic, not "
                        "a calibrated return probability."
                    ),
                ),
                _check(
                    "warn",
                    "runtime_short_market_confirmation_window",
                    (
                        "Market confirmation uses 22 common sessions and "
                        "must remain supporting context."
                    ),
                ),
                _check(
                    "warn",
                    "runtime_partial_fundamental_comparability",
                    (
                        "TSM annual TWD observations are separate from US "
                        "quarterly USD comparison lanes."
                    ),
                ),
            ]
        )
    if domain_instance:
        instance_summary = domain_instance.get("summary", {})
        checks.append(
            _check(
                "pass" if instance_summary.get("instance_status") == "domain_analyst_instance_review_ready" else "warn",
                "domain_instance_contract_ready",
                str(instance_summary.get("instance_status")),
            )
        )
        checks.append(_check("pass" if instance_summary.get("can_scale_to_other_domains_now") is False else "fail", "domain_instance_no_scaling", "Scaling remains disabled."))
        checks.append(_check("pass" if instance_summary.get("can_trade") is False else "fail", "domain_instance_no_trading", "Instance contract keeps trading disabled."))
    if regime_scenario:
        regime_summary = regime_scenario.get("summary", {})
        checks.append(
            _check(
                "pass" if regime_scenario.get("mode") == "domain_analyst_regime_scenario_packet" else "fail",
                "regime_scenario_artifact_type",
                str(regime_scenario.get("mode")),
            )
        )
        checks.append(
            _check(
                "pass" if str(regime_summary.get("packet_status", "")).startswith("domain_analyst_regime_scenario_ready") else "warn",
                "regime_scenario_review_ready",
                str(regime_summary.get("packet_status")),
            )
        )
        checks.append(
            _check(
                "pass" if regime_summary.get("probability_mass_valid") is True else "fail",
                "regime_scenario_probability_mass_valid",
                str(regime_summary.get("probability_mass_valid")),
            )
        )
        checks.append(_check("pass" if regime_summary.get("can_create_execution_recommendation") is False else "fail", "regime_scenario_no_execution_recommendation", "Regime/scenario packet has no execution authority."))
        checks.append(_check("pass" if regime_summary.get("can_trade") is False else "fail", "regime_scenario_no_trading", "Regime/scenario packet keeps trading disabled."))
    if architecture_map:
        arch_summary = architecture_map.get("summary", {})
        checks.append(_check("pass" if arch_summary.get("can_clone_domain_profiles_now") is False else "fail", "architecture_no_domain_cloning", "Architecture keeps cloning disabled."))
        checks.append(_check("pass" if arch_summary.get("can_trade") is False else "fail", "architecture_no_trading", "Architecture keeps trading disabled."))
    return checks


def _packet_status(
    domain_intake: dict[str, Any],
    domain_instance: dict[str, Any] | None,
    checks: list[dict[str, str]],
) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "blocked_domain_thesis_review"
    report = domain_intake.get("analyst_report") or {}
    missing = domain_intake.get("summary", {}).get("required_evidence_missing") or []
    if missing or report.get("recommendation") in {"needs_more_data", "blocked"}:
        return "domain_thesis_review_needs_more_evidence"
    if any(check["status"] == "warn" for check in checks):
        return "domain_thesis_review_ready_with_cautions"
    return "domain_thesis_review_ready"


def _decision_guidance(status: str, checks: list[dict[str, str]]) -> dict[str, Any]:
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    failures = [check["code"] for check in checks if check["status"] == "fail"]
    if failures:
        action = "fix_failed_checks_before_thesis_review"
    elif status == "domain_thesis_review_needs_more_evidence":
        action = "collect_missing_domain_evidence_before_standardization"
    elif warnings:
        action = "manual_review_required_before_standardization"
    else:
        action = "manual_review_can_accept_template_candidate"
    reasons = [
        f"Packet status is {status}.",
        "This packet reviews the domain/sector thesis only.",
        "Ticker bridge, learning promotion, execution recommendations, and trading remain separate.",
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
        "No live collector is started.",
        "No source extraction, claim/event/entity extraction, or evidence promotion is executed.",
        "No sector-to-ticker bridge is executed by this packet.",
        "No ticker thesis, execution recommendation, allocation, price target, paper order, broker call, or live trade is generated.",
        "No learning memory, analyst-weight update, model training, tuning, or production config write is performed.",
        "No new domain analyst profile is cloned or enabled.",
    ]


def _commands(
    domain_intake_json: str | Path,
    reasoning_snapshot_json: str | Path | None = None,
) -> dict[str, str]:
    reasoning_arg = (
        f"--reasoning-snapshot-json {reasoning_snapshot_json} "
        if reasoning_snapshot_json
        else ""
    )
    return {
        "rerun_domain_thesis_review": (
            "python run_agent_domain_analyst_thesis_review_packet.py "
            f"--domain-intake-json {domain_intake_json} "
            f"{reasoning_arg}"
            "--domain-instance-contract-json reports\\dean_os\\domain_analyst_instance_contract_current\\latest.json "
            "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
            "--output-dir reports\\dean_os\\domain_analyst_thesis_review_packet_current"
        ),
        "next_separate_ticker_bridge_after_manual_review": (
            "python run_agent_sector_to_ticker_bridge.py "
            "--research-batch-json PATH_TO_REVIEWED_RESEARCH_BATCH "
            "--domain-profile semiconductor_ai_infrastructure --sector semiconductor "
            "--output-dir reports\\dean_os\\sector_thesis_to_ticker_basket_current"
        ),
    }


def _operator_next_steps(status: str, checks: list[dict[str, str]]) -> list[str]:
    if status == "blocked_domain_thesis_review":
        return ["Fix failed thesis-review checks before using this analyst instance as a reusable template."]
    warnings = [check["code"] for check in checks if check["status"] == "warn"]
    if status == "domain_thesis_review_needs_more_evidence":
        return ["Collect missing domain evidence lanes, rerun intake, then rerun this thesis review packet."]
    steps = ["Manually review the thesis text, evidence lanes, risks, and contradicting evidence examples."]
    if warnings:
        steps.append("Resolve or explicitly accept warning checks before standardization: " + ", ".join(warnings) + ".")
    steps.append("If accepted, mark this semiconductor domain-thesis packet as the first reusable analyst-template candidate.")
    steps.append("Only after that, run the separate sector-to-ticker bridge; do not clone other domains yet.")
    return steps


def _active_regime_field_count(regime_scenario: dict[str, Any] | None) -> int:
    if not regime_scenario:
        return 0
    fields = regime_scenario.get("regime_context_vector", {}).get("fields", {})
    return sum(1 for item in fields.values() if isinstance(item, dict) and float(item.get("intensity") or 0.0) > 0)


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
