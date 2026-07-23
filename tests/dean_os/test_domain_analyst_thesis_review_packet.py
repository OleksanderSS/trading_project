from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dean_os.analyst_core.domain_analyst_thesis_review_packet import DomainAnalystThesisReviewPacket


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _artifact_descriptor(path: Path, payload: dict) -> dict:
    _write_json(path, payload)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _evidence_item(evidence_id: str, *, stance_hint: str = "positive", evidence_type: str = "sector_demand") -> dict:
    return {
        "evidence_id": evidence_id,
        "source_type": "news",
        "source": "fixture://source",
        "published_at": "2026-05-01T00:00:00+00:00",
        "as_of": "2026-06-19T00:00:00+00:00",
        "domain_id": "semiconductor_ai_infrastructure",
        "tickers": [],
        "sectors": ["semiconductor"],
        "evidence_type": evidence_type,
        "summary": f"{evidence_type} evidence summary",
        "stance_hint": stance_hint,
        "strength": 0.7,
        "freshness_score": 0.8,
        "directness": "sector",
        "reliability_score": 0.7,
        "limitations": ["review_only"],
        "blocked_windows": [],
    }


def _domain_intake(**summary_overrides) -> dict:
    summary = {
        "intake_status": "domain_analyst_intake_ready",
        "domain_id": "semiconductor_ai_infrastructure",
        "document_count": 6,
        "evidence_item_count": 6,
        "ticker_direct_count": 0,
        "sector_or_domain_count": 5,
        "macro_policy_context_count": 1,
        "required_evidence_missing": [],
        "analyst_report_created": True,
        "can_run_domain_analyst": True,
        "can_create_direct_ticker_thesis_without_bridge": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "domain_intake_fixture",
        "mode": "domain_analyst_intake_packet",
        "inputs": {"domain_id": "semiconductor_ai_infrastructure", "sectors": ["semiconductor"]},
        "summary": summary,
        "source_gate_context": {
            "available": True,
            "gate_status": "source_evidence_ready_for_domain_research",
            "can_enter_domain_research": True,
            "safe_downstream_boundary": True,
            "warnings": [],
        },
        "domain_profile_snapshot": {
            "domain_id": "semiconductor_ai_infrastructure",
            "required_evidence_types": [
                "sector_demand",
                "capex_cycle",
                "supply_chain",
                "policy_or_geopolitical",
                "market_confirmation",
            ],
            "useful_evidence_types": ["hyperscaler_capex", "inventory_cycle"],
            "ticker_universe_hint": ["AMD", "NVDA", "TSM"],
        },
        "evidence_type_summary": {
            "sector_demand": 2,
            "capex_cycle": 1,
            "supply_chain": 1,
            "policy_or_geopolitical": 1,
            "market_confirmation": 1,
        },
        "directness_summary": {"sector": 5, "policy": 1},
        "evidence_items": [
            _evidence_item("ev_support_1", evidence_type="sector_demand"),
            _evidence_item("ev_support_2", evidence_type="capex_cycle"),
            _evidence_item("ev_contra_1", stance_hint="negative", evidence_type="policy_or_geopolitical"),
        ],
        "analyst_report": {
            "report_id": "analyst_report_fixture",
            "agent_name": "semiconductor_ai_infrastructure_working_analyst",
            "domain_id": "semiconductor_ai_infrastructure",
            "as_of": "2026-06-19T00:00:00+00:00",
            "horizon_days": 180,
            "domain_profile_version": "0.1.0",
            "thesis": {
                "thesis_id": "thesis_fixture",
                "domain_id": "semiconductor_ai_infrastructure",
                "as_of": "2026-06-19T00:00:00+00:00",
                "horizon_days": 180,
                "stance": "mixed",
                "expected_direction": "mixed",
                "confidence": 0.68,
                "thesis": "Semiconductor AI infrastructure thesis is mixed but reviewable.",
                "key_drivers": ["AI demand", "capex cycle", "supply chain"],
                "supporting_evidence_ids": ["ev_support_1", "ev_support_2"],
                "contradicting_evidence_ids": ["ev_contra_1"],
                "assumptions": ["Sector thesis is not a ticker forecast."],
                "risks": ["Export controls can reduce confidence."],
                "blind_spots": [],
                "data_quality": "medium",
                "review_required": True,
            },
            "ticker_basket": {
                "basket_status": "partial_basket_ready",
                "direct_ready_count": 0,
                "basket_candidate_count": 3,
                "blocked_count": 0,
            },
            "recommendation": "partial_ready_for_review",
            "review_required": True,
            "live_execution_allowed": False,
        },
    }


def _domain_instance(**summary_overrides) -> dict:
    summary = {
        "instance_status": "domain_analyst_instance_review_ready",
        "domain_id": "semiconductor_ai_infrastructure",
        "can_reuse_as_template_after_manual_review": True,
        "can_scale_to_other_domains_now": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {"mode": "domain_analyst_instance_contract", "summary": summary}


def _architecture() -> dict:
    return {
        "mode": "current_architecture_map",
        "summary": {"can_clone_domain_profiles_now": False, "can_trade": False},
    }


def _regime_scenario() -> dict:
    return {
        "run_id": "regime_scenario_fixture",
        "mode": "domain_analyst_regime_scenario_packet",
        "summary": {
            "packet_status": "domain_analyst_regime_scenario_ready_with_review_items",
            "probability_mass_valid": True,
            "scenario_node_count": 8,
            "evidence_gap_count": 2,
            "can_create_execution_recommendation": False,
            "can_trade": False,
        },
        "regime_context_vector": {
            "fields": {
                "ai_tech_cycle": {
                    "state": "capex_boom",
                    "intensity": 0.8,
                    "trend": "rising",
                    "confidence": "high",
                    "evidence_ids": ["event:ai_demand"],
                },
                "geopolitical_state": {
                    "state": "sanctions_chokepoint_risk",
                    "intensity": 0.6,
                    "trend": "rising",
                    "confidence": "medium",
                    "evidence_ids": ["event:export_controls"],
                },
            }
        },
        "scenario_outcome_graph": {
            "scenario_probabilities": {
                "base_case_continuation": 0.5,
                "upside_acceleration": 0.3,
                "downside_constraint": 0.2,
            },
            "probability_mass_check": {"sum": 1.0, "valid": True},
            "horizons": ["1d", "5d", "20d", "60d", "120d"],
        },
        "evidence_gap_priorities": [
            {"gap_id": "gap:1", "priority": "high", "description": "Check earnings revisions."},
            {"gap_id": "gap:2", "priority": "medium", "description": "Check export-control exemptions."},
        ],
        "domain_analyst_report_extension": {"valuation_expectation_gap": {"status": "requires_market_confirmation"}},
    }


def _semiconductor_runtime(tmp_path: Path) -> dict:
    source_dir = tmp_path / "runtime_sources"
    source_artifacts = {
        "fundamental": _artifact_descriptor(
            source_dir / "fundamental.json",
            {"summary": {"source_fact_count": 29}},
        ),
        "macro": _artifact_descriptor(
            source_dir / "macro.json",
            {"summary": {"accepted_series_count": 27}},
        ),
        "sector_market": _artifact_descriptor(
            source_dir / "market.json",
            {
                "summary": {
                    "common_session_count": 22,
                    "lookback_sessions": 20,
                },
                "metrics": [
                    {
                        "name": "sector_median_return_20_session",
                        "value": 3.56,
                    },
                    {
                        "name": "sector_positive_breadth",
                        "value": 0.75,
                    },
                    {
                        "name": "sector_median_excess_return_vs_qqq",
                        "value": 6.84,
                    },
                    {
                        "name": "sector_return_dispersion_20_session",
                        "value": 5.47,
                    },
                    {
                        "name": "nvda_return_20_session",
                        "value": -8.39,
                    },
                ],
            },
        ),
        "semiconductor_news": _artifact_descriptor(
            source_dir / "news.json",
            {
                "lane_review": [
                    {
                        "evidence_type": "sector_demand",
                        "status": "eligible",
                        "candidate_count": 5,
                        "strong_candidate_count": 3,
                        "independent_strong_source_count": 3,
                        "independent_strong_sources": [
                            "bloomberg",
                            "cnbc",
                            "reuters",
                        ],
                    }
                ]
            },
        ),
        "derived_ratios": _artifact_descriptor(
            source_dir / "ratios.json",
            {
                "summary": {
                    "derived_ratio_count": 2,
                    "multi_ticker_comparison_lane_count": 1,
                    "full_cohort_comparison_lane_count": 0,
                    "can_claim_full_cohort_comparability": False,
                },
                "ratios": [
                    {
                        "ticker": "AMD",
                        "ratio_name": "operating_margin",
                        "value": 0.144,
                        "comparison_period_class": "quarterly_Q1",
                        "source_currency": "USD",
                    },
                    {
                        "ticker": "TSM",
                        "ratio_name": "operating_margin",
                        "value": 0.5083,
                        "comparison_period_class": "annual",
                        "source_currency": "TWD",
                    },
                ],
            },
        ),
        "official_policy": _artifact_descriptor(
            source_dir / "policy.json",
            {
                "corroboration": {
                    "policy_lane_eligible": True,
                    "combined_independent_source_count": 2,
                    "combined_independent_sources": [
                        "bloomberg",
                        "us_bureau_industry_security",
                    ],
                }
            },
        ),
        "excluded_pipeline_case": _artifact_descriptor(
            source_dir / "excluded_pipeline_case.json",
            {"summary": {"ticker": "AMD", "case_status": "blocked"}},
        ),
    }
    report = _domain_intake()["analyst_report"]
    report["evidence"] = _domain_intake()["evidence_items"]
    report["ticker_basket"]["candidates"] = [
        {
            "ticker": ticker,
            "candidate_status": "basket_candidate",
            "expected_direction": "mixed",
            "confidence": 0.35,
            "ticker_specific_evidence_ids": [],
            "required_missing_evidence": [
                "ticker_specific_evidence"
            ],
            "blocked_reasons": [
                "Only sector/domain evidence is available."
            ],
        }
        for ticker in ("AMD", "INTC", "NVDA", "TSM")
    ]
    return {
        "run_id": "semiconductor_runtime_fixture",
        "mode": "semiconductor_analyst_runtime",
        "domain_id": "semiconductor_ai_infrastructure",
        "status": "semiconductor_analysis_partial_ready_for_review",
        "source_artifacts": source_artifacts,
        "summary": {
            "recommendation": "partial_ready_for_review",
            "evidence_count": 3,
            "required_lane_count": 5,
            "satisfied_required_lane_count": 5,
            "missing_required_evidence": [],
            "sector_thesis_ready": True,
            "can_create_ticker_forecast": False,
            "can_train": False,
            "can_tune": False,
            "can_trade": False,
        },
        "evidence_lane_coverage": {
            "required_lanes": [
                {
                    "evidence_type": lane,
                    "status": "satisfied",
                    "eligible_evidence_count": 1,
                    "all_context_item_count": 1,
                }
                for lane in (
                    "sector_demand",
                    "capex_cycle",
                    "supply_chain",
                    "policy_or_geopolitical",
                    "market_confirmation",
                )
            ],
            "eligible_evidence_type_counts": {
                "sector_demand": 1,
                "capex_cycle": 1,
                "supply_chain": 1,
                "policy_or_geopolitical": 1,
                "market_confirmation": 1,
            },
            "all_evidence_type_counts": {
                "sector_demand": 1,
                "capex_cycle": 1,
                "supply_chain": 1,
                "policy_or_geopolitical": 1,
                "market_confirmation": 1,
                "fundamental_context": 1,
            },
        },
        "integration_boundary": {
            "training_allowed": False,
            "tuning_allowed": False,
            "automatic_trading_allowed": False,
            "ticker_model_case_is_sector_evidence": False,
        },
        "safety": {"review_only": True},
        "analyst_report": report,
    }


def _write_inputs(tmp_path: Path, *, intake: dict | None = None, instance: dict | None = None) -> dict[str, Path]:
    return {
        "intake": _write_json(tmp_path / "intake" / "latest.json", intake or _domain_intake()),
        "instance": _write_json(tmp_path / "instance" / "latest.json", instance or _domain_instance()),
        "architecture": _write_json(tmp_path / "architecture" / "latest.json", _architecture()),
        "regime": _write_json(tmp_path / "regime" / "latest.json", _regime_scenario()),
    }


def test_domain_analyst_thesis_review_packet_marks_sector_thesis_review_ready(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystThesisReviewPacket(tmp_path / "reports").build(
        domain_intake_json=paths["intake"],
        domain_instance_contract_json=paths["instance"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["packet_status"] == "domain_thesis_review_ready"
    assert payload["summary"]["can_standardize_domain_template_after_manual_review"] is True
    assert payload["summary"]["can_prepare_separate_ticker_bridge_after_manual_review"] is True
    assert payload["summary"]["can_create_direct_ticker_thesis_without_bridge"] is False
    assert payload["summary"]["can_create_analyst_research_recommendation"] is True
    assert payload["summary"]["can_create_execution_recommendation"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["ticker_bridge_boundary"]["ticker_direct_count"] == 0
    assert payload["supporting_evidence_examples"][0]["evidence_id"] == "ev_support_1"
    assert any(check["code"] == "sector_domain_thesis_not_ticker_forced" and check["status"] == "pass" for check in payload["review_checks"])


def test_domain_analyst_thesis_review_packet_attaches_regime_scenario_context(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystThesisReviewPacket(tmp_path / "reports").build(
        domain_intake_json=paths["intake"],
        domain_instance_contract_json=paths["instance"],
        regime_scenario_json=paths["regime"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["packet_status"] == "domain_thesis_review_ready"
    assert payload["summary"]["regime_scenario_status"] == "domain_analyst_regime_scenario_ready_with_review_items"
    assert payload["summary"]["active_regime_field_count"] == 2
    assert payload["summary"]["scenario_probability_mass_valid"] is True
    assert payload["summary"]["can_use_regime_scenario_context_for_review"] is True
    assert payload["regime_scenario_context"]["available"] is True
    assert payload["regime_scenario_context"]["scenario_probabilities"]["upside_acceleration"] == 0.3
    assert payload["regime_scenario_context"]["top_evidence_gaps"][0]["priority"] == "high"
    assert any(check["code"] == "regime_scenario_probability_mass_valid" and check["status"] == "pass" for check in payload["review_checks"])
    assert payload["summary"]["can_create_execution_recommendation"] is False
    assert payload["summary"]["can_trade"] is False


def test_domain_analyst_thesis_review_packet_blocks_unsafe_intake(tmp_path):
    paths = _write_inputs(tmp_path, intake=_domain_intake(can_trade=True))

    payload = DomainAnalystThesisReviewPacket(tmp_path / "reports").build(
        domain_intake_json=paths["intake"],
        domain_instance_contract_json=paths["instance"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["packet_status"] == "blocked_domain_thesis_review"
    assert payload["summary"]["can_standardize_domain_template_after_manual_review"] is False
    assert any(check["code"] == "no_trading" and check["status"] == "fail" for check in payload["review_checks"])


def test_domain_analyst_thesis_review_packet_needs_more_evidence_when_required_lane_missing(tmp_path):
    intake = _domain_intake(required_evidence_missing=["capex_cycle"])
    intake["evidence_type_summary"]["capex_cycle"] = 0
    paths = _write_inputs(tmp_path, intake=intake)

    payload = DomainAnalystThesisReviewPacket(tmp_path / "reports").build(
        domain_intake_json=paths["intake"],
        domain_instance_contract_json=paths["instance"],
        architecture_map_json=paths["architecture"],
        save=False,
    )

    assert payload["summary"]["packet_status"] == "domain_thesis_review_needs_more_evidence"
    assert payload["summary"]["can_standardize_domain_template_after_manual_review"] is False
    capex = [lane for lane in payload["evidence_lane_coverage"]["required_lanes"] if lane["evidence_type"] == "capex_cycle"][0]
    assert capex["status"] == "missing"


def test_domain_thesis_review_consumes_current_semiconductor_runtime(
    tmp_path,
):
    runtime_path = _write_json(
        tmp_path / "runtime" / "latest.json",
        _semiconductor_runtime(tmp_path),
    )
    instance_path = _write_json(
        tmp_path / "instance" / "latest.json",
        _domain_instance(),
    )
    architecture_path = _write_json(
        tmp_path / "architecture" / "latest.json",
        _architecture(),
    )

    payload = DomainAnalystThesisReviewPacket(
        tmp_path / "reports"
    ).build(
        domain_intake_json=runtime_path,
        domain_instance_contract_json=instance_path,
        architecture_map_json=architecture_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == (
        "domain_thesis_review_ready_with_cautions"
    )
    assert payload["inputs"]["review_source_mode"] == (
        "semiconductor_analyst_runtime"
    )
    assert payload["linked_artifact_verification"][
        "all_hashes_match"
    ] is True
    analytical = payload["analytical_review"]
    assert analytical["assessment_status"] == (
        "sector_thesis_reviewable_with_cautions"
    )
    assert analytical["scope_decision"] == "sector_thesis_only"
    assert analytical["ticker_decision"] == (
        "no_direct_ticker_thesis"
    )
    assert analytical["market_snapshot"][
        "sector_median_return_percent"
    ] == 3.56
    assert analytical["fundamental_ratio_review"][
        "quarterly_comparable_ratios"
    ][0]["ticker"] == "AMD"
    assert analytical["fundamental_ratio_review"][
        "separate_annual_ratios"
    ][0]["ticker"] == "TSM"
    assert payload["ticker_bridge_boundary"][
        "ticker_direct_count"
    ] == 0
    assert {
        item["ticker"]
        for item in payload["ticker_bridge_boundary"][
            "ticker_candidates"
        ]
    } == {"AMD", "INTC", "NVDA", "TSM"}
    assert any(
        check["code"] == "runtime_ticker_model_case_excluded"
        and check["status"] == "pass"
        for check in payload["review_checks"]
    )


def test_domain_analyst_thesis_review_packet_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystThesisReviewPacket(tmp_path / "reports").build(
        domain_intake_json=paths["intake"],
        domain_instance_contract_json=paths["instance"],
        architecture_map_json=paths["architecture"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can trade: False" in markdown
    assert "Ticker Bridge Boundary" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_thesis_review_packet.py"),
            "--domain-intake-json",
            str(paths["intake"]),
            "--domain-instance-contract-json",
            str(paths["instance"]),
            "--architecture-map-json",
            str(paths["architecture"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Packet status: domain_thesis_review_ready" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
