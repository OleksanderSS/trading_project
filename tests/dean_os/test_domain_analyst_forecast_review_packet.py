from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.analyst_core.domain_analyst_forecast_review_packet import DomainAnalystForecastReviewPacket


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _thesis_review(**summary_overrides) -> dict:
    summary = {
        "packet_status": "domain_thesis_review_ready",
        "domain_id": "semiconductor_ai_infrastructure",
        "thesis_id": "thesis_fixture",
        "thesis_stance": "mixed",
        "expected_direction": "mixed",
        "confidence": 0.7,
        "supporting_evidence_count": 2,
        "contradicting_evidence_count": 1,
        "ticker_direct_count": 0,
        "required_evidence_missing": [],
        "manual_review_required": True,
        "can_create_direct_ticker_thesis_without_bridge": False,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "thesis_review_fixture",
        "mode": "domain_analyst_thesis_review_packet",
        "created_at": "2026-06-19T00:00:00+00:00",
        "summary": summary,
        "thesis_snapshot": {
            "thesis_id": "thesis_fixture",
            "domain_id": "semiconductor_ai_infrastructure",
            "as_of": "2026-06-19T00:00:00+00:00",
            "horizon_days": 180,
            "stance": "mixed",
            "expected_direction": "mixed",
            "confidence": 0.7,
            "thesis": "Semiconductor AI infrastructure demand is mixed but reviewable over the next 180 days.",
            "key_drivers": ["AI demand", "hyperscaler capex", "export controls"],
            "assumptions": ["Sector thesis is not a ticker forecast."],
        },
        "evidence_lane_coverage": {
            "required_lanes": [
                {"evidence_type": "sector_demand", "count": 2, "status": "covered"},
                {"evidence_type": "capex_cycle", "count": 1, "status": "covered"},
            ],
            "required_evidence_missing": [],
        },
        "supporting_evidence_examples": [
            {
                "evidence_id": "ev_support_1",
                "source_type": "news",
                "published_at": "2026-06-01T00:00:00+00:00",
                "evidence_type": "sector_demand",
                "directness": "sector",
                "stance_hint": "positive",
                "summary": "AI demand supports semiconductor infrastructure.",
                "limitations": ["review_only"],
            }
        ],
        "contradicting_evidence_examples": [
            {
                "evidence_id": "ev_contra_1",
                "source_type": "report",
                "published_at": "2026-06-02T00:00:00+00:00",
                "evidence_type": "policy_or_geopolitical",
                "directness": "policy",
                "stance_hint": "negative",
                "summary": "Export controls may reduce demand.",
                "limitations": ["review_only"],
            }
        ],
        "risk_and_blind_spot_review": {
            "risks": ["Export controls can reduce confidence."],
            "blind_spots": [],
            "assumptions": ["Sector thesis is not a ticker forecast."],
        },
    }


def _vertical_slice() -> dict:
    return {
        "run_id": "vertical_fixture",
        "mode": "domain_analyst_vertical_slice_run",
        "summary": {
            "run_status": "domain_analyst_candidate_complete_pending_manual_acceptance",
            "domain_id": "semiconductor_ai_infrastructure",
            "can_create_recommendation": False,
            "can_trade": False,
        },
        "synthetic_fixture_audit": {
            "has_synthetic_marker": False,
            "has_fixture_marker": False,
            "has_smoke_label": False,
        },
    }


def _regime_scenario() -> dict:
    return {
        "run_id": "regime_scenario_fixture",
        "mode": "domain_analyst_regime_scenario_packet",
        "summary": {
            "packet_status": "domain_analyst_regime_scenario_ready_with_review_items",
            "probability_mass_valid": True,
            "can_create_execution_recommendation": False,
            "can_trade": False,
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
            {"gap_id": "gap:1", "priority": "high", "description": "Check earnings revisions."}
        ],
    }


def _write_inputs(tmp_path: Path, *, thesis: dict | None = None, vertical: dict | None = None) -> dict[str, Path]:
    return {
        "thesis": _write_json(tmp_path / "thesis" / "latest.json", thesis or _thesis_review()),
        "vertical": _write_json(tmp_path / "vertical" / "latest.json", vertical or _vertical_slice()),
        "regime": _write_json(tmp_path / "regime" / "latest.json", _regime_scenario()),
    }


def test_forecast_review_packet_creates_review_only_expectation_ledger(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystForecastReviewPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        vertical_slice_json=paths["vertical"],
        save=False,
    )

    assert payload["summary"]["packet_status"] == "forecast_review_ready_with_cautions_pending_outcomes"
    assert payload["summary"]["forecast_candidate_count"] == 1
    assert payload["summary"]["can_promote_learning_now"] is False
    assert payload["summary"]["can_create_analyst_research_recommendation"] is True
    assert payload["summary"]["can_create_analyst_self_improvement_proposal"] is True
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_create_execution_recommendation"] is False
    assert payload["summary"]["can_create_buy_sell_hold_recommendation"] is False
    assert payload["summary"]["can_create_recommendation"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["naming_contract"]["preferred_term"] == "thesis_expectation_or_forecast_candidate"
    assert payload["naming_contract"]["allowed_recommendation_term"] == "review_only_analyst_recommendation"
    assert payload["forecast_candidates"][0]["is_investment_recommendation"] is False
    assert payload["forecast_candidates"][0]["is_review_only_analyst_recommendation"] is True
    assert "self_improvement_proposal" in payload["forecast_candidates"][0]["allowed_review_outputs"]
    assert payload["forecast_candidates"][0]["learning_use"] == "candidate_case_only_until_outcome_and_human_causal_review"
    assert any(plane["plane_id"] == "luck_vs_skill" for plane in payload["analyst_control_planes"])


def test_forecast_review_packet_freezes_regime_scenario_context(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystForecastReviewPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        vertical_slice_json=paths["vertical"],
        regime_scenario_json=paths["regime"],
        save=False,
    )

    assert payload["summary"]["regime_scenario_context_available"] is True
    assert payload["summary"]["scenario_evidence_gap_count"] == 1
    assert payload["summary"]["scenario_probability_mass_valid"] is True
    assert payload["summary"]["self_check_horizon_count"] == 5
    candidate_context = payload["forecast_candidates"][0]["regime_scenario_context"]
    assert candidate_context["scenario_probabilities"]["base_case_continuation"] == 0.5
    assert candidate_context["self_check_horizons"] == ["1d", "5d", "20d", "60d", "120d"]
    assert candidate_context["top_evidence_gaps"][0]["priority"] == "high"
    assert any(plane["plane_id"] == "regime_scenario_context" for plane in payload["analyst_control_planes"])
    assert any(check["code"] == "regime_scenario_probability_mass_valid" and check["status"] == "pass" for check in payload["review_checks"])
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_trade"] is False


def test_forecast_review_packet_defines_lucky_hit_and_wrong_reason_taxonomy(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystForecastReviewPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        vertical_slice_json=paths["vertical"],
        save=False,
    )

    buckets = {item["bucket_id"] for item in payload["outcome_taxonomy"]}
    assert "correct_for_stated_reasons" in buckets
    assert "correct_but_lucky_or_wrong_reason" in buckets
    assert "incorrect_forecast" in buckets
    assert "inconclusive_or_not_mature" in buckets
    assert "unfalsifiable_or_underspecified" in buckets
    assert "data_unavailable" in buckets


def test_forecast_review_packet_blocks_unsafe_thesis_review(tmp_path):
    paths = _write_inputs(tmp_path, thesis=_thesis_review(can_write_learning_memory=True, can_trade=True))

    payload = DomainAnalystForecastReviewPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        vertical_slice_json=paths["vertical"],
        save=False,
    )

    assert payload["summary"]["packet_status"] == "blocked_forecast_review"
    assert any(check["code"] == "no_learning_write" and check["status"] == "fail" for check in payload["review_checks"])
    assert any(check["code"] == "no_trading" and check["status"] == "fail" for check in payload["review_checks"])


def test_forecast_review_packet_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystForecastReviewPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        vertical_slice_json=paths["vertical"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can promote learning now: False" in markdown
    assert "Can trade: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_forecast_review_packet.py"),
            "--domain-thesis-review-json",
            str(paths["thesis"]),
            "--vertical-slice-json",
            str(paths["vertical"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Packet status: forecast_review_ready_with_cautions_pending_outcomes" in result.stdout
    assert "Can create analyst research recommendation: True" in result.stdout
    assert "Can create execution recommendation: False" in result.stdout
    assert "Can write learning memory: False" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
