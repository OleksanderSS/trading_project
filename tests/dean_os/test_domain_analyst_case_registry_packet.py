from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dean_os.domain_analyst_case_registry_packet import DomainAnalystCaseRegistryPacket


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _thesis_review(**summary_overrides) -> dict:
    summary = {
        "packet_status": "domain_thesis_review_ready",
        "domain_id": "semiconductor_ai_infrastructure",
        "thesis_stance": "mixed",
        "expected_direction": "mixed",
        "confidence": 0.7,
        "supporting_evidence_count": 2,
        "contradicting_evidence_count": 1,
        "ticker_direct_count": 0,
        "required_evidence_missing": [],
        "manual_review_required": True,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
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
            "expected_direction": "mixed",
            "confidence": 0.7,
            "key_drivers": ["AI demand", "capex cycle"],
            "assumptions": ["summer liquidity may be different"],
        },
        "supporting_evidence_examples": [
            {
                "evidence_id": "ev_support_1",
                "source_type": "news",
                "published_at": "2026-07-10T00:00:00+00:00",
                "evidence_type": "sector_demand",
                "directness": "sector",
                "stance_hint": "positive",
                "summary": "AI demand supports sector demand.",
                "limitations": ["review_only"],
            }
        ],
        "contradicting_evidence_examples": [
            {
                "evidence_id": "ev_contra_1",
                "source_type": "report",
                "published_at": "2026-08-02T00:00:00+00:00",
                "evidence_type": "policy_or_geopolitical",
                "directness": "policy",
                "stance_hint": "negative",
                "summary": "Export controls may reduce demand.",
                "limitations": ["review_only"],
            }
        ],
    }


def _template_packet() -> dict:
    return {
        "run_id": "template_fixture",
        "mode": "domain_analyst_template_standardization_packet",
        "inputs": {
            "domain_thesis_review_run_id": (
                "thesis_review_fixture"
            )
        },
        "summary": {
            "candidate_status": "ready_for_manual_template_acceptance",
            "domain_id": "semiconductor_ai_infrastructure",
            "can_mark_template_accepted_now": False,
            "can_write_learning_memory": False,
            "can_trade": False,
        },
    }


def _forecast_review(**summary_overrides) -> dict:
    summary = {
        "packet_status": "forecast_review_ready_with_cautions_pending_outcomes",
        "domain_id": "semiconductor_ai_infrastructure",
        "forecast_candidate_count": 1,
        "can_create_analyst_research_recommendation": True,
        "can_create_analyst_self_improvement_proposal": True,
        "can_write_learning_memory": False,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }
    summary.update(summary_overrides)
    return {
        "run_id": "forecast_review_fixture",
        "mode": "domain_analyst_forecast_review_packet",
        "created_at": "2026-06-19T00:00:00+00:00",
        "summary": summary,
        "forecast_candidates": [
            {
                "expectation_id": "domain_expectation:thesis_fixture",
                "expectation_type": "domain_or_sector_thesis_expectation",
                "source_artifact_mode": "domain_analyst_thesis_review_packet",
                "source_run_id": "thesis_review_fixture",
                "domain_id": "semiconductor_ai_infrastructure",
                "thesis_id": "thesis_fixture",
                "as_of": "2026-06-19T00:00:00+00:00",
                "horizon_days": 180,
                "stance": "mixed",
                "expected_direction": "mixed",
                "confidence": 0.7,
                "key_drivers": ["AI demand", "capex cycle"],
                "assumptions": ["Sector thesis is not a ticker forecast."],
                "evidence_balance": {
                    "supporting_count": 2,
                    "contradicting_count": 1,
                    "required_evidence_missing": [],
                    "ticker_direct_count": 0,
                },
                "required_outcome_observations": [
                    "Observe whether the domain/sector outcome moved in the expected direction.",
                    "Check whether the stated key drivers actually materialized.",
                ],
                "invalidation_triggers": ["Outcome appears correct but the named causal drivers did not materialize."],
                "allowed_future_labels": [
                    "correct_for_stated_reasons",
                    "correct_but_lucky_or_wrong_reason",
                    "incorrect_forecast",
                ],
                "evaluation_scope": "domain_or_sector_level",
                "learning_use": "candidate_case_only_until_outcome_and_human_causal_review",
            }
        ],
        "outcome_taxonomy": [
            {"bucket_id": "correct_for_stated_reasons", "meaning": "Correct direction and stated reasons."},
            {"bucket_id": "correct_but_lucky_or_wrong_reason", "meaning": "Correct direction, wrong or unsupported reasons."},
            {"bucket_id": "incorrect_forecast", "meaning": "Incorrect direction."},
            {"bucket_id": "inconclusive_or_not_mature", "meaning": "Not mature or inconclusive."},
            {"bucket_id": "unfalsifiable_or_underspecified", "meaning": "Underspecified."},
            {"bucket_id": "data_unavailable", "meaning": "Data unavailable."},
        ],
    }


def _outcome_evaluation() -> dict:
    return {
        "run_id": "outcome_fixture",
        "mode": "analyst_outcome_evaluation_loop",
        "outcome_evaluation": {
            "status_counts": {"evaluable": 3},
            "evaluations": [
                {
                    "record_id": "rec_hit",
                    "agent_name": "analyst",
                    "topic": "AI cycle",
                    "expected_direction": "bullish",
                    "horizon_days": 30,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "target_at": "2026-02-01T00:00:00+00:00",
                    "tickers": ["AMD"],
                    "context_tags": ["ai_cycle"],
                    "regime_tags": ["risk_on"],
                    "status": "evaluable",
                    "realized_return": 0.12,
                    "outcome_label": "hit",
                },
                {
                    "record_id": "rec_miss",
                    "agent_name": "analyst",
                    "topic": "Export control",
                    "expected_direction": "bullish",
                    "horizon_days": 30,
                    "created_at": "2026-03-01T00:00:00+00:00",
                    "target_at": "2026-04-01T00:00:00+00:00",
                    "tickers": ["AMD"],
                    "context_tags": ["policy"],
                    "regime_tags": ["risk_off"],
                    "status": "evaluable",
                    "realized_return": -0.08,
                    "outcome_label": "miss",
                },
                {
                    "record_id": "rec_inconclusive",
                    "agent_name": "analyst",
                    "topic": "Supply chain",
                    "expected_direction": "bullish",
                    "horizon_days": 30,
                    "created_at": "2026-05-01T00:00:00+00:00",
                    "target_at": "2026-06-01T00:00:00+00:00",
                    "tickers": ["AMD"],
                    "context_tags": ["supply_chain"],
                    "regime_tags": [],
                    "status": "evaluable",
                    "realized_return": 0.003,
                    "outcome_label": "inconclusive",
                },
            ],
        },
    }


def _write_inputs(
    tmp_path: Path,
    *,
    thesis: dict | None = None,
    template: dict | None = None,
    forecast: dict | None = None,
    outcome: dict | None = None,
) -> dict[str, Path | None]:
    paths: dict[str, Path | None] = {
        "thesis": _write_json(tmp_path / "thesis" / "latest.json", thesis or _thesis_review()),
        "template": _write_json(tmp_path / "template" / "latest.json", template or _template_packet()),
        "forecast": None,
        "outcome": None,
    }
    if forecast is not None:
        paths["forecast"] = _write_json(tmp_path / "forecast" / "latest.json", forecast)
    if outcome is not None:
        paths["outcome"] = _write_json(tmp_path / "outcome" / "latest.json", outcome)
    return paths


def test_case_registry_starts_with_pending_domain_case_and_observations(tmp_path):
    paths = _write_inputs(tmp_path)

    payload = DomainAnalystCaseRegistryPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        domain_template_standardization_json=paths["template"],
        save=False,
    )

    assert payload["summary"]["registry_status"] == "case_registry_ready_pending_outcomes"
    assert payload["summary"]["outcome_bucket_counts"]["pending_domain_outcome"] == 1
    assert payload["summary"]["source_observation_count"] == 2
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_train_from_hits_only"] is False
    assert payload["summary"]["can_drop_miss_cases"] is False
    assert payload["source_observation_entries"][0]["seasonality_context"]["season_tag"] == "summer"
    assert "summer_vacation_liquidity_context" in payload["source_observation_entries"][0]["seasonality_context"]["tags"]
    case = payload["case_entries"][0]
    assert case["prospective_registration_status"] == (
        "registered_before_outcome"
    )
    assert case["monitoring_horizons_days"] == [30, 90, 180]
    assert [
        item["horizon_days"]
        for item in case["evaluation_schedule"]
    ] == [30, 90, 180]
    assert case["source_artifact_sha256"]
    assert payload["prospective_registration_contract"][
        "automatic_trading_allowed"
    ] is False


def test_case_registry_keeps_hit_miss_and_inconclusive_buckets(tmp_path):
    paths = _write_inputs(tmp_path, outcome=_outcome_evaluation())

    payload = DomainAnalystCaseRegistryPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        domain_template_standardization_json=paths["template"],
        outcome_evaluation_json=paths["outcome"],
        save=False,
    )

    assert payload["summary"]["registry_status"] == "case_registry_ready_with_outcome_buckets"
    assert payload["summary"]["outcome_bucket_counts"]["hit"] == 1
    assert payload["summary"]["outcome_bucket_counts"]["miss"] == 1
    assert payload["summary"]["outcome_bucket_counts"]["inconclusive"] == 1
    assert any(case["outcome_bucket"] == "miss" for case in payload["case_entries"])
    assert any(check["code"] == "miss_cases_retained" and check["status"] == "pass" for check in payload["review_checks"])


def test_case_registry_registers_forecast_expectation_as_pending_case(tmp_path):
    paths = _write_inputs(tmp_path, forecast=_forecast_review())

    payload = DomainAnalystCaseRegistryPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        domain_template_standardization_json=paths["template"],
        domain_forecast_review_json=paths["forecast"],
        save=False,
    )

    assert payload["summary"]["registry_status"] == "case_registry_ready_pending_outcomes"
    assert payload["summary"]["expectation_case_count"] == 1
    assert payload["summary"]["outcome_bucket_counts"]["pending_expectation_outcome"] == 1
    assert payload["summary"]["can_create_analyst_learning_recommendation"] is True
    assert payload["summary"]["can_create_analyst_self_improvement_proposal"] is True
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_create_execution_recommendation"] is False
    case = payload["case_entries"][0]
    assert case["case_type"] == "domain_thesis_expectation"
    assert case["learning_use"] == "frozen_expectation_case_only_until_outcome_and_human_causal_review"
    assert "learning_recommendation" in case["allowed_review_outputs"]
    taxonomy = {item["bucket_id"] for item in case["outcome_taxonomy"]}
    assert "correct_for_stated_reasons" in taxonomy
    assert "correct_but_lucky_or_wrong_reason" in taxonomy
    assert any(check["code"] == "forecast_taxonomy_separates_luck_vs_skill" and check["status"] == "pass" for check in payload["review_checks"])
    assert any(check["code"] == "forecast_allows_review_only_analyst_recommendations" and check["status"] == "pass" for check in payload["review_checks"])


def test_case_registry_blocks_if_thesis_review_allows_learning_write(tmp_path):
    paths = _write_inputs(tmp_path, thesis=_thesis_review(can_write_learning_memory=True))

    payload = DomainAnalystCaseRegistryPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        domain_template_standardization_json=paths["template"],
        save=False,
    )

    assert payload["summary"]["registry_status"] == "blocked_case_registry"
    assert any(check["code"] == "thesis_review_no_learning_write" and check["status"] == "fail" for check in payload["review_checks"])


def test_case_registry_blocks_stale_template_lineage(tmp_path):
    stale_template = _template_packet()
    stale_template["inputs"][
        "domain_thesis_review_run_id"
    ] = "older_thesis_review"
    paths = _write_inputs(tmp_path, template=stale_template)

    payload = DomainAnalystCaseRegistryPacket(
        tmp_path / "reports"
    ).build(
        domain_thesis_review_json=paths["thesis"],
        domain_template_standardization_json=paths["template"],
        save=False,
    )

    assert payload["summary"]["registry_status"] == (
        "blocked_case_registry"
    )
    assert any(
        check["code"]
        == "template_bound_to_current_thesis_review"
        and check["status"] == "fail"
        for check in payload["review_checks"]
    )


def test_case_registry_saves_markdown_and_cli_runs(tmp_path):
    paths = _write_inputs(tmp_path, forecast=_forecast_review(), outcome=_outcome_evaluation())

    payload = DomainAnalystCaseRegistryPacket(tmp_path / "reports").build(
        domain_thesis_review_json=paths["thesis"],
        domain_template_standardization_json=paths["template"],
        domain_forecast_review_json=paths["forecast"],
        outcome_evaluation_json=paths["outcome"],
    )
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Can train from hits only: False" in markdown
    assert "Can write learning memory: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_domain_analyst_case_registry_packet.py"),
            "--domain-thesis-review-json",
            str(paths["thesis"]),
            "--domain-template-standardization-json",
            str(paths["template"]),
            "--domain-forecast-review-json",
            str(paths["forecast"]),
            "--outcome-evaluation-json",
            str(paths["outcome"]),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Registry status: case_registry_ready_with_outcome_buckets" in result.stdout
    assert "expectations=1" in result.stdout
    assert "Can train from hits only: False" in result.stdout
    assert "Can create analyst learning recommendation: True" in result.stdout
    assert "Can create execution recommendation: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
