from __future__ import annotations

import hashlib
import json

import pandas as pd

from dean_os.historical_replay_outcome_review import HistoricalReplayOutcomeReview


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_primary_matured_checkpoint_is_unobservable_not_falsified(tmp_path):
    gate_path = tmp_path / "gate.json"
    registration_path = tmp_path / "registration.json"
    prices = tmp_path / "prices.csv"
    pipeline = tmp_path / "pipeline_features.parquet"
    prices.write_text(
        "datetime,ticker,close\n2026-05-14T00:00:00+00:00,NVDA,100\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "datetime": ["2026-05-14T00:00:00+00:00"],
            "ticker": ["AMAT"],
            "returns_1d": [0.01],
            "news_impact_score_1d": [0.2],
        }
    ).to_parquet(pipeline, index=False)
    gate = {
        "run_id": "gate_1",
        "hypothesis_review": [
            {
                "hypothesis_id": "h1",
                "hypothesis": "AMAT demand persists",
                "measurement_spec": {
                    "primary_horizon_days": 20,
                    "target_metrics": [
                        "amat_consensus_sales_revision",
                        "amat_or_equipment_basket_relative_total_return",
                    ],
                },
            }
        ],
    }
    _write(gate_path, gate)
    registration = {
        "contract": "dean_world_model_replay_registration_bridge_v1",
        "source_gate": {"run_id": "gate_1", "sha256": _sha(gate_path)},
        "deferred_historical_tasks": [
            {"task_id": "task_20", "due_at": "2026-06-03T00:00:00+00:00"}
        ],
        "registration_plan": [
            {
                "task_id": "task_20",
                "hypothesis_id": "h1",
                "horizon_days": 20,
                "due_at": "2026-06-03T00:00:00+00:00",
            }
        ],
    }
    _write(registration_path, registration)

    payload = HistoricalReplayOutcomeReview(tmp_path / "out").build(
        review_gate_json=gate_path,
        registration_json=registration_path,
        price_paths=[prices],
        pipeline_paths=[pipeline],
        save=False,
    )

    assert payload["summary"]["primary_outcome_count"] == 1
    outcome = payload["outcomes"][0]
    assert outcome["result_label"] == "unobservable"
    assert outcome["observable"] is False
    assert "point_in_time_consensus_estimate_baseline_and_checkpoint" in (
        outcome["missing_point_in_time_evidence"]
    )
    assert payload["summary"]["outcome_scoring_performed"] is False
    assert payload["safety"]["can_trade"] is False
    context = payload["checkpoint_reviews"][0]["pipeline_context"]
    assert context["status"] == "partial_target_universe_secondary_context_only"
    assert context["can_replace_missing_primary_outcome_evidence"] is False


def test_predeclared_basket_price_window_is_calculated_without_scoring(tmp_path):
    gate_path = tmp_path / "gate.json"
    registration_path = tmp_path / "registration.json"
    prices = tmp_path / "prices.csv"
    pd.DataFrame(
        [
            {"datetime": date, "ticker": ticker, "close": close}
            for ticker, before, after in (
                ("AMAT", 100, 98),
                ("LRCX", 100, 97),
                ("KLAC", 100, 99),
                ("ASML", 100, 96),
                ("SOXX", 100, 99),
            )
            for date, close in (
                ("2026-06-24T00:00:00+00:00", before),
                ("2026-06-26T00:00:00+00:00", after),
            )
        ]
    ).to_csv(prices, index=False)
    gate = {
        "run_id": "gate_2",
        "hypothesis_review": [
            {
                "hypothesis_id": "h2",
                "measurement_spec": {
                    "primary_horizon_days": 20,
                    "target_metrics": [
                        "median_public_company_capex_plan_revision_pct",
                        "capital_equipment_basket_relative_total_return",
                    ],
                    "measurement_context": {
                        "capital_equipment_basket": {
                            "members": ["AMAT", "LRCX", "KLAC", "ASML"],
                            "minimum_checkpoint_coverage": 3,
                            "benchmark": "SOXX",
                        }
                    },
                    "relative_return_direction_contract": {
                        "contract": "dean_relative_return_direction_contract_v1",
                        "status": "calibrated_pre_outcome_direction_contract",
                        "expected_direction": "negative",
                        "horizon_days": 20,
                        "neutral_band_absolute_return": 0.01,
                        "blockers": [],
                    },
                },
            }
        ],
    }
    _write(gate_path, gate)
    registration = {
        "contract": "dean_world_model_replay_registration_bridge_v1",
        "source_gate": {"run_id": "gate_2", "sha256": _sha(gate_path)},
        "deferred_historical_tasks": [
            {"task_id": "task_1", "due_at": "2026-06-26T07:53:06+00:00"}
        ],
        "registration_plan": [
            {
                "task_id": "task_1",
                "hypothesis_id": "h2",
                "event_anchor_at": "2026-06-25T07:53:06+00:00",
                "horizon_days": 1,
                "due_at": "2026-06-26T07:53:06+00:00",
            }
        ],
    }
    _write(registration_path, registration)

    payload = HistoricalReplayOutcomeReview(tmp_path / "out").build(
        review_gate_json=gate_path,
        registration_json=registration_path,
        price_paths=[prices],
        pipeline_paths=[],
        save=False,
    )

    observation = payload["checkpoint_reviews"][0]["price_observation"]
    assert observation["status"] == "checkpoint_price_window_observed"
    assert observation["ticker_windows"][0]["baseline_session"] == "2026-06-24"
    assert observation["ticker_windows"][0]["checkpoint_session"] == "2026-06-26"
    relative = observation["relative_return_observation"]
    assert relative["relative_price_return"] < 0
    assert relative["relative_total_return"] == (
        (1 + relative["basket_price_return"])
        / (1 + relative["benchmark_price_return"])
        - 1
    )
    assert relative["active_return_spread_percentage_points"] != (
        relative["relative_total_return"]
    )
    assert relative["claim_relation"]["classification"] == "support"
    assert observation["automatic_hypothesis_scoring_allowed"] is False
