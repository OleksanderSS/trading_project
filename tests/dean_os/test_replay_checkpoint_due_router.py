from __future__ import annotations

import json
import hashlib

import pandas as pd

from dean_os.replays.replay_checkpoint_due_router import ReplayCheckpointDueRouter


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _inputs(tmp_path, due_at="2026-07-15T07:53:06+00:00"):
    registration = tmp_path / "registration.json"
    gate = tmp_path / "gate.json"
    _write(
        registration,
        {
            "contract": "dean_world_model_replay_registration_bridge_v1",
            "registration_plan": [
                {
                    "task_id": "task_20",
                    "hypothesis_id": "h1",
                    "horizon_days": 20,
                    "due_at": due_at,
                }
            ],
            "deferred_historical_tasks": [],
        },
    )
    _write(
        gate,
        {
            "hypothesis_review": [
                {
                    "hypothesis_id": "h1",
                    "measurement_spec": {
                        "primary_horizon_days": 20,
                        "target_metrics": [
                            "capital_equipment_basket_relative_total_return"
                        ],
                        "measurement_context": {
                            "capital_equipment_basket": {
                                "members": ["AMAT", "LRCX", "KLAC", "ASML"],
                                "minimum_checkpoint_coverage": 3,
                                "benchmark": "SOXX",
                            }
                        },
                    },
                }
            ]
        },
    )
    return registration, gate


def _prices(path, session="2026-07-15T00:00:00+00:00"):
    pd.DataFrame(
        {
            "datetime": [session] * 4,
            "ticker": ["AMAT", "LRCX", "KLAC", "SOXX"],
            "close": [100.0, 100.0, 100.0, 100.0],
        }
    ).to_csv(path, index=False)


def test_future_checkpoint_is_silent_even_when_due_soon(tmp_path):
    registration, gate = _inputs(tmp_path)
    prices = tmp_path / "prices.csv"
    _prices(prices)

    payload = ReplayCheckpointDueRouter(tmp_path / "out").build(
        registration,
        gate,
        as_of="2026-07-13T21:00:00+00:00",
        verified_price_paths=[prices],
        save=False,
    )

    assert payload["routes"][0]["route_state"] == "future_silent"
    assert payload["routes"][0]["due_soon"] is True
    assert payload["chief_review_inbox"]["pending_decisions"] == []
    assert payload["summary"]["due_soon_silent_count"] == 1


def test_due_checkpoint_waits_until_verified_session_is_available(tmp_path):
    registration, gate = _inputs(tmp_path)
    old_prices = tmp_path / "old_prices.csv"
    _prices(old_prices, "2026-07-14T00:00:00+00:00")

    payload = ReplayCheckpointDueRouter(tmp_path / "out").build(
        registration,
        gate,
        as_of="2026-07-15T21:00:00+00:00",
        verified_price_paths=[old_prices],
        save=False,
    )

    assert payload["routes"][0]["route_state"] == (
        "due_waiting_for_verified_checkpoint_data"
    )
    assert payload["summary"]["operator_decision_count"] == 0
    assert len(payload["chief_review_inbox"]["data_accrual_actions"]) == 1


def test_due_checkpoint_routes_after_verified_session_close(tmp_path):
    registration, gate = _inputs(tmp_path)
    prices = tmp_path / "prices.csv"
    _prices(prices)

    before_close = ReplayCheckpointDueRouter(tmp_path / "out1").build(
        registration,
        gate,
        as_of="2026-07-15T19:59:00+00:00",
        verified_price_paths=[prices],
        save=False,
    )
    after_close = ReplayCheckpointDueRouter(tmp_path / "out2").build(
        registration,
        gate,
        as_of="2026-07-15T20:01:00+00:00",
        verified_price_paths=[prices],
        save=False,
    )

    assert before_close["routes"][0]["route_state"] == (
        "due_waiting_for_verified_checkpoint_data"
    )
    assert after_close["routes"][0]["route_state"] == (
        "matured_pending_outcome_review"
    )
    assert after_close["routes"][0]["checkpoint_data"]["checkpoint_session"] == (
        "2026-07-15"
    )
    assert len(after_close["chief_review_inbox"]["pending_decisions"]) == 1


def test_existing_review_does_not_reappear(tmp_path):
    registration, gate = _inputs(tmp_path)
    prices = tmp_path / "prices.csv"
    outcome = tmp_path / "outcome.json"
    _prices(prices)
    _write(
        outcome,
        {
            "contract": "dean_historical_replay_outcome_review_v1",
            "inputs": {
                "registration": {
                    "sha256": hashlib.sha256(registration.read_bytes()).hexdigest()
                }
            },
            "checkpoint_reviews": [
                {
                    "task_id": "task_20",
                    "review_status": "primary_outcome_observed",
                    "result_label": "support",
                }
            ]
        },
    )

    payload = ReplayCheckpointDueRouter(tmp_path / "out").build(
        registration,
        gate,
        as_of="2026-07-16T21:00:00+00:00",
        verified_price_paths=[prices],
        outcome_json_paths=[outcome],
        save=False,
    )

    assert payload["routes"][0]["route_state"] == "reviewed_support"
    assert payload["summary"]["reviewed_checkpoint_count"] == 1
    assert payload["chief_review_inbox"]["pending_decisions"] == []


def test_outcome_from_different_registration_cannot_suppress_task(tmp_path):
    registration, gate = _inputs(tmp_path)
    prices = tmp_path / "prices.csv"
    outcome = tmp_path / "outcome.json"
    _prices(prices)
    _write(
        outcome,
        {
            "inputs": {"registration": {"sha256": "wrong"}},
            "checkpoint_reviews": [
                {"task_id": "task_20", "result_label": "support"}
            ],
        },
    )

    payload = ReplayCheckpointDueRouter(tmp_path / "out").build(
        registration,
        gate,
        as_of="2026-07-16T21:00:00+00:00",
        verified_price_paths=[prices],
        outcome_json_paths=[outcome],
        save=False,
    )

    assert payload["routes"][0]["route_state"] == (
        "matured_pending_outcome_review"
    )
    assert payload["outcome_review_inventory"][0]["status"] == (
        "registration_lineage_mismatch"
    )
