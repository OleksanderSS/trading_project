from __future__ import annotations

import json

import pandas as pd

from dean_os.verified_market_source_router import VerifiedMarketSourceRouter


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifacts(tmp_path):
    lifecycle = tmp_path / "lifecycle.json"
    registration = tmp_path / "registration.json"
    gate = tmp_path / "gate.json"
    refresh = tmp_path / "refresh.json"
    policy = tmp_path / "policy.json"
    _write(
        lifecycle,
        {
            "review_inbox": {
                "data_actions": [
                    {
                        "task_id": "task_60",
                        "hypothesis_id": "h1",
                        "due_at": "2026-07-13T20:13:54+00:00",
                    }
                ]
            }
        },
    )
    _write(
        registration,
        {
            "registration_plan": [
                {
                    "task_id": "task_60",
                    "hypothesis_id": "h1",
                    "due_at": "2026-07-13T20:13:54+00:00",
                }
            ]
        },
    )
    _write(
        gate,
        {
            "hypothesis_review": [
                {
                    "hypothesis_id": "h1",
                    "measurement_spec": {
                        "target_metrics": [
                            "amat_or_equipment_basket_relative_total_return"
                        ]
                    },
                }
            ]
        },
    )
    _write(
        refresh,
        {
            "inputs": {"apply_refresh": True},
            "refresh_jobs": [
                {"task_id": "task_60", "provider": "yahoo_finance"}
            ],
        },
    )
    _write(
        policy,
        {
            "contract": "dean_verified_market_source_policy_v1",
            "providers": [
                {
                    "provider_id": "yahoo_finance",
                    "rank": 1,
                    "maximum_attempts_per_task": 1,
                    "automatic_execution_allowed": True,
                },
                {
                    "provider_id": "local_validated_snapshot",
                    "rank": 2,
                    "maximum_attempts_per_task": 1,
                    "automatic_execution_allowed": False,
                },
            ],
            "failover_policy": {"automatic_multi_provider_loop_allowed": False},
        },
    )
    return lifecycle, registration, gate, refresh, policy


def _build(tmp_path, *, local=None):
    lifecycle, registration, gate, refresh, policy = _artifacts(tmp_path)
    return VerifiedMarketSourceRouter(tmp_path / "out").build(
        lifecycle_json=lifecycle,
        registration_json=registration,
        review_gate_json=gate,
        source_policy_json=policy,
        previous_refresh_json_paths=[refresh],
        local_snapshot_paths=[local] if local else [],
        as_of="2026-07-14T21:00:00+00:00",
        save=False,
    )


def test_failed_network_provider_routes_to_bounded_local_snapshot(tmp_path):
    payload = _build(tmp_path)

    assert payload["summary"]["status"] == (
        "awaiting_operator_supplied_verified_snapshot"
    )
    route = payload["routes"][0]
    assert route["selected_provider"]["provider_id"] == (
        "local_validated_snapshot"
    )
    assert route["automatic_failover_executed"] is False
    assert payload["next_system_actions"][0]["automatic_execution_allowed"] is False


def test_valid_local_snapshot_is_ready_for_existing_lifecycle(tmp_path):
    snapshot = tmp_path / "snapshot.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-07-14T00:00:00+00:00"],
            "ticker": ["AMAT"],
            "close": [200.0],
        }
    ).to_csv(snapshot, index=False)

    payload = _build(tmp_path, local=snapshot)

    assert payload["summary"]["status"] == "verified_local_snapshot_ready"
    validation = payload["routes"][0]["local_snapshot_validation"]
    assert validation["valid"] is True
    assert validation["eligible_sessions"] == ["2026-07-14"]
    assert payload["next_system_actions"][0]["automatic_execution_allowed"] is True


def test_pre_due_or_incomplete_local_snapshot_is_rejected(tmp_path):
    snapshot = tmp_path / "snapshot.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-07-13T00:00:00+00:00"],
            "ticker": ["OTHER"],
            "close": [200.0],
        }
    ).to_csv(snapshot, index=False)

    payload = _build(tmp_path, local=snapshot)

    assert payload["summary"]["status"] == "local_snapshot_rejected"
    issues = payload["routes"][0]["local_snapshot_validation"]["candidates"][0][
        "issues"
    ]
    assert "missing_tickers:AMAT" in issues
    assert "no_complete_post_due_closed_session" in issues
