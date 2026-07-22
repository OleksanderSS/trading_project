from __future__ import annotations

import json

import pandas as pd

from dean_os.hypothesis_measurement_policy_preparer import (
    HypothesisMeasurementPolicyPreparer,
)


def _write_draft(path, *, direction="negative"):
    payload = {
        "contract": "dean_world_model_hypothesis_resolution_specs_v2",
        "source_packet": {"run_id": "packet", "sha256": "packet_hash"},
        "source_review_gate": {"run_id": "gate", "sha256": "gate_hash"},
        "resolutions": {
            "h1": {
                "measurement_spec": {
                    "primary_horizon_days": 20,
                    "target_metrics": ["basket_relative_total_return"],
                    "relative_return_expected_direction": direction,
                    "measurement_context": {
                        "trigger_event_at": "2026-01-01T15:00:00+00:00",
                        "capital_equipment_basket": {
                            "members": ["A", "B", "C"],
                            "benchmark": "BM",
                        },
                    },
                },
                "registration_blockers": [],
            },
            "h2": {
                "measurement_spec": {
                    "primary_horizon_days": 20,
                    "target_metrics": ["consensus_revision"],
                },
                "registration_blockers": [],
            },
        },
        "safety": {"can_trade": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prices(path):
    sessions = pd.bdate_range("2024-01-02", "2025-12-31", tz="UTC")
    rows = []
    for index, session in enumerate(sessions):
        benchmark = 100 * (1.0003 ** index)
        cycle = ((index % 29) - 14) / 1000
        for offset, ticker in enumerate(("A", "B", "C")):
            rows.append(
                {
                    "datetime": session,
                    "ticker": ticker,
                    "close": benchmark * (1 + cycle * (offset + 1) / 3),
                }
            )
        rows.append({"datetime": session, "ticker": "BM", "close": benchmark})
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_preparer_attaches_contract_without_mutating_source(tmp_path):
    draft = tmp_path / "draft.json"
    prices = tmp_path / "pipeline_features.parquet"
    _write_draft(draft)
    _prices(prices)
    before = draft.read_bytes()

    payload = HypothesisMeasurementPolicyPreparer(tmp_path / "out").build(
        draft,
        pipeline_paths=[prices],
        price_paths=[],
        save=False,
    )

    assert draft.read_bytes() == before
    measurement = payload["resolutions"]["h1"]["measurement_spec"]
    contract = measurement["relative_return_direction_contract"]
    assert contract["expected_direction"] == "negative"
    assert contract["neutral_band_absolute_return"] > 0
    summary = payload["measurement_policy_preparation"]["summary"]
    assert summary["relative_return_contract_ready_count"] == 1
    assert summary["blocked_hypothesis_count"] == 0
    assert payload["resolutions"]["h2"]["measurement_spec"].get(
        "relative_return_direction_contract"
    ) is None


def test_preparer_blocks_missing_direction_instead_of_guessing(tmp_path):
    draft = tmp_path / "draft.json"
    prices = tmp_path / "prices.parquet"
    _write_draft(draft, direction="")
    _prices(prices)

    payload = HypothesisMeasurementPolicyPreparer(tmp_path / "out").build(
        draft,
        price_paths=[prices],
        pipeline_paths=[],
        save=False,
    )

    spec = payload["resolutions"]["h1"]
    assert "relative_return_expected_direction_missing" in spec[
        "registration_blockers"
    ]
    assert "relative_return_direction_contract" not in spec["measurement_spec"]
    assert payload["measurement_policy_preparation"]["summary"][
        "blocked_hypothesis_count"
    ] == 1
