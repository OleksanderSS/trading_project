from __future__ import annotations

import pandas as pd

from dean_os.learning import LearningStore
from dean_os.outcome_readiness_gate import OutcomeReadinessGate
from dean_os.schemas import AgentLearningRecord


def _seed_record(tmp_path, *, created_at="2026-01-01T00:00:00+00:00", horizon_days=30) -> None:
    LearningStore(tmp_path / "learning.sqlite").add_record(
        AgentLearningRecord(
            record_id="record_1",
            agent_name="specialist_research",
            note_id="note_1",
            expected_direction="bullish",
            horizon_days=horizon_days,
            created_at=created_at,
            metadata={
                "analyst_learning_bridge": True,
                "profile": "generalist_base_analyst",
                "tickers": ["AMD"],
                "context_tags": ["ai_cycle"],
            },
        )
    )


def _prices(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_outcome_readiness_gate_reports_ready_for_dry_run(tmp_path):
    _seed_record(tmp_path)
    prices = _prices(
        tmp_path / "prices.csv",
        [
            {"datetime": "2026-01-01T00:00:00+00:00", "ticker": "AMD", "close": 100.0},
            {"datetime": "2026-02-02T00:00:00+00:00", "ticker": "AMD", "close": 110.0},
        ],
    )

    payload = OutcomeReadinessGate(tmp_path / "reports").build(
        learning_path=tmp_path / "learning.sqlite",
        market_data_path=prices,
        as_of="2026-02-02T00:00:00+00:00",
        save=False,
    )

    assert payload["summary"]["readiness_status"] == "ready_for_outcome_dry_run"
    assert payload["summary"]["evaluable_count"] == 1
    assert payload["commands"]["outcome_dry_run"]
    assert payload["commands"]["outcome_apply_after_dry_run_review"]
    assert LearningStore(tmp_path / "learning.sqlite").get_record("record_1").outcome_label is None


def test_outcome_readiness_gate_waits_for_horizon(tmp_path):
    _seed_record(tmp_path, horizon_days=60)
    prices = _prices(
        tmp_path / "prices.csv",
        [
            {"datetime": "2026-01-01T00:00:00+00:00", "ticker": "AMD", "close": 100.0},
            {"datetime": "2026-01-15T00:00:00+00:00", "ticker": "AMD", "close": 105.0},
        ],
    )

    payload = OutcomeReadinessGate(tmp_path / "reports").build(
        learning_path=tmp_path / "learning.sqlite",
        market_data_path=prices,
        as_of="2026-01-15T00:00:00+00:00",
        save=False,
    )

    assert payload["summary"]["readiness_status"] == "waiting_for_horizon"
    assert payload["summary"]["can_run_outcome_dry_run"] is False
    assert payload["commands"]["outcome_apply_after_dry_run_review"] is None


def test_outcome_readiness_gate_blocks_when_prices_are_not_newer(tmp_path):
    _seed_record(tmp_path, created_at="2026-02-01T00:00:00+00:00")
    prices = _prices(
        tmp_path / "prices.csv",
        [{"datetime": "2026-01-15T00:00:00+00:00", "ticker": "AMD", "close": 100.0}],
    )

    payload = OutcomeReadinessGate(tmp_path / "reports").build(
        learning_path=tmp_path / "learning.sqlite",
        market_data_path=prices,
        save=False,
    )

    assert payload["summary"]["readiness_status"] == "blocked_need_newer_prices"
    assert payload["summary"]["status_counts"]["no_price_after_created_at"] == 1


def test_outcome_readiness_gate_handles_missing_market_data(tmp_path):
    _seed_record(tmp_path)

    payload = OutcomeReadinessGate(tmp_path / "reports").build(
        learning_path=tmp_path / "learning.sqlite",
        market_data_path=tmp_path / "missing.csv",
        save=False,
    )

    assert payload["summary"]["readiness_status"] == "blocked_missing_market_data"
    assert payload["summary"]["pending_record_count"] == 1
    assert payload["summary"]["outcome_write_performed"] is False
