from __future__ import annotations

import pandas as pd
import pytest

from dean_os.analyst_core.analyst_outcome_evaluation_loop import AnalystOutcomeEvaluationLoop
from dean_os.learning import LearningStore
from dean_os.schemas import AgentLearningRecord


def _write_prices(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        [
            {"datetime": "2026-01-01T00:00:00+00:00", "ticker": "AMD", "close": 100.0},
            {"datetime": "2026-01-15T00:00:00+00:00", "ticker": "AMD", "close": 105.0},
            {"datetime": "2026-02-02T00:00:00+00:00", "ticker": "AMD", "close": 112.0},
        ]
    ).to_csv(path, index=False)
    return path


def _seed_records(tmp_path, horizon_days: int = 30) -> None:
    store = LearningStore(tmp_path / "learning.sqlite")
    store.add_record(
        AgentLearningRecord(
            record_id="analyst_record_1",
            agent_name="specialist_research",
            note_id="note_1",
            expected_direction="bullish",
            horizon_days=horizon_days,
            created_at="2026-01-01T00:00:00+00:00",
            metadata={
                "analyst_learning_bridge": True,
                "profile": "generalist_base_analyst",
                "topic": "ai cycle",
                "tickers": ["AMD"],
                "context_tags": ["ai_cycle"],
                "regime_tags": ["rising_market"],
            },
        )
    )
    store.add_record(
        AgentLearningRecord(
            record_id="non_analyst_record_1",
            agent_name="manual_case",
            note_id="note_2",
            expected_direction="bearish",
            horizon_days=horizon_days,
            created_at="2026-01-01T00:00:00+00:00",
            metadata={"topic": "manual case", "tickers": ["AMD"]},
        )
    )


def test_analyst_outcome_loop_dry_run_filters_to_analyst_records(tmp_path):
    price_path = _write_prices(tmp_path)
    _seed_records(tmp_path)

    payload = AnalystOutcomeEvaluationLoop(tmp_path / "out").run(
        learning_path=tmp_path / "learning.sqlite",
        memory_path=tmp_path / "memory.sqlite",
        market_data_path=price_path,
        as_of="2026-02-02T00:00:00+00:00",
        save=False,
    )
    records = {record.record_id: record for record in LearningStore(tmp_path / "learning.sqlite").list_records()}

    assert payload["evaluation_gate"]["status"] == "dry_run_ready"
    assert payload["outcome_evaluation"]["pending_record_count"] == 1
    assert payload["outcome_evaluation"]["evaluable_count"] == 1
    assert records["analyst_record_1"].outcome_label is None
    assert records["non_analyst_record_1"].outcome_label is None


def test_analyst_outcome_loop_apply_updates_and_audits_analyst_record(tmp_path):
    price_path = _write_prices(tmp_path)
    _seed_records(tmp_path)

    payload = AnalystOutcomeEvaluationLoop(tmp_path / "out").run(
        learning_path=tmp_path / "learning.sqlite",
        memory_path=tmp_path / "memory.sqlite",
        market_data_path=price_path,
        as_of="2026-02-02T00:00:00+00:00",
        apply=True,
        save=False,
    )
    records = {record.record_id: record for record in LearningStore(tmp_path / "learning.sqlite").list_records()}

    assert payload["evaluation_gate"]["status"] == "applied"
    assert payload["outcome_evaluation"]["updated_count"] == 1
    assert records["analyst_record_1"].outcome_label == "hit"
    assert records["analyst_record_1"].metadata["analyst_outcome_evaluation_loop"]["run_id"] == payload["run_id"]
    assert records["non_analyst_record_1"].outcome_label is None
    assert payload["profile_outcomes"]["generalist_base_analyst"]["hit_rate"] == 1.0
    assert payload["context_performance"]["overall"]["completed_count"] == 1


def test_analyst_outcome_loop_historical_diagnostic_is_dry_run_by_default(tmp_path):
    price_path = _write_prices(tmp_path)
    _seed_records(tmp_path, horizon_days=60)

    waiting = AnalystOutcomeEvaluationLoop(tmp_path / "out").run(
        learning_path=tmp_path / "learning.sqlite",
        memory_path=tmp_path / "memory.sqlite",
        market_data_path=price_path,
        as_of="2026-01-15T00:00:00+00:00",
        save=False,
    )
    diagnostic = AnalystOutcomeEvaluationLoop(tmp_path / "out").run(
        learning_path=tmp_path / "learning.sqlite",
        memory_path=tmp_path / "memory.sqlite",
        market_data_path=price_path,
        as_of="2026-01-15T00:00:00+00:00",
        historical_diagnostic=True,
        save=False,
    )

    assert waiting["evaluation_gate"]["status"] == "waiting_for_horizon"
    assert diagnostic["evaluation_gate"]["status"] == "historical_diagnostic_ready"
    assert LearningStore(tmp_path / "learning.sqlite").get_record("analyst_record_1").outcome_label is None


def test_analyst_outcome_loop_blocks_historical_diagnostic_apply_without_override(tmp_path):
    price_path = _write_prices(tmp_path)
    _seed_records(tmp_path, horizon_days=60)

    with pytest.raises(ValueError, match="historical_diagnostic is dry-run"):
        AnalystOutcomeEvaluationLoop(tmp_path / "out").run(
            learning_path=tmp_path / "learning.sqlite",
            memory_path=tmp_path / "memory.sqlite",
            market_data_path=price_path,
            as_of="2026-01-15T00:00:00+00:00",
            historical_diagnostic=True,
            apply=True,
            save=False,
        )
