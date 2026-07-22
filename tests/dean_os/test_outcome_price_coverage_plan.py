from __future__ import annotations

import json

from dean_os.outcome_price_coverage_plan import OutcomePriceCoveragePlan


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _write_prices(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["datetime,ticker,close"]
    lines.extend(f"{row['datetime']},{row['ticker']},{row['close']}" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _readiness_artifact(path, market_data_path):
    return _write_json(
        path,
        {
            "mode": "outcome_readiness_gate",
            "inputs": {
                "learning_path": "learning.sqlite",
                "memory_path": "memory.sqlite",
                "market_data_path": market_data_path,
                "latest_processed_prices": "1d",
                "tickers": ["AAPL", "AMD"],
                "close_col": "close",
                "datetime_col": "datetime",
                "neutral_band": 0.01,
            },
            "summary": {
                "readiness_status": "blocked_need_newer_prices",
                "pending_record_count": 1,
                "status_counts": {"no_price_after_created_at": 1},
            },
            "pending_records": [
                {
                    "record_id": "record_1",
                    "agent_name": "specialist_research",
                    "expected_direction": "bullish",
                    "horizon_days": 365,
                    "created_at": "2026-06-13T07:06:38+00:00",
                    "profile": "generalist_base_analyst",
                    "tickers": ["AAPL", "AMD"],
                    "context_tags": ["ai_cycle"],
                }
            ],
            "dry_run_outcome_evaluation": {
                "as_of": "2026-05-04T15:45:00+00:00",
                "evaluations": [
                    {
                        "record_id": "record_1",
                        "status": "no_price_after_created_at",
                        "created_at": "2026-06-13T07:06:38+00:00",
                        "horizon_days": 365,
                        "due_at": "2027-06-13T07:06:38+00:00",
                        "latest_price_at": "2026-05-04T15:45:00+00:00",
                        "tickers": ["AAPL", "AMD"],
                    }
                ],
            },
        },
    )


def test_outcome_price_plan_requests_refresh_after_learning_creation(tmp_path):
    prices = _write_prices(
        tmp_path / "prices.csv",
        [
            {"datetime": "2026-05-04T15:45:00+00:00", "ticker": "AAPL", "close": 100.0},
            {"datetime": "2026-05-04T15:45:00+00:00", "ticker": "AMD", "close": 80.0},
        ],
    )
    readiness = _readiness_artifact(tmp_path / "readiness.json", prices)

    payload = OutcomePriceCoveragePlan(tmp_path / "reports").build(readiness_path=readiness, save=False)

    assert payload["summary"]["plan_status"] == "needs_price_refresh_after_record_creation"
    assert payload["summary"]["tickers_need_price_after_creation"] == ["AAPL", "AMD"]
    task_ids = {task["task_id"] for task in payload["coverage_tasks"]}
    assert "refresh_prices_after_learning_creation" in task_ids
    assert "rerun_outcome_readiness_after_price_refresh" in task_ids
    assert "run_agent_market_freshness.py" in payload["commands"]["inspect_current_market_freshness"]


def test_outcome_price_plan_waits_when_prices_are_after_creation_but_before_due(tmp_path):
    prices = _write_prices(
        tmp_path / "prices.csv",
        [
            {"datetime": "2026-06-14T15:45:00+00:00", "ticker": "AAPL", "close": 101.0},
            {"datetime": "2026-06-14T15:45:00+00:00", "ticker": "AMD", "close": 81.0},
        ],
    )
    readiness = _readiness_artifact(tmp_path / "readiness.json", prices)

    payload = OutcomePriceCoveragePlan(tmp_path / "reports").build(readiness_path=readiness, save=False)

    assert payload["summary"]["plan_status"] == "waiting_for_outcome_horizon"
    assert payload["summary"]["can_run_outcome_readiness_now"] is True
    assert payload["summary"]["tickers_waiting_for_horizon"] == ["AAPL", "AMD"]


def test_outcome_price_plan_marks_coverage_ready_after_due_prices(tmp_path):
    prices = _write_prices(
        tmp_path / "prices.csv",
        [
            {"datetime": "2027-06-14T15:45:00+00:00", "ticker": "AAPL", "close": 130.0},
            {"datetime": "2027-06-14T15:45:00+00:00", "ticker": "AMD", "close": 110.0},
        ],
    )
    readiness = _readiness_artifact(tmp_path / "readiness.json", prices)

    payload = OutcomePriceCoveragePlan(tmp_path / "reports").build(readiness_path=readiness, save=False)

    assert payload["summary"]["plan_status"] == "coverage_ready_for_outcome_readiness_rerun"
    assert all(item["status"] == "ready_for_outcome_check" for item in payload["ticker_coverage"])


def test_outcome_price_plan_blocks_without_readiness_artifact(tmp_path):
    payload = OutcomePriceCoveragePlan(tmp_path / "reports").build(
        readiness_path=tmp_path / "missing.json",
        save=False,
    )

    assert payload["summary"]["plan_status"] == "blocked_no_readiness_artifact"
    assert payload["summary"]["task_count"] == 1
    assert payload["coverage_tasks"][0]["task_id"] == "run_outcome_readiness_gate_first"
