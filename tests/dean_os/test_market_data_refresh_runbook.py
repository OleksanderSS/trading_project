from __future__ import annotations

import json

from dean_os.market_data_refresh_runbook import LIVE_COLLECTOR_INVENTORY, MarketDataRefreshRunbook


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _readiness_artifact(path):
    return _write_json(
        path,
        {
            "mode": "outcome_readiness_gate",
            "inputs": {
                "learning_path": "learning.sqlite",
                "memory_path": "memory.sqlite",
                "market_data_path": "old_prices.parquet",
                "tickers": ["AAPL", "AMD"],
                "close_col": "close",
                "datetime_col": "datetime",
                "neutral_band": 0.01,
            },
        },
    )


def _coverage_plan(path, readiness_path, *, status="needs_price_refresh_after_record_creation"):
    return _write_json(
        path,
        {
            "mode": "outcome_price_coverage_plan",
            "inputs": {
                "readiness_path": readiness_path,
                "market_data_path": "old_prices.parquet",
                "latest_processed_prices": "1d",
                "tickers": ["AAPL", "AMD"],
                "close_col": "close",
                "datetime_col": "datetime",
            },
            "summary": {
                "plan_status": status,
                "tickers_need_price_after_creation": ["AAPL", "AMD"] if status == "needs_price_refresh_after_record_creation" else [],
                "missing_tickers": [],
                "minimum_price_after_created_at": "2026-06-13T07:06:38+00:00",
                "earliest_due_at": "2027-06-13T07:06:38+00:00",
                "latest_due_at": "2027-06-13T07:06:38+00:00",
            },
            "ticker_coverage": [
                {"ticker": "AAPL", "status": status},
                {"ticker": "AMD", "status": status},
            ],
            "commands": {"rebuild_this_plan": "python run_agent_outcome_price_coverage.py --readiness-json readiness.json"},
        },
    )


def _collector_inventory(path, *, enabled=True, class_found=True):
    return _write_json(
        path,
        {
            "mode": "collector_inventory_agent",
            "collector_inventory": {
                "summary": {"status": "ok", "enabled_missing_classes": []},
                "configured_collectors": [
                    {
                        "name": "yahoo_finance",
                        "type": "yahoo_finance",
                        "enabled": enabled,
                        "critical": True,
                        "class_found": class_found,
                        "class_name": "YFCollector",
                        "module_path": "src/data/collectors/yf_collector.py",
                        "table_name": "market_data_raw",
                        "cache_ttl": 900,
                        "cache_duration_minutes": 15,
                        "recommended_use": "pipeline_price_feed",
                        "schedule_hint": "pipeline: per approved market-data refresh and timeframe cadence",
                        "notes": [],
                    }
                ],
            },
        },
    )


def test_market_data_refresh_runbook_ready_with_enabled_price_feed(tmp_path):
    readiness = _readiness_artifact(tmp_path / "readiness.json")
    coverage = _coverage_plan(tmp_path / "coverage.json", readiness)
    inventory = _collector_inventory(tmp_path / "inventory.json")

    payload = MarketDataRefreshRunbook(tmp_path / "reports").build(
        coverage_plan_path=coverage,
        collector_inventory_path=inventory,
        price_globs=[],
        save=False,
    )

    assert payload["summary"]["runbook_status"] == "refresh_runbook_ready"
    assert payload["summary"]["primary_price_feed"] == "yahoo_finance"
    assert payload["summary"]["can_refresh_automatically"] is False
    assert payload["summary"]["collector_run_performed"] is False
    task_ids = {task["task_id"] for task in payload["operator_tasks"]}
    assert "produce_refreshed_price_artifact" in task_ids
    assert "validate_refreshed_artifact" in task_ids
    assert "PATH_TO_REFRESHED_PRICE_FILE" in payload["commands"]["rerun_outcome_readiness_after_refresh"]


def test_market_data_refresh_runbook_blocks_without_coverage_plan(tmp_path):
    payload = MarketDataRefreshRunbook(tmp_path / "reports").build(
        coverage_plan_path=tmp_path / "missing.json",
        collector_inventory_path=None,
        price_globs=[],
        save=False,
    )

    assert payload["summary"]["runbook_status"] == "blocked_no_coverage_plan"
    assert payload["operator_tasks"][0]["task_id"] == "build_outcome_price_coverage_plan"


def test_market_data_refresh_runbook_requests_inventory_when_missing(tmp_path):
    readiness = _readiness_artifact(tmp_path / "readiness.json")
    coverage = _coverage_plan(tmp_path / "coverage.json", readiness)

    payload = MarketDataRefreshRunbook(tmp_path / "reports").build(
        coverage_plan_path=coverage,
        collector_inventory_path=tmp_path / "missing_inventory.json",
        price_globs=[],
        save=False,
    )

    assert payload["summary"]["runbook_status"] == "blocked_missing_collector_inventory"
    task_ids = {task["task_id"] for task in payload["operator_tasks"]}
    assert "build_collector_inventory" in task_ids
    assert "produce_refreshed_price_artifact" in task_ids


def test_live_inventory_reads_current_collector_config(tmp_path):
    """The default path must scan collectors.yaml, not a frozen snapshot.

    The snapshot producer (CollectorInventoryAgent) was archived on 2026-07-24, so a
    saved artifact reports whatever the config looked like when it was last written.
    """
    readiness = _readiness_artifact(tmp_path / "readiness.json")
    coverage = _coverage_plan(tmp_path / "coverage.json", readiness)
    config = tmp_path / "collectors.yaml"
    config.write_text(
        "collectors:\n"
        "  yahoo_finance:\n"
        "    type: yahoo_finance\n"
        "    enabled: true\n"
        "    critical: true\n",
        encoding="utf-8",
    )

    collectors_dir = tmp_path / "collectors"
    collectors_dir.mkdir()
    (collectors_dir / "yf_collector.py").write_text(
        "class YFCollector:\n"
        "    collector_type = 'yahoo_finance'\n"
        "    data_type = 'market_data'\n",
        encoding="utf-8",
    )

    payload = MarketDataRefreshRunbook(tmp_path / "reports").build(
        coverage_plan_path=coverage,
        collector_inventory_path=LIVE_COLLECTOR_INVENTORY,
        price_globs=[],
        save=False,
        collector_config_path=config,
        collectors_dir=collectors_dir,
    )

    assert payload["inputs"]["collector_inventory_source"] == "live"
    assert payload["inputs"]["collector_inventory_as_of"] is not None
    assert payload["summary"]["collector_inventory_status"] == "ok"
    assert payload["summary"]["primary_price_feed"] == "yahoo_finance"
    assert payload["inputs"]["collector_inventory_config_path"] == str(config)


def test_live_inventory_blocks_on_unreadable_collector_config(tmp_path):
    readiness = _readiness_artifact(tmp_path / "readiness.json")
    coverage = _coverage_plan(tmp_path / "coverage.json", readiness)

    payload = MarketDataRefreshRunbook(tmp_path / "reports").build(
        coverage_plan_path=coverage,
        collector_inventory_path=LIVE_COLLECTOR_INVENTORY,
        price_globs=[],
        save=False,
        collector_config_path=tmp_path / "no_such_collectors.yaml",
        collectors_dir=tmp_path / "no_such_dir",
    )

    assert payload["summary"]["runbook_status"] == "blocked_unreadable_collector_config"
    assert payload["validation"]["can_plan"] is False


def test_snapshot_inventory_is_never_reported_as_current(tmp_path):
    readiness = _readiness_artifact(tmp_path / "readiness.json")
    coverage = _coverage_plan(tmp_path / "coverage.json", readiness)
    inventory = _collector_inventory(tmp_path / "inventory.json")

    payload = MarketDataRefreshRunbook(tmp_path / "reports").build(
        coverage_plan_path=coverage,
        collector_inventory_path=inventory,
        price_globs=[],
        save=False,
    )

    assert payload["inputs"]["collector_inventory_source"] == "snapshot"
    assert payload["inputs"]["collector_inventory_as_of"] is not None
    assert payload["summary"]["collector_inventory_status"] == "snapshot_ok"


def test_runbook_no_longer_points_at_the_archived_inventory_agent(tmp_path):
    readiness = _readiness_artifact(tmp_path / "readiness.json")
    coverage = _coverage_plan(tmp_path / "coverage.json", readiness)
    inventory = _collector_inventory(tmp_path / "inventory.json")

    payload = MarketDataRefreshRunbook(tmp_path / "reports").build(
        coverage_plan_path=coverage,
        collector_inventory_path=inventory,
        price_globs=[],
        save=False,
    )

    commands = " ".join(str(value) for value in payload["commands"].values())
    tasks = json.dumps(payload["operator_tasks"])
    assert "run_agent_collector_inventory.py" not in commands
    assert "run_agent_collector_inventory.py" not in tasks


def test_market_data_refresh_runbook_rechecks_when_coverage_ready(tmp_path):
    readiness = _readiness_artifact(tmp_path / "readiness.json")
    coverage = _coverage_plan(tmp_path / "coverage.json", readiness, status="coverage_ready_for_outcome_readiness_rerun")
    inventory = _collector_inventory(tmp_path / "inventory.json")

    payload = MarketDataRefreshRunbook(tmp_path / "reports").build(
        coverage_plan_path=coverage,
        collector_inventory_path=inventory,
        price_globs=[],
        save=False,
    )

    assert payload["summary"]["runbook_status"] == "no_refresh_required_recheck_readiness"
    assert payload["summary"]["can_run_outcome_readiness_now"] is True
    assert payload["operator_tasks"][0]["task_id"] == "schedule_horizon_coverage_monitoring"
