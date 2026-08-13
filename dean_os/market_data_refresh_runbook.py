from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_COVERAGE_PLAN_PATH = "reports/dean_os/outcome_price_coverage_plan/latest.json"
# Sentinel for collector_inventory_path: scan src/config/collectors.yaml now rather
# than read a saved snapshot. The snapshot's producer (CollectorInventoryAgent) was
# archived on 2026-07-24, so any file under reports/dean_os/collector_inventory/ is
# frozen at whatever the collector config looked like when it was last written.
LIVE_COLLECTOR_INVENTORY = "live"
DEFAULT_PRICE_GLOBS = [
    "data/processed/prices_*.parquet",
    "data/processed/prices_*.csv",
    "data/colab/**/stage2_prices_*.parquet",
    "data/dean_os/replay_prices/*.parquet",
]
REFRESHED_PRICE_PLACEHOLDER = "PATH_TO_REFRESHED_PRICE_FILE"


class MarketDataRefreshRunbook:
    """Read-only runbook for clearing market-price coverage blockers."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/market_data_refresh_runbook"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        coverage_plan_path: str | Path = DEFAULT_COVERAGE_PLAN_PATH,
        collector_inventory_path: str | Path | None = LIVE_COLLECTOR_INVENTORY,
        price_globs: list[str] | None = None,
        max_price_artifacts: int = 25,
        refreshed_price_placeholder: str = REFRESHED_PRICE_PLACEHOLDER,
        save: bool = True,
        collector_config_path: str | Path | None = None,
        collectors_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        coverage_plan = _load_optional_json(coverage_plan_path)
        readiness_artifact = _load_optional_json(coverage_plan.get("inputs", {}).get("readiness_path")) if coverage_plan else {}
        collector_inventory = _resolve_collector_inventory(
            collector_inventory_path,
            collector_config_path=collector_config_path,
            collectors_dir=collectors_dir,
        )
        requirements = _requirements(coverage_plan)
        price_feeds = _price_feeds(collector_inventory)
        known_artifacts = _known_price_artifacts(
            price_globs or DEFAULT_PRICE_GLOBS,
            current_market_data_path=coverage_plan.get("inputs", {}).get("market_data_path") if coverage_plan else None,
            max_price_artifacts=max_price_artifacts,
        )
        validation = _validation(coverage_plan, collector_inventory, requirements, price_feeds)
        tasks = _tasks(validation, requirements, price_feeds, refreshed_price_placeholder)
        payload = {
            "run_id": _run_id("market_data_refresh_runbook"),
            "created_at": utc_now_iso(),
            "mode": "market_data_refresh_runbook",
            "inputs": {
                "coverage_plan_path": str(coverage_plan_path),
                "collector_inventory_path": str(collector_inventory_path) if collector_inventory_path else None,
                "collector_inventory_source": collector_inventory.get("inventory_source"),
                "collector_inventory_config_path": collector_inventory.get("config_path"),
                "collector_inventory_as_of": collector_inventory.get("scanned_at")
                or collector_inventory.get("snapshot_written_at"),
                "price_globs": price_globs or DEFAULT_PRICE_GLOBS,
                "max_price_artifacts": max_price_artifacts,
                "refreshed_price_placeholder": refreshed_price_placeholder,
            },
            "summary": {
                "runbook_status": validation["status"],
                "coverage_plan_status": coverage_plan.get("summary", {}).get("plan_status") if coverage_plan else None,
                "collector_inventory_status": _collector_inventory_status(collector_inventory),
                "required_tickers": requirements["required_tickers"],
                "minimum_timestamp_exclusive": requirements.get("minimum_timestamp_exclusive"),
                "earliest_due_at": requirements.get("earliest_due_at"),
                "latest_due_at": requirements.get("latest_due_at"),
                "primary_price_feed": (price_feeds[0]["name"] if price_feeds else None),
                "enabled_price_feed_count": sum(1 for feed in price_feeds if feed.get("enabled")),
                "known_price_artifact_count": len(known_artifacts),
                "task_count": len(tasks),
                "can_refresh_automatically": False,
                "can_run_outcome_readiness_now": validation["status"]
                in {"no_refresh_required_recheck_readiness", "horizon_monitoring_only"},
                "can_apply_outcomes": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "outcome_write_performed": False,
                "learning_write_performed": False,
                "review_action_write_performed": False,
                "proposal_enqueue_performed": False,
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
            },
            "validation": validation,
            "requirements": requirements,
            "price_feed_candidates": price_feeds,
            "known_price_artifacts": known_artifacts,
            "refresh_options": _refresh_options(requirements, price_feeds, refreshed_price_placeholder),
            "operator_tasks": tasks,
            "acceptance_criteria": _acceptance_criteria(refreshed_price_placeholder),
            "commands": _commands(
                coverage_plan=coverage_plan,
                readiness_artifact=readiness_artifact,
                coverage_plan_path=coverage_plan_path,
                collector_inventory_path=collector_inventory_path,
                requirements=requirements,
                refreshed_price_placeholder=refreshed_price_placeholder,
            ),
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(validation, requirements),
        }
        if save:
            self.save(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_market_data_refresh_runbook_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_market_data_refresh_runbook_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Market Data Refresh Runbook",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Runbook status: `{summary.get('runbook_status')}`",
        f"- Coverage plan status: `{summary.get('coverage_plan_status')}`",
        f"- Required tickers: {', '.join(summary.get('required_tickers', [])) or 'none'}",
        f"- Minimum timestamp exclusive: `{summary.get('minimum_timestamp_exclusive')}`",
        f"- Earliest due_at: `{summary.get('earliest_due_at')}`",
        f"- Primary price feed: `{summary.get('primary_price_feed')}`",
        f"- Can refresh automatically: {summary.get('can_refresh_automatically')}",
        "",
        "## Price Feed Candidates",
        "",
    ]
    for feed in payload.get("price_feed_candidates", []):
        lines.append(
            f"- `{feed.get('name')}` type=`{feed.get('type')}` enabled={feed.get('enabled')} "
            f"class_found={feed.get('class_found')} table=`{feed.get('table_name')}`"
        )
    lines.extend(["", "## Operator Tasks", ""])
    for task in payload.get("operator_tasks", []):
        lines.append(f"- `{task.get('priority')}` {task.get('task_id')}: {task.get('description')}")
    lines.extend(["", "## Acceptance Criteria", ""])
    lines.extend(f"- {item}" for item in payload.get("acceptance_criteria", []))
    lines.extend(["", "## Commands", ""])
    for key, command in payload.get("commands", {}).items():
        if command:
            lines.append(f"- {key}: `{command}`")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    if not path:
        return {}
    try:
        return DeanPaths.load_json(path)
    except Exception:
        return {}


def _resolve_collector_inventory(
    collector_inventory_path: str | Path | None,
    collector_config_path: str | Path | None = None,
    collectors_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Scan the collector config live, or read an explicitly requested snapshot.

    A snapshot is only as current as the file's mtime, and nothing regenerates it
    any more, so it is tagged ``inventory_source: snapshot`` and its own age is
    reported. Live scans carry ``inventory_source: live``.
    """
    if not collector_inventory_path:
        return {}
    if str(collector_inventory_path) == LIVE_COLLECTOR_INVENTORY:
        from dean_os.collector_inventory_scan import (
            DEFAULT_COLLECTORS_DIR,
            DEFAULT_CONFIG_PATH,
            inspect_collector_inventory,
        )

        inventory = inspect_collector_inventory(
            config_path=collector_config_path or DEFAULT_CONFIG_PATH,
            collectors_dir=collectors_dir or DEFAULT_COLLECTORS_DIR,
        )
        inventory["inventory_source"] = "live"
        inventory["scanned_at"] = utc_now_iso()
        return inventory

    inventory = _load_optional_json(collector_inventory_path)
    if not inventory:
        return {}
    inventory = dict(inventory.get("collector_inventory") or inventory)
    inventory["inventory_source"] = "snapshot"
    inventory["snapshot_path"] = str(collector_inventory_path)
    inventory["snapshot_written_at"] = _file_mtime_iso(collector_inventory_path)
    return inventory


def _file_mtime_iso(path: str | Path) -> str | None:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return None


def _requirements(coverage_plan: dict[str, Any]) -> dict[str, Any]:
    if not coverage_plan:
        return {
            "required_tickers": [],
            "needs_price_after_creation": [],
            "missing_tickers": [],
            "minimum_timestamp_exclusive": None,
            "earliest_due_at": None,
            "latest_due_at": None,
            "current_market_data_path": None,
            "latest_processed_prices": "1d",
            "close_col": "close",
            "datetime_col": "datetime",
        }
    summary = coverage_plan.get("summary", {})
    inputs = coverage_plan.get("inputs", {})
    required_tickers = _unique_upper(inputs.get("tickers", []) or summary.get("tickers_need_price_after_creation", []))
    if not required_tickers:
        required_tickers = _unique_upper(item.get("ticker") for item in coverage_plan.get("ticker_coverage", []))
    return {
        "required_tickers": required_tickers,
        "needs_price_after_creation": _unique_upper(summary.get("tickers_need_price_after_creation", [])),
        "missing_tickers": _unique_upper(summary.get("missing_tickers", [])),
        "minimum_timestamp_exclusive": summary.get("minimum_price_after_created_at")
        or coverage_plan.get("coverage_targets", {}).get("latest_created_at"),
        "earliest_due_at": summary.get("earliest_due_at") or coverage_plan.get("coverage_targets", {}).get("earliest_due_at"),
        "latest_due_at": summary.get("latest_due_at") or coverage_plan.get("coverage_targets", {}).get("latest_due_at"),
        "current_market_data_path": inputs.get("market_data_path"),
        "latest_processed_prices": inputs.get("latest_processed_prices") or "1d",
        "close_col": inputs.get("close_col") or "close",
        "datetime_col": inputs.get("datetime_col") or "datetime",
    }


def _price_feeds(collector_inventory: dict[str, Any]) -> list[dict[str, Any]]:
    inventory = collector_inventory.get("collector_inventory", collector_inventory)
    records = inventory.get("configured_collectors", []) if isinstance(inventory, dict) else []
    feeds = []
    for record in records:
        if record.get("recommended_use") != "pipeline_price_feed":
            continue
        feeds.append(
            {
                "name": record.get("name"),
                "type": record.get("type"),
                "enabled": bool(record.get("enabled")),
                "critical": bool(record.get("critical")),
                "class_found": bool(record.get("class_found")),
                "class_name": record.get("class_name"),
                "module_path": record.get("module_path"),
                "table_name": record.get("table_name"),
                "cache_ttl": record.get("cache_ttl"),
                "cache_duration_minutes": record.get("cache_duration_minutes"),
                "schedule_hint": record.get("schedule_hint"),
                "notes": record.get("notes", []),
            }
        )
    return sorted(feeds, key=lambda item: (not item.get("enabled"), not item.get("critical"), str(item.get("name"))))


def _known_price_artifacts(
    price_globs: list[str],
    current_market_data_path: str | Path | None,
    max_price_artifacts: int,
) -> list[dict[str, Any]]:
    candidates: dict[str, Path] = {}
    for pattern in price_globs:
        for path in Path(".").glob(pattern):
            if path.is_file():
                candidates[str(path)] = path
    current_resolved = str(Path(current_market_data_path)) if current_market_data_path else None
    artifacts = []
    for path in candidates.values():
        try:
            stat = path.stat()
        except OSError:
            continue
        artifacts.append(
            {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                "inferred_interval": _infer_interval(path.name),
                "filename_timestamp": _infer_filename_timestamp(path.name),
                "is_current_market_data_path": current_resolved is not None and str(path) == current_resolved,
            }
        )
    return sorted(artifacts, key=lambda item: item["last_modified"], reverse=True)[: max(0, max_price_artifacts)]


def _validation(
    coverage_plan: dict[str, Any],
    collector_inventory: dict[str, Any],
    requirements: dict[str, Any],
    price_feeds: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not coverage_plan:
        return {"status": "blocked_no_coverage_plan", "can_plan": False, "reasons": ["Outcome price coverage plan was not found."]}
    coverage_status = coverage_plan.get("summary", {}).get("plan_status")
    if coverage_status == "coverage_ready_for_outcome_readiness_rerun":
        return {
            "status": "no_refresh_required_recheck_readiness",
            "can_plan": True,
            "reasons": ["Price coverage plan says refreshed coverage is ready for an OutcomeReadinessGate rerun."],
        }
    if coverage_status == "waiting_for_outcome_horizon":
        return {
            "status": "horizon_monitoring_only",
            "can_plan": True,
            "reasons": ["Prices exist after learning creation, but production outcome horizons have not elapsed."],
        }
    if not requirements["required_tickers"]:
        return {"status": "blocked_no_required_tickers", "can_plan": False, "reasons": ["No required tickers were found."]}
    if not collector_inventory:
        return {
            "status": "blocked_missing_collector_inventory",
            "can_plan": True,
            "reasons": [
                "Collector inventory is empty: src/config/collectors.yaml could not be read, "
                "or collector_inventory_path was set to None."
            ],
        }
    if collector_inventory.get("summary", {}).get("status") == "unavailable":
        return {
            "status": "blocked_unreadable_collector_config",
            "can_plan": False,
            "reasons": [
                str(collector_inventory.get("summary", {}).get("reason") or "Collector config is unreadable.")
            ],
        }
    enabled_price_feeds = [feed for feed in price_feeds if feed.get("enabled") and feed.get("class_found")]
    if not enabled_price_feeds:
        reasons.append("No enabled local pipeline price feed with a discovered class was found.")
        return {"status": "blocked_no_enabled_price_feed", "can_plan": True, "reasons": reasons}
    reasons.append("Enabled local price feed exists; operator can refresh a separate price artifact and rerun readiness checks.")
    return {"status": "refresh_runbook_ready", "can_plan": True, "reasons": reasons}


def _tasks(
    validation: dict[str, Any],
    requirements: dict[str, Any],
    price_feeds: list[dict[str, Any]],
    refreshed_price_placeholder: str,
) -> list[dict[str, Any]]:
    status = validation["status"]
    if status == "blocked_no_coverage_plan":
        return [
            _task(
                "build_outcome_price_coverage_plan",
                "high",
                "Build OutcomePriceCoveragePlan first so required tickers and timestamps are explicit.",
                {"expected_artifact": DEFAULT_COVERAGE_PLAN_PATH},
            )
        ]
    tasks = []
    if status == "blocked_missing_collector_inventory":
        tasks.append(
            _task(
                "build_collector_inventory",
                "high",
                "Make src/config/collectors.yaml readable so the live collector scan can run, "
                "or pass a saved inventory JSON via collector_inventory_path.",
                {"expected_input": "src/config/collectors.yaml"},
            )
        )
    if status == "blocked_no_enabled_price_feed":
        tasks.append(
            _task(
                "identify_price_feed_or_manual_artifact",
                "high",
                "Identify an approved price feed or provide a manually refreshed local CSV/parquet artifact.",
                {"required_tickers": requirements["required_tickers"]},
            )
        )
    if status in {"refresh_runbook_ready", "blocked_missing_collector_inventory", "blocked_no_enabled_price_feed"}:
        tasks.append(
            _task(
                "produce_refreshed_price_artifact",
                "high",
                "Create a separate refreshed local price artifact rather than overwriting the old artifact before validation.",
                {
                    "target_path_placeholder": refreshed_price_placeholder,
                    "tickers": requirements["required_tickers"],
                    "minimum_timestamp_exclusive": requirements.get("minimum_timestamp_exclusive"),
                    "source_candidates": [feed["name"] for feed in price_feeds],
                },
            )
        )
        tasks.append(
            _task(
                "validate_refreshed_artifact",
                "high",
                "Run market freshness and OutcomePriceCoveragePlan against the refreshed artifact before any outcome evaluation.",
                {"required_status": "coverage_ready_for_outcome_readiness_rerun or waiting_for_outcome_horizon"},
            )
        )
    if requirements.get("latest_due_at"):
        tasks.append(
            _task(
                "schedule_horizon_coverage_monitoring",
                "medium",
                "Keep local price coverage active until the configured outcome horizon is due.",
                {"latest_due_at": requirements.get("latest_due_at")},
            )
        )
    if status in {"no_refresh_required_recheck_readiness", "horizon_monitoring_only"}:
        tasks.append(
            _task(
                "rerun_outcome_readiness",
                "medium",
                "Rerun OutcomeReadinessGate and inspect statuses before any outcome apply ceremony.",
                {"expected_artifact": "reports/dean_os/outcome_readiness_gate_after_price_refresh/latest.json"},
            )
        )
    return tasks


def _refresh_options(
    requirements: dict[str, Any],
    price_feeds: list[dict[str, Any]],
    refreshed_price_placeholder: str,
) -> list[dict[str, Any]]:
    options = [
        {
            "option_id": "separate_refreshed_price_artifact",
            "recommended": True,
            "description": "Create a new local CSV/parquet price artifact and point readiness checks at it.",
            "requires_network": "depends_on_operator_refresh_method",
            "writes_project_config": False,
            "runs_heavy_pipeline": False,
            "target": {
                "path_placeholder": refreshed_price_placeholder,
                "tickers": requirements["required_tickers"],
                "minimum_timestamp_exclusive": requirements.get("minimum_timestamp_exclusive"),
            },
        }
    ]
    for feed in price_feeds:
        options.append(
            {
                "option_id": f"{feed.get('name')}_isolated_price_feed_refresh",
                "recommended": bool(feed.get("enabled") and feed.get("class_found")),
                "description": (
                    f"Use the `{feed.get('name')}` collector as an isolated price refresh source only after operator approval."
                ),
                "requires_network": True,
                "writes_project_config": False,
                "runs_heavy_pipeline": False,
                "target": {
                    "collector": feed.get("name"),
                    "type": feed.get("type"),
                    "table_name": feed.get("table_name"),
                    "module_path": feed.get("module_path"),
                },
            }
        )
    options.append(
        {
            "option_id": "do_not_run_heavy_pipeline_for_refresh",
            "recommended": True,
            "description": "Do not run the full trading pipeline just to clear outcome price coverage.",
            "requires_network": False,
            "writes_project_config": False,
            "runs_heavy_pipeline": False,
            "target": {"avoid": ["run_progressive_pipeline.py", "full pipeline stage execution", "broker/live execution"]},
        }
    )
    return options


def _commands(
    coverage_plan: dict[str, Any],
    readiness_artifact: dict[str, Any],
    coverage_plan_path: str | Path,
    collector_inventory_path: str | Path | None,
    requirements: dict[str, Any],
    refreshed_price_placeholder: str,
) -> dict[str, str | None]:
    tickers = requirements["required_tickers"]
    latest_processed_prices = requirements.get("latest_processed_prices") or "1d"
    close_col = requirements.get("close_col") or "close"
    datetime_col = requirements.get("datetime_col") or "datetime"
    readiness_inputs = readiness_artifact.get("inputs", {})
    readiness_after_refresh = _readiness_after_refresh_command(readiness_inputs, tickers, refreshed_price_placeholder)
    market_freshness_after_refresh = [
        "python run_agent_market_freshness.py",
        "--market-data-path",
        refreshed_price_placeholder,
        "--latest-processed-prices",
        latest_processed_prices,
    ]
    if tickers:
        market_freshness_after_refresh.extend(["--tickers", *tickers])
    if requirements.get("minimum_timestamp_exclusive"):
        market_freshness_after_refresh.extend(["--as-of", str(requirements["minimum_timestamp_exclusive"])])
    if close_col != "close":
        market_freshness_after_refresh.extend(["--close-col", close_col])
    if datetime_col != "datetime":
        market_freshness_after_refresh.extend(["--datetime-col", datetime_col])
    market_freshness_after_refresh.extend(["--output-dir", "reports/dean_os/market_freshness_after_price_refresh"])

    return {
        # There is no run_agent_collector_inventory.py: the agent was archived on
        # 2026-07-24. The inventory is now scanned in-process on every build, so the
        # operator has nothing to refresh by hand.
        "refresh_collector_inventory": (
            "not required: dean_os.collector_inventory_scan.inspect_collector_inventory() "
            "runs during this build and reads src/config/collectors.yaml directly"
        ),
        "inspect_refreshed_market_freshness": " ".join(market_freshness_after_refresh),
        "rerun_outcome_readiness_after_refresh": readiness_after_refresh,
        "rebuild_price_coverage_after_refresh": (
            "python run_agent_outcome_price_coverage.py "
            f"--readiness-json reports/dean_os/outcome_readiness_gate_after_price_refresh/latest.json "
            f"--market-data-path {refreshed_price_placeholder} --output-dir reports/dean_os/outcome_price_coverage_plan_after_refresh"
        ),
        "rebuild_this_runbook": f"python run_agent_market_data_refresh_runbook.py --coverage-plan-json {coverage_plan_path}",
        "current_coverage_plan_command": coverage_plan.get("commands", {}).get("rebuild_this_plan") if coverage_plan else None,
    }


def _readiness_after_refresh_command(readiness_inputs: dict[str, Any], tickers: list[str], refreshed_price_placeholder: str) -> str | None:
    if not readiness_inputs:
        return None
    command = [
        "python run_agent_outcome_readiness.py",
        "--learning-store",
        str(readiness_inputs.get("learning_path") or "data/dean_os/agent_learning.sqlite"),
        "--memory-store",
        str(readiness_inputs.get("memory_path") or "data/dean_os/recommendation_memory.sqlite"),
        "--market-data-path",
        refreshed_price_placeholder,
    ]
    if readiness_inputs.get("latest_processed_prices"):
        command.extend(["--latest-processed-prices", str(readiness_inputs["latest_processed_prices"])])
    command.extend(["--tickers", *(tickers or [str(ticker).upper() for ticker in readiness_inputs.get("tickers", [])])])
    if readiness_inputs.get("as_of"):
        command.extend(["--as-of", str(readiness_inputs["as_of"])])
    if readiness_inputs.get("close_col") and readiness_inputs.get("close_col") != "close":
        command.extend(["--close-col", str(readiness_inputs["close_col"])])
    if readiness_inputs.get("datetime_col") and readiness_inputs.get("datetime_col") != "datetime":
        command.extend(["--datetime-col", str(readiness_inputs["datetime_col"])])
    if readiness_inputs.get("neutral_band") and float(readiness_inputs["neutral_band"]) != 0.01:
        command.extend(["--neutral-band", str(readiness_inputs["neutral_band"])])
    if readiness_inputs.get("profile"):
        command.extend(["--profile", str(readiness_inputs["profile"])])
    if readiness_inputs.get("agent_names"):
        command.extend(["--agent-names", *[str(name) for name in readiness_inputs["agent_names"]]])
    if readiness_inputs.get("include_non_analyst_records"):
        command.append("--include-non-analyst-records")
    if readiness_inputs.get("historical_diagnostic"):
        command.append("--historical-diagnostic")
    command.extend(["--output-dir", "reports/dean_os/outcome_readiness_gate_after_price_refresh"])
    return " ".join(command)


def _acceptance_criteria(refreshed_price_placeholder: str) -> list[str]:
    return [
        f"`{refreshed_price_placeholder}` exists as a local CSV/parquet file and is not merely the old stale artifact.",
        "The refreshed artifact contains every required ticker.",
        "Each required ticker has a timestamp strictly after the coverage plan minimum created_at timestamp.",
        "MarketDataFreshnessAgent can read the refreshed artifact without missing ticker rows.",
        "OutcomePriceCoveragePlan no longer reports `needs_price_refresh_after_record_creation` after the refreshed artifact is used.",
        "OutcomeReadinessGate is rerun before any outcome apply ceremony or calibration proposal.",
    ]


def _operator_notes() -> list[str]:
    return [
        "This runbook is read-only and never runs collectors, network calls, pipeline stages, config writes, learning writes, or broker actions.",
        "It is acceptable for an operator or a separate approved collector task to refresh prices, but that refresh is outside this command.",
        "Do not overwrite the stale price artifact until the refreshed artifact passes market freshness and outcome readiness checks.",
    ]


def _recommendations(validation: dict[str, Any], requirements: dict[str, Any]) -> list[str]:
    status = validation["status"]
    if status == "refresh_runbook_ready":
        return [
            "Refresh a separate local price artifact for the required tickers, then run the generated freshness and readiness commands.",
            "Keep outcome apply and calibration blocked until readiness no longer reports stale/created-after price blockers.",
        ]
    if status == "blocked_missing_collector_inventory":
        return ["Run CollectorInventoryAgent first, then rebuild this runbook."]
    if status == "blocked_no_enabled_price_feed":
        return ["Provide an approved manual price artifact or fix/enable a local pipeline price feed before outcome checks."]
    if status == "horizon_monitoring_only":
        return [
            f"Keep collecting prices until at least {requirements.get('earliest_due_at')} before production outcome labels.",
            "Rerun readiness periodically, but do not apply outcome labels before the horizon policy is satisfied.",
        ]
    if status == "no_refresh_required_recheck_readiness":
        return ["Rerun OutcomeReadinessGate against the refreshed artifact and review the dry-run output."]
    if status == "blocked_no_coverage_plan":
        return ["Build OutcomePriceCoveragePlan first."]
    return validation.get("reasons", []) or ["Resolve blockers, then rebuild this runbook."]


def _collector_inventory_status(collector_inventory: dict[str, Any]) -> str:
    if not collector_inventory:
        return "missing"
    inventory = collector_inventory.get("collector_inventory", collector_inventory)
    status = str(inventory.get("summary", {}).get("status") or "unknown")
    # A snapshot's status describes the config as it was when the file was written,
    # not as it is now, so never let it read as a current "ok".
    if collector_inventory.get("inventory_source") == "snapshot":
        return f"snapshot_{status}"
    return status


def _task(task_id: str, priority: str, description: str, target: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": task_id, "priority": priority, "description": description, "target": target}


def _infer_interval(name: str) -> str | None:
    match = re.search(r"(?:prices|stage2_prices)_([0-9]+[a-zA-Z]+)_", name)
    return match.group(1) if match else None


def _infer_filename_timestamp(name: str) -> str | None:
    match = re.search(r"_(20\d{6}_\d{6})", name)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}T{raw[9:11]}:{raw[11:13]}:{raw[13:15]}"


def _unique_upper(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        ticker = str(value).strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            result.append(ticker)
    return result


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
