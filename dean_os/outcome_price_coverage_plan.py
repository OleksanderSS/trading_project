from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.agents.market_data_freshness import inspect_market_data_freshness
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


DEFAULT_READINESS_PATH = "reports/dean_os/outcome_readiness_gate/latest.json"


class OutcomePriceCoveragePlan:
    """Read-only plan for the market data coverage needed by outcome evaluation."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/outcome_price_coverage_plan"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        readiness_path: str | Path = DEFAULT_READINESS_PATH,
        market_data_path: str | Path | None = None,
        latest_processed_prices: str | None = None,
        tickers: list[str] | None = None,
        close_col: str | None = None,
        datetime_col: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        readiness_artifact = _load_optional_json(readiness_path)
        readiness_inputs = readiness_artifact.get("inputs", {}) if readiness_artifact else {}
        resolved_market_data_path = market_data_path or readiness_inputs.get("market_data_path")
        resolved_latest_processed_prices = latest_processed_prices or readiness_inputs.get("latest_processed_prices") or "1d"
        resolved_close_col = close_col or readiness_inputs.get("close_col") or "close"
        resolved_datetime_col = datetime_col or readiness_inputs.get("datetime_col") or "datetime"
        requested_tickers = _requested_tickers(readiness_artifact, tickers or [])
        records = _pending_records(readiness_artifact)
        coverage_targets = _coverage_targets(records, requested_tickers)
        market_snapshot = _market_snapshot(
            market_data_path=resolved_market_data_path,
            latest_processed_prices=resolved_latest_processed_prices,
            tickers=requested_tickers,
            close_col=resolved_close_col,
            datetime_col=resolved_datetime_col,
            target_as_of=coverage_targets.get("latest_created_at") or coverage_targets.get("earliest_due_at"),
        )
        ticker_coverage = _ticker_coverage(records, requested_tickers, market_snapshot, readiness_artifact)
        validation = _validation(readiness_artifact, records, requested_tickers, resolved_market_data_path, ticker_coverage)
        tasks = _coverage_tasks(validation, coverage_targets, ticker_coverage, requested_tickers)
        payload = {
            "run_id": _run_id("outcome_price_coverage_plan"),
            "created_at": utc_now_iso(),
            "mode": "outcome_price_coverage_plan",
            "inputs": {
                "readiness_path": str(readiness_path),
                "market_data_path": str(resolved_market_data_path) if resolved_market_data_path else None,
                "latest_processed_prices": resolved_latest_processed_prices,
                "tickers": requested_tickers,
                "close_col": resolved_close_col,
                "datetime_col": resolved_datetime_col,
            },
            "summary": {
                "plan_status": validation["status"],
                "readiness_status": readiness_artifact.get("summary", {}).get("readiness_status") if readiness_artifact else None,
                "pending_record_count": len(records),
                "ticker_count": len(requested_tickers),
                "task_count": len(tasks),
                "minimum_price_after_created_at": _iso(coverage_targets.get("latest_created_at")),
                "earliest_due_at": _iso(coverage_targets.get("earliest_due_at")),
                "latest_due_at": _iso(coverage_targets.get("latest_due_at")),
                "market_latest_timestamp": market_snapshot.get("latest_timestamp"),
                "missing_tickers": [item["ticker"] for item in ticker_coverage if item["status"] == "missing_ticker_rows"],
                "tickers_need_price_after_creation": [
                    item["ticker"] for item in ticker_coverage if item["status"] == "needs_price_after_created_at"
                ],
                "tickers_waiting_for_horizon": [
                    item["ticker"] for item in ticker_coverage if item["status"] == "waiting_for_outcome_horizon"
                ],
                "can_run_outcome_readiness_now": validation["status"]
                in {"coverage_ready_for_outcome_readiness_rerun", "waiting_for_outcome_horizon"},
                "can_apply_outcomes": False,
                "outcome_write_performed": False,
                "learning_write_performed": False,
                "review_action_write_performed": False,
                "proposal_enqueue_performed": False,
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
            },
            "validation": validation,
            "coverage_targets": _json_targets(coverage_targets),
            "market_data_snapshot": market_snapshot,
            "ticker_coverage": ticker_coverage,
            "coverage_tasks": tasks,
            "acceptance_criteria": _acceptance_criteria(),
            "commands": _commands(
                readiness_artifact=readiness_artifact,
                readiness_path=readiness_path,
                market_data_path=resolved_market_data_path,
                latest_processed_prices=resolved_latest_processed_prices,
                requested_tickers=requested_tickers,
                close_col=resolved_close_col,
                datetime_col=resolved_datetime_col,
                coverage_targets=coverage_targets,
            ),
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(validation, coverage_targets),
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
        rendered_md = render_outcome_price_coverage_plan_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_outcome_price_coverage_plan_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Outcome Price Coverage Plan",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Plan status: `{summary.get('plan_status')}`",
        f"- Readiness status: `{summary.get('readiness_status')}`",
        f"- Pending records: {summary.get('pending_record_count')}",
        f"- Required tickers: {', '.join(payload.get('inputs', {}).get('tickers', [])) or 'none'}",
        f"- Market latest timestamp: `{summary.get('market_latest_timestamp')}`",
        f"- Minimum price after created_at: `{summary.get('minimum_price_after_created_at')}`",
        f"- Earliest due_at: `{summary.get('earliest_due_at')}`",
        f"- Latest due_at: `{summary.get('latest_due_at')}`",
        "",
        "## Ticker Coverage",
        "",
    ]
    for item in payload.get("ticker_coverage", []):
        lines.append(
            f"- `{item.get('ticker')}` status=`{item.get('status')}` latest=`{item.get('latest_price_at')}` "
            f"created_after=`{item.get('required_after_created_at')}` due_until=`{item.get('required_until_due_at')}`"
        )
    lines.extend(["", "## Coverage Tasks", ""])
    for task in payload.get("coverage_tasks", []):
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


def _load_optional_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _requested_tickers(readiness_artifact: dict[str, Any], explicit_tickers: list[str]) -> list[str]:
    values: list[Any] = list(explicit_tickers)
    if readiness_artifact:
        values.extend(readiness_artifact.get("inputs", {}).get("tickers", []) or [])
        for record in readiness_artifact.get("pending_records", []):
            values.extend(record.get("tickers", []) or [])
        for evaluation in readiness_artifact.get("dry_run_outcome_evaluation", {}).get("evaluations", []):
            values.extend(evaluation.get("tickers", []) or [])
    return _unique_upper(values)


def _pending_records(readiness_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not readiness_artifact:
        return []
    evaluations = {
        item.get("record_id"): item
        for item in readiness_artifact.get("dry_run_outcome_evaluation", {}).get("evaluations", [])
        if item.get("record_id")
    }
    records: list[dict[str, Any]] = []
    for record in readiness_artifact.get("pending_records", []):
        evaluation = evaluations.get(record.get("record_id"), {})
        created_at = _parse_datetime(record.get("created_at") or evaluation.get("created_at"))
        horizon_days = int(record.get("horizon_days") or evaluation.get("horizon_days") or 0)
        due_at = _parse_datetime(evaluation.get("due_at"))
        if created_at and due_at is None and horizon_days:
            due_at = created_at + timedelta(days=horizon_days)
        records.append(
            {
                "record_id": record.get("record_id"),
                "agent_name": record.get("agent_name"),
                "created_at": created_at,
                "horizon_days": horizon_days,
                "due_at": due_at,
                "tickers": _unique_upper(record.get("tickers", []) or evaluation.get("tickers", []) or []),
                "evaluation_status": evaluation.get("status"),
                "evaluation_latest_price_at": evaluation.get("latest_price_at"),
            }
        )
    return records


def _coverage_targets(records: list[dict[str, Any]], requested_tickers: list[str]) -> dict[str, Any]:
    created_dates = [record["created_at"] for record in records if record.get("created_at")]
    due_dates = [record["due_at"] for record in records if record.get("due_at")]
    by_ticker: dict[str, dict[str, datetime | None]] = {}
    for ticker in requested_tickers:
        matching_records = [record for record in records if not record.get("tickers") or ticker in record.get("tickers", [])]
        created = [record["created_at"] for record in matching_records if record.get("created_at")]
        due = [record["due_at"] for record in matching_records if record.get("due_at")]
        by_ticker[ticker] = {
            "latest_created_at": max(created) if created else None,
            "latest_due_at": max(due) if due else None,
            "earliest_due_at": min(due) if due else None,
        }
    return {
        "latest_created_at": max(created_dates) if created_dates else None,
        "earliest_created_at": min(created_dates) if created_dates else None,
        "earliest_due_at": min(due_dates) if due_dates else None,
        "latest_due_at": max(due_dates) if due_dates else None,
        "by_ticker": by_ticker,
    }


def _market_snapshot(
    market_data_path: str | Path | None,
    latest_processed_prices: str | None,
    tickers: list[str],
    close_col: str,
    datetime_col: str,
    target_as_of: datetime | None,
) -> dict[str, Any]:
    if not market_data_path:
        return {"status": "unavailable", "stale": True, "reason": "No market data path was provided.", "per_ticker_latest": {}}
    try:
        return inspect_market_data_freshness(
            market_data_path=market_data_path,
            latest_processed_prices=latest_processed_prices,
            tickers=tickers,
            as_of=target_as_of or datetime.now(UTC),
            max_age_hours=72.0,
            close_col=close_col,
            datetime_col=datetime_col,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "stale": True,
            "reason": f"Could not inspect market data: {type(exc).__name__}: {exc}",
            "market_data_path": str(market_data_path),
            "per_ticker_latest": {},
        }


def _ticker_coverage(
    records: list[dict[str, Any]],
    requested_tickers: list[str],
    market_snapshot: dict[str, Any],
    readiness_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    fallback_latest = _parse_datetime(readiness_artifact.get("dry_run_outcome_evaluation", {}).get("as_of")) if readiness_artifact else None
    per_ticker_latest = market_snapshot.get("per_ticker_latest", {}) or {}
    targets = _coverage_targets(records, requested_tickers)
    result: list[dict[str, Any]] = []
    for ticker in requested_tickers:
        ticker_metrics = per_ticker_latest.get(ticker, {})
        latest_at = _parse_datetime(ticker_metrics.get("latest_timestamp")) or fallback_latest
        ticker_targets = targets["by_ticker"].get(ticker, {})
        required_after_created = ticker_targets.get("latest_created_at")
        required_until_due = ticker_targets.get("latest_due_at")
        status = _ticker_status(latest_at, required_after_created, required_until_due, bool(ticker_metrics))
        result.append(
            {
                "ticker": ticker,
                "status": status,
                "latest_price_at": _iso(latest_at),
                "latest_close": ticker_metrics.get("latest_close"),
                "row_count": ticker_metrics.get("row_count"),
                "required_after_created_at": _iso(required_after_created),
                "required_until_due_at": _iso(required_until_due),
            }
        )
    return result


def _ticker_status(
    latest_at: datetime | None,
    required_after_created: datetime | None,
    required_until_due: datetime | None,
    has_local_rows: bool,
) -> str:
    if not latest_at or not has_local_rows:
        return "missing_ticker_rows"
    if required_after_created and latest_at <= required_after_created:
        return "needs_price_after_created_at"
    if required_until_due and latest_at < required_until_due:
        return "waiting_for_outcome_horizon"
    return "ready_for_outcome_check"


def _validation(
    readiness_artifact: dict[str, Any],
    records: list[dict[str, Any]],
    requested_tickers: list[str],
    market_data_path: str | Path | None,
    ticker_coverage: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not readiness_artifact:
        return {"status": "blocked_no_readiness_artifact", "can_plan": False, "reasons": ["Outcome readiness artifact was not found."]}
    if not records:
        return {"status": "blocked_no_pending_records", "can_plan": False, "reasons": ["No pending learning records were found."]}
    if not requested_tickers:
        return {"status": "blocked_no_requested_tickers", "can_plan": False, "reasons": ["No requested tickers were found."]}
    if not market_data_path:
        return {"status": "blocked_missing_market_data_reference", "can_plan": True, "reasons": ["No market data path is available."]}
    statuses = {item["status"] for item in ticker_coverage}
    if "missing_ticker_rows" in statuses:
        reasons.append("At least one requested ticker has no local price rows.")
        return {"status": "blocked_missing_ticker_prices", "can_plan": True, "reasons": reasons}
    if "needs_price_after_created_at" in statuses:
        reasons.append("Local market data ends before at least one pending learning record was created.")
        return {"status": "needs_price_refresh_after_record_creation", "can_plan": True, "reasons": reasons}
    if "waiting_for_outcome_horizon" in statuses:
        reasons.append("Prices exist after record creation, but the configured outcome horizons have not elapsed.")
        return {"status": "waiting_for_outcome_horizon", "can_plan": True, "reasons": reasons}
    reasons.append("Local price coverage is sufficient to rerun outcome readiness.")
    return {"status": "coverage_ready_for_outcome_readiness_rerun", "can_plan": True, "reasons": reasons}


def _coverage_tasks(
    validation: dict[str, Any],
    coverage_targets: dict[str, Any],
    ticker_coverage: list[dict[str, Any]],
    requested_tickers: list[str],
) -> list[dict[str, Any]]:
    status = validation["status"]
    tasks: list[dict[str, Any]] = []
    if status == "blocked_no_readiness_artifact":
        tasks.append(
            _task(
                "run_outcome_readiness_gate_first",
                "high",
                "Run the outcome readiness gate before building a price coverage plan.",
                {"expected_artifact": DEFAULT_READINESS_PATH},
            )
        )
        return tasks
    if status == "blocked_missing_market_data_reference":
        tasks.append(
            _task(
                "provide_market_data_path",
                "high",
                "Provide a local CSV or Parquet price file so coverage can be checked without network access.",
                {"requested_tickers": requested_tickers},
            )
        )
    missing = [item["ticker"] for item in ticker_coverage if item["status"] == "missing_ticker_rows"]
    if missing:
        tasks.append(
            _task(
                "add_missing_ticker_price_rows",
                "high",
                "Add local price rows for every requested ticker before outcome evaluation can be trusted.",
                {"missing_tickers": missing},
            )
        )
    needs_after_creation = [item["ticker"] for item in ticker_coverage if item["status"] == "needs_price_after_created_at"]
    if needs_after_creation:
        tasks.append(
            _task(
                "refresh_prices_after_learning_creation",
                "high",
                "Refresh local price data so each ticker has at least one timestamp strictly after the latest learning record creation time.",
                {
                    "tickers": needs_after_creation,
                    "minimum_timestamp_exclusive": _iso(coverage_targets.get("latest_created_at")),
                },
            )
        )
    if coverage_targets.get("latest_due_at"):
        tasks.append(
            _task(
                "maintain_prices_until_outcome_due_dates",
                "medium",
                "Keep collecting prices until the learning-record horizons are due before applying production outcome labels.",
                {
                    "earliest_due_at": _iso(coverage_targets.get("earliest_due_at")),
                    "latest_due_at": _iso(coverage_targets.get("latest_due_at")),
                },
            )
        )
    if status not in {"blocked_no_readiness_artifact", "blocked_no_pending_records", "blocked_no_requested_tickers"}:
        tasks.append(
            _task(
                "rerun_outcome_readiness_after_price_refresh",
                "medium",
                "Rerun OutcomeReadinessGate after refreshing prices and inspect the dry-run status before any apply ceremony.",
                {"expected_status_before_apply": "ready_for_outcome_dry_run"},
            )
        )
    return tasks


def _commands(
    readiness_artifact: dict[str, Any],
    readiness_path: str | Path,
    market_data_path: str | Path | None,
    latest_processed_prices: str,
    requested_tickers: list[str],
    close_col: str,
    datetime_col: str,
    coverage_targets: dict[str, Any],
) -> dict[str, str | None]:
    market_check: list[str] | None = None
    if market_data_path:
        market_check = [
            "python run_agent_market_freshness.py",
            "--market-data-path",
            str(market_data_path),
            "--latest-processed-prices",
            latest_processed_prices,
        ]
        if requested_tickers:
            market_check.extend(["--tickers", *requested_tickers])
        target_as_of = coverage_targets.get("latest_created_at") or coverage_targets.get("earliest_due_at")
        if target_as_of:
            market_check.extend(["--as-of", _iso(target_as_of)])
        if close_col != "close":
            market_check.extend(["--close-col", close_col])
        if datetime_col != "datetime":
            market_check.extend(["--datetime-col", datetime_col])
        market_check.extend(["--output-dir", "reports/dean_os/market_freshness_outcome_prices"])

    readiness_command = _readiness_command(readiness_artifact)
    return {
        "inspect_current_market_freshness": " ".join(market_check) if market_check else None,
        "rerun_outcome_readiness_after_refresh": readiness_command,
        "rebuild_this_plan": f"python run_agent_outcome_price_coverage.py --readiness-json {readiness_path}",
    }


def _readiness_command(readiness_artifact: dict[str, Any]) -> str | None:
    if not readiness_artifact:
        return None
    inputs = readiness_artifact.get("inputs", {})
    command = [
        "python run_agent_outcome_readiness.py",
        "--learning-store",
        str(inputs.get("learning_path") or "data/dean_os/agent_learning.sqlite"),
        "--memory-store",
        str(inputs.get("memory_path") or "data/dean_os/recommendation_memory.sqlite"),
    ]
    if inputs.get("market_data_path"):
        command.extend(["--market-data-path", str(inputs["market_data_path"])])
    if inputs.get("latest_processed_prices"):
        command.extend(["--latest-processed-prices", str(inputs["latest_processed_prices"])])
    if inputs.get("tickers"):
        command.extend(["--tickers", *[str(ticker).upper() for ticker in inputs["tickers"]]])
    if inputs.get("as_of"):
        command.extend(["--as-of", str(inputs["as_of"])])
    if inputs.get("close_col") and inputs.get("close_col") != "close":
        command.extend(["--close-col", str(inputs["close_col"])])
    if inputs.get("datetime_col") and inputs.get("datetime_col") != "datetime":
        command.extend(["--datetime-col", str(inputs["datetime_col"])])
    if inputs.get("neutral_band") and float(inputs["neutral_band"]) != 0.01:
        command.extend(["--neutral-band", str(inputs["neutral_band"])])
    if inputs.get("limit") is not None:
        command.extend(["--limit", str(inputs["limit"])])
    if inputs.get("profile"):
        command.extend(["--profile", str(inputs["profile"])])
    if inputs.get("agent_names"):
        command.extend(["--agent-names", *[str(name) for name in inputs["agent_names"]]])
    if inputs.get("include_non_analyst_records"):
        command.append("--include-non-analyst-records")
    if inputs.get("historical_diagnostic"):
        command.append("--historical-diagnostic")
    command.extend(["--output-dir", "reports/dean_os/outcome_readiness_gate_after_price_refresh"])
    return " ".join(command)


def _acceptance_criteria() -> list[str]:
    return [
        "Every requested ticker has readable local price rows.",
        "For every pending learning record, each relevant ticker has a price timestamp strictly after record created_at.",
        "For production outcome labels, each relevant ticker has coverage through the record due_at horizon.",
        "OutcomeReadinessGate must be rerun and reviewed before any outcome apply ceremony.",
        "This plan never writes learning records, review actions, proposals, config, pipeline outputs, or broker actions.",
    ]


def _operator_notes() -> list[str]:
    return [
        "This is a read-only planning artifact for price coverage, not a collector or pipeline runner.",
        "Having prices after created_at only proves the learning record is not temporally impossible to evaluate.",
        "Production outcome learning still requires the configured horizon to elapse unless a run is explicitly marked historical diagnostic.",
    ]


def _recommendations(validation: dict[str, Any], coverage_targets: dict[str, Any]) -> list[str]:
    status = validation["status"]
    if status == "needs_price_refresh_after_record_creation":
        return [
            f"Refresh local prices after {_iso(coverage_targets.get('latest_created_at'))} for the requested tickers.",
            "After refresh, rerun OutcomeReadinessGate; do not apply outcome labels while it still reports no_price_after_created_at.",
        ]
    if status == "waiting_for_outcome_horizon":
        return [
            f"Keep coverage running until at least {_iso(coverage_targets.get('earliest_due_at'))} before production outcome apply.",
            "You may rerun readiness as a monitoring check, but not as final learning truth before due_at.",
        ]
    if status == "coverage_ready_for_outcome_readiness_rerun":
        return ["Rerun OutcomeReadinessGate and inspect the dry-run output before considering an apply ceremony."]
    if status == "blocked_missing_ticker_prices":
        return ["Add missing ticker rows to the local price cache, then rebuild this coverage plan."]
    if status == "blocked_no_readiness_artifact":
        return ["Run OutcomeReadinessGate first so the plan can use real pending learning records."]
    return validation.get("reasons", []) or ["Resolve coverage blockers, then rebuild this plan."]


def _json_targets(targets: dict[str, Any]) -> dict[str, Any]:
    return {
        "latest_created_at": _iso(targets.get("latest_created_at")),
        "earliest_created_at": _iso(targets.get("earliest_created_at")),
        "earliest_due_at": _iso(targets.get("earliest_due_at")),
        "latest_due_at": _iso(targets.get("latest_due_at")),
        "by_ticker": {
            ticker: {key: _iso(value) for key, value in ticker_targets.items()}
            for ticker, ticker_targets in targets.get("by_ticker", {}).items()
        },
    }


def _task(task_id: str, priority: str, description: str, target: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": task_id, "priority": priority, "description": description, "target": target}


def _unique_upper(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = str(value).strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            result.append(ticker)
    return result


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(value: Any) -> str | None:
    parsed = _parse_datetime(value)
    return parsed.isoformat() if parsed else None


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
