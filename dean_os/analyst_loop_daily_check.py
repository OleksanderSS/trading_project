from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.agent_learning_loop_runbook import AgentLearningLoopRunbook, DEFAULT_STAGE_PATHS
from dean_os.agents.market_data_freshness import inspect_market_data_freshness
from dean_os.event_log import EventLog
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


DEFAULT_EVENT_LOG_PATH = "logs/dean_os/events.jsonl"


class AnalystLoopDailyCheck:
    """Read-only daily operator check for the analyst learning loop."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/analyst_loop_daily_check"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        stage_paths: dict[str, str | Path | None] | None = None,
        market_data_path: str | Path | None = None,
        latest_processed_prices: str | None = "1d",
        tickers: list[str] | None = None,
        as_of: str | datetime | None = None,
        max_age_hours: float = 72.0,
        close_col: str = "close",
        datetime_col: str = "datetime",
        event_log_path: str | Path = DEFAULT_EVENT_LOG_PATH,
        event_limit: int = 10,
        save: bool = True,
    ) -> dict[str, Any]:
        resolved_stage_paths = _resolve_stage_paths(stage_paths)
        runbook = AgentLearningLoopRunbook().build(stage_paths=resolved_stage_paths, save=False)
        market_freshness = inspect_market_data_freshness(
            market_data_path=market_data_path,
            latest_processed_prices=latest_processed_prices,
            tickers=tickers or [],
            as_of=_parse_as_of(as_of),
            max_age_hours=max_age_hours,
            close_col=close_col,
            datetime_col=datetime_col,
        )
        evidence_pack = _summarize_evidence_pack(resolved_stage_paths.get("evidence_pack", ""))
        profile_scorecard = _summarize_profile_scorecard(resolved_stage_paths.get("profile_scorecard", ""))
        event_log = _summarize_event_log(event_log_path, event_limit)
        blockers, warnings = _assess_daily_state(
            runbook=runbook,
            market_freshness=market_freshness,
            evidence_pack=evidence_pack,
            profile_scorecard=profile_scorecard,
            event_log=event_log,
        )
        decision = "blocked" if blockers else "needs_operator_review" if warnings else "safe_to_continue"
        position = runbook.get("loop_position", {})
        payload = {
            "run_id": _run_id("analyst_loop_daily_check"),
            "created_at": utc_now_iso(),
            "mode": "analyst_loop_daily_check",
            "inputs": {
                "stage_paths": resolved_stage_paths,
                "market_data_path": str(market_data_path) if market_data_path else None,
                "latest_processed_prices": latest_processed_prices,
                "tickers": [ticker.upper() for ticker in tickers or [] if str(ticker).strip()],
                "as_of": _parse_as_of(as_of).isoformat(),
                "max_age_hours": max_age_hours,
                "close_col": close_col,
                "datetime_col": datetime_col,
                "event_log_path": str(event_log_path),
                "event_limit": event_limit,
            },
            "summary": {
                "decision": decision,
                "blocker_count": len(blockers),
                "warning_count": len(warnings),
                "current_stage": position.get("stage_id"),
                "current_status": position.get("status"),
                "next_command": position.get("next_command"),
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
                "stage_execution_performed": False,
            },
            "checks": {
                "learning_loop": _compact_runbook(runbook),
                "market_freshness": market_freshness,
                "evidence_pack": evidence_pack,
                "profile_scorecard": profile_scorecard,
                "event_log": event_log,
            },
            "blockers": blockers,
            "warnings": warnings,
            "operator_actions": _operator_actions(decision, position, blockers, warnings, market_freshness),
            "recommendations": _recommendations(decision, blockers, warnings),
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
        rendered_md = render_analyst_loop_daily_check_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_analyst_loop_daily_check_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    checks = payload.get("checks", {})
    market = checks.get("market_freshness", {})
    loop = checks.get("learning_loop", {})
    lines = [
        "# DEAN-OS Analyst Loop Daily Check",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Decision: `{summary.get('decision')}`",
        f"- Current stage: `{summary.get('current_stage')}`",
        f"- Current status: `{summary.get('current_status')}`",
        f"- Market data status: `{market.get('status')}`",
        f"- Config write performed: {summary.get('config_write_performed')}",
        f"- Pipeline run performed: {summary.get('pipeline_run_performed')}",
        f"- Broker access performed: {summary.get('broker_access_performed')}",
        "",
        "## Next Action",
        "",
        f"- Command: `{summary.get('next_command')}`",
        f"- Stop reason: {loop.get('stop_reason') or 'none'}",
        "",
        "## Blockers",
        "",
    ]
    blockers = payload.get("blockers", [])
    lines.extend(f"- {item.get('code')}: {item.get('message')}" for item in blockers)
    if not blockers:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings", [])
    lines.extend(f"- {item.get('code')}: {item.get('message')}" for item in warnings)
    if not warnings:
        lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _resolve_stage_paths(stage_paths: dict[str, str | Path | None] | None) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in {**DEFAULT_STAGE_PATHS, **(stage_paths or {})}.items()
        if value is not None
    }


def _compact_runbook(runbook: dict[str, Any]) -> dict[str, Any]:
    summary = runbook.get("summary", {})
    position = runbook.get("loop_position", {})
    return {
        "stage_count": summary.get("stage_count"),
        "available_artifact_count": summary.get("available_artifact_count"),
        "current_stage": summary.get("current_stage"),
        "current_status": summary.get("current_status"),
        "stop_reason": position.get("stop_reason"),
        "next_command": position.get("next_command"),
        "operator_action": position.get("operator_action"),
        "stages": [
            {
                "stage_id": stage.get("stage_id"),
                "status": stage.get("status"),
                "artifact_exists": stage.get("artifact_exists"),
                "is_stop": stage.get("is_stop"),
            }
            for stage in runbook.get("stages", [])
        ],
    }


def _summarize_evidence_pack(path: str | Path) -> dict[str, Any]:
    payload, status = _load_json_payload(path)
    if status != "loaded":
        return {"status": status, "path": str(path), "agent_lab_ready": False, "warnings": [status]}
    coverage = payload.get("coverage", {})
    analyst_inputs = payload.get("analyst_inputs", {})
    warnings: list[str] = []
    if not coverage.get("agent_lab_ready"):
        warnings.append("agent_lab_not_ready")
    if coverage.get("data_quality") and coverage.get("data_quality") != "clean":
        warnings.append(f"data_quality_{coverage.get('data_quality')}")
    if coverage.get("missing_requested_tickers"):
        warnings.append("missing_requested_tickers")
    return {
        "status": "ready" if coverage.get("agent_lab_ready") else "blocked",
        "path": str(path),
        "document_count": int(coverage.get("document_count", 0) or 0),
        "data_quality": coverage.get("data_quality"),
        "agent_lab_ready": bool(coverage.get("agent_lab_ready")),
        "source_types": sorted((coverage.get("by_source_type") or {}).keys()),
        "tickers": coverage.get("tickers", []),
        "missing_requested_tickers": coverage.get("missing_requested_tickers", []),
        "candidate_profiles": (analyst_inputs.get("manager_plan") or {}).get("candidate_profiles", []),
        "warnings": warnings,
    }


def _summarize_profile_scorecard(path: str | Path) -> dict[str, Any]:
    payload, status = _load_json_payload(path)
    if status != "loaded":
        return {"status": status, "path": str(path), "warnings": [status]}
    summary = payload.get("summary", {})
    ready_profiles = summary.get("activation_ready_profiles", []) or []
    keep_candidate = summary.get("keep_candidate_profiles", []) or []
    blocked_profiles = summary.get("blocked_profiles", []) or []
    if ready_profiles:
        status_value = "ready_profiles_pending_review"
    elif keep_candidate:
        status_value = "keep_candidate"
    elif blocked_profiles or summary.get("profile_count", 0):
        status_value = "gated"
    else:
        status_value = "no_profiles"
    warnings: list[str] = []
    if ready_profiles:
        warnings.append("ready_profiles_require_manual_review")
    if status_value == "no_profiles":
        warnings.append("no_profiles_scored")
    return {
        "status": status_value,
        "path": str(path),
        "profile_count": int(summary.get("profile_count", 0) or 0),
        "activation_ready_profiles": ready_profiles,
        "keep_candidate_profiles": keep_candidate,
        "blocked_profiles": blocked_profiles,
        "warnings": warnings,
    }


def _summarize_event_log(path: str | Path, limit: int) -> dict[str, Any]:
    log = EventLog(path)
    summary = log.summary()
    recent = log.read(limit=max(limit, 0)) if limit else []
    return {
        "status": "available" if summary.get("event_count", 0) else "empty",
        "log_path": str(path),
        "event_count": summary.get("event_count", 0),
        "event_counts": summary.get("event_counts", {}),
        "source_counts": summary.get("source_counts", {}),
        "latest_event": summary.get("latest_event"),
        "recent_events": recent,
    }


def _load_json_payload(path: str | Path) -> tuple[dict[str, Any], str]:
    resolved = Path(path)
    if not resolved.exists():
        return {}, "missing_artifact"
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return {}, "unreadable_json"
    if not isinstance(payload, dict):
        return {}, "invalid_json"
    return payload, "loaded"


def _assess_daily_state(
    runbook: dict[str, Any],
    market_freshness: dict[str, Any],
    evidence_pack: dict[str, Any],
    profile_scorecard: dict[str, Any],
    event_log: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    loop = runbook.get("loop_position", {})
    loop_status = str(loop.get("status") or "")
    loop_stage = str(loop.get("stage_id") or "")
    hard_loop_statuses = {"missing_artifact", "unreadable_json", "invalid_json", "blocked", "blocked_need_newer_prices"}
    soft_loop_statuses = {
        "waiting_for_horizon",
        "no_pending_records",
        "no_ready_profiles",
        "no_calibration_proposals",
        "operation_queue_empty",
        "no_manual_tasks_in_scope",
        "gated",
        "no_profiles",
    }
    if loop_status in hard_loop_statuses:
        blockers.append(_issue("learning_loop_blocked", f"{loop_stage} is {loop_status}: {loop.get('stop_reason')}"))
    elif loop_status in soft_loop_statuses:
        warnings.append(_issue("learning_loop_waiting", f"{loop_stage} is {loop_status}: {loop.get('stop_reason')}"))
    elif loop.get("stop_reason"):
        warnings.append(_issue("learning_loop_manual_review", str(loop.get("stop_reason"))))

    market_status = market_freshness.get("status")
    if market_status == "unavailable":
        blockers.append(_issue("market_data_unavailable", str(market_freshness.get("reason", "Market data unavailable."))))
    elif market_status == "stale":
        warnings.append(
            _issue(
                "market_data_stale",
                f"Market data age {market_freshness.get('age_hours')}h exceeds {market_freshness.get('max_age_hours')}h or tickers are missing.",
            )
        )

    if evidence_pack.get("status") in {"missing_artifact", "unreadable_json", "invalid_json", "blocked"}:
        blockers.append(_issue("evidence_pack_not_ready", f"Evidence pack status is {evidence_pack.get('status')}."))
    for warning in evidence_pack.get("warnings", []):
        if warning != "agent_lab_not_ready":
            warnings.append(_issue("evidence_pack_warning", str(warning)))

    if profile_scorecard.get("status") in {"missing_artifact", "unreadable_json", "invalid_json"}:
        warnings.append(_issue("profile_scorecard_missing", f"Profile scorecard status is {profile_scorecard.get('status')}."))
    for warning in profile_scorecard.get("warnings", []):
        warnings.append(_issue("profile_scorecard_warning", str(warning)))

    if event_log.get("status") == "empty":
        warnings.append(_issue("event_log_empty", "No DEAN-OS events were found in the configured log."))
    return _dedupe_issues(blockers), _dedupe_issues(warnings)


def _operator_actions(
    decision: str,
    position: dict[str, Any],
    blockers: list[dict[str, str]],
    warnings: list[dict[str, str]],
    market_freshness: dict[str, Any],
) -> list[str]:
    actions = [f"Decision is {decision}; do not skip review gates."]
    if blockers:
        actions.append("Resolve blockers before running the next analyst-learning stage.")
    elif warnings:
        actions.append("Review warnings before spending resources on new analyst runs.")
    else:
        actions.append("It is safe to inspect the next command, but still keep apply/config changes gated.")
    next_command = position.get("next_command")
    if next_command:
        actions.append(f"Next safe command from runbook: {next_command}")
    if market_freshness.get("status") in {"stale", "unavailable"}:
        actions.append("Refresh or inspect local market data before trusting outcome evaluation or regime context.")
    return actions


def _recommendations(decision: str, blockers: list[dict[str, str]], warnings: list[dict[str, str]]) -> list[str]:
    if decision == "blocked":
        return [
            "Treat the stop as a safety success, not a failure.",
            "Fix the first blocker before running more analyst/promotion/calibration commands.",
            "Do not apply learning, enqueue calibration, or change config from this report.",
        ]
    if decision == "needs_operator_review":
        return [
            "Review warnings and decide whether the next read-only command is worth running.",
            "Prefer improving evidence quality or data freshness before expanding specialist profiles.",
            "Keep all promotion/calibration steps behind explicit review.",
        ]
    return [
        "No blockers or warnings were detected by the daily check.",
        "Continue with the runbook next command only as a reviewed operation.",
        "Keep live trading and production config writes disabled unless explicitly approved.",
    ]


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _dedupe_issues(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        key = (item.get("code", ""), item.get("message", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _parse_as_of(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
