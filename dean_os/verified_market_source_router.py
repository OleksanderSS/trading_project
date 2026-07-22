from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.replays.replay_checkpoint_due_router import measurement_price_requirements
from dean_os.schemas import utc_now_iso


DEFAULT_SOURCE_POLICY_PATH = (
    "dean_os/config/replay_verified_market_sources.template.json"
)


class VerifiedMarketSourceRouter:
    """Choose the next bounded provider without silently changing evidence."""

    contract = "dean_verified_market_source_router_v1"

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/verified_market_source_router_current"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        lifecycle_json: str | Path,
        registration_json: str | Path,
        review_gate_json: str | Path,
        source_policy_json: str | Path = DEFAULT_SOURCE_POLICY_PATH,
        previous_refresh_json_paths: list[str | Path] | None = None,
        local_snapshot_paths: list[str | Path] | None = None,
        as_of: str,
        save: bool = True,
    ) -> dict[str, Any]:
        lifecycle_path = Path(lifecycle_json)
        registration_path = Path(registration_json)
        gate_path = Path(review_gate_json)
        policy_path = Path(source_policy_json)
        lifecycle = _load(lifecycle_path)
        registration = _load(registration_path)
        gate = _load(gate_path)
        policy = _load(policy_path)
        cutoff = parse_timezone_aware(as_of)
        if cutoff is None:
            raise ValueError("source router as_of must be timezone-aware")
        cutoff = cutoff.astimezone(timezone.utc)
        _validate_policy(policy)

        previous_paths = [Path(item) for item in previous_refresh_json_paths or []]
        candidates = [Path(item) for item in local_snapshot_paths or []]
        attempts = _provider_attempts(previous_paths)
        tasks = _waiting_tasks(lifecycle, registration, gate)
        routes = [
            _route_task(
                task,
                providers=policy["providers"],
                attempts=attempts.get(task["task_id"], {}),
                candidates=candidates,
                cutoff=cutoff,
            )
            for task in tasks
        ]
        status = (
            "verified_local_snapshot_ready"
            if any(item["route_state"] == "verified_local_snapshot_ready" for item in routes)
            else "awaiting_operator_supplied_verified_snapshot"
            if any(
                item["route_state"] == "awaiting_operator_supplied_verified_snapshot"
                for item in routes
            )
            else "local_snapshot_rejected"
            if any(item["route_state"] == "local_snapshot_rejected" for item in routes)
            else "network_provider_available"
            if any(item["route_state"] == "network_provider_available" for item in routes)
            else "all_declared_providers_exhausted"
            if routes
            else "no_waiting_source_route"
        )
        created_at = utc_now_iso()
        run_id = "verified_market_source_router_" + created_at.replace(
            ":", ""
        ).replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "verified_market_source_router",
            "contract": self.contract,
            "inputs": {
                "lifecycle_json": str(lifecycle_path),
                "registration_json": str(registration_path),
                "review_gate_json": str(gate_path),
                "source_policy_json": str(policy_path),
                "previous_refresh_json_paths": [str(item) for item in previous_paths],
                "local_snapshot_paths": [str(item) for item in candidates],
                "as_of": cutoff.isoformat(),
            },
            "summary": {
                "status": status,
                "waiting_task_count": len(tasks),
                "ready_local_snapshot_count": sum(
                    item["route_state"] == "verified_local_snapshot_ready"
                    for item in routes
                ),
                "awaiting_local_snapshot_count": sum(
                    item["route_state"]
                    == "awaiting_operator_supplied_verified_snapshot"
                    for item in routes
                ),
                "automatic_provider_loop_allowed": False,
                "can_trade": False,
            },
            "provider_policy": policy,
            "provider_attempts": attempts,
            "routes": routes,
            "next_system_actions": _next_actions(routes),
            "safety": {
                "routing_only": True,
                "network_access_performed": False,
                "local_snapshot_ingested": False,
                "pipeline_context_substituted": False,
                "outcome_scoring_performed": False,
                "learning_write_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=run_id,
            )
        return payload


def validate_local_market_snapshot(
    path: str | Path,
    *,
    required_tickers: list[str],
    due_at: str,
    as_of: datetime,
) -> dict[str, Any]:
    candidate = Path(path)
    result: dict[str, Any] = {
        "path": str(candidate),
        "exists": candidate.is_file(),
        "sha256": _sha256(candidate) if candidate.is_file() else None,
        "valid": False,
        "issues": [],
    }
    if not candidate.is_file():
        result["issues"].append("snapshot_missing")
        return result
    try:
        frame = _read_table(candidate)
    except Exception as exc:
        result["issues"].append("snapshot_unreadable:" + type(exc).__name__)
        return result
    required_columns = {"datetime", "ticker", "close"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        result["issues"].append("missing_columns:" + ",".join(missing_columns))
        return result
    raw_datetime = frame["datetime"]
    if any(_timestamp_is_naive(value) for value in raw_datetime.dropna().tolist()):
        result["issues"].append("naive_datetime")
    parsed = pd.to_datetime(raw_datetime, utc=True, errors="coerce")
    if parsed.isna().any():
        result["issues"].append("invalid_or_naive_datetime")
    close = pd.to_numeric(frame["close"], errors="coerce")
    if close.isna().any() or any(not math.isfinite(float(value)) for value in close.dropna()):
        result["issues"].append("non_finite_close")
    tickers = frame["ticker"].astype(str).str.upper()
    present = sorted(tickers.dropna().unique().tolist())
    missing_tickers = sorted(set(required_tickers) - set(present))
    if missing_tickers:
        result["issues"].append("missing_tickers:" + ",".join(missing_tickers))
    due = parse_timezone_aware(due_at)
    if due is None:
        result["issues"].append("invalid_due_at")
    eligible_sessions: list[str] = []
    if due is not None and not parsed.isna().all():
        session_dates = parsed.dt.date
        for session_date in sorted(set(item for item in session_dates if pd.notna(item))):
            close_at = _session_close_utc(session_date)
            if session_date < due.date():
                continue
            if session_date == due.date() and due.astimezone(timezone.utc) > close_at:
                continue
            if close_at > as_of:
                continue
            rows = frame.loc[session_dates == session_date]
            covered = set(rows["ticker"].astype(str).str.upper())
            if set(required_tickers).issubset(covered):
                eligible_sessions.append(session_date.isoformat())
    if not eligible_sessions:
        result["issues"].append("no_complete_post_due_closed_session")
    result.update(
        {
            "row_count": len(frame),
            "present_tickers": present,
            "required_tickers": required_tickers,
            "eligible_sessions": eligible_sessions,
            "valid": not result["issues"],
        }
    )
    return result


def _route_task(
    task: dict[str, Any],
    *,
    providers: list[dict[str, Any]],
    attempts: dict[str, int],
    candidates: list[Path],
    cutoff: datetime,
) -> dict[str, Any]:
    provider_routes = []
    selected = None
    validation = None
    for provider in sorted(providers, key=lambda item: int(item.get("rank") or 999)):
        provider_id = str(provider.get("provider_id"))
        used = int(attempts.get(provider_id, 0))
        limit = int(provider.get("maximum_attempts_per_task") or 0)
        exhausted = used >= limit
        route = {
            "provider_id": provider_id,
            "rank": provider.get("rank"),
            "attempts_used": used,
            "attempt_limit": limit,
            "exhausted": exhausted,
            "automatic_execution_allowed": bool(
                provider.get("automatic_execution_allowed")
            ),
        }
        provider_routes.append(route)
        if selected is not None or exhausted:
            continue
        selected = provider
        if provider_id == "local_validated_snapshot" and candidates:
            validations = [
                validate_local_market_snapshot(
                    path,
                    required_tickers=task["required_tickers"],
                    due_at=task["due_at"],
                    as_of=cutoff,
                )
                for path in candidates
            ]
            validation = next((item for item in validations if item["valid"]), None)
            if validation is None:
                validation = {"valid": False, "candidates": validations}
        break
    if selected is None:
        state = "all_declared_providers_exhausted"
    elif selected.get("provider_id") == "local_validated_snapshot":
        state = (
            "verified_local_snapshot_ready"
            if validation and validation.get("valid")
            else "local_snapshot_rejected"
            if candidates
            else "awaiting_operator_supplied_verified_snapshot"
        )
    else:
        state = "network_provider_available"
    return {
        **task,
        "route_state": state,
        "selected_provider": selected,
        "provider_routes": provider_routes,
        "local_snapshot_validation": validation,
        "automatic_failover_executed": False,
    }


def _waiting_tasks(
    lifecycle: dict[str, Any], registration: dict[str, Any], gate: dict[str, Any]
) -> list[dict[str, Any]]:
    plans = {
        str(item.get("task_id")): item
        for item in registration.get("registration_plan") or []
    }
    specs = {
        str(item.get("hypothesis_id")): item.get("measurement_spec") or {}
        for item in gate.get("hypothesis_review") or []
    }
    tasks = []
    for action in (lifecycle.get("review_inbox") or {}).get("data_actions") or []:
        task_id = str(action.get("task_id"))
        plan = plans.get(task_id) or {}
        hypothesis_id = str(plan.get("hypothesis_id") or action.get("hypothesis_id") or "")
        requirements = measurement_price_requirements(specs.get(hypothesis_id, {}))
        tasks.append(
            {
                "task_id": task_id,
                "hypothesis_id": hypothesis_id,
                "due_at": str(plan.get("due_at") or action.get("due_at")),
                "required_tickers": requirements.get("required_tickers") or [],
                "requirement_type": requirements.get("requirement_type"),
            }
        )
    return tasks


def _provider_attempts(paths: list[Path]) -> dict[str, dict[str, int]]:
    attempts: dict[str, dict[str, int]] = {}
    for path in paths:
        if not path.is_file():
            continue
        payload = _load(path)
        if not (payload.get("inputs") or {}).get("apply_refresh"):
            continue
        for job in payload.get("refresh_jobs") or []:
            task_id = str(job.get("task_id") or "")
            provider = str(job.get("provider") or "")
            if task_id and provider:
                bucket = attempts.setdefault(task_id, {})
                bucket[provider] = bucket.get(provider, 0) + 1
    return attempts


def _next_actions(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in routes:
        state = item["route_state"]
        if state == "awaiting_operator_supplied_verified_snapshot":
            result.append(
                {
                    "action_type": "supply_local_verified_market_snapshot",
                    "task_id": item["task_id"],
                    "required_tickers": item["required_tickers"],
                    "automatic_execution_allowed": False,
                }
            )
        elif state == "verified_local_snapshot_ready":
            result.append(
                {
                    "action_type": "rerun_replay_outcome_lifecycle_with_validated_snapshot",
                    "task_id": item["task_id"],
                    "snapshot": item["local_snapshot_validation"],
                    "automatic_execution_allowed": True,
                }
            )
        elif state == "local_snapshot_rejected":
            result.append(
                {
                    "action_type": "repair_or_replace_local_snapshot",
                    "task_id": item["task_id"],
                    "automatic_execution_allowed": False,
                }
            )
    return result


def _validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("contract") != "dean_verified_market_source_policy_v1":
        raise ValueError("unsupported verified market source policy contract")
    providers = policy.get("providers") or []
    ids = [str(item.get("provider_id")) for item in providers]
    if not providers or len(ids) != len(set(ids)):
        raise ValueError("provider policy must contain unique providers")
    if (policy.get("failover_policy") or {}).get(
        "automatic_multi_provider_loop_allowed"
    ) is not False:
        raise ValueError("automatic multi-provider loops must be disabled")


def _session_close_utc(session_date: date) -> datetime:
    return datetime.combine(
        session_date, time(16, 0), tzinfo=ZoneInfo("America/New_York")
    ).astimezone(timezone.utc)


def _timestamp_is_naive(value: Any) -> bool:
    try:
        return pd.Timestamp(value).tzinfo is None
    except Exception:
        return False


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Verified Market Source Router",
        "",
        f"- Status: `{summary['status']}`",
        f"- Waiting tasks: `{summary['waiting_task_count']}`",
        f"- Ready local snapshots: `{summary['ready_local_snapshot_count']}`",
        f"- Awaiting local snapshots: `{summary['awaiting_local_snapshot_count']}`",
        "",
    ]
    for item in payload["routes"]:
        selected = item.get("selected_provider") or {}
        lines.append(
            f"- `{item['task_id']}`: {item['route_state']} / provider={selected.get('provider_id')}"
        )
    lines.extend(
        [
            "",
            "Provider failover is bounded and never loops automatically.",
            "Pipeline context cannot replace the primary verified outcome source.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


__all__ = ["VerifiedMarketSourceRouter", "validate_local_market_snapshot"]
