from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


DEFAULT_REGISTRATION_PATH = (
    "reports/dean_os/world_model_replay_registration_approved_current/latest.json"
)
DEFAULT_REVIEW_GATE_PATH = (
    "reports/dean_os/world_model_replay_review_gate_approved_current/latest.json"
)
DEFAULT_OUTCOME_REVIEW_PATH = (
    "reports/dean_os/historical_replay_outcome_review_current/latest.json"
)
DEFAULT_VERIFIED_PRICE_PATH = (
    "data/dean_os/historical_outcome_market_snapshots/latest.parquet"
)
DEFAULT_PIPELINE_PATHS = (
    "data/colab/accumulated/main_database/features.parquet",
    "data/colab/regenerated/semiconductor_clean_1d_stage23/features.parquet",
)


class ReplayCheckpointDueRouter:
    """Route replay checkpoints without mistaking a clock deadline for evidence.

    The router is deliberately read-only.  A market checkpoint becomes ready only
    when a verified daily-price artifact contains a qualifying market session and
    that session has closed by ``as_of``.  Pipeline features are inventoried as
    secondary context, but never silently replace the verified outcome lane.
    """

    contract = "dean_replay_checkpoint_due_router_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/replay_checkpoint_due_router_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        registration_json: str | Path,
        review_gate_json: str | Path,
        *,
        as_of: str,
        verified_price_paths: Iterable[str | Path] = (),
        pipeline_paths: Iterable[str | Path] = (),
        outcome_json_paths: Iterable[str | Path] = (),
        due_soon_days: int = 3,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = parse_timezone_aware(as_of)
        if cutoff is None:
            raise ValueError("router as_of must be timezone-aware")
        cutoff = cutoff.astimezone(timezone.utc)
        if due_soon_days < 0:
            raise ValueError("due_soon_days must be non-negative")

        registration_path = Path(registration_json)
        gate_path = Path(review_gate_json)
        registration = _load(registration_path)
        gate = _load(gate_path)
        if registration.get("contract") != "dean_world_model_replay_registration_bridge_v1":
            raise ValueError("unsupported replay registration contract")
        source_gate = registration.get("source_gate") or {}
        bound_gate_sha = source_gate.get("sha256")
        if bound_gate_sha and bound_gate_sha != _sha256(gate_path):
            raise ValueError("registration source gate SHA-256 does not match supplied gate")

        specs = _measurement_specs(gate)
        price_paths = [Path(path) for path in verified_price_paths]
        pipeline_paths = [Path(path) for path in pipeline_paths]
        outcome_paths = [Path(path) for path in outcome_json_paths]
        sessions, price_inventory = _verified_price_sessions(price_paths)
        pipeline_inventory = [_table_inventory(path) for path in pipeline_paths]
        reviewed, outcome_review_inventory = _reviewed_checkpoints(
            outcome_paths,
            registration_sha256=_sha256(registration_path),
        )
        deferred = {
            str(item.get("task_id"))
            for item in registration.get("deferred_historical_tasks") or []
        }

        routes = []
        for task in registration.get("registration_plan") or []:
            task_id = str(task.get("task_id") or "")
            hypothesis_id = str(task.get("hypothesis_id") or "")
            spec = specs.get(hypothesis_id, {})
            route = _route_task(
                task,
                spec=spec,
                cutoff=cutoff,
                sessions=sessions,
                reviewed=reviewed.get(task_id),
                due_soon_days=due_soon_days,
            )
            route["registration_lineage"] = (
                "deferred_historical_review" if task_id in deferred else "registered_or_existing"
            )
            routes.append(route)

        counts = Counter(route["route_state"] for route in routes)
        matured = [
            _chief_checkpoint_item(route)
            for route in routes
            if route["route_state"] == "matured_pending_outcome_review"
        ]
        waiting = [
            _chief_checkpoint_item(route)
            for route in routes
            if route["route_state"] == "due_waiting_for_verified_checkpoint_data"
        ]
        # Due-soon checkpoints are intentionally excluded from pending decisions.
        # They remain machine-readable schedule context, not operator work.
        due_soon = [
            _chief_checkpoint_item(route)
            for route in routes
            if route["route_state"] == "future_silent" and route["due_soon"]
        ]
        reviewed_count = sum(
            count for state, count in counts.items() if state.startswith("reviewed_")
        )
        inbox_status = (
            "matured_checkpoints_require_outcome_review"
            if matured
            else "due_checkpoints_waiting_for_verified_data"
            if waiting
            else "no_checkpoint_action_required"
        )

        created_at = utc_now_iso()
        run_id = "replay_checkpoint_due_router_" + created_at.replace(":", "").replace(
            "+00:00", "Z"
        )
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "replay_checkpoint_due_router",
            "contract": self.contract,
            "inputs": {
                "as_of": cutoff.isoformat(),
                "due_soon_days": due_soon_days,
                "registration": _path_ref(registration_path),
                "review_gate": _path_ref(gate_path),
                "verified_price_paths": [_path_ref(path) for path in price_paths],
                "pipeline_paths": [_path_ref(path) for path in pipeline_paths],
                "outcome_json_paths": [_path_ref(path) for path in outcome_paths],
            },
            "summary": {
                "task_count": len(routes),
                "route_state_counts": dict(sorted(counts.items())),
                "future_silent_count": counts.get("future_silent", 0),
                "due_soon_silent_count": len(due_soon),
                "waiting_for_verified_data_count": len(waiting),
                "matured_pending_outcome_review_count": len(matured),
                "reviewed_checkpoint_count": reviewed_count,
                "operator_decision_count": len(matured),
                "automatic_outcome_scoring_allowed": False,
                "can_trade": False,
            },
            "price_inventory": price_inventory,
            "pipeline_inventory": pipeline_inventory,
            "outcome_review_inventory": outcome_review_inventory,
            "routes": routes,
            "chief_review_inbox": {
                "status": inbox_status,
                "matured_checkpoints": matured,
                "data_accrual_actions": waiting,
                "pending_decisions": [
                    {
                        **item,
                        "decision_type": "checkpoint_outcome_review",
                        "allowed_decisions": [
                            "record_observed_outcome",
                            "record_unobservable_outcome",
                            "defer_for_missing_evidence",
                        ],
                    }
                    for item in matured
                ],
                "due_soon_silent_count": len(due_soon),
                "future_checkpoints_are_operator_actions": False,
            },
            "routing_policy": {
                "future": "Keep silent in the operator inbox, including due-soon checkpoints.",
                "market_maturity": (
                    "A due timestamp is necessary but insufficient. A declared market checkpoint "
                    "also requires a qualifying verified daily-price session whose US market close "
                    "is not later than as_of."
                ),
                "missing_verified_data": (
                    "Route to data accrual, not outcome judgment. Pipeline data remains secondary "
                    "context and cannot silently replace the verified price lane."
                ),
                "reviewed": "Do not re-open a task already present in a supplied outcome review.",
            },
            "safety": {
                "review_only": True,
                "price_collection_performed": False,
                "pipeline_execution_performed": False,
                "outcome_scoring_performed": False,
                "outcome_write_performed": False,
                "learning_write_performed": False,
                "registration_performed": False,
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


def _route_task(
    task: dict[str, Any],
    *,
    spec: dict[str, Any],
    cutoff: datetime,
    sessions: dict[str, set[date]],
    reviewed: dict[str, Any] | None,
    due_soon_days: int,
) -> dict[str, Any]:
    due = parse_timezone_aware(str(task.get("due_at") or ""))
    if due is None:
        raise ValueError(f"task {task.get('task_id')} has invalid due_at")
    due = due.astimezone(timezone.utc)
    horizon = int(task.get("horizon_days") or 0)
    primary_horizon = spec.get("primary_horizon_days")
    requirements = measurement_price_requirements(spec)
    checkpoint = _checkpoint_availability(
        due=due,
        cutoff=cutoff,
        sessions=sessions,
        requirements=requirements,
    )
    if reviewed is not None:
        label = str(reviewed.get("result_label") or reviewed.get("review_status") or "recorded")
        state = "reviewed_" + _slug(label)
    elif cutoff < due:
        state = "future_silent"
    elif requirements["required_tickers"] and not checkpoint["ready"]:
        state = "due_waiting_for_verified_checkpoint_data"
    else:
        state = "matured_pending_outcome_review"

    seconds_until_due = (due - cutoff).total_seconds()
    due_soon = 0 <= seconds_until_due <= due_soon_days * 86400
    return {
        "task_id": task.get("task_id"),
        "hypothesis_id": task.get("hypothesis_id"),
        "horizon_days": horizon,
        "checkpoint_role": (
            "primary_hypothesis_outcome"
            if primary_horizon is not None and horizon == int(primary_horizon)
            else "intermediate_event_response_checkpoint"
        ),
        "due_at": due.isoformat(),
        "route_state": state,
        "operator_action_required": state == "matured_pending_outcome_review",
        "due_soon": due_soon,
        "days_until_due": max(0.0, seconds_until_due / 86400),
        "price_requirements": requirements,
        "checkpoint_data": checkpoint,
        "existing_review": reviewed,
    }


def _checkpoint_availability(
    *,
    due: datetime,
    cutoff: datetime,
    sessions: dict[str, set[date]],
    requirements: dict[str, Any],
) -> dict[str, Any]:
    tickers = requirements["required_tickers"]
    if not tickers:
        return {
            "ready": True,
            "status": "no_verified_price_gate_declared",
            "checkpoint_session": None,
            "covered_tickers": [],
            "missing_tickers": [],
        }

    candidate_dates = sorted(set().union(*(sessions.get(ticker, set()) for ticker in tickers)))
    for session_date in candidate_dates:
        close_at = _us_session_close_utc(session_date)
        if session_date < due.date():
            continue
        if session_date == due.date() and due > close_at:
            continue
        if close_at > cutoff:
            continue
        covered = [ticker for ticker in tickers if session_date in sessions.get(ticker, set())]
        benchmark = requirements.get("benchmark")
        members = requirements.get("members") or []
        if benchmark:
            ready = benchmark in covered and sum(ticker in covered for ticker in members) >= int(
                requirements.get("minimum_member_coverage") or len(members)
            )
        else:
            ready = all(ticker in covered for ticker in tickers)
        if ready:
            return {
                "ready": True,
                "status": "verified_checkpoint_session_available",
                "checkpoint_session": session_date.isoformat(),
                "session_close_at": close_at.isoformat(),
                "covered_tickers": covered,
                "missing_tickers": [ticker for ticker in tickers if ticker not in covered],
            }
    latest = {
        ticker: max(values).isoformat() if values else None
        for ticker, values in ((ticker, sessions.get(ticker, set())) for ticker in tickers)
    }
    return {
        "ready": False,
        "status": "verified_checkpoint_session_not_available",
        "checkpoint_session": None,
        "covered_tickers": [],
        "missing_tickers": tickers,
        "latest_verified_session_by_ticker": latest,
    }


def _us_session_close_utc(session_date: date) -> datetime:
    local = datetime.combine(session_date, time(16, 0), tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def _measurement_specs(gate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("hypothesis_id")): dict(item.get("measurement_spec") or {})
        for item in gate.get("hypothesis_review") or []
    }


def measurement_price_requirements(spec: dict[str, Any]) -> dict[str, Any]:
    context = spec.get("measurement_context") or {}
    basket = context.get("capital_equipment_basket") or {}
    raw_members = basket.get("members") or []
    members = [
        str(item.get("ticker") if isinstance(item, dict) else item).upper()
        for item in raw_members
        if (item.get("ticker") if isinstance(item, dict) else item)
    ]
    benchmark = str(basket.get("benchmark") or "").upper() or None
    if members and benchmark:
        return {
            "requirement_type": "basket_relative_return",
            "members": members,
            "benchmark": benchmark,
            "minimum_member_coverage": int(
                basket.get("minimum_checkpoint_coverage") or len(members)
            ),
            "required_tickers": list(dict.fromkeys([*members, benchmark])),
        }
    metrics = [str(item).lower() for item in spec.get("target_metrics") or []]
    if any("amat_or_equipment_basket_relative_total_return" in item for item in metrics):
        return {
            "requirement_type": "single_ticker_or_basket_relative_return",
            "members": [],
            "benchmark": None,
            "minimum_member_coverage": 1,
            "required_tickers": ["AMAT"],
        }
    return {
        "requirement_type": "no_verified_price_gate_declared",
        "members": [],
        "benchmark": None,
        "minimum_member_coverage": 0,
        "required_tickers": [],
    }


def _verified_price_sessions(paths: list[Path]) -> tuple[dict[str, set[date]], list[dict[str, Any]]]:
    sessions: dict[str, set[date]] = {}
    inventory = []
    for path in paths:
        item = _table_inventory(path)
        item["lane_authority"] = "verified_outcome_price"
        inventory.append(item)
        if not path.is_file() or item.get("status") != "readable":
            continue
        frame = _read_table(path)
        if "datetime" not in frame or "ticker" not in frame:
            continue
        dates = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dt.date
        tickers = frame["ticker"].astype(str).str.upper()
        for ticker, session_date in zip(tickers, dates):
            if pd.isna(session_date):
                continue
            sessions.setdefault(ticker, set()).add(session_date)
    return sessions, inventory


def _reviewed_checkpoints(
    paths: list[Path], *, registration_sha256: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    reviewed: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            inventory.append(
                {"path": str(path), "status": "missing", "accepted_for_deduplication": False}
            )
            continue
        try:
            payload = _load(path)
        except Exception as exc:
            inventory.append(
                {
                    "path": str(path),
                    "status": "unreadable",
                    "error": repr(exc),
                    "accepted_for_deduplication": False,
                }
            )
            continue
        binding = (payload.get("inputs") or {}).get("registration") or {}
        bound_sha = binding.get("sha256")
        accepted = bound_sha == registration_sha256
        inventory.append(
            {
                "path": str(path),
                "status": "lineage_verified" if accepted else "registration_lineage_mismatch",
                "contract": payload.get("contract"),
                "bound_registration_sha256": bound_sha,
                "expected_registration_sha256": registration_sha256,
                "accepted_for_deduplication": accepted,
            }
        )
        if not accepted:
            continue
        for key in ("checkpoint_reviews", "outcomes"):
            for item in payload.get(key) or []:
                task_id = str(item.get("task_id") or "")
                if not task_id:
                    continue
                existing = reviewed.setdefault(
                    task_id,
                    {
                        "task_id": task_id,
                        "source_paths": [],
                        "review_status": None,
                        "result_label": None,
                    },
                )
                existing["source_paths"].append(str(path))
                if item.get("review_status") is not None:
                    existing["review_status"] = item.get("review_status")
                if item.get("result_label") is not None:
                    existing["result_label"] = item.get("result_label")
    return reviewed, inventory


def _table_inventory(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        item["status"] = "missing"
        return item
    try:
        frame = _read_table(path)
        item.update(
            {
                "status": "readable",
                "sha256": _sha256(path),
                "row_count": len(frame),
                "columns": list(frame.columns),
                "tickers": sorted(frame["ticker"].dropna().astype(str).str.upper().unique().tolist())
                if "ticker" in frame
                else [],
                "latest_datetime": _latest_datetime(frame),
            }
        )
    except Exception as exc:
        item.update({"status": "unreadable", "error": repr(exc)})
    return item


def _latest_datetime(frame: pd.DataFrame) -> str | None:
    if "datetime" not in frame:
        return None
    values = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dropna()
    return values.max().isoformat() if not values.empty else None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _chief_checkpoint_item(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": route.get("task_id"),
        "hypothesis_id": route.get("hypothesis_id"),
        "horizon_days": route.get("horizon_days"),
        "checkpoint_role": route.get("checkpoint_role"),
        "due_at": route.get("due_at"),
        "route_state": route.get("route_state"),
        "checkpoint_session": (route.get("checkpoint_data") or {}).get("checkpoint_session"),
    }


def _path_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_") or "recorded"


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    inbox = payload["chief_review_inbox"]
    lines = [
        "# Replay Checkpoint Due Router",
        "",
        f"- As of: `{payload['inputs']['as_of']}`",
        f"- Tasks: `{summary['task_count']}`",
        f"- Future and silent: `{summary['future_silent_count']}`",
        f"- Due soon but silent: `{summary['due_soon_silent_count']}`",
        f"- Waiting for verified data: `{summary['waiting_for_verified_data_count']}`",
        f"- Matured outcome reviews: `{summary['matured_pending_outcome_review_count']}`",
        f"- Already reviewed: `{summary['reviewed_checkpoint_count']}`",
        "",
        "## Operator Inbox",
        "",
        f"- Status: `{inbox['status']}`",
    ]
    for item in inbox["matured_checkpoints"]:
        lines.append(
            f"- REVIEW `{item['task_id']}`: {item['horizon_days']}d, session={item['checkpoint_session']}"
        )
    for item in inbox["data_accrual_actions"]:
        lines.append(f"- DATA `{item['task_id']}`: verified checkpoint session not available")
    lines.extend(
        [
            "",
            "Future checkpoints are intentionally absent from the operator action list.",
            "Pipeline artifacts are retained as secondary context and do not replace verified outcome prices.",
            "No outcome scoring, learning update, registration, or trading was performed.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


__all__ = ["ReplayCheckpointDueRouter", "measurement_price_requirements"]
