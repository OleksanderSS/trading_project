from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


REVIEW_INTERVAL_HOURS = {
    "clean_market_15m_60m_1d": 24,
    "sector_market_evidence": 24,
    "issuer_sec_fundamentals": 168,
    "macro_context": 168,
    "semiconductor_news": 24,
    "official_policy": 168,
    "industry_operational_metrics": 168,
}


class ProspectiveAccumulationScheduleBuilder:
    """Turn an accumulation runbook into a non-executing due-work manifest."""

    contract = "dean_prospective_accumulation_schedule_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/prospective_accumulation_schedule_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        runbook_path: str | Path,
        *,
        as_of: str,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = parse_timezone_aware(as_of)
        if cutoff is None:
            raise ValueError("schedule as_of must be timezone-aware")
        path = Path(runbook_path)
        runbook = _load(path)
        if runbook.get("contract") != "dean_prospective_accumulation_runbook_v1":
            raise ValueError("unsupported accumulation runbook contract")

        work = [_schedule_lane(lane, cutoff) for lane in runbook.get("collection_lanes") or []]
        _apply_dependencies(work)
        due = sorted(
            (item for item in work if item["authorization_request_ready"]),
            key=lambda item: (item["priority"], item["lane_id"]),
        )
        created_at = utc_now_iso()
        run_id = "prospective_accumulation_schedule_" + created_at.replace(":", "").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "prospective_accumulation_schedule",
            "contract": self.contract,
            "inputs": {
                "as_of": cutoff.isoformat(),
                "runbook": {"path": str(path), "sha256": _sha256(path)},
            },
            "summary": {
                "lane_count": len(work),
                "due_lane_count": sum(item["due_for_review"] for item in work),
                "authorization_request_count": len(due),
                "dependency_blocked_count": sum(bool(item.get("dependency_blocked_by")) for item in work),
                "automatic_execution_allowed": False,
                "scheduler_write_performed": False,
                "collector_execution_performed": False,
                "can_trade": False,
            },
            "policy": {
                "interval_meaning": "maximum operational review interval, not a claim that the source economically updates on this cadence",
                "event_override": "material source events may request an earlier reviewed refresh",
                "checkpoint_override": "pre-due source review requires coverage review even when a lane is not stale",
                "authorization": "due means ready to request execution authorization, never permission to execute",
            },
            "lanes": work,
            "authorization_requests": [
                {
                    "lane_id": item["lane_id"],
                    "priority": item["priority"],
                    "reason": item["due_reason"],
                    "command": item["command"],
                    "network_or_external_access_may_occur": item["lane_id"]
                    in {"clean_market_15m_60m_1d", "issuer_sec_fundamentals", "official_policy"},
                    "approved": False,
                }
                for item in due
            ],
            "safety": {
                "review_only": True,
                "command_execution_performed": False,
                "scheduler_registration_performed": False,
                "config_write_performed": False,
                "outcome_evaluation_performed": False,
                "learning_write_performed": False,
                "broker_access_performed": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=run_id,
            )
        return payload


def _schedule_lane(lane: dict[str, Any], cutoff: Any) -> dict[str, Any]:
    lane_id = str(lane.get("lane_id"))
    interval = REVIEW_INTERVAL_HOURS.get(lane_id, 168)
    created_at = parse_timezone_aware(str(lane.get("artifact_created_at") or ""))
    next_review = created_at + timedelta(hours=interval) if created_at else None
    due = created_at is None or (next_review is not None and cutoff >= next_review)
    priority = {
        "clean_market_15m_60m_1d": 1,
        "semiconductor_news": 2,
        "issuer_sec_fundamentals": 3,
        "macro_context": 4,
        "official_policy": 5,
        "sector_market_evidence": 6,
        "industry_operational_metrics": 7,
    }.get(lane_id, 99)
    return {
        "lane_id": lane_id,
        "priority": priority,
        "review_interval_hours": interval,
        "artifact_created_at": created_at.isoformat() if created_at else None,
        "next_review_at": next_review.isoformat() if next_review else None,
        "due_for_review": bool(due),
        "due_reason": "artifact_missing" if created_at is None else ("review_interval_elapsed" if due else "not_due"),
        "runner_exists": bool(lane.get("runner_exists")),
        "command_executable": bool(lane.get("command_executable")),
        "missing_parameters": list(lane.get("missing_parameters") or []),
        "command": lane.get("command"),
        "dependency_blocked_by": [],
        "authorization_request_ready": bool(
            due and lane.get("runner_exists") and lane.get("command_executable")
        ),
        "automatic_execution_allowed": False,
    }


def _apply_dependencies(work: list[dict[str, Any]]) -> None:
    by_id = {item["lane_id"]: item for item in work}
    sector = by_id.get("sector_market_evidence")
    market = by_id.get("clean_market_15m_60m_1d")
    if sector and market and market["due_for_review"]:
        sector["dependency_blocked_by"] = ["clean_market_15m_60m_1d"]
        sector["authorization_request_ready"] = False
        sector["due_reason"] = "refresh_after_clean_market_snapshot"
    for item in work:
        if item["due_for_review"] and item["runner_exists"] and not item["command_executable"]:
            item["authorization_request_ready"] = False
            if not item["dependency_blocked_by"]:
                item["due_reason"] = "command_parameters_unresolved"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Prospective Accumulation Schedule",
        "",
        f"- Due lanes: `{summary['due_lane_count']}`",
        f"- Authorization requests: `{summary['authorization_request_count']}`",
        f"- Dependency blocked: `{summary['dependency_blocked_count']}`",
        "- Automatic execution allowed: `false`",
        "",
        "## Requests",
        "",
    ]
    lines.extend(
        f"- `{item['priority']}` `{item['lane_id']}`: {item['reason']}"
        for item in payload["authorization_requests"]
    )
    return "\n".join(lines) + "\n"


__all__ = ["ProspectiveAccumulationScheduleBuilder", "REVIEW_INTERVAL_HOURS"]
