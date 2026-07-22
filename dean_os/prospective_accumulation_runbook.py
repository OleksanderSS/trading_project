from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


class ProspectiveAccumulationRunbookBuilder:
    """Build a review-only, checkpoint-bound evidence accumulation runbook."""

    contract = "dean_prospective_accumulation_runbook_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/prospective_accumulation_runbook_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        evidence_plan_path: str | Path,
        checkpoint_monitor_path: str | Path,
        *,
        as_of: str,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = parse_timezone_aware(as_of)
        if cutoff is None:
            raise ValueError("runbook as_of must be timezone-aware")
        plan_path = Path(evidence_plan_path)
        monitor_path = Path(checkpoint_monitor_path)
        plan = _load(plan_path)
        monitor = _load(monitor_path)
        if plan.get("contract") != "dean_replay_outcome_evidence_plan_v1":
            raise ValueError("unsupported replay evidence plan contract")
        if monitor.get("contract") != "dean_replay_checkpoint_monitor_v1":
            raise ValueError("unsupported replay checkpoint monitor contract")
        plan_sha = _sha256(plan_path)
        bound_sha = ((monitor.get("inputs") or {}).get("evidence_plan") or {}).get("sha256")
        if bound_sha != plan_sha:
            raise ValueError("checkpoint monitor is not bound to the current evidence plan")

        lanes = [_inspect_lane(spec, cutoff) for spec in _lane_specs()]
        task_count = int((monitor.get("summary") or {}).get("task_count") or 0)
        due = _checkpoint_dates(monitor)
        commands_ready = sum(lane["runner_exists"] for lane in lanes)
        artifacts_ready = sum(lane["artifact_exists"] for lane in lanes)
        created_at = utc_now_iso()
        run_id = "prospective_accumulation_runbook_" + created_at.replace(":", "").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "prospective_accumulation_runbook",
            "contract": self.contract,
            "inputs": {
                "as_of": cutoff.isoformat(),
                "evidence_plan": {"path": str(plan_path), "sha256": plan_sha},
                "checkpoint_monitor": {"path": str(monitor_path), "sha256": _sha256(monitor_path)},
            },
            "summary": {
                "replay_task_count": task_count,
                "lane_count": len(lanes),
                "runner_ready_count": commands_ready,
                "artifact_present_count": artifacts_ready,
                "missing_runner_count": len(lanes) - commands_ready,
                "missing_artifact_count": len(lanes) - artifacts_ready,
                "nearest_pre_due_review": due.get("nearest_pre_due_review"),
                "nearest_outcome_review": due.get("nearest_outcome_review"),
                "accumulation_can_start": task_count > 0 and commands_ready > 0,
                "automatic_execution_allowed": False,
                "early_outcome_evaluation_allowed": False,
                "can_trade": False,
            },
            "checkpoint_dates": due,
            "collection_lanes": lanes,
            "operator_sequence": _operator_sequence(lanes),
            "scheduling_policy": {
                "market_prices": "refresh each reviewed market-data session; retain immutable 15m/60m/1d snapshots",
                "news": "refresh daily and after material domain events; preserve publication and ingestion timestamps",
                "macro": "refresh after registered-series releases; never infer a vintage timestamp",
                "issuer_filings": "refresh after accepted filings and before each pre-due review",
                "official_policy": "refresh after official publication and before each pre-due review",
                "industry_operations": "manual source onboarding until a reviewed structured feed exists",
                "checkpoint_rule": "run source-coverage review at pre_due; evaluate outcomes only at or after due_at",
            },
            "safety": {
                "review_only": True,
                "collector_execution_performed": False,
                "scheduler_write_performed": False,
                "config_write_performed": False,
                "pipeline_run_performed": False,
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


def _lane_specs() -> list[dict[str, Any]]:
    return [
        {
            "lane_id": "clean_market_15m_60m_1d",
            "role": "outcome_and_market_context",
            "runner": "run_agent_clean_yahoo_market_snapshot.py",
            "artifact": "reports/dean_os/clean_market_snapshot_current/latest.json",
            "command": "python run_agent_clean_yahoo_market_snapshot.py --ticker ASML --ticker MU --ticker NVDA --ticker TSM --timeframe 15m --timeframe 60m --timeframe 1d",
            "command_executable": True,
            "missing_parameters": [],
            "boundary": "Networked, identity-validated snapshot; no legacy Stage1 write.",
        },
        {
            "lane_id": "sector_market_evidence",
            "role": "market_confirmation",
            "runner": "run_agent_saved_sector_market_evidence.py",
            "artifact": "reports/dean_os/saved_sector_market_evidence_producer_current/latest.json",
            "command": "python run_agent_saved_sector_market_evidence.py <REPAIR_ARTIFACT> --as-of <AS_OF>",
            "command_executable": False,
            "missing_parameters": ["repair_artifact_path", "as_of"],
            "boundary": "Normalize only an immutable reviewed market snapshot.",
        },
        {
            "lane_id": "issuer_sec_fundamentals",
            "role": "issuer_fundamental_outcomes",
            "runner": "run_agent_sec_companyfacts_snapshot.py",
            "artifact": "reports/dean_os/saved_sec_fundamental_evidence_merger_current/latest.json",
            "command": "python run_agent_sec_companyfacts_snapshot.py <FILING_INDEX> --as-of <AS_OF>",
            "command_executable": False,
            "missing_parameters": ["filing_index_path", "as_of", "SEC_USER_AGENT"],
            "boundary": "Network snapshot first; filing/accession-bound producers second.",
        },
        {
            "lane_id": "macro_context",
            "role": "regime_and_expectation_context",
            "runner": "run_agent_saved_macro_evidence_producer.py",
            "artifact": "reports/dean_os/saved_macro_evidence_producer_current/latest.json",
            "command": "python run_agent_saved_macro_evidence_producer.py <SAVED_MACRO_SOURCE> --as-of <AS_OF>",
            "command_executable": False,
            "missing_parameters": ["source_path", "as_of"],
            "boundary": "Saved registered series only; exact vintage rules remain mandatory.",
        },
        {
            "lane_id": "semiconductor_news",
            "role": "event_and_mechanism_evidence",
            "runner": "run_agent_saved_semiconductor_news_evidence.py",
            "artifact": "reports/dean_os/saved_semiconductor_news_evidence_producer_current/latest.json",
            "command": "python run_agent_saved_semiconductor_news_evidence.py <SAVED_NEWS_SOURCE> --as-of <AS_OF>",
            "command_executable": False,
            "missing_parameters": ["source_path", "as_of"],
            "boundary": "Two independent strong sources; headline alone is not a fact.",
        },
        {
            "lane_id": "official_policy",
            "role": "policy_event_evidence",
            "runner": "run_agent_bis_policy_snapshot.py",
            "artifact": "reports/dean_os/saved_official_policy_evidence_producer_current/latest.json",
            "command": "python run_agent_bis_policy_snapshot.py <OFFICIAL_BIS_URL> --published-at <PUBLISHED_AT> --user-agent <USER_AGENT>",
            "command_executable": False,
            "missing_parameters": ["official_bis_url", "published_at", "user_agent"],
            "boundary": "Official immutable source plus corroboration; publication cutoff enforced.",
        },
        {
            "lane_id": "industry_operational_metrics",
            "role": "orders_capacity_utilization_lead_times",
            "runner": "run_agent_industry_operational_source_coverage.py",
            "artifact": "reports/dean_os/industry_operational_source_coverage_current/latest.json",
            "command": "python run_agent_industry_operational_source_coverage.py",
            "command_executable": True,
            "missing_parameters": [],
            "boundary": "Coverage discovery only until a reviewed structured source feed is connected.",
        },
    ]


def _inspect_lane(spec: dict[str, str], cutoff: datetime) -> dict[str, Any]:
    runner = Path(spec["runner"])
    artifact = Path(spec["artifact"])
    created_at = None
    age_days = None
    artifact_contract = None
    if artifact.exists():
        try:
            payload = _load(artifact)
            created_at = parse_timezone_aware(str(payload.get("created_at") or ""))
            artifact_contract = payload.get("contract") or payload.get("producer_contract")
            if created_at is not None:
                age_days = max(0.0, round((cutoff - created_at).total_seconds() / 86400, 3))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {
        **spec,
        "runner_exists": runner.is_file(),
        "artifact_exists": artifact.is_file(),
        "artifact_created_at": created_at.isoformat() if created_at else None,
        "artifact_age_days": age_days,
        "artifact_contract": artifact_contract,
        "automatic_execution_allowed": False,
    }


def _checkpoint_dates(monitor: dict[str, Any]) -> dict[str, Any]:
    tasks = monitor.get("tasks") or []
    pre_due = sorted({task.get("pre_due_source_review") for task in tasks if task.get("pre_due_source_review")})
    due = sorted({task.get("due_outcome_review") for task in tasks if task.get("due_outcome_review")})
    return {
        "pre_due_source_reviews": pre_due,
        "outcome_reviews": due,
        "nearest_pre_due_review": pre_due[0] if pre_due else None,
        "nearest_outcome_review": due[0] if due else None,
    }


def _operator_sequence(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = [
        "clean_market_15m_60m_1d",
        "issuer_sec_fundamentals",
        "semiconductor_news",
        "macro_context",
        "official_policy",
        "sector_market_evidence",
        "industry_operational_metrics",
    ]
    by_id = {lane["lane_id"]: lane for lane in lanes}
    return [
        {
            "priority": index,
            "lane_id": lane_id,
            "status": (
                "ready_for_operator"
                if by_id[lane_id]["runner_exists"] and by_id[lane_id].get("command_executable")
                else ("command_parameters_unresolved" if by_id[lane_id]["runner_exists"] else "runner_missing")
            ),
            "command": by_id[lane_id]["command"],
            "requires_review": True,
        }
        for index, lane_id in enumerate(order, start=1)
    ]


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
        "# Prospective Evidence Accumulation Runbook",
        "",
        f"- Replay tasks: `{summary['replay_task_count']}`",
        f"- Ready runners: `{summary['runner_ready_count']}/{summary['lane_count']}`",
        f"- Present artifacts: `{summary['artifact_present_count']}/{summary['lane_count']}`",
        f"- Nearest pre-due review: `{summary['nearest_pre_due_review']}`",
        f"- Nearest outcome review: `{summary['nearest_outcome_review']}`",
        "- Automatic execution allowed: `false`",
        "- Can trade: `false`",
        "",
        "## Operator sequence",
        "",
    ]
    for item in payload["operator_sequence"]:
        lines.append(f"{item['priority']}. `{item['lane_id']}` — `{item['status']}`")
    return "\n".join(lines) + "\n"


__all__ = ["ProspectiveAccumulationRunbookBuilder"]
