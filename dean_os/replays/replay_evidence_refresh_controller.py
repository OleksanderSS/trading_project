from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.clean_yahoo_market_snapshot import CleanYahooMarketSnapshot
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.replays.replay_checkpoint_due_router import measurement_price_requirements
from dean_os.replays.replay_outcome_lifecycle_orchestrator import (
    ReplayOutcomeLifecycleOrchestrator,
)
from dean_os.schemas import utc_now_iso


class ReplayEvidenceRefreshController:
    """Execute at most one allowlisted evidence refresh pass per invocation."""

    contract = "dean_replay_evidence_refresh_controller_v1"

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/replay_evidence_refresh_controller_current"
        ),
        snapshot_artifact_dir: str | Path = (
            "data/dean_os/historical_outcome_market_snapshots"
        ),
        snapshot_report_dir: str | Path = (
            "reports/dean_os/historical_outcome_market_snapshot_current"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.snapshot_artifact_dir = Path(snapshot_artifact_dir)
        self.snapshot_report_dir = Path(snapshot_report_dir)

    async def build(
        self,
        *,
        lifecycle_json: str | Path,
        registration_json: str | Path,
        review_gate_json: str | Path,
        as_of: str,
        pipeline_paths: list[str | Path],
        prior_outcome_json_paths: list[str | Path],
        packet_json: str | Path | None = None,
        journal_path: str | Path = "data/dean_os/system_journal.jsonl",
        apply_refresh: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        lifecycle_path = Path(lifecycle_json)
        registration_path = Path(registration_json)
        gate_path = Path(review_gate_json)
        lifecycle = _load(lifecycle_path)
        registration = _load(registration_path)
        gate = _load(gate_path)
        cutoff = parse_timezone_aware(as_of)
        if cutoff is None:
            raise ValueError("refresh as_of must be timezone-aware")
        if lifecycle.get("contract") != "dean_replay_outcome_lifecycle_v1":
            raise ValueError("unsupported replay outcome lifecycle contract")

        jobs = _refresh_jobs(lifecycle, registration, gate)
        authority_issues = _authority_issues(jobs)
        can_execute = bool(jobs) and not authority_issues
        snapshot: dict[str, Any] | None = None
        refreshed_lifecycle: dict[str, Any] | None = None
        refresh_failure: dict[str, Any] | None = None
        execution_status = (
            "not_requested"
            if not apply_refresh
            else "no_refresh_action"
            if not jobs
            else "blocked_by_refresh_authority"
            if authority_issues
            else "refresh_started"
        )

        if apply_refresh and can_execute:
            tickers = sorted(
                {ticker for job in jobs for ticker in job["tickers"]}
            )
            try:
                snapshot = await CleanYahooMarketSnapshot(
                    artifact_dir=self.snapshot_artifact_dir,
                    report_dir=self.snapshot_report_dir,
                ).build(
                    tickers=tickers,
                    timeframes=["1d"],
                    end_date=cutoff,
                    max_download_attempts=1,
                    save=save,
                )
                snapshot_path = (snapshot.get("snapshot") or {}).get("path")
                if not snapshot_path:
                    raise RuntimeError(
                        "verified refresh did not produce a snapshot path"
                    )
                refreshed_lifecycle = ReplayOutcomeLifecycleOrchestrator().build(
                    registration_json=registration_path,
                    review_gate_json=gate_path,
                    packet_json=packet_json,
                    as_of=as_of,
                    verified_price_paths=[snapshot_path],
                    pipeline_paths=pipeline_paths,
                    prior_outcome_json_paths=prior_outcome_json_paths,
                    journal_path=journal_path,
                    save=save,
                )
                execution_status = "single_refresh_pass_completed"
            except Exception as exc:
                snapshot = None
                refreshed_lifecycle = None
                refresh_failure = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retry_automatically": False,
                    "next_action": (
                        "Use an alternate verified market source or ingest a "
                        "point-in-time snapshot; do not reinterpret missing data."
                    ),
                }
                execution_status = "single_refresh_pass_failed"

        created_at = utc_now_iso()
        run_id = "replay_evidence_refresh_" + created_at.replace(":", "").replace(
            "+00:00", "Z"
        )
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "replay_evidence_refresh_controller",
            "contract": self.contract,
            "inputs": {
                "lifecycle_json": str(lifecycle_path),
                "registration_json": str(registration_path),
                "review_gate_json": str(gate_path),
                "as_of": cutoff.isoformat(),
                "pipeline_paths": [str(item) for item in pipeline_paths],
                "prior_outcome_json_paths": [
                    str(item) for item in prior_outcome_json_paths
                ],
                "apply_refresh": apply_refresh,
            },
            "summary": {
                "status": execution_status,
                "refresh_job_count": len(jobs),
                "authority_issue_count": len(authority_issues),
                "refresh_executed": bool(apply_refresh and can_execute),
                "snapshot_created": snapshot is not None,
                "lifecycle_rerun": refreshed_lifecycle is not None,
                "post_refresh_lifecycle_status": (
                    ((refreshed_lifecycle or {}).get("summary") or {}).get("status")
                ),
                "refresh_failure_recorded": refresh_failure is not None,
                "automatic_looping_allowed": False,
                "can_trade": False,
            },
            "refresh_jobs": jobs,
            "authority_issues": authority_issues,
            "snapshot": _artifact_summary(snapshot),
            "refreshed_lifecycle": _artifact_summary(refreshed_lifecycle),
            "refresh_failure": refresh_failure,
            "refresh_policy": {
                "allowed_provider": "yahoo_finance",
                "allowed_timeframes": ["1d"],
                "ticker_source": "hash_bound_hypothesis_measurement_spec_only",
                "maximum_passes_per_invocation": 1,
                "database_write_allowed": False,
                "broker_access_allowed": False,
                "causal_approval_allowed": False,
                "rule_promotion_allowed": False,
            },
            "safety": {
                "review_only_after_collection": True,
                "network_access_performed": bool(apply_refresh and can_execute),
                "refresh_apply_requested": apply_refresh,
                "legacy_database_write_performed": False,
                "outcome_scoring_performed": False,
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "broker_access_performed": False,
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


def _refresh_jobs(
    lifecycle: dict[str, Any],
    registration: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    recommendations = {
        str(item.get("task_id")): item
        for item in lifecycle.get("system_recommendations") or []
        if item.get("action_type") == "refresh_verified_checkpoint_evidence"
        and item.get("task_id")
    }
    plans = {
        str(item.get("task_id")): item
        for item in registration.get("registration_plan") or []
        if item.get("task_id")
    }
    specs = {
        str(item.get("hypothesis_id")): item.get("measurement_spec") or {}
        for item in gate.get("hypothesis_review") or []
        if item.get("hypothesis_id")
    }
    jobs = []
    for task_id, recommendation in recommendations.items():
        plan = plans.get(task_id) or {}
        hypothesis_id = str(plan.get("hypothesis_id") or "")
        requirements = measurement_price_requirements(specs.get(hypothesis_id, {}))
        jobs.append(
            {
                "job_id": "refresh_" + task_id,
                "task_id": task_id,
                "hypothesis_id": hypothesis_id,
                "due_at": plan.get("due_at"),
                "provider": "yahoo_finance",
                "timeframe": "1d",
                "tickers": requirements.get("required_tickers") or [],
                "measurement_requirement_type": requirements.get(
                    "requirement_type"
                ),
                "source_recommendation": recommendation,
                "maximum_attempts_this_invocation": 1,
            }
        )
    return jobs


def _authority_issues(jobs: list[dict[str, Any]]) -> list[str]:
    issues = []
    for job in jobs:
        if job.get("provider") != "yahoo_finance":
            issues.append(f"{job.get('job_id')}:provider_not_allowlisted")
        if job.get("timeframe") != "1d":
            issues.append(f"{job.get('job_id')}:timeframe_not_allowlisted")
        if not job.get("tickers"):
            issues.append(f"{job.get('job_id')}:no_declared_measurement_tickers")
    return issues


def _artifact_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "run_id": payload.get("run_id"),
        "contract": payload.get("contract"),
        "summary": payload.get("summary"),
        "saved_paths": payload.get("saved_paths"),
        "snapshot": payload.get("snapshot"),
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Replay Evidence Refresh Controller",
        "",
        f"- Status: `{summary['status']}`",
        f"- Refresh jobs: `{summary['refresh_job_count']}`",
        f"- Authority issues: `{summary['authority_issue_count']}`",
        f"- Refresh executed: `{summary['refresh_executed']}`",
        f"- Snapshot created: `{summary['snapshot_created']}`",
        f"- Lifecycle rerun: `{summary['lifecycle_rerun']}`",
        f"- Post-refresh status: `{summary['post_refresh_lifecycle_status']}`",
        "",
    ]
    for job in payload["refresh_jobs"]:
        lines.append(
            f"- `{job['task_id']}`: {','.join(job['tickers'])} / {job['timeframe']}"
        )
    failure = payload.get("refresh_failure")
    if failure:
        lines.extend(
            [
                "",
                f"- Failure: `{failure.get('error_type')}` — {failure.get('error')}",
                f"- Automatic retry: `{failure.get('retry_automatically')}`",
                f"- Next action: {failure.get('next_action')}",
            ]
        )
    lines.extend(
        [
            "",
            "At most one refresh pass is permitted per invocation.",
            "No database write, causal approval, rule promotion, broker access, or trading is permitted.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


__all__ = ["ReplayEvidenceRefreshController"]
