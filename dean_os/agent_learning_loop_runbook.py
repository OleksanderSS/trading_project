from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_STAGE_PATHS = {
    "evidence_pack": "reports/dean_os/analyst_evidence_pack/latest.json",
    "analyst_profiles": "reports/dean_os/analyst_profiles/latest.json",
    "profile_scorecard": "reports/dean_os/analyst_profile_scorecard/latest.json",
    "learning_bridge": "reports/dean_os/analyst_learning_bridge/latest.json",
    "review_approved_learning": "reports/dean_os/review_approved_learning/latest.json",
    "outcome_evaluation": "reports/dean_os/analyst_outcome_evaluation/latest.json",
    "calibration_gate": "reports/dean_os/analyst_calibration_gate/latest.json",
    "calibration_proposals": "reports/dean_os/calibration_proposals/latest.json",
    "calibration_review": "reports/dean_os/calibration_review_lifecycle/latest.json",
    "manual_backlog": "reports/dean_os/manual_implementation_backlog/latest.json",
}


class AgentLearningLoopRunbook:
    """Read-only operator runbook for the safe analyst learning loop."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/agent_learning_loop_runbook"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        stage_paths: dict[str, str | Path | None] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        resolved_paths = {
            key: str(value)
            for key, value in {**DEFAULT_STAGE_PATHS, **(stage_paths or {})}.items()
            if value is not None
        }
        stages = [_build_stage(stage_id, resolved_paths.get(stage_id, "")) for stage_id in _stage_order()]
        loop_position = _loop_position(stages)
        payload = {
            "run_id": _run_id("agent_learning_loop_runbook"),
            "created_at": utc_now_iso(),
            "mode": "agent_learning_loop_runbook",
            "inputs": {"stage_paths": resolved_paths},
            "summary": {
                "stage_count": len(stages),
                "available_artifact_count": sum(1 for stage in stages if stage["artifact_exists"]),
                "current_stage": loop_position.get("stage_id"),
                "current_status": loop_position.get("status"),
                "next_command": loop_position.get("next_command"),
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
            },
            "loop_position": loop_position,
            "stages": stages,
            "stop_conditions": _stop_conditions(),
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(loop_position),
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
        rendered_md = render_agent_learning_loop_runbook_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_agent_learning_loop_runbook_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    position = payload.get("loop_position", {})
    lines = [
        "# DEAN-OS Agent Learning Loop Runbook",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Current stage: `{summary.get('current_stage')}`",
        f"- Current status: `{summary.get('current_status')}`",
        f"- Config write performed: {summary.get('config_write_performed')}",
        f"- Pipeline run performed: {summary.get('pipeline_run_performed')}",
        "",
        "## Next Action",
        "",
        f"- {position.get('operator_action')}",
        f"- Command: `{position.get('next_command')}`",
        "",
        "## Stages",
        "",
    ]
    for stage in payload.get("stages", []):
        lines.extend(
            [
                f"### {stage.get('index')}. {stage.get('title')}",
                "",
                f"- Artifact: `{stage.get('artifact_path')}`",
                f"- Exists: {stage.get('artifact_exists')}",
                f"- Status: `{stage.get('status')}`",
                f"- Stop reason: {stage.get('stop_reason') or 'none'}",
                f"- Next command: `{stage.get('next_command')}`",
                "",
            ]
        )
    lines.extend(["## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _stage_order() -> list[str]:
    return [
        "evidence_pack",
        "analyst_profiles",
        "profile_scorecard",
        "learning_bridge",
        "review_approved_learning",
        "outcome_evaluation",
        "calibration_gate",
        "calibration_proposals",
        "calibration_review",
        "manual_backlog",
    ]


def _stage_metadata(stage_id: str) -> dict[str, str]:
    metadata = {
        "evidence_pack": {
            "title": "Analyst Evidence Pack",
            "command": "python run_agent_analyst_evidence_pack.py --materials docs/research --news-data DATA_NEWS --macro-data DATA_MACRO --tickers AMD NVDA --sectors semiconductor --tags ai_cycle",
        },
        "analyst_profiles": {
            "title": "Analyst Profile Manager",
            "command": "python run_agent_analyst_profiles.py reports/dean_os/analyst_evidence_pack/latest.json --output-dir reports/dean_os/analyst_profiles",
        },
        "profile_scorecard": {
            "title": "Analyst Profile Scorecard",
            "command": "python run_agent_analyst_scorecard.py --profile-runs-dir reports/dean_os/analyst_profiles --output-dir reports/dean_os/analyst_profile_scorecard",
        },
        "learning_bridge": {
            "title": "Analyst Learning Promotion Bridge",
            "command": "python run_agent_analyst_learning_bridge.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite",
        },
        "review_approved_learning": {
            "title": "Review-Approved Learning Loop",
            "command": "python run_agent_review_approved_learning.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite",
        },
        "outcome_evaluation": {
            "title": "Analyst Outcome Evaluation Loop",
            "command": "python run_agent_analyst_outcome_loop.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --latest-processed-prices 1d",
        },
        "calibration_gate": {
            "title": "Analyst Calibration Gate",
            "command": "python run_agent_analyst_calibration_gate.py --profile-scorecard-json reports/dean_os/analyst_profile_scorecard/latest.json --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite",
        },
        "calibration_proposals": {
            "title": "Calibration Proposals",
            "command": "python run_agent_calibration_proposals.py reports/dean_os/analyst_calibration_gate/latest.json --operations-store data/dean_os/operation_queue.sqlite",
        },
        "calibration_review": {
            "title": "Calibration Review Lifecycle",
            "command": "python run_agent_calibration_review_lifecycle.py --operations-store data/dean_os/operation_queue.sqlite --dry-run-proposals",
        },
        "manual_backlog": {
            "title": "Manual Implementation Backlog",
            "command": "python run_agent_manual_implementation_backlog.py --operations-store data/dean_os/operation_queue.sqlite",
        },
    }
    return metadata[stage_id]


def _build_stage(stage_id: str, artifact_path: str) -> dict[str, Any]:
    metadata = _stage_metadata(stage_id)
    path = Path(artifact_path)
    payload = _load_payload(path)
    status = _status_for_stage(stage_id, payload)
    stop_reason = _stop_reason(stage_id, status, payload, path.exists())
    return {
        "index": _stage_order().index(stage_id) + 1,
        "stage_id": stage_id,
        "title": metadata["title"],
        "artifact_path": artifact_path,
        "artifact_exists": path.exists(),
        "mode": payload.get("mode") if payload else None,
        "status": status,
        "stop_reason": stop_reason,
        "is_stop": bool(stop_reason),
        "next_command": metadata["command"],
        "share_sections": _share_sections(stage_id),
        "safety_contract": _safety_contract(stage_id),
    }


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"mode": "unreadable_json"}
    return payload if isinstance(payload, dict) else {"mode": "invalid_json"}


def _status_for_stage(stage_id: str, payload: dict[str, Any]) -> str:
    if not payload:
        return "missing_artifact"
    if payload.get("mode") in {"unreadable_json", "invalid_json"}:
        return payload["mode"]
    if stage_id == "evidence_pack":
        coverage = payload.get("coverage", {})
        if coverage.get("agent_lab_ready"):
            return "ready"
        return "blocked"
    if stage_id == "analyst_profiles":
        runs = payload.get("profile_runs", [])
        if any(item.get("status") == "completed" for item in runs):
            return "completed"
        return "blocked"
    if stage_id == "profile_scorecard":
        summary = payload.get("summary", {})
        if summary.get("activation_ready_profiles"):
            return "ready_profiles"
        if summary.get("profile_count", 0):
            return "gated"
        return "no_profiles"
    if stage_id == "learning_bridge":
        return payload.get("promotion_gate", {}).get("status", "unknown")
    if stage_id == "review_approved_learning":
        return payload.get("loop_gate", {}).get("status", "unknown")
    if stage_id == "outcome_evaluation":
        return payload.get("evaluation_gate", {}).get("status", "unknown")
    if stage_id == "calibration_gate":
        summary = payload.get("summary", {})
        if summary.get("ready_for_review_profiles"):
            return "ready_for_review"
        if summary.get("blocked_profiles"):
            return "blocked"
        return "gated"
    if stage_id == "calibration_proposals":
        return payload.get("proposal_gate", {}).get("status", "unknown")
    if stage_id == "calibration_review":
        return payload.get("lifecycle_gate", {}).get("status", "unknown")
    if stage_id == "manual_backlog":
        return payload.get("backlog_gate", {}).get("status", "unknown")
    return "unknown"


def _stop_reason(stage_id: str, status: str, payload: dict[str, Any], artifact_exists: bool) -> str:
    if not artifact_exists:
        return "Artifact is missing; run this stage command next."
    if status in {"unreadable_json", "invalid_json"}:
        return "Artifact exists but cannot be read as a valid JSON object."
    blocking_statuses = {
        "blocked": "Gate is blocked; inspect artifact blockers before continuing.",
        "blocked_need_newer_prices": "Outcome evaluation needs newer prices after thesis creation.",
        "waiting_for_horizon": "Learning record horizon has not elapsed.",
        "no_pending_records": "No pending analyst learning records exist yet.",
        "no_ready_profiles": "No profile is ready for calibration proposal.",
        "no_calibration_proposals": "No calibration proposals are available for review.",
        "operation_queue_empty": "Operation queue has no approved calibration implementation task.",
        "no_manual_tasks_in_scope": "No approved manual implementation task is in scope.",
        # profile_scorecard / calibration_gate / calibration_proposals can emit
        # these two -- they were missing here, so _loop_position() silently
        # walked past a not-actually-ready stage instead of stopping on it.
        # analyst_loop_daily_check.py's soft_loop_statuses already expects
        # both to surface as a stop reason.
        "gated": "Profiles exist but none have cleared the gate yet.",
        "no_profiles": "No profiles are available at this stage yet.",
    }
    if status in blocking_statuses:
        return blocking_statuses[status]
    if stage_id == "manual_backlog" and status == "manual_implementation_required":
        return "Loop reached manual implementation boundary; open a separate manual PR/config change."
    return ""


def _loop_position(stages: list[dict[str, Any]]) -> dict[str, Any]:
    for stage in stages:
        if stage["is_stop"]:
            return {
                "stage_id": stage["stage_id"],
                "title": stage["title"],
                "status": stage["status"],
                "artifact_path": stage["artifact_path"],
                "stop_reason": stage["stop_reason"],
                "next_command": stage["next_command"],
                "operator_action": _operator_action(stage),
            }
    final_stage = stages[-1]
    return {
        "stage_id": final_stage["stage_id"],
        "title": final_stage["title"],
        "status": final_stage["status"],
        "artifact_path": final_stage["artifact_path"],
        "stop_reason": "All visible gates passed; manual review remains required before any implementation.",
        "next_command": final_stage["next_command"],
        "operator_action": "Review final backlog and do not auto-apply config changes.",
    }


def _operator_action(stage: dict[str, Any]) -> str:
    if not stage["artifact_exists"]:
        return f"Run the {stage['title']} command and share the key sections."
    if stage["stage_id"] == "manual_backlog":
        return "If tasks exist, create a separate manual PR/config change; otherwise wait for approved proposals."
    return "Inspect the stop reason and resolve that gate before continuing."


def _share_sections(stage_id: str) -> list[str]:
    sections = {
        "evidence_pack": ["coverage", "analyst_inputs", "warnings", "recommendations"],
        "analyst_profiles": ["profile_plan", "profile_runs", "recommendations"],
        "profile_scorecard": ["summary", "profiles", "recommendations"],
        "learning_bridge": ["promotion_gate", "sources", "recommendations"],
        "review_approved_learning": ["loop_gate", "review_actions", "final_bridge"],
        "outcome_evaluation": ["evaluation_gate", "outcome_evaluation.status_counts", "profile_outcomes"],
        "calibration_gate": ["summary", "profiles", "recommendations"],
        "calibration_proposals": ["proposal_gate", "proposals", "recommendations"],
        "calibration_review": ["lifecycle_gate", "action_results", "approved_waiting_manual_implementation"],
        "manual_backlog": ["backlog_gate", "tasks", "recommendations"],
    }
    return sections[stage_id]


def _safety_contract(stage_id: str) -> list[str]:
    shared = ["No broker access.", "No heavy pipeline run.", "No production config write."]
    if stage_id in {"calibration_review", "manual_backlog"}:
        return [*shared, "Approval is not implementation.", "Manual PR/config change remains separate."]
    if stage_id in {"learning_bridge", "review_approved_learning", "outcome_evaluation"}:
        return [*shared, "Dry-run by default.", "Learning writes require explicit apply/review gates."]
    return shared


def _stop_conditions() -> list[str]:
    return [
        "Missing artifact: run that stage command first.",
        "Unreviewed source: record a real review action before promotion.",
        "No newer prices: wait for newer local prices or use a diagnostic-only historical path.",
        "No completed outcomes: keep collecting outcomes before calibration.",
        "No ready profiles: do not create calibration proposals.",
        "No approved proposals: no manual implementation backlog item exists.",
        "Manual implementation required: open a separate reviewed PR/config task.",
    ]


def _operator_notes() -> list[str]:
    return [
        "This runbook is read-only and does not execute any stage.",
        "Use it to decide which saved report or command to inspect next.",
        "A blocked gate is a success condition when evidence is insufficient.",
        "Do not skip from analyst notes directly to weight/config changes.",
    ]


def _recommendations(position: dict[str, Any]) -> list[str]:
    return [
        f"Current stop: {position.get('title')} -> {position.get('status')}.",
        position.get("stop_reason") or "Review the next command before continuing.",
        "Continue with the listed next_command only if the stop reason is resolved.",
    ]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
