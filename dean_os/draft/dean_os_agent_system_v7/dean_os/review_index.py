from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_REVIEW_SOURCES = {
    "domain_analyst": "reports/dean_os/domain_analyst/latest.json",
    "pipeline_tuning_controller": "reports/dean_os/pipeline_tuning_controller/latest.json",
    "pipeline_model_case": (
        "reports/dean_os/pipeline_model_case_packet_current/latest.json"
    ),
    "pipeline_model_feedback": (
        "reports/dean_os/pipeline_model_feedback_packet_current/latest.json"
    ),
}


class ReviewIndexBuilder:
    """Builds a discoverable index of latest DEAN-OS review artifacts.

    This is a read-only discovery layer. It reads latest review artifacts and writes
    a local index artifact. It does not approve, execute, tune, train, or write
    learning memory.
    """

    def __init__(
        self,
        sources: dict[str, str | Path] | None = None,
        output_dir: str | Path = "reports/dean_os/review_index",
    ):
        self.sources = sources or DEFAULT_REVIEW_SOURCES
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        entries = [self._entry(name, path) for name, path in self.sources.items()]
        available = [entry for entry in entries if entry["available"]]
        missing = [entry for entry in entries if not entry["available"]]

        payload = {
            "run_id": _run_id("review_index"),
            "created_at": utc_now_iso(),
            "mode": "review_index",
            "summary": {
                "source_count": len(entries),
                "available_count": len(available),
                "missing_count": len(missing),
                "ready_for_chief_review": bool(available),
                "missing_sources": [entry["source_name"] for entry in missing],
            },
            "entries": entries,
            "next_actions": _next_actions(available, missing),
            "safety": {
                "review_only": True,
                "read_existing_artifacts_only": True,
                "live_execution_allowed": False,
                "broker_access_performed": False,
                "production_config_write_performed": False,
                "learning_write_performed": False,
                "training_or_tuning_run_performed": False,
            },
        }

        if save:
            markdown = render_review_index_markdown(payload)
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=markdown,
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths

        return json_ready(payload)

    def _entry(self, name: str, path: str | Path) -> dict[str, Any]:
        artifact_path = Path(path)
        if not artifact_path.exists():
            return {
                "source_name": name,
                "path": str(artifact_path),
                "available": False,
                "artifact_mode": None,
                "run_id": None,
                "created_at": None,
                "recommendation": None,
                "status": "missing",
                "summary": {},
                "safety": {},
                "errors": ["latest artifact not found"],
            }

        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "source_name": name,
                "path": str(artifact_path),
                "available": False,
                "artifact_mode": None,
                "run_id": None,
                "created_at": None,
                "recommendation": None,
                "status": "invalid_json",
                "summary": {},
                "safety": {},
                "errors": [repr(exc)],
            }

        return {
            "source_name": name,
            "path": str(artifact_path),
            "available": True,
            "artifact_mode": payload.get("mode"),
            "run_id": payload.get("run_id"),
            "created_at": payload.get("created_at"),
            "recommendation": _recommendation(payload),
            "status": _status(payload),
            "summary": _summary(payload),
            "safety": _safety(payload),
            "errors": [],
        }


def render_review_index_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Review Index",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Created at: `{payload.get('created_at')}`",
        f"- Available sources: `{summary.get('available_count')}` / `{summary.get('source_count')}`",
        f"- Ready for chief review: `{summary.get('ready_for_chief_review')}`",
        "",
        "## Entries",
        "",
        "| Source | Available | Mode | Status | Recommendation | Path |",
        "|---|---:|---|---|---|---|",
    ]

    for entry in payload.get("entries", []):
        lines.append(
            "| {source} | {available} | {mode} | {status} | {recommendation} | `{path}` |".format(
                source=entry.get("source_name"),
                available=entry.get("available"),
                mode=entry.get("artifact_mode"),
                status=entry.get("status"),
                recommendation=entry.get("recommendation"),
                path=entry.get("path"),
            )
        )

    lines.extend(["", "## Next Actions", ""])
    for action in payload.get("next_actions", []):
        lines.append(f"- {action}")

    lines.extend(["", "## Safety", ""])
    safety = payload.get("safety") or {}
    for key in sorted(safety):
        lines.append(f"- {key}: `{safety[key]}`")

    return "\n".join(lines).strip() + "\n"


def _recommendation(payload: dict[str, Any]) -> str | None:
    if (
        payload.get("mode") == "pipeline_model_feedback_packet"
        and isinstance(payload.get("summary"), dict)
    ):
        status = payload["summary"].get("packet_status")
        return (
            "repair_feedback_or_binding"
            if status == "pipeline_model_feedback_blocked"
            else "await_manual_feedback"
            if status
            == "pipeline_model_feedback_ready_pending_manual_feedback"
            else "review_proposal_only_learning_candidates"
        )
    if (
        payload.get("mode") == "pipeline_model_case_packet"
        and isinstance(payload.get("summary"), dict)
    ):
        return payload["summary"].get("review_disposition")
    if "analyst_report" in payload and isinstance(payload["analyst_report"], dict):
        return payload["analyst_report"].get("recommendation")
    if "pipeline_tuning_plan" in payload and isinstance(payload["pipeline_tuning_plan"], dict):
        return payload["pipeline_tuning_plan"].get("status")
    if "analytical_report" in payload and isinstance(payload["analytical_report"], dict):
        return payload["analytical_report"].get("verdict")
    if "pipeline_report" in payload and isinstance(payload["pipeline_report"], dict):
        return payload["pipeline_report"].get("verdict")
    return None


def _status(payload: dict[str, Any]) -> str:
    if (
        payload.get("mode") == "pipeline_model_feedback_packet"
        and isinstance(payload.get("summary"), dict)
    ):
        return str(
            payload["summary"].get("packet_status") or "unknown"
        )
    if (
        payload.get("mode") == "pipeline_model_case_packet"
        and isinstance(payload.get("summary"), dict)
    ):
        return str(
            payload["summary"].get("case_status") or "unknown"
        )
    if "analyst_report" in payload and isinstance(payload["analyst_report"], dict):
        basket = payload["analyst_report"].get("ticker_basket") or {}
        return str(basket.get("basket_status") or payload["analyst_report"].get("recommendation") or "unknown")
    if "pipeline_tuning_plan" in payload and isinstance(payload["pipeline_tuning_plan"], dict):
        return str(payload["pipeline_tuning_plan"].get("status") or "unknown")
    return "available"


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    if (
        payload.get("mode") == "pipeline_model_feedback_packet"
        and isinstance(payload.get("summary"), dict)
    ):
        summary = payload["summary"]
        return {
            "case_id": summary.get("case_id"),
            "case_status": summary.get("case_status"),
            "packet_status": summary.get("packet_status"),
            "manual_feedback_record_count": summary.get(
                "manual_feedback_record_count"
            ),
            "learning_candidate_proposal_count": summary.get(
                "learning_candidate_proposal_count"
            ),
            "can_route_to_existing_analyst_learning_apply_loop": (
                summary.get(
                    "can_route_to_existing_analyst_learning_apply_loop"
                )
            ),
            "can_apply_learning": summary.get("can_apply_learning"),
            "can_write_learning_memory": summary.get(
                "can_write_learning_memory"
            ),
            "can_launch_model_variant_now": summary.get(
                "can_launch_model_variant_now"
            ),
            "can_trade": summary.get("can_trade"),
        }
    if (
        payload.get("mode") == "pipeline_model_case_packet"
        and isinstance(payload.get("summary"), dict)
    ):
        summary = payload["summary"]
        case = (
            payload.get("case")
            if isinstance(payload.get("case"), dict)
            else {}
        )
        return {
            "case_id": summary.get("case_id"),
            "case_status": summary.get("case_status"),
            "case_classification": summary.get(
                "case_classification"
            ),
            "result_label": summary.get("result_label"),
            "blocked_metric_planes": summary.get(
                "blocked_metric_planes",
                [],
            ),
            "root_cause_categories": summary.get(
                "root_cause_categories",
                [],
            ),
            "lineage": case.get("lineage", {}),
            "evaluated_at": case.get("evaluated_at"),
            "can_write_learning_memory": summary.get(
                "can_write_learning_memory"
            ),
            "can_launch_model_variant_now": summary.get(
                "can_launch_model_variant_now"
            ),
            "can_trade": summary.get("can_trade"),
        }
    if "analyst_report" in payload and isinstance(payload["analyst_report"], dict):
        report = payload["analyst_report"]
        thesis = report.get("thesis") or {}
        basket = report.get("ticker_basket") or {}
        return {
            "domain_id": report.get("domain_id"),
            "thesis_stance": thesis.get("stance"),
            "expected_direction": thesis.get("expected_direction"),
            "confidence": thesis.get("confidence"),
            "basket_status": basket.get("basket_status"),
            "candidate_count": len(basket.get("candidates") or []),
        }
    if "pipeline_tuning_plan" in payload and isinstance(payload["pipeline_tuning_plan"], dict):
        plan = payload["pipeline_tuning_plan"]
        return {
            "plan_status": plan.get("status"),
            "target": plan.get("target"),
            "plane_count": len(plan.get("planes") or []),
            "action_proposal_count": len(payload.get("action_proposals") or []),
        }
    return {}


def _safety(payload: dict[str, Any]) -> dict[str, Any]:
    safety = {}
    if isinstance(payload.get("safety"), dict):
        safety.update(payload["safety"])
    if isinstance(payload.get("artifact_safety"), dict):
        safety.update(payload["artifact_safety"])
    return safety


def _next_actions(available: list[dict[str, Any]], missing: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    if missing:
        actions.append("Run missing review producers before ChiefReview synthesis: " + ", ".join(entry["source_name"] for entry in missing))
    if available:
        actions.append("ChiefReview can inspect available artifacts and decide whether follow-up evidence or validation is needed.")
    actions.append("Do not execute trades, tune models, promote models, or write learning memory from this index.")
    return actions


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
