from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_learning_promotion_bridge import AnalystLearningPromotionBridge
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_performance import AgentPerformanceByContext
from dean_os.draft.dean_os_agent_system_v7.dean_os.review_actions import ReviewActionStore
from dean_os.schemas import ReviewActionRecord, utc_now_iso
from dean_os.utils import json_ready


class ReviewApprovedLearningLoop:
    """Auditable review ceremony around analyst learning promotion.

    The loop deliberately separates preview, human review action, and apply.
    It can record a review action only when the caller explicitly asks for it,
    and learning promotion still goes through AnalystLearningPromotionBridge.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/review_approved_learning"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        profile_run_path: str | Path | None = None,
        agent_lab_report_path: str | Path | None = None,
        learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
        review_actions_path: str | Path = "data/dean_os/review_actions.sqlite",
        operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
        memory_path: str | Path = "data/dean_os/recommendation_memory.sqlite",
        reviewer: str = "human",
        review_notes: str = "",
        mark_reviewed: bool = False,
        needs_more_data_request: str | None = None,
        apply: bool = False,
        allow_weak_notes: bool = False,
        allow_duplicates: bool = False,
        default_horizon_days: int = 365,
        include_context_summary: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        _validate_review_intent(mark_reviewed, needs_more_data_request, review_notes)

        bridge = AnalystLearningPromotionBridge(output_dir=self.output_dir / "bridge")
        pre_review = bridge.run(
            profile_run_path=profile_run_path,
            agent_lab_report_path=agent_lab_report_path,
            learning_path=learning_path,
            review_actions_path=review_actions_path,
            operations_path=operations_path,
            apply=False,
            allow_weak_notes=allow_weak_notes,
            allow_duplicates=allow_duplicates,
            default_horizon_days=default_horizon_days,
            save=False,
        )

        recorded_actions = self._record_review_actions(
            pre_review=pre_review,
            review_actions_path=review_actions_path,
            operations_path=operations_path,
            reviewer=reviewer,
            review_notes=review_notes,
            mark_reviewed=mark_reviewed,
            needs_more_data_request=needs_more_data_request,
        )

        final_bridge = bridge.run(
            profile_run_path=profile_run_path,
            agent_lab_report_path=agent_lab_report_path,
            learning_path=learning_path,
            review_actions_path=review_actions_path,
            operations_path=operations_path,
            apply=apply,
            allow_weak_notes=allow_weak_notes,
            allow_duplicates=allow_duplicates,
            default_horizon_days=default_horizon_days,
            save=True,
        )
        context_summary = (
            AgentPerformanceByContext(learning_path, memory_path).build_summary()
            if include_context_summary
            else None
        )
        payload = {
            "run_id": _run_id("review_approved_learning"),
            "created_at": utc_now_iso(),
            "mode": "review_approved_learning_loop",
            "inputs": {
                "profile_run_path": str(profile_run_path) if profile_run_path else None,
                "agent_lab_report_path": str(agent_lab_report_path) if agent_lab_report_path else None,
                "learning_path": str(learning_path),
                "review_actions_path": str(review_actions_path),
                "operations_path": str(operations_path),
                "memory_path": str(memory_path),
                "reviewer": reviewer,
                "mark_reviewed": mark_reviewed,
                "needs_more_data_request": needs_more_data_request,
                "apply": apply,
                "allow_weak_notes": allow_weak_notes,
                "allow_duplicates": allow_duplicates,
                "default_horizon_days": default_horizon_days,
                "include_context_summary": include_context_summary,
            },
            "loop_gate": _loop_gate(final_bridge, recorded_actions, mark_reviewed, bool(needs_more_data_request), apply),
            "pre_review_bridge": _compact_bridge(pre_review),
            "review_actions": [action.model_dump(mode="json") for action in recorded_actions],
            "final_bridge": final_bridge,
            "context_performance": context_summary,
            "recommendations": _recommendations(
                final_bridge=final_bridge,
                recorded_actions=recorded_actions,
                mark_reviewed=mark_reviewed,
                needs_more_data=bool(needs_more_data_request),
                apply=apply,
            ),
        }
        if save:
            self.save(payload)
        return payload

    def _record_review_actions(
        self,
        pre_review: dict[str, Any],
        review_actions_path: str | Path,
        operations_path: str | Path,
        reviewer: str,
        review_notes: str,
        mark_reviewed: bool,
        needs_more_data_request: str | None,
    ) -> list[ReviewActionRecord]:
        if not mark_reviewed and not needs_more_data_request:
            return []

        store = ReviewActionStore(
            db_path=review_actions_path,
            operations_path=operations_path,
            event_log_path=None,
        )
        actions: list[ReviewActionRecord] = []
        for source in pre_review.get("sources", []):
            source_id = source.get("source_id")
            if not source_id:
                continue
            review = source.get("review", {})
            if needs_more_data_request:
                if review.get("needs_more_data"):
                    continue
                actions.append(
                    store.needs_more_data(
                        source_type="agent_lab_report",
                        source_id=source_id,
                        data_request=needs_more_data_request,
                        notes=review_notes,
                        reviewer=reviewer,
                    )
                )
                continue
            if mark_reviewed and not review.get("reviewed"):
                actions.append(
                    store.mark_reviewed(
                        source_type="agent_lab_report",
                        source_id=source_id,
                        notes=review_notes,
                        reviewer=reviewer,
                    )
                )
        return actions

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
        rendered_md = render_review_approved_learning_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_review_approved_learning_markdown(payload: dict[str, Any]) -> str:
    gate = payload.get("loop_gate", {})
    final_gate = payload.get("final_bridge", {}).get("promotion_gate", {})
    lines = [
        "# DEAN-OS Review-Approved Learning Loop",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{gate.get('status')}`",
        f"- Apply: {payload.get('inputs', {}).get('apply')}",
        f"- Review actions: {gate.get('review_action_count', 0)}",
        f"- Candidates: {final_gate.get('candidate_count', 0)}",
        f"- Promotable: {final_gate.get('promotable_count', 0)}",
        f"- Promoted: {final_gate.get('promoted_count', 0)}",
        "",
        "## Sources",
        "",
    ]
    for source in payload.get("final_bridge", {}).get("sources", []):
        lines.append(
            f"- `{source.get('source_id')}` profile={source.get('profile')} "
            f"reviewed={source.get('review', {}).get('reviewed')} promoted={source.get('promoted_count')}"
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _validate_review_intent(
    mark_reviewed: bool,
    needs_more_data_request: str | None,
    review_notes: str,
) -> None:
    if mark_reviewed and needs_more_data_request:
        raise ValueError("Choose either mark_reviewed or needs_more_data_request, not both.")
    if mark_reviewed and not review_notes.strip():
        raise ValueError("review_notes is required when marking an Agent Lab report reviewed.")
    if needs_more_data_request is not None and not needs_more_data_request.strip():
        raise ValueError("needs_more_data_request cannot be blank.")


def _compact_bridge(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": payload.get("run_id"),
        "promotion_gate": payload.get("promotion_gate", {}),
        "sources": [
            {
                "source_id": source.get("source_id"),
                "profile": source.get("profile"),
                "report_json": source.get("report_json"),
                "review": source.get("review", {}),
                "candidate_count": source.get("candidate_count"),
                "promotable_count": source.get("promotable_count"),
                "promoted_count": source.get("promoted_count"),
            }
            for source in payload.get("sources", [])
        ],
        "recommendations": payload.get("recommendations", []),
    }


def _loop_gate(
    final_bridge: dict[str, Any],
    recorded_actions: list[ReviewActionRecord],
    mark_reviewed: bool,
    needs_more_data: bool,
    apply: bool,
) -> dict[str, Any]:
    promotion_gate = final_bridge.get("promotion_gate", {})
    bridge_status = promotion_gate.get("status")
    if needs_more_data:
        status = "needs_more_data_recorded"
    elif bridge_status == "applied":
        status = "applied"
    elif bridge_status == "dry_run_ready" and mark_reviewed:
        status = "reviewed_ready_to_apply"
    elif bridge_status == "dry_run_ready":
        status = "preview_ready_to_apply"
    elif bridge_status == "blocked":
        status = "blocked"
    else:
        status = bridge_status or "unknown"
    return {
        "status": status,
        "bridge_status": bridge_status,
        "apply_requested": apply,
        "review_action_count": len(recorded_actions),
        "review_action_ids": [action.action_id for action in recorded_actions],
        "can_apply": promotion_gate.get("can_apply", False),
        "promoted_count": promotion_gate.get("promoted_count", 0),
    }


def _recommendations(
    final_bridge: dict[str, Any],
    recorded_actions: list[ReviewActionRecord],
    mark_reviewed: bool,
    needs_more_data: bool,
    apply: bool,
) -> list[str]:
    gate = final_bridge.get("promotion_gate", {})
    if needs_more_data:
        return ["Data request recorded. Do not promote this source until the open data gap is resolved or voided."]
    if gate.get("status") == "blocked":
        return ["Promotion is blocked; review candidate blockers before recording new learning."]
    if gate.get("status") == "dry_run_ready" and not apply:
        if recorded_actions or mark_reviewed:
            return ["Review action recorded. Rerun with --apply only after a final sanity check of the promotable notes."]
        return ["Preview is promotable because the source was already reviewed. Use --apply only when you accept the audit trail."]
    if gate.get("status") == "applied":
        return [
            "Learning records were promoted as pending outcomes.",
            "Do not change agent weights until outcome evaluation completes on the relevant horizon.",
            "Use context performance as monitoring, not as immediate proof of skill.",
        ]
    return final_bridge.get("recommendations", []) or ["No additional recommendation."]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
