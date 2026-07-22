from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.learning import LearningStore, direction_from_note
from dean_os.review_actions import ReviewActionStore
from dean_os.schemas import AgentLabRunReport, AgentLearningRecord, ResearchNote, ReviewActionRecord, utc_now_iso
from dean_os.utils import json_ready


class AnalystLearningPromotionBridge:
    """Promotes reviewed analyst notes into learning records.

    This bridge is intentionally conservative. It does not run analysts, does
    not evaluate outcomes, and does not write learning records unless the caller
    passes apply=True and the source Agent Lab report has a non-voided
    mark_reviewed action.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/analyst_learning_bridge"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        profile_run_path: str | Path | None = None,
        agent_lab_report_path: str | Path | None = None,
        learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
        review_actions_path: str | Path = "data/dean_os/review_actions.sqlite",
        operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
        require_review: bool = True,
        apply: bool = False,
        allow_weak_notes: bool = False,
        allow_duplicates: bool = False,
        default_horizon_days: int = 365,
        save: bool = True,
    ) -> dict[str, Any]:
        sources = _resolve_sources(profile_run_path, agent_lab_report_path)
        review_actions = ReviewActionStore(
            db_path=review_actions_path,
            operations_path=operations_path,
            event_log_path=None,
        ).list_actions()
        learning_store = LearningStore(learning_path)
        existing_note_ids = {
            record.note_id for record in learning_store.list_records()
        } if not allow_duplicates else set()

        source_results: list[dict[str, Any]] = []
        promoted_records: list[AgentLearningRecord] = []
        for source in sources:
            result, promoted = self._process_source(
                source=source,
                review_actions=review_actions,
                learning_store=learning_store,
                existing_note_ids=existing_note_ids,
                require_review=require_review,
                apply=apply,
                allow_weak_notes=allow_weak_notes,
                default_horizon_days=default_horizon_days,
            )
            source_results.append(result)
            promoted_records.extend(promoted)
            existing_note_ids.update(record.note_id for record in promoted)

        candidate_count = sum(len(item.get("candidates", [])) for item in source_results)
        promotable_count = sum(
            1
            for item in source_results
            for candidate in item.get("candidates", [])
            if candidate.get("can_promote")
        )
        blocked_count = sum(
            1
            for item in source_results
            for candidate in item.get("candidates", [])
            if not candidate.get("can_promote")
        )
        payload = {
            "run_id": _run_id("analyst_learning_bridge"),
            "created_at": utc_now_iso(),
            "mode": "analyst_learning_promotion_bridge",
            "inputs": {
                "profile_run_path": str(profile_run_path) if profile_run_path else None,
                "agent_lab_report_path": str(agent_lab_report_path) if agent_lab_report_path else None,
                "learning_path": str(learning_path),
                "review_actions_path": str(review_actions_path),
                "require_review": require_review,
                "apply": apply,
                "allow_weak_notes": allow_weak_notes,
                "allow_duplicates": allow_duplicates,
                "default_horizon_days": default_horizon_days,
            },
            "promotion_gate": _promotion_gate(
                source_results=source_results,
                candidate_count=candidate_count,
                promotable_count=promotable_count,
                blocked_count=blocked_count,
                apply=apply,
                promoted_count=len(promoted_records),
            ),
            "sources": source_results,
            "promoted_records": [record.model_dump(mode="json") for record in promoted_records],
            "recommendations": _recommendations(candidate_count, promotable_count, blocked_count, apply, promoted_records),
        }
        if save:
            self.save(payload)
        return payload

    def _process_source(
        self,
        source: dict[str, Any],
        review_actions: list[ReviewActionRecord],
        learning_store: LearningStore,
        existing_note_ids: set[str],
        require_review: bool,
        apply: bool,
        allow_weak_notes: bool,
        default_horizon_days: int,
    ) -> tuple[dict[str, Any], list[AgentLearningRecord]]:
        report_path = Path(source["report_json"])
        report = AgentLabRunReport(**json.loads(report_path.read_text(encoding="utf-8")))
        review = _review_state(report.run_id, review_actions)
        candidates = [
            _candidate_from_note(
                note=note,
                report=report,
                source=source,
                review=review,
                require_review=require_review,
                allow_weak_notes=allow_weak_notes,
                existing_note_ids=existing_note_ids,
                default_horizon_days=default_horizon_days,
            )
            for note in report.research_notes
        ]
        promoted_records: list[AgentLearningRecord] = []
        if apply:
            for candidate, note in zip(candidates, report.research_notes, strict=False):
                if not candidate["can_promote"]:
                    continue
                record = learning_store.create_record_from_note(
                    note=note,
                    expected_direction=candidate["expected_direction"],
                    horizon_days=candidate["horizon_days"],
                    metadata=candidate["metadata"],
                    lifecycle_status="validated",
                    lifecycle_actor=(
                        "review_actions:"
                        + ",".join(candidate["metadata"].get("review_action_ids", []))
                    ),
                    lifecycle_reason=(
                        "Promoted only after reviewed AgentLab source passed "
                        "the analyst learning promotion gate."
                    ),
                )
                promoted_records.append(record)

        return {
            "source_type": "agent_lab_report",
            "source_id": report.run_id,
            "profile": source.get("profile"),
            "profile_run_id": source.get("profile_run_id"),
            "evidence_pack_run_id": source.get("evidence_pack_run_id"),
            "evidence_pack_path": source.get("evidence_pack_path"),
            "report_json": str(report_path),
            "review": review,
            "note_count": len(report.research_notes),
            "candidate_count": len(candidates),
            "promotable_count": sum(1 for candidate in candidates if candidate["can_promote"]),
            "promoted_count": len(promoted_records),
            "candidates": candidates,
        }, promoted_records

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
        rendered_md = render_analyst_learning_bridge_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_analyst_learning_bridge_markdown(payload: dict[str, Any]) -> str:
    gate = payload.get("promotion_gate", {})
    lines = [
        "# DEAN-OS Analyst Learning Promotion Bridge",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{gate.get('status')}`",
        f"- Apply: {payload.get('inputs', {}).get('apply')}",
        f"- Candidates: {gate.get('candidate_count', 0)}",
        f"- Promotable: {gate.get('promotable_count', 0)}",
        f"- Promoted: {gate.get('promoted_count', 0)}",
        "",
        "## Sources",
        "",
    ]
    for source in payload.get("sources", []):
        lines.extend(
            [
                f"- `{source.get('source_id')}` profile={source.get('profile')} reviewed={source.get('review', {}).get('reviewed')} "
                f"promotable={source.get('promotable_count')} promoted={source.get('promoted_count')}",
            ]
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _resolve_sources(
    profile_run_path: str | Path | None,
    agent_lab_report_path: str | Path | None,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if profile_run_path:
        profile_payload = json.loads(Path(profile_run_path).read_text(encoding="utf-8"))
        evidence_pack = profile_payload.get("evidence_pack", {})
        for run in profile_payload.get("profile_runs", []):
            if run.get("runner") != "agent_lab" or run.get("status") != "completed" or not run.get("report_json"):
                continue
            sources.append(
                {
                    "profile": run.get("profile"),
                    "profile_run_id": profile_payload.get("run_id"),
                    "evidence_pack_run_id": evidence_pack.get("run_id"),
                    "evidence_pack_path": evidence_pack.get("path"),
                    "report_json": run["report_json"],
                }
            )
    if agent_lab_report_path:
        sources.append(
            {
                "profile": "manual_agent_lab_report",
                "profile_run_id": None,
                "evidence_pack_run_id": None,
                "evidence_pack_path": None,
                "report_json": str(agent_lab_report_path),
            }
        )
    if not sources:
        raise ValueError("No completed Agent Lab report sources were found.")
    return sources


def _review_state(run_id: str, review_actions: list[ReviewActionRecord]) -> dict[str, Any]:
    matching = [
        action
        for action in review_actions
        if action.source_type == "agent_lab_report"
        and action.source_id == run_id
        and action.status != "voided"
    ]
    reviewed = [action for action in matching if action.action_type == "mark_reviewed"]
    needs_more_data = [action for action in matching if action.action_type == "needs_more_data"]
    return {
        "reviewed": bool(reviewed),
        "needs_more_data": bool(needs_more_data),
        "action_count": len(matching),
        "review_action_ids": [action.action_id for action in matching],
    }


def _candidate_from_note(
    note: ResearchNote,
    report: AgentLabRunReport,
    source: dict[str, Any],
    review: dict[str, Any],
    require_review: bool,
    allow_weak_notes: bool,
    existing_note_ids: set[str],
    default_horizon_days: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    if require_review and not review["reviewed"]:
        blockers.append("source_agent_lab_report_not_marked_reviewed")
    if review["needs_more_data"]:
        blockers.append("source_has_open_needs_more_data_action")
    if note.data_quality == "weak" and not allow_weak_notes:
        blockers.append("weak_note_data_quality")
    if note.note_id in existing_note_ids:
        blockers.append("duplicate_note_id")

    expected_direction = direction_from_note(note)
    horizon_days = note.horizon_days or default_horizon_days
    metadata = {
        "analyst_learning_bridge": True,
        "source_type": "agent_lab_report",
        "source_id": report.run_id,
        "profile": source.get("profile"),
        "profile_run_id": source.get("profile_run_id"),
        "evidence_pack_run_id": source.get("evidence_pack_run_id"),
        "evidence_pack_path": source.get("evidence_pack_path"),
        "review_action_ids": review.get("review_action_ids", []),
        "reviewed": review.get("reviewed", False),
        "tickers": note.tickers,
        "sectors": note.sectors,
        "context_tags": report.summary.get("context_tags", []),
        "regime_tags": report.summary.get("regime_tags", []),
        "note_data_quality": note.data_quality,
        "patterns": note.patterns,
    }
    return {
        "note_id": note.note_id,
        "agent_name": note.agent_name,
        "topic": note.topic,
        "data_quality": note.data_quality,
        "confidence": note.confidence,
        "expected_direction": expected_direction,
        "horizon_days": horizon_days,
        "can_promote": not blockers,
        "blockers": blockers,
        "metadata": metadata,
    }


def _promotion_gate(
    source_results: list[dict[str, Any]],
    candidate_count: int,
    promotable_count: int,
    blocked_count: int,
    apply: bool,
    promoted_count: int,
) -> dict[str, Any]:
    if not source_results:
        status = "no_sources"
    elif candidate_count == 0:
        status = "no_candidates"
    elif promotable_count == 0:
        status = "blocked"
    elif apply:
        status = "applied"
    else:
        status = "dry_run_ready"
    return {
        "status": status,
        "candidate_count": candidate_count,
        "promotable_count": promotable_count,
        "blocked_count": blocked_count,
        "promoted_count": promoted_count,
        "can_apply": promotable_count > 0,
    }


def _recommendations(
    candidate_count: int,
    promotable_count: int,
    blocked_count: int,
    apply: bool,
    promoted_records: list[AgentLearningRecord],
) -> list[str]:
    if candidate_count == 0:
        return ["No research notes were available for promotion."]
    if promotable_count == 0:
        return ["Do not apply promotion; resolve review/data-quality blockers first."]
    if not apply:
        return ["Dry-run is ready. Review candidates, then rerun with --apply only if the source review is approved."]
    recommendations = [f"Promoted {len(promoted_records)} learning record(s)."]
    if blocked_count:
        recommendations.append("Some candidates were still blocked; review blockers before assuming full promotion.")
    recommendations.append("Run context performance after outcomes become available, not immediately after promotion.")
    return recommendations


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
