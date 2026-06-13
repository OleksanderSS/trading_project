from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


DEFAULT_LEARNING_BRIDGE_PATH = "reports/dean_os/analyst_learning_bridge/latest.json"
DEFAULT_PROFILE_RUN_PATH = "reports/dean_os/analyst_profiles/latest.json"


class AnalystReviewInbox:
    """Read-only inbox for analyst/profile reports awaiting human review."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/analyst_review_inbox"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        learning_bridge_path: str | Path | None = DEFAULT_LEARNING_BRIDGE_PATH,
        profile_run_path: str | Path | None = DEFAULT_PROFILE_RUN_PATH,
        review_actions_path: str | Path = "data/dean_os/review_actions.sqlite",
        learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
        operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
        save: bool = True,
    ) -> dict[str, Any]:
        action_index = _read_review_actions(review_actions_path)
        sources, source_status = _resolve_sources(learning_bridge_path, profile_run_path)
        items = [
            _build_inbox_item(
                source=source,
                action_index=action_index,
                profile_run_path=profile_run_path,
                learning_path=learning_path,
                review_actions_path=review_actions_path,
                operations_path=operations_path,
            )
            for source in sources
        ]
        groups = _group_items(items)
        summary = _summary(groups, items, source_status, action_index)
        payload = {
            "run_id": _run_id("analyst_review_inbox"),
            "created_at": utc_now_iso(),
            "mode": "analyst_review_inbox",
            "inputs": {
                "learning_bridge_path": str(learning_bridge_path) if learning_bridge_path else None,
                "profile_run_path": str(profile_run_path) if profile_run_path else None,
                "review_actions_path": str(review_actions_path),
                "learning_path": str(learning_path),
                "operations_path": str(operations_path),
            },
            "summary": summary,
            "review_actions": {
                "status": action_index["status"],
                "path": str(review_actions_path),
                "source_count": len(action_index["by_source"]),
                "action_count": action_index["action_count"],
            },
            "groups": groups,
            "items": items,
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(summary),
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
        rendered_md = render_analyst_review_inbox_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_analyst_review_inbox_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    groups = payload.get("groups", {})
    lines = [
        "# DEAN-OS Analyst Review Inbox",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Sources: {summary.get('source_count')}",
        f"- Ready for manual review: {summary.get('ready_for_manual_review_count')}",
        f"- Needs more data candidates: {summary.get('needs_more_data_candidate_count')}",
        f"- Not reviewable yet: {summary.get('not_reviewable_yet_count')}",
        f"- Review action write performed: {summary.get('review_action_write_performed')}",
        "",
    ]
    for group_name in ["ready_for_manual_review", "needs_more_data_candidate", "not_reviewable_yet"]:
        lines.extend([f"## {group_name}", ""])
        entries = groups.get(group_name, [])
        if not entries:
            lines.append("- none")
        for item in entries:
            lines.append(
                f"- `{item.get('source_id')}` profile={item.get('profile')} "
                f"notes={item.get('note_count')} candidates={item.get('candidate_count')} reason={item.get('group_reason')}"
            )
        lines.append("")
    lines.extend(["## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _resolve_sources(
    learning_bridge_path: str | Path | None,
    profile_run_path: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if learning_bridge_path:
        bridge_path = Path(learning_bridge_path)
        if bridge_path.exists():
            bridge = _load_json(bridge_path)
            sources = bridge.get("sources", []) if isinstance(bridge, dict) else []
            if sources:
                return [dict(source, source_origin="learning_bridge") for source in sources], {
                    "status": "loaded_learning_bridge",
                    "path": str(bridge_path),
                }
    if profile_run_path:
        profile_path = Path(profile_run_path)
        if profile_path.exists():
            profile = _load_json(profile_path)
            sources = _sources_from_profile_run(profile, profile_path)
            if sources:
                return sources, {"status": "loaded_profile_run", "path": str(profile_path)}
    return [], {
        "status": "no_sources",
        "learning_bridge_path": str(learning_bridge_path) if learning_bridge_path else None,
        "profile_run_path": str(profile_run_path) if profile_run_path else None,
    }


def _sources_from_profile_run(profile: dict[str, Any], profile_path: Path) -> list[dict[str, Any]]:
    evidence_pack = profile.get("evidence_pack", {})
    sources: list[dict[str, Any]] = []
    for run in profile.get("profile_runs", []):
        if run.get("runner") != "agent_lab" or run.get("status") != "completed":
            continue
        report_json = run.get("report_json")
        if not report_json:
            continue
        report = _load_report_summary(report_json)
        sources.append(
            {
                "source_origin": "profile_run",
                "source_type": "agent_lab_report",
                "source_id": run.get("agent_lab_run_id") or report.get("run_id"),
                "profile": run.get("profile"),
                "profile_run_id": profile.get("run_id"),
                "evidence_pack_run_id": evidence_pack.get("run_id"),
                "evidence_pack_path": evidence_pack.get("path"),
                "report_json": report_json,
                "review": {},
                "note_count": run.get("note_count", report.get("note_count", 0)),
                "candidate_count": run.get("note_count", report.get("note_count", 0)),
                "promotable_count": 0,
                "promoted_count": 0,
                "candidates": [],
                "profile_run_path": str(profile_path),
            }
        )
    return sources


def _build_inbox_item(
    source: dict[str, Any],
    action_index: dict[str, Any],
    profile_run_path: str | Path | None,
    learning_path: str | Path,
    review_actions_path: str | Path,
    operations_path: str | Path,
) -> dict[str, Any]:
    source_id = source.get("source_id")
    report_info = _load_report_summary(source.get("report_json"))
    review = _merge_review_state(source.get("review", {}), action_index["by_source"].get(source_id, []))
    candidate_summary = _candidate_summary(source.get("candidates", []))
    note_count = int(source.get("note_count") or report_info.get("note_count") or len(report_info.get("research_notes", [])) or 0)
    candidate_count = int(source.get("candidate_count") or len(source.get("candidates", [])) or note_count)
    structural_blockers = _structural_blockers(source, report_info, note_count, candidate_count)
    classification, group_reason = _classify_item(
        review=review,
        candidate_summary=candidate_summary,
        structural_blockers=structural_blockers,
        candidate_count=candidate_count,
    )
    item = {
        "group": classification,
        "group_reason": group_reason,
        "source_type": source.get("source_type", "agent_lab_report"),
        "source_id": source_id,
        "profile": source.get("profile"),
        "profile_run_id": source.get("profile_run_id"),
        "evidence_pack_run_id": source.get("evidence_pack_run_id"),
        "evidence_pack_path": source.get("evidence_pack_path"),
        "report_json": source.get("report_json"),
        "report_exists": bool(report_info.get("exists")),
        "report_status": report_info.get("status"),
        "note_count": note_count,
        "candidate_count": candidate_count,
        "promotable_count": int(source.get("promotable_count") or 0),
        "promoted_count": int(source.get("promoted_count") or 0),
        "review": review,
        "candidate_summary": candidate_summary,
        "structural_blockers": structural_blockers,
        "suggested_commands": _suggested_commands(
            source_id=source_id,
            profile_run_path=profile_run_path or source.get("profile_run_path"),
            learning_path=learning_path,
            review_actions_path=review_actions_path,
            operations_path=operations_path,
        ),
    }
    return item


def _read_review_actions(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {"status": "missing_store", "action_count": 0, "by_source": {}}
    try:
        conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT payload FROM review_actions ORDER BY rowid").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {"status": f"unreadable_store:{type(exc).__name__}", "action_count": 0, "by_source": {}}

    by_source: dict[str, list[dict[str, Any]]] = {}
    action_count = 0
    for row in rows:
        try:
            action = json.loads(row["payload"])
        except Exception:
            continue
        if action.get("source_type") != "agent_lab_report" or action.get("status") == "voided":
            continue
        source_id = action.get("source_id")
        if not source_id:
            continue
        by_source.setdefault(source_id, []).append(action)
        action_count += 1
    return {"status": "loaded", "action_count": action_count, "by_source": by_source}


def _merge_review_state(artifact_review: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    if actions:
        reviewed = [action for action in actions if action.get("action_type") == "mark_reviewed"]
        needs_more_data = [action for action in actions if action.get("action_type") == "needs_more_data"]
        return {
            "reviewed": bool(reviewed),
            "needs_more_data": bool(needs_more_data),
            "action_count": len(actions),
            "review_action_ids": [action.get("action_id") for action in actions if action.get("action_id")],
            "source": "review_actions_store",
        }
    return {
        "reviewed": bool(artifact_review.get("reviewed")),
        "needs_more_data": bool(artifact_review.get("needs_more_data")),
        "action_count": int(artifact_review.get("action_count", 0) or 0),
        "review_action_ids": artifact_review.get("review_action_ids", []),
        "source": "learning_bridge_artifact",
    }


def _candidate_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    agent_counts: Counter[str] = Counter()
    for candidate in candidates:
        quality_counts[str(candidate.get("data_quality", "unknown"))] += 1
        agent_counts[str(candidate.get("agent_name", "unknown"))] += 1
        for blocker in candidate.get("blockers", []):
            blocker_counts[str(blocker)] += 1
    return {
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "data_quality_counts": dict(sorted(quality_counts.items())),
        "agent_counts": dict(sorted(agent_counts.items())),
        "only_missing_review": bool(blocker_counts)
        and set(blocker_counts) == {"source_agent_lab_report_not_marked_reviewed"},
    }


def _classify_item(
    review: dict[str, Any],
    candidate_summary: dict[str, Any],
    structural_blockers: list[str],
    candidate_count: int,
) -> tuple[str, str]:
    if structural_blockers:
        return "not_reviewable_yet", ", ".join(structural_blockers)
    if review.get("reviewed"):
        return "not_reviewable_yet", "already_reviewed"
    if review.get("needs_more_data"):
        return "needs_more_data_candidate", "open_needs_more_data_action"
    blockers = set(candidate_summary.get("blocker_counts", {}))
    if not candidate_count:
        return "not_reviewable_yet", "no_candidates"
    if not blockers or candidate_summary.get("only_missing_review"):
        return "ready_for_manual_review", "requires_human_citation_and_thesis_review"
    return "needs_more_data_candidate", "candidate_blockers_beyond_review_gate"


def _structural_blockers(
    source: dict[str, Any],
    report_info: dict[str, Any],
    note_count: int,
    candidate_count: int,
) -> list[str]:
    blockers: list[str] = []
    if not source.get("source_id"):
        blockers.append("missing_source_id")
    if not source.get("report_json"):
        blockers.append("missing_report_json")
    elif not report_info.get("exists"):
        blockers.append("report_json_missing")
    elif report_info.get("status") != "loaded":
        blockers.append(str(report_info.get("status")))
    if note_count <= 0 and candidate_count <= 0:
        blockers.append("no_notes_or_candidates")
    return blockers


def _suggested_commands(
    source_id: str | None,
    profile_run_path: str | Path | None,
    learning_path: str | Path,
    review_actions_path: str | Path,
    operations_path: str | Path,
) -> dict[str, str | None]:
    if not source_id:
        return {"inspect_report": None, "mark_reviewed_preview": None, "needs_more_data_preview": None}
    profile_arg = f"--profile-run-json {profile_run_path}" if profile_run_path else ""
    base = (
        f"python run_agent_review_approved_learning.py {profile_arg} "
        f"--learning-store {learning_path} --review-actions-store {review_actions_path} "
        f"--operations-store {operations_path}"
    ).replace("  ", " ").strip()
    return {
        "inspect_report": "Open the listed report_json and verify citations/thesis before recording any review action.",
        "mark_reviewed_preview": f'{base} --mark-reviewed --review-notes "Reviewed citations and accepted for pending outcome tracking."',
        "needs_more_data_preview": f'{base} --needs-more-data "Add stronger citations or missing source coverage before learning promotion." --review-notes "Current source is too thin."',
    }


def _group_items(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {"ready_for_manual_review": [], "needs_more_data_candidate": [], "not_reviewable_yet": []}
    for item in items:
        groups[item["group"]].append(item)
    return groups


def _summary(
    groups: dict[str, list[dict[str, Any]]],
    items: list[dict[str, Any]],
    source_status: dict[str, Any],
    action_index: dict[str, Any],
) -> dict[str, Any]:
    ready = len(groups["ready_for_manual_review"])
    needs_more_data = len(groups["needs_more_data_candidate"])
    not_reviewable = len(groups["not_reviewable_yet"])
    if ready:
        status = "ready_for_manual_review"
    elif needs_more_data:
        status = "needs_more_data"
    elif items:
        status = "no_reviewable_sources"
    else:
        status = "empty"
    return {
        "status": status,
        "source_count": len(items),
        "ready_for_manual_review_count": ready,
        "needs_more_data_candidate_count": needs_more_data,
        "not_reviewable_yet_count": not_reviewable,
        "source_status": source_status,
        "review_action_store_status": action_index["status"],
        "review_action_write_performed": False,
        "learning_write_performed": False,
        "config_write_performed": False,
        "pipeline_run_performed": False,
        "broker_access_performed": False,
    }


def _load_report_summary(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"status": "missing_path", "exists": False}
    resolved = Path(path)
    if not resolved.exists():
        return {"status": "missing_report", "exists": False, "path": str(path)}
    try:
        payload = _load_json(resolved)
    except Exception:
        return {"status": "unreadable_report_json", "exists": True, "path": str(path)}
    research_notes = payload.get("research_notes", []) if isinstance(payload, dict) else []
    return {
        "status": "loaded",
        "exists": True,
        "path": str(path),
        "run_id": payload.get("run_id"),
        "note_count": int(payload.get("note_count") or len(research_notes) or 0),
        "research_notes": research_notes,
        "summary": payload.get("summary", {}),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _operator_notes() -> list[str]:
    return [
        "This inbox is read-only and never records review actions.",
        "Use it to decide whether to inspect a report, mark it reviewed, or request more data.",
        "A source in ready_for_manual_review is not automatically approved.",
        "A mark-reviewed command should be run only after checking citations, thesis quality, and source coverage.",
    ]


def _recommendations(summary: dict[str, Any]) -> list[str]:
    if summary["status"] == "ready_for_manual_review":
        return [
            "Inspect ready reports manually before recording mark-reviewed.",
            "If citations or coverage are weak, use the needs-more-data command instead of approval.",
            "After review actions are recorded, rerun the learning bridge in dry-run mode before apply.",
        ]
    if summary["status"] == "needs_more_data":
        return [
            "Do not mark these sources reviewed yet.",
            "Improve evidence coverage or rerun the evidence/profile flow, then rebuild the inbox.",
        ]
    if summary["status"] == "no_reviewable_sources":
        return ["No source currently needs review; continue with the daily check or build better evidence."]
    return ["No analyst review sources were found. Run analyst profiles or learning bridge dry-run first."]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
