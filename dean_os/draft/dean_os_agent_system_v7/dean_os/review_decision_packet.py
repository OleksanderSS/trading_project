from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_INBOX_PATH = "reports/dean_os/analyst_review_inbox/latest.json"


class ReviewDecisionPacket:
    """Read-only packet for deciding whether an analyst source can be reviewed."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/review_decision_packet"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        inbox_path: str | Path = DEFAULT_INBOX_PATH,
        source_id: str | None = None,
        max_notes: int = 6,
        max_citations_per_note: int = 3,
        max_text_chars: int = 500,
        save: bool = True,
    ) -> dict[str, Any]:
        inbox = _load_json(Path(inbox_path))
        item = _select_inbox_item(inbox, source_id)
        report = _load_report(item.get("report_json"))
        evidence_pack = _load_optional_json(item.get("evidence_pack_path"))
        notes = _summarize_notes(
            report.get("research_notes", []),
            item.get("candidate_summary", {}),
            max_notes=max_notes,
            max_citations_per_note=max_citations_per_note,
            max_text_chars=max_text_chars,
        )
        evidence_summary = _evidence_summary(evidence_pack)
        review_checks = _review_checks(item=item, notes=notes, evidence_summary=evidence_summary)
        decision_guidance = _decision_guidance(review_checks)
        payload = {
            "run_id": _run_id("review_decision_packet"),
            "created_at": utc_now_iso(),
            "mode": "review_decision_packet",
            "inputs": {
                "inbox_path": str(inbox_path),
                "source_id": source_id,
                "max_notes": max_notes,
                "max_citations_per_note": max_citations_per_note,
                "max_text_chars": max_text_chars,
            },
            "summary": {
                "source_id": item.get("source_id"),
                "profile": item.get("profile"),
                "packet_status": decision_guidance["status"],
                "recommended_review_action": decision_guidance["recommended_review_action"],
                "manual_review_required": True,
                "review_action_write_performed": False,
                "learning_write_performed": False,
                "proposal_enqueue_performed": False,
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
            },
            "source": {
                "source_type": item.get("source_type"),
                "source_id": item.get("source_id"),
                "profile": item.get("profile"),
                "profile_run_id": item.get("profile_run_id"),
                "evidence_pack_run_id": item.get("evidence_pack_run_id"),
                "evidence_pack_path": item.get("evidence_pack_path"),
                "report_json": item.get("report_json"),
                "group": item.get("group"),
                "group_reason": item.get("group_reason"),
                "review": item.get("review", {}),
                "candidate_summary": item.get("candidate_summary", {}),
                "suggested_commands": item.get("suggested_commands", {}),
            },
            "report_summary": _report_summary(report),
            "evidence_pack": evidence_summary,
            "notes": notes,
            "review_checks": review_checks,
            "decision_guidance": decision_guidance,
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(decision_guidance),
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
        rendered_md = render_review_decision_packet_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_review_decision_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Review Decision Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Source ID: `{summary.get('source_id')}`",
        f"- Profile: `{summary.get('profile')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Recommended action: `{summary.get('recommended_review_action')}`",
        f"- Review action write performed: {summary.get('review_action_write_performed')}",
        "",
        "## Checks",
        "",
    ]
    for check in payload.get("review_checks", []):
        lines.append(f"- `{check.get('status')}` {check.get('code')}: {check.get('message')}")
    lines.extend(["", "## Notes", ""])
    for note in payload.get("notes", []):
        lines.extend(
            [
                f"### {note.get('agent_name')} / {note.get('topic')}",
                "",
                f"- Thesis: {note.get('thesis')}",
                f"- Confidence: {note.get('confidence')} | quality={note.get('data_quality')}",
                f"- Citations: {note.get('citation_count')}",
                "",
            ]
        )
    lines.extend(["## Command Previews", ""])
    for key, command in payload.get("source", {}).get("suggested_commands", {}).items():
        if command:
            lines.append(f"- {key}: `{command}`")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    lines.extend(["", "## Decision Rationale", ""])
    lines.extend(f"- {item}" for item in guidance.get("reasons", []))
    return "\n".join(lines).strip() + "\n"


def _select_inbox_item(inbox: dict[str, Any], source_id: str | None) -> dict[str, Any]:
    items = inbox.get("items", [])
    if not items:
        raise ValueError("Review inbox has no items.")
    if source_id:
        for item in items:
            if item.get("source_id") == source_id:
                return item
        raise ValueError(f"Source not found in inbox: {source_id}")
    ready = inbox.get("groups", {}).get("ready_for_manual_review", [])
    return ready[0] if ready else items[0]


def _load_report(path: str | Path | None) -> dict[str, Any]:
    if not path:
        raise ValueError("Selected inbox item has no report_json path.")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Agent Lab report not found: {path}")
    return _load_json(resolved)


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    try:
        return _load_json(resolved)
    except Exception:
        return {}


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    reports = report.get("reports", [])
    return {
        "run_id": report.get("run_id"),
        "document_count": int(report.get("document_count", 0) or 0),
        "chunk_count": int(report.get("chunk_count", 0) or 0),
        "note_count": int(report.get("note_count", len(report.get("research_notes", []))) or 0),
        "agent_reports": [
            {
                "agent_name": item.get("agent_name"),
                "verdict": item.get("verdict"),
                "confidence": item.get("confidence"),
                "data_quality_score": item.get("data_quality_score"),
                "reason_preview": _truncate(" ".join(item.get("reasons", [])[:2]), 260),
            }
            for item in reports
        ],
    }


def _summarize_notes(
    notes: list[dict[str, Any]],
    candidate_summary: dict[str, Any],
    max_notes: int,
    max_citations_per_note: int,
    max_text_chars: int,
) -> list[dict[str, Any]]:
    blocker_counts = candidate_summary.get("blocker_counts", {})
    summarized: list[dict[str, Any]] = []
    for note in notes[: max(max_notes, 0)]:
        citations = note.get("citations", [])
        summarized.append(
            {
                "note_id": note.get("note_id"),
                "agent_name": note.get("agent_name"),
                "topic": note.get("topic"),
                "thesis": _truncate(note.get("thesis", ""), max_text_chars),
                "patterns": note.get("patterns", []),
                "tickers": note.get("tickers", []),
                "sectors": note.get("sectors", []),
                "horizon_days": note.get("horizon_days"),
                "confidence": note.get("confidence"),
                "data_quality": note.get("data_quality"),
                "citation_count": len(citations),
                "citations": [_summarize_citation(citation, max_text_chars) for citation in citations[:max_citations_per_note]],
                "risks": [_truncate(item, max_text_chars) for item in note.get("risks", [])[:3]],
                "blind_spots": [_truncate(item, max_text_chars) for item in note.get("blind_spots", [])[:3]],
                "review_blockers": blocker_counts,
            }
        )
    return summarized


def _summarize_citation(citation: dict[str, Any], max_text_chars: int) -> dict[str, Any]:
    return {
        "source_id": citation.get("source_id"),
        "source_type": citation.get("source_type"),
        "title": _truncate(citation.get("title", ""), 220),
        "uri": citation.get("uri"),
        "timestamp": citation.get("timestamp"),
        "excerpt": _truncate(citation.get("excerpt", ""), max_text_chars),
    }


def _evidence_summary(evidence_pack: dict[str, Any]) -> dict[str, Any]:
    coverage = evidence_pack.get("coverage", {}) if evidence_pack else {}
    return {
        "available": bool(evidence_pack),
        "run_id": evidence_pack.get("run_id") if evidence_pack else None,
        "document_count": int(coverage.get("document_count", 0) or 0),
        "data_quality": coverage.get("data_quality"),
        "agent_lab_ready": bool(coverage.get("agent_lab_ready")),
        "source_types": coverage.get("by_source_type", {}),
        "tickers": coverage.get("tickers", []),
        "missing_requested_tickers": coverage.get("missing_requested_tickers", []),
        "date_range": coverage.get("date_range", {}),
        "warning_count": int(coverage.get("warning_count", 0) or 0),
        "dropped_count": int(coverage.get("dropped_count", 0) or 0),
    }


def _review_checks(
    item: dict[str, Any],
    notes: list[dict[str, Any]],
    evidence_summary: dict[str, Any],
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    group = item.get("group")
    if group == "ready_for_manual_review":
        checks.append(_check("pass", "inbox_ready", "Inbox classified this source as ready for manual review."))
    elif group == "needs_more_data_candidate":
        checks.append(_check("fail", "needs_more_data_candidate", "Inbox says this source needs more data before review approval."))
    else:
        checks.append(_check("fail", "not_reviewable_yet", f"Inbox group is {group}."))

    if evidence_summary.get("available"):
        checks.append(_check("pass", "evidence_pack_available", "Evidence pack is available for source review."))
    else:
        checks.append(_check("warn", "evidence_pack_missing", "Evidence pack was not available from the inbox item."))
    if evidence_summary.get("data_quality") not in {"clean", "strong"}:
        checks.append(_check("warn", "evidence_quality_not_clean", f"Evidence quality is {evidence_summary.get('data_quality')}."))
    elif evidence_summary.get("data_quality") == "strong":
        checks.append(_check("pass", "evidence_quality_strong", "Evidence quality is strong."))
    if evidence_summary.get("missing_requested_tickers"):
        missing = ", ".join(evidence_summary.get("missing_requested_tickers", []))
        checks.append(_check("warn", "missing_requested_tickers", f"Evidence pack is missing requested tickers: {missing}."))

    if not notes:
        checks.append(_check("fail", "no_notes", "Agent Lab report has no research notes to review."))
    else:
        strong_count = sum(1 for note in notes if note.get("data_quality") == "strong")
        citation_count = sum(int(note.get("citation_count", 0) or 0) for note in notes)
        if strong_count:
            checks.append(_check("pass", "strong_notes_present", f"{strong_count} reviewed note(s) have strong data quality."))
        else:
            checks.append(_check("warn", "no_strong_notes", "No summarized note has strong data quality."))
        if citation_count:
            checks.append(_check("pass", "citations_present", f"Summarized notes include {citation_count} citation(s)."))
        else:
            checks.append(_check("fail", "no_citations", "Summarized notes include no citations."))

    if item.get("review", {}).get("reviewed"):
        checks.append(_check("fail", "already_reviewed", "Source already has a mark-reviewed action."))
    if item.get("review", {}).get("needs_more_data"):
        checks.append(_check("fail", "open_needs_more_data", "Source already has an open needs-more-data action."))
    return checks


def _decision_guidance(checks: list[dict[str, str]]) -> dict[str, Any]:
    fails = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    if fails:
        status = "needs_more_data_recommended"
        action = "needs_more_data"
    elif warnings:
        status = "manual_review_with_warnings"
        action = "operator_decides"
    else:
        status = "reviewable"
        action = "mark_reviewed_candidate"
    return {
        "status": status,
        "recommended_review_action": action,
        "fail_count": len(fails),
        "warning_count": len(warnings),
        "pass_count": sum(1 for check in checks if check["status"] == "pass"),
        "reasons": [check["message"] for check in [*fails, *warnings] if check.get("message")],
    }


def _recommendations(guidance: dict[str, Any]) -> list[str]:
    if guidance["recommended_review_action"] == "needs_more_data":
        return [
            "Do not mark this source reviewed yet.",
            "Use the needs-more-data preview or rerun evidence/profile generation with stronger sources.",
            "Rebuild the inbox and packet after the evidence gap is resolved.",
        ]
    if guidance["recommended_review_action"] == "operator_decides":
        return [
            "Warnings exist; inspect citations and ticker/source coverage before marking reviewed.",
            "If warnings are acceptable for a diagnostic-only learning record, record explicit review notes.",
            "If warnings are material, choose needs-more-data instead.",
        ]
    return [
        "This source is a candidate for mark-reviewed after manual citation/thesis inspection.",
        "After recording review, rerun the learning bridge dry-run before any apply.",
        "Do not change agent weights until outcomes mature and calibration gates pass.",
    ]


def _operator_notes() -> list[str]:
    return [
        "This packet is read-only and does not record the review decision.",
        "The recommended action is guidance, not an executed command.",
        "Manual review should verify citations, ticker coverage, thesis quality, and risks.",
        "Mark-reviewed only means the source may enter pending learning; it is not a trading signal.",
    ]


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _load_json(path: Path) -> dict[str, Any]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    payload = DeanPaths.load_json(path)
    return payload


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
