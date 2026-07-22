from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.draft.dean_os_agent_system_v7.dean_os.ticker_focused_research_note_builder import (
    DEFAULT_RESEARCH_BATCH,
    _counts,
)
from dean_os.utils import json_ready

DEFAULT_FOCUSED_NOTES = "reports/dean_os/ticker_focused_research_notes_current/latest.json"


class TickerFocusedReplayExamBridge:
    """Creates a read-only replay-exam overlay from ticker-focused notes."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/ticker_focused_replay_exam_bridge"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        research_batch_path: str | Path = DEFAULT_RESEARCH_BATCH,
        focused_notes_path: str | Path = DEFAULT_FOCUSED_NOTES,
        save: bool = True,
    ) -> dict[str, Any]:
        research_batch = _load_json(research_batch_path)
        focused_notes_report = _load_json(focused_notes_path)
        notes_by_key = _index_focused_notes(focused_notes_report.get("focused_notes", []))
        overlays = [
            _build_overlay(batch_run=run, notes_by_key=notes_by_key)
            for run in research_batch.get("runs", [])
        ]
        summary = _summary(research_batch, focused_notes_report, overlays)
        payload = {
            "run_id": _run_id("ticker_focused_replay_exam_bridge"),
            "created_at": utc_now_iso(),
            "mode": "ticker_focused_replay_exam_bridge",
            "inputs": {
                "research_batch_path": str(research_batch_path),
                "focused_notes_path": str(focused_notes_path),
            },
            "summary": summary,
            "batch_context": {
                "summary": research_batch.get("summary", {}),
                "focused_notes_summary": focused_notes_report.get("summary", {}),
            },
            "run_overlays": overlays,
            "issue_counts": _issue_counts(overlays),
            "tasks": _tasks(summary, overlays),
            "commands": _commands(research_batch, focused_notes_report, overlays),
            "safety": {
                "read_only": True,
                "original_replay_outputs_mutated": False,
                "data_mutation_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "pipeline_run_performed": False,
                "learning_write_performed": False,
                "operation_proposal_created": False,
                "config_write_performed": False,
                "broker_access_performed": False,
            },
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
        rendered_md = render_ticker_focused_replay_exam_bridge_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_ticker_focused_replay_exam_bridge_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Ticker-Focused Replay Exam Bridge",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('bridge_status')}`",
        f"- Runs: {summary.get('run_count')}",
        f"- Overlay-ready runs: {summary.get('overlay_ready_count')}",
        f"- Blocked runs: {summary.get('blocked_overlay_count')}",
        f"- Focused directional runs: {summary.get('focused_directional_count')}",
        "",
        "## Overlay Comparison",
        "",
    ]
    for overlay in payload.get("run_overlays", [])[:12]:
        focused = overlay.get("focused_exam", {})
        original = overlay.get("original_exam", {})
        lines.append(
            f"- as_of=`{overlay.get('as_of')}` ticker=`{overlay.get('price_ticker')}` "
            f"status=`{overlay.get('overlay_status')}` "
            f"original=`{original.get('research_stance')}/{original.get('exam_verdict')}` "
            f"focused=`{focused.get('research_stance')}/{focused.get('exam_verdict')}`"
        )
    lines.extend(["", "## Tasks", ""])
    tasks = payload.get("tasks", [])
    lines.extend(f"- `{task.get('priority')}` {task.get('task_id')}: {task.get('description')}" for task in tasks) if tasks else lines.append("- None.")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _build_overlay(batch_run: dict[str, Any], notes_by_key: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    price_ticker = str(batch_run.get("price_ticker") or "").upper()
    as_of = str(batch_run.get("as_of") or "")
    run_id = str(batch_run.get("run_id") or "")
    note = notes_by_key.get(("run_id", run_id)) or notes_by_key.get(("asof_ticker", f"{as_of}|{price_ticker}")) or {}
    original_exam = _original_exam(batch_run)
    focused_exam, issues = _focused_exam(batch_run, note)
    return {
        "run_id": run_id,
        "as_of": as_of,
        "horizon_days": batch_run.get("horizon_days"),
        "price_ticker": price_ticker,
        "price_expected_direction": batch_run.get("price_expected_direction"),
        "outcome_label": batch_run.get("outcome_label"),
        "realized_return": batch_run.get("realized_return"),
        "overlay_status": _overlay_status(issues),
        "original_exam": original_exam,
        "focused_exam": focused_exam,
        "comparison": _comparison(original_exam, focused_exam),
        "focused_note": _compact_note(note),
        "issues": issues,
    }


def _original_exam(batch_run: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_stance": batch_run.get("research_stance"),
        "research_expected_direction": batch_run.get("research_expected_direction"),
        "research_confidence": batch_run.get("research_confidence"),
        "research_data_quality": batch_run.get("research_data_quality"),
        "ticker_specificity": batch_run.get("ticker_specificity"),
        "research_price_agreement": batch_run.get("research_price_agreement"),
        "exam_verdict": batch_run.get("exam_verdict"),
        "learning_gate_status": batch_run.get("learning_gate_status"),
    }


def _focused_exam(batch_run: dict[str, Any], note: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    if not note:
        issues.append("missing_focused_note")
    elif note.get("note_status") != "focused_note_ready":
        issues.extend(note.get("issues", []) or ["focused_note_not_ready"])
    price_direction = str(batch_run.get("price_expected_direction") or "neutral")
    quality_warnings = batch_run.get("quality_warnings", []) or []
    outcome_label = batch_run.get("outcome_label")

    if issues:
        stance = "insufficient_data"
        research_direction = "neutral"
        agreement = _agreement(research_direction, price_direction)
        verdict = "focused_note_blocked"
        learning_gate = _blocked_learning_gate(issues)
        confidence = note.get("confidence", 0.0) if note else 0.0
        data_quality = note.get("data_quality", "weak") if note else "weak"
        thesis = note.get("thesis", "No focused note was available.") if note else "No focused note was available."
    else:
        stance = str(note.get("research_stance") or "mixed")
        research_direction = str(note.get("expected_direction") or _direction_from_stance(stance))
        agreement = _agreement(research_direction, price_direction)
        verdict = _exam_verdict(
            research_direction=research_direction,
            price_direction=price_direction,
            agreement=agreement,
            outcome_label=outcome_label,
            quality_warnings=quality_warnings,
        )
        learning_gate = _learning_gate(batch_run, quality_warnings)
        confidence = note.get("confidence", 0.0)
        data_quality = note.get("data_quality", "weak")
        thesis = note.get("thesis")

    return (
        {
            "selected_note_agent": note.get("agent_name") if note else None,
            "selected_note_id": note.get("note_id") if note else None,
            "research_thesis": thesis,
            "research_stance": stance,
            "research_expected_direction": research_direction,
            "research_confidence": confidence,
            "research_data_quality": data_quality,
            "ticker_specificity": "single_ticker" if not issues and note.get("price_ticker") else "none",
            "price_expected_direction": price_direction,
            "research_price_agreement": agreement,
            "exam_verdict": verdict,
            "learning_gate": learning_gate,
        },
        issues,
    )


def _agreement(research_direction: str, price_direction: str) -> str:
    if research_direction == "neutral":
        return "research_inconclusive"
    if price_direction == "neutral":
        return "price_inconclusive"
    if research_direction == price_direction:
        return "aligned"
    return "conflict"


def _direction_from_stance(stance: str) -> str:
    if stance == "constructive":
        return "bullish"
    if stance == "risk_heavy":
        return "bearish"
    return "neutral"


def _exam_verdict(
    research_direction: str,
    price_direction: str,
    agreement: str,
    outcome_label: Any,
    quality_warnings: list[Any],
) -> str:
    if quality_warnings:
        return "diagnostic_only_quality_warning"
    if research_direction == "neutral" and price_direction != "neutral":
        return "price_only_candidate_not_research_confirmed"
    if agreement == "conflict":
        return "research_price_conflict"
    if outcome_label == "hit":
        return "aligned_hit" if agreement == "aligned" else "price_hit_research_inconclusive"
    if outcome_label == "miss":
        return "aligned_miss_review_needed" if agreement == "aligned" else "price_miss_research_inconclusive"
    return "review_required"


def _learning_gate(batch_run: dict[str, Any], quality_warnings: list[Any]) -> dict[str, Any]:
    if quality_warnings:
        return {
            "status": "blocked",
            "can_write_learning_memory": False,
            "reason": "Price-quality warnings make this focused replay overlay diagnostic only.",
        }
    if batch_run.get("evaluation_status") != "evaluated":
        return {
            "status": "pending_outcome",
            "can_write_learning_memory": False,
            "reason": "Outcome window is not evaluable yet.",
        }
    return {
        "status": "review_required",
        "can_write_learning_memory": False,
        "reason": "A human-reviewed learning apply ceremony must approve any memory write.",
    }


def _blocked_learning_gate(issues: list[str]) -> dict[str, Any]:
    return {
        "status": "blocked_focused_note",
        "can_write_learning_memory": False,
        "reason": "Focused note is not ready: " + ", ".join(issues),
    }


def _overlay_status(issues: list[str]) -> str:
    if not issues:
        return "focused_overlay_ready"
    if "missing_focused_note" in issues:
        return "blocked_missing_focused_note"
    return "blocked_focused_note_not_ready"


def _comparison(original_exam: dict[str, Any], focused_exam: dict[str, Any]) -> dict[str, Any]:
    return {
        "stance_changed": original_exam.get("research_stance") != focused_exam.get("research_stance"),
        "direction_changed": original_exam.get("research_expected_direction") != focused_exam.get("research_expected_direction"),
        "agreement_changed": original_exam.get("research_price_agreement") != focused_exam.get("research_price_agreement"),
        "verdict_changed": original_exam.get("exam_verdict") != focused_exam.get("exam_verdict"),
        "specificity_improved": original_exam.get("ticker_specificity") != "single_ticker"
        and focused_exam.get("ticker_specificity") == "single_ticker",
    }


def _compact_note(note: dict[str, Any]) -> dict[str, Any]:
    return {
        "note_id": note.get("note_id"),
        "note_status": note.get("note_status"),
        "price_ticker": note.get("price_ticker"),
        "direct_document_count": note.get("direct_document_count"),
        "citation_count": note.get("citation_count"),
        "confidence": note.get("confidence"),
        "data_quality": note.get("data_quality"),
        "limitations": note.get("limitations", []),
    }


def _summary(
    research_batch: dict[str, Any],
    focused_notes_report: dict[str, Any],
    overlays: list[dict[str, Any]],
) -> dict[str, Any]:
    run_count = len(overlays)
    status_counts = _counts(overlay.get("overlay_status") for overlay in overlays)
    ready_count = status_counts.get("focused_overlay_ready", 0)
    blocked_count = run_count - ready_count
    original_directional = sum(1 for item in overlays if item.get("original_exam", {}).get("research_expected_direction") in {"bullish", "bearish"})
    focused_directional = sum(1 for item in overlays if item.get("focused_exam", {}).get("research_expected_direction") in {"bullish", "bearish"})
    focused_inconclusive = sum(1 for item in overlays if item.get("focused_exam", {}).get("research_expected_direction") == "neutral")
    if run_count == 0:
        status = "no_runs"
        next_action = "provide_selected_replay_batch_and_focused_notes"
    elif ready_count == run_count:
        status = "focused_overlay_ready"
        next_action = "prepare_runner_integration_review"
    elif ready_count:
        status = "partial_focused_overlay_ready"
        next_action = "integrate_ready_overlay_and_backfill_blocked_windows"
    else:
        status = "blocked_no_focused_overlay"
        next_action = "build_or_backfill_focused_notes"
    return {
        "bridge_status": status,
        "next_action": next_action,
        "run_count": run_count,
        "overlay_ready_count": ready_count,
        "blocked_overlay_count": blocked_count,
        "original_directional_count": original_directional,
        "focused_directional_count": focused_directional,
        "focused_inconclusive_count": focused_inconclusive,
        "status_counts": status_counts,
        "focused_exam_verdict_counts": _counts(overlay.get("focused_exam", {}).get("exam_verdict") for overlay in overlays),
        "original_exam_verdict_counts": _counts(overlay.get("original_exam", {}).get("exam_verdict") for overlay in overlays),
        "price_ticker_counts": _counts(overlay.get("price_ticker") for overlay in overlays),
        "specificity_improved_count": sum(1 for overlay in overlays if overlay.get("comparison", {}).get("specificity_improved")),
        "research_batch_summary": research_batch.get("summary", {}),
        "focused_notes_summary": focused_notes_report.get("summary", {}),
        "can_integrate_runner": ready_count > 0,
        "can_change_analyst_weights": False,
        "can_write_learning_memory": False,
    }


def _issue_counts(overlays: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for overlay in overlays:
        counts.update(overlay.get("issues", []))
    return dict(sorted(counts.items()))


def _tasks(summary: dict[str, Any], overlays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    ready = [item for item in overlays if item.get("overlay_status") == "focused_overlay_ready"]
    blocked = [item for item in overlays if item.get("overlay_status") != "focused_overlay_ready"]
    if ready:
        tasks.append(
            {
                "priority": "P0",
                "task_id": "review_focused_overlay_runner_integration",
                "description": "Review how ready focused notes change replay exams before wiring them into HistoricalResearchReplayRunner.",
                "affected_runs": [item.get("as_of") for item in ready],
                "affected_price_tickers": sorted({item.get("price_ticker") for item in ready if item.get("price_ticker")}),
            }
        )
    if blocked:
        tasks.append(
            {
                "priority": "P1",
                "task_id": "keep_blocked_windows_out_of_calibration",
                "description": "Keep weak or missing focused-note windows out of calibration until direct evidence is backfilled.",
                "affected_runs": [item.get("as_of") for item in blocked],
                "affected_price_tickers": sorted({item.get("price_ticker") for item in blocked if item.get("price_ticker")}),
            }
        )
    if summary.get("can_integrate_runner"):
        tasks.append(
            {
                "priority": "P2",
                "task_id": "rerun_attribution_after_runner_overlay",
                "description": "After runner integration, rerun selected replay, attribution audit, focused notes, bridge, and readiness gate.",
            }
        )
    return tasks


def _commands(
    research_batch: dict[str, Any],
    focused_notes_report: dict[str, Any],
    overlays: list[dict[str, Any]],
) -> dict[str, str | None]:
    ready_dates = [str(item.get("as_of")) for item in overlays if item.get("overlay_status") == "focused_overlay_ready"]
    return {
        "rebuild_focused_notes": (
            "python run_agent_ticker_focused_notes.py "
            f"--research-batch-json {focused_notes_report.get('inputs', {}).get('research_batch_path', DEFAULT_RESEARCH_BATCH)} "
            "--output-dir reports\\dean_os\\ticker_focused_research_notes_current"
        ),
        "rerun_bridge": (
            "python run_agent_ticker_focused_replay_bridge.py "
            f"--research-batch-json {focused_notes_report.get('inputs', {}).get('research_batch_path', DEFAULT_RESEARCH_BATCH)} "
            "--focused-notes-json reports\\dean_os\\ticker_focused_research_notes_current\\latest.json "
            "--output-dir reports\\dean_os\\ticker_focused_replay_exam_bridge_current"
        ),
        "ready_as_of_dates": " ".join(ready_dates) if ready_dates else None,
        "source_price_data_path": research_batch.get("inputs", {}).get("price_data_path"),
    }


def _recommendations(summary: dict[str, Any]) -> list[str]:
    status = summary.get("bridge_status")
    if status == "focused_overlay_ready":
        return ["Review the overlay, then wire focused notes into the replay runner before calibration."]
    if status == "partial_focused_overlay_ready":
        return [
            "Use the ready overlay runs to design runner integration, but keep blocked windows out of calibration.",
            "Backfill direct evidence for blocked windows before judging analyst skill across the full selected batch.",
        ]
    if status == "blocked_no_focused_overlay":
        return ["Build or backfill focused notes before replay-exam integration."]
    return ["Provide selected replay and focused-note artifacts before building the bridge."]


def _index_focused_notes(notes: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for note in notes:
        run_id = str(note.get("run_id") or "")
        as_of = str(note.get("as_of") or "")
        price_ticker = str(note.get("price_ticker") or "").upper()
        if run_id:
            indexed[("run_id", run_id)] = note
        if as_of and price_ticker:
            indexed[("asof_ticker", f"{as_of}|{price_ticker}")] = note
    return indexed


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
