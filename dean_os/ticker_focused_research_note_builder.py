from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.ticker_specific_attribution_audit import DEFAULT_RESEARCH_BATCH
from dean_os.utils import json_ready


class TickerFocusedResearchNoteBuilder:
    """Builds ticker-focused note candidates from existing replay evidence packs.

    This is a read-only bridge between broad Agent Lab notes and ticker-level
    replay calibration. It does not mutate replay outputs; it creates a separate
    artifact that can be reviewed before any runner starts trusting a directional
    thesis as ticker-specific.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/ticker_focused_research_notes"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        research_batch_path: str | Path = DEFAULT_RESEARCH_BATCH,
        min_direct_documents: int = 3,
        max_citations_per_note: int = 5,
        save: bool = True,
    ) -> dict[str, Any]:
        research_batch = _load_json(research_batch_path)
        notes = [
            _build_run_note(
                batch_run=run,
                min_direct_documents=min_direct_documents,
                max_citations_per_note=max_citations_per_note,
            )
            for run in research_batch.get("runs", [])
        ]
        summary = _summary(notes)
        payload = {
            "run_id": _run_id("ticker_focused_research_notes"),
            "created_at": utc_now_iso(),
            "mode": "ticker_focused_research_note_builder",
            "inputs": {
                "research_batch_path": str(research_batch_path),
                "min_direct_documents": min_direct_documents,
                "max_citations_per_note": max_citations_per_note,
            },
            "summary": summary,
            "batch_context": {
                "summary": research_batch.get("summary", {}),
                "inputs": research_batch.get("inputs", {}),
            },
            "focused_notes": notes,
            "issue_counts": _issue_counts(notes),
            "tasks": _tasks(summary, notes),
            "commands": _commands(research_batch, notes),
            "safety": {
                "read_only": True,
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
        rendered_md = render_ticker_focused_notes_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_ticker_focused_notes_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Ticker-Focused Research Notes",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('builder_status')}`",
        f"- Runs: {summary.get('run_count')}",
        f"- Focused-note ready: {summary.get('ready_note_count')}",
        f"- Weak direct evidence: {summary.get('weak_direct_evidence_count')}",
        "",
        "## Focused Notes",
        "",
    ]
    for note in payload.get("focused_notes", [])[:12]:
        lines.append(
            f"- as_of=`{note.get('as_of')}` ticker=`{note.get('price_ticker')}` "
            f"status=`{note.get('note_status')}` direct_docs={note.get('direct_document_count')} "
            f"quality=`{note.get('data_quality')}` confidence={note.get('confidence')}"
        )
        if note.get("thesis"):
            lines.append(f"  Thesis: {note.get('thesis')}")
    lines.extend(["", "## Tasks", ""])
    tasks = payload.get("tasks", [])
    lines.extend(f"- `{task.get('priority')}` {task.get('task_id')}: {task.get('description')}" for task in tasks) if tasks else lines.append("- None.")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _build_run_note(batch_run: dict[str, Any], min_direct_documents: int, max_citations_per_note: int) -> dict[str, Any]:
    full_run = _load_full_run(batch_run)
    evidence_pack = _load_evidence_pack(full_run)
    documents = evidence_pack.get("documents", [])
    exam = full_run.get("research_exam", {})
    selected_note = _selected_note(full_run)
    price_ticker = _price_ticker(batch_run, full_run)
    direct_documents = _rank_direct_documents(documents, price_ticker)
    citations = [_citation(document) for document in direct_documents[:max_citations_per_note]]
    direct_count = len(direct_documents)
    issues = _issues(price_ticker, direct_count, min_direct_documents, full_run)
    status = _note_status(issues)
    confidence = _confidence(direct_count, min_direct_documents, len(citations))
    data_quality = _data_quality(direct_documents, direct_count, min_direct_documents)
    return {
        "note_id": _note_id(batch_run, price_ticker),
        "agent_name": "ticker_focused_research_note_builder",
        "run_id": batch_run.get("run_id") or full_run.get("run_id"),
        "as_of": batch_run.get("as_of") or full_run.get("inputs", {}).get("as_of"),
        "horizon_days": batch_run.get("horizon_days") or full_run.get("inputs", {}).get("horizon_days"),
        "price_ticker": price_ticker,
        "topic": f"{price_ticker or 'UNKNOWN'} ticker-focused replay evidence",
        "thesis": _thesis(price_ticker, status, exam, direct_count, data_quality),
        "note_status": status,
        "expected_direction": exam.get("research_expected_direction", "neutral"),
        "research_stance": exam.get("research_stance", "mixed"),
        "confidence": confidence,
        "data_quality": data_quality,
        "tickers": [price_ticker] if price_ticker else [],
        "sectors": sorted({sector for doc in direct_documents for sector in _strings(doc.get("sectors", []))}),
        "patterns": selected_note.get("patterns", []),
        "tailwinds": selected_note.get("tailwinds", []),
        "headwinds": selected_note.get("headwinds", []),
        "risks": _risks(status, selected_note, issues),
        "limitations": _limitations(issues, selected_note),
        "direct_document_count": direct_count,
        "source_type_counts": _counts(doc.get("source_type") for doc in direct_documents),
        "citations": citations,
        "citation_count": len(citations),
        "direct_document_titles": [str(doc.get("title", ""))[:180] for doc in direct_documents[:max_citations_per_note]],
        "inherited_context": {
            "selected_note_agent": exam.get("selected_note_agent"),
            "selected_note_tickers": _upper_list(selected_note.get("tickers", [])),
            "selected_note_ticker_count": len(_upper_list(selected_note.get("tickers", []))),
            "selected_note_thesis": selected_note.get("thesis"),
            "original_ticker_specificity": exam.get("ticker_specificity"),
            "original_exam_verdict": exam.get("exam_verdict"),
        },
        "issues": issues,
    }


def _load_full_run(batch_run: dict[str, Any]) -> dict[str, Any]:
    raw_path = batch_run.get("saved_paths", {}).get("json")
    if raw_path and Path(raw_path).exists():
        return _load_json(raw_path)
    return {}


def _load_evidence_pack(full_run: dict[str, Any]) -> dict[str, Any]:
    latest = full_run.get("evidence_pack", {}).get("saved_paths", {}).get("latest_json")
    if latest and Path(latest).exists():
        return _load_json(latest)
    return full_run.get("evidence_pack", {})


def _selected_note(full_run: dict[str, Any]) -> dict[str, Any]:
    selected_agent = full_run.get("research_exam", {}).get("selected_note_agent")
    notes = full_run.get("agent_lab", {}).get("research_notes", [])
    selected = [note for note in notes if note.get("agent_name") == selected_agent]
    if selected:
        return selected[-1]
    return notes[-1] if notes else {}


def _price_ticker(batch_run: dict[str, Any], full_run: dict[str, Any]) -> str:
    return str(batch_run.get("price_ticker") or full_run.get("price_replay", {}).get("decision", {}).get("ticker") or "").upper()


def _rank_direct_documents(documents: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    if not ticker:
        return []
    direct = [document for document in documents if ticker in _upper_list(document.get("tickers", []))]
    return sorted(direct, key=lambda document: _document_score(document, ticker), reverse=True)


def _document_score(document: dict[str, Any], ticker: str) -> tuple[float, str]:
    title = str(document.get("title") or "")
    text = str(document.get("text") or "")
    source_type = str(document.get("source_type") or "")
    score = 1.0
    if ticker in title.upper():
        score += 3.0
    if ticker in text.upper():
        score += 2.0
    if source_type in {"filing", "transcript"}:
        score += 2.0
    elif source_type in {"news", "article", "report"}:
        score += 1.0
    if document.get("published_at"):
        score += 0.5
    return (score, title)


def _citation(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(document.get("document_id") or _hash(document.get("title"), document.get("uri"))),
        "source_type": str(document.get("source_type") or "news"),
        "title": str(document.get("title") or "Untitled source")[:180],
        "uri": document.get("uri"),
        "timestamp": document.get("published_at"),
        "excerpt": _excerpt(document),
    }


def _excerpt(document: dict[str, Any], max_chars: int = 280) -> str | None:
    text = " ".join(str(document.get("text") or "").split())
    return text[:max_chars] if text else None


def _issues(price_ticker: str, direct_count: int, min_direct_documents: int, full_run: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not full_run:
        issues.append("missing_full_run_json")
    if not price_ticker:
        issues.append("missing_price_ticker")
    if direct_count == 0:
        issues.append("no_direct_price_ticker_documents")
    elif direct_count < min_direct_documents:
        issues.append("weak_direct_price_ticker_documents")
    return issues


def _note_status(issues: list[str]) -> str:
    if "missing_full_run_json" in issues:
        return "blocked_missing_run"
    if "missing_price_ticker" in issues or "no_direct_price_ticker_documents" in issues:
        return "blocked_missing_direct_evidence"
    if "weak_direct_price_ticker_documents" in issues:
        return "blocked_weak_direct_evidence"
    return "focused_note_ready"


def _confidence(direct_count: int, min_direct_documents: int, citation_count: int) -> float:
    if direct_count < min_direct_documents:
        return round(min(0.35 + 0.05 * direct_count, 0.5), 3)
    return round(min(0.55 + 0.04 * direct_count + 0.03 * citation_count, 0.82), 3)


def _data_quality(direct_documents: list[dict[str, Any]], direct_count: int, min_direct_documents: int) -> str:
    if direct_count < min_direct_documents:
        return "weak"
    source_types = {str(document.get("source_type") or "") for document in direct_documents}
    if direct_count >= 5 and len(source_types) >= 2:
        return "strong"
    return "partial"


def _thesis(price_ticker: str, status: str, exam: dict[str, Any], direct_count: int, data_quality: str) -> str:
    ticker = price_ticker or "the selected ticker"
    direction = str(exam.get("research_expected_direction") or "neutral")
    stance = str(exam.get("research_stance") or "mixed")
    if status != "focused_note_ready":
        return f"{ticker} does not yet have enough direct pre-as_of evidence for a ticker-focused replay thesis."
    if direction == "bullish":
        return (
            f"Direct pre-as_of evidence for {ticker} is sufficient for a ticker-focused constructive review "
            f"({direct_count} direct documents, {data_quality} quality); keep broader sector context separate."
        )
    if direction == "bearish":
        return (
            f"Direct pre-as_of evidence for {ticker} is sufficient for a ticker-focused risk review "
            f"({direct_count} direct documents, {data_quality} quality); keep broader sector context separate."
        )
    return (
        f"Direct pre-as_of evidence for {ticker} is sufficient to review the ticker directly, but the inherited "
        f"research stance remains {stance}/{direction}; do not force a directional signal."
    )


def _risks(status: str, selected_note: dict[str, Any], issues: list[str]) -> list[str]:
    risks = [str(item) for item in selected_note.get("risks", []) if str(item).strip()]
    if status != "focused_note_ready":
        risks.append("Ticker-focused thesis is blocked until direct evidence improves.")
    if "weak_direct_price_ticker_documents" in issues:
        risks.append("Direct evidence count is below the configured threshold.")
    return list(dict.fromkeys(risks))


def _limitations(issues: list[str], selected_note: dict[str, Any]) -> list[str]:
    limitations = list(issues)
    selected_tickers = _upper_list(selected_note.get("tickers", []))
    if len(selected_tickers) > 1:
        limitations.append("original_selected_note_was_basket_or_sector")
    if not limitations:
        limitations.append("focused_note_is_candidate_only_until_runner_integration")
    return list(dict.fromkeys(limitations))


def _summary(notes: list[dict[str, Any]]) -> dict[str, Any]:
    run_count = len(notes)
    status_counts = _counts(note.get("note_status") for note in notes)
    ready = status_counts.get("focused_note_ready", 0)
    weak = status_counts.get("blocked_weak_direct_evidence", 0)
    if run_count == 0:
        status = "no_runs"
        next_action = "provide_selected_research_replay_batch"
    elif ready == run_count:
        status = "focused_notes_ready"
        next_action = "wire_focused_notes_into_replay_exam"
    elif ready:
        status = "partial_focused_notes_ready"
        next_action = "backfill_weak_windows_then_wire_focused_notes"
    else:
        status = "blocked_no_ticker_focused_notes"
        next_action = "backfill_direct_ticker_evidence"
    return {
        "builder_status": status,
        "next_action": next_action,
        "run_count": run_count,
        "ready_note_count": ready,
        "weak_direct_evidence_count": weak,
        "status_counts": status_counts,
        "price_ticker_counts": _counts(note.get("price_ticker") for note in notes),
        "can_replace_selected_research_note": ready == run_count and run_count > 0,
        "can_change_analyst_weights": False,
        "can_write_learning_memory": False,
    }


def _issue_counts(notes: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for note in notes:
        counts.update(note.get("issues", []))
    return dict(sorted(counts.items()))


def _tasks(summary: dict[str, Any], notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    weak_notes = [note for note in notes if note.get("note_status") == "blocked_weak_direct_evidence"]
    ready_notes = [note for note in notes if note.get("note_status") == "focused_note_ready"]
    if ready_notes:
        tasks.append(
            {
                "priority": "P0",
                "task_id": "wire_focused_notes_into_replay_exam",
                "description": "Add an explicit review path for using ticker-focused notes instead of broad basket notes in historical research replay.",
                "affected_runs": [note.get("as_of") for note in ready_notes],
                "affected_price_tickers": sorted({note.get("price_ticker") for note in ready_notes if note.get("price_ticker")}),
            }
        )
    if weak_notes:
        tasks.append(
            {
                "priority": "P1",
                "task_id": "backfill_direct_price_ticker_documents",
                "description": "Add enough direct pre-as_of documents for weak ticker windows before including them in calibration.",
                "affected_runs": [note.get("as_of") for note in weak_notes],
                "affected_price_tickers": sorted({note.get("price_ticker") for note in weak_notes if note.get("price_ticker")}),
            }
        )
    if summary.get("ready_note_count", 0) < summary.get("run_count", 0):
        tasks.append(
            {
                "priority": "P2",
                "task_id": "rerun_note_builder_after_backfill",
                "description": "Rerun ticker-focused note building after evidence backfill or ticker-note integration changes.",
            }
        )
    return tasks


def _commands(research_batch: dict[str, Any], notes: list[dict[str, Any]]) -> dict[str, str | None]:
    ready_dates = [str(note.get("as_of")) for note in notes if note.get("note_status") == "focused_note_ready" and note.get("as_of")]
    inputs = research_batch.get("inputs", {})
    rebuild = (
        "python run_agent_ticker_focused_notes.py "
        "--research-batch-json reports\\dean_os\\historical_research_replay_batch_evidence_window_selected_after_directionality_fix\\latest.json "
        "--output-dir reports\\dean_os\\ticker_focused_research_notes_current"
    )
    return {
        "rebuild_ticker_focused_notes": rebuild,
        "ready_as_of_dates": " ".join(ready_dates) if ready_dates else None,
        "integration_note": (
            "Focused notes are review artifacts only until HistoricalResearchReplayRunner explicitly consumes them."
        ),
        "source_price_data_path": inputs.get("price_data_path"),
    }


def _recommendations(summary: dict[str, Any]) -> list[str]:
    status = summary.get("builder_status")
    if status == "focused_notes_ready":
        return ["Wire focused notes into the replay exam, then rerun attribution and readiness gates before calibration."]
    if status == "partial_focused_notes_ready":
        return [
            "Use ready focused notes as an integration smoke, but keep weak early windows out of calibration.",
            "Backfill direct evidence for weak windows before judging analyst skill across the full selected batch.",
        ]
    if status == "blocked_no_ticker_focused_notes":
        return ["Backfill direct ticker evidence before trying to build ticker-focused notes."]
    return ["Provide selected-window research replay outputs before building focused notes."]


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _upper_list(values: Any) -> list[str]:
    return sorted({str(value).strip().upper() for value in _strings(values)})


def _strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _counts(values: Any) -> dict[str, int]:
    counts: Counter[str] = Counter(str(value) for value in values if value)
    return dict(sorted(counts.items()))


def _note_id(batch_run: dict[str, Any], ticker: str) -> str:
    seed = "|".join(str(item) for item in (batch_run.get("run_id"), batch_run.get("as_of"), ticker))
    return "ticker_focused_" + sha256(seed.encode("utf-8")).hexdigest()[:16]


def _hash(*values: Any) -> str:
    return sha256("|".join(str(value or "") for value in values).encode("utf-8")).hexdigest()[:16]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
