from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_RESEARCH_BATCH = "reports/dean_os/historical_research_replay_batch_evidence_window_selected_after_directionality_fix/latest.json"


class TickerSpecificAttributionAudit:
    """Audits whether selected research replay theses are backed by ticker-specific evidence."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/ticker_specific_attribution_audit"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        research_batch_path: str | Path = DEFAULT_RESEARCH_BATCH,
        min_direct_documents: int = 3,
        max_selected_note_tickers: int = 2,
        save: bool = True,
    ) -> dict[str, Any]:
        research_batch = _load_json(research_batch_path)
        run_audits = [
            _run_audit(
                batch_run=run,
                min_direct_documents=min_direct_documents,
                max_selected_note_tickers=max_selected_note_tickers,
            )
            for run in research_batch.get("runs", [])
        ]
        summary = _summary(research_batch, run_audits)
        payload = {
            "run_id": _run_id("ticker_specific_attribution_audit"),
            "created_at": utc_now_iso(),
            "mode": "ticker_specific_attribution_audit",
            "inputs": {
                "research_batch_path": str(research_batch_path),
                "min_direct_documents": min_direct_documents,
                "max_selected_note_tickers": max_selected_note_tickers,
            },
            "summary": summary,
            "batch_context": {
                "summary": research_batch.get("summary", {}),
                "inputs": research_batch.get("inputs", {}),
            },
            "run_audits": run_audits,
            "issue_counts": _issue_counts(run_audits),
            "attribution_tasks": _tasks(summary, run_audits),
            "commands": _commands(research_batch, run_audits),
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
        rendered_md = render_ticker_specific_attribution_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_ticker_specific_attribution_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Ticker-Specific Attribution Audit",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('attribution_status')}`",
        f"- Runs: {summary.get('run_count')}",
        f"- Ticker-ready runs: {summary.get('ticker_ready_run_count')}",
        f"- Basket-note runs: {summary.get('basket_note_run_count')}",
        f"- Weak direct evidence runs: {summary.get('weak_direct_evidence_run_count')}",
        "",
        "## Top Issues",
        "",
    ]
    issue_counts = payload.get("issue_counts", {})
    lines.extend(f"- `{key}`: {value}" for key, value in issue_counts.items()) if issue_counts else lines.append("- None.")
    lines.extend(["", "## Run Audits", ""])
    for item in payload.get("run_audits", [])[:12]:
        lines.append(
            f"- as_of=`{item.get('as_of')}` price_ticker=`{item.get('price_ticker')}` "
            f"status=`{item.get('attribution_status')}` direct_docs={item.get('direct_document_count')} "
            f"note_tickers={item.get('selected_note_ticker_count')}"
        )
    lines.extend(["", "## Tasks", ""])
    tasks = payload.get("attribution_tasks", [])
    lines.extend(f"- `{task.get('priority')}` {task.get('task_id')}: {task.get('description')}" for task in tasks) if tasks else lines.append("- None.")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _run_audit(batch_run: dict[str, Any], min_direct_documents: int, max_selected_note_tickers: int) -> dict[str, Any]:
    raw_run_path = batch_run.get("saved_paths", {}).get("json")
    run_path = Path(raw_run_path) if raw_run_path else None
    full_run = _load_json(run_path) if run_path and run_path.exists() and run_path.is_file() else {}
    evidence_pack = _load_evidence_pack(full_run)
    documents = evidence_pack.get("documents", [])
    exam = full_run.get("research_exam", {})
    focused_overlay = full_run.get("focused_research_exam_overlay") or {}
    selected_note = _selected_note(full_run)
    price_ticker = str(batch_run.get("price_ticker") or full_run.get("price_replay", {}).get("decision", {}).get("ticker") or "").upper()
    direct_docs = _direct_documents(documents, price_ticker)
    focused_overlay_applied = bool(exam.get("focused_overlay_applied"))
    selected_note_tickers = _selected_note_tickers(
        selected_note=selected_note,
        price_ticker=price_ticker,
        focused_overlay_applied=focused_overlay_applied,
        exam=exam,
    )
    coverage = evidence_pack.get("coverage", full_run.get("evidence_pack", {}).get("coverage", {}))
    direct_document_count = len(direct_docs)
    direct_ratio = direct_document_count / len(documents) if documents else 0.0
    issues = _issues(
        price_ticker=price_ticker,
        direct_document_count=direct_document_count,
        selected_note_tickers=selected_note_tickers,
        min_direct_documents=min_direct_documents,
        max_selected_note_tickers=max_selected_note_tickers,
        exam=exam,
    )
    status = _attribution_status(issues)
    return {
        "run_id": batch_run.get("run_id") or full_run.get("run_id"),
        "as_of": batch_run.get("as_of") or full_run.get("inputs", {}).get("as_of"),
        "horizon_days": batch_run.get("horizon_days"),
        "price_ticker": price_ticker,
        "research_stance": batch_run.get("research_stance") or exam.get("research_stance"),
        "research_expected_direction": batch_run.get("research_expected_direction") or exam.get("research_expected_direction"),
        "research_price_agreement": batch_run.get("research_price_agreement") or exam.get("research_price_agreement"),
        "exam_verdict": batch_run.get("exam_verdict") or exam.get("exam_verdict"),
        "outcome_label": batch_run.get("outcome_label"),
        "realized_return": batch_run.get("realized_return"),
        "selected_note_agent": exam.get("selected_note_agent"),
        "selected_note_source": "focused_overlay" if focused_overlay_applied else "agent_lab",
        "focused_overlay_applied": focused_overlay_applied,
        "focused_overlay_status": exam.get("focused_overlay_status") or focused_overlay.get("overlay_status"),
        "selected_note_tickers": selected_note_tickers,
        "selected_note_ticker_count": len(selected_note_tickers),
        "selected_note_patterns": selected_note.get("patterns", []),
        "selected_note_tailwinds": selected_note.get("tailwinds", []),
        "selected_note_headwinds": selected_note.get("headwinds", []),
        "selected_note_citation_count": _selected_note_citation_count(selected_note, focused_overlay, focused_overlay_applied),
        "evidence_document_count": len(documents) or batch_run.get("evidence_document_count"),
        "direct_document_count": direct_document_count,
        "direct_document_ratio": round(direct_ratio, 6),
        "direct_document_titles": [str(doc.get("title", ""))[:180] for doc in direct_docs[:5]],
        "coverage_by_ticker": coverage.get("by_ticker", {}),
        "coverage_missing_requested_tickers": coverage.get("missing_requested_tickers", []),
        "attribution_status": status,
        "issues": issues,
    }


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


def _direct_documents(documents: list[dict[str, Any]], price_ticker: str) -> list[dict[str, Any]]:
    if not price_ticker:
        return []
    return [doc for doc in documents if price_ticker in _upper_list(doc.get("tickers", []))]


def _selected_note_tickers(
    selected_note: dict[str, Any],
    price_ticker: str,
    focused_overlay_applied: bool,
    exam: dict[str, Any],
) -> list[str]:
    if focused_overlay_applied and price_ticker and exam.get("ticker_specificity") == "single_ticker":
        return [price_ticker]
    if focused_overlay_applied and price_ticker and exam.get("focused_overlay_status") != "focused_overlay_ready":
        return [price_ticker]
    return _upper_list(selected_note.get("tickers", []))


def _selected_note_citation_count(
    selected_note: dict[str, Any],
    focused_overlay: dict[str, Any],
    focused_overlay_applied: bool,
) -> int:
    if focused_overlay_applied:
        return int(focused_overlay.get("focused_note", {}).get("citation_count") or 0)
    return int(selected_note.get("citation_count", 0) or 0)


def _issues(
    price_ticker: str,
    direct_document_count: int,
    selected_note_tickers: list[str],
    min_direct_documents: int,
    max_selected_note_tickers: int,
    exam: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    if not price_ticker:
        issues.append("missing_price_ticker")
    if direct_document_count == 0:
        issues.append("no_direct_price_ticker_documents")
    elif direct_document_count < min_direct_documents:
        issues.append("weak_direct_price_ticker_documents")
    if price_ticker and selected_note_tickers and price_ticker not in selected_note_tickers:
        issues.append("selected_note_missing_price_ticker")
    if len(selected_note_tickers) > max_selected_note_tickers:
        issues.append("selected_note_is_basket_or_sector")
    if exam.get("ticker_specificity") == "basket_or_sector":
        issues.append("exam_ticker_specificity_basket")
    if exam.get("research_expected_direction") in {"bullish", "bearish"} and len(selected_note_tickers) > max_selected_note_tickers:
        issues.append("directional_thesis_not_ticker_specific")
    return issues


def _attribution_status(issues: list[str]) -> str:
    hard = {"missing_price_ticker", "no_direct_price_ticker_documents", "selected_note_missing_price_ticker"}
    if hard.intersection(issues):
        return "blocked_missing_direct_attribution"
    if "weak_direct_price_ticker_documents" in issues:
        return "blocked_weak_direct_evidence"
    if "directional_thesis_not_ticker_specific" in issues or "selected_note_is_basket_or_sector" in issues:
        return "needs_ticker_specific_attribution"
    return "ticker_specific_ready"


def _summary(research_batch: dict[str, Any], run_audits: list[dict[str, Any]]) -> dict[str, Any]:
    run_count = len(run_audits)
    status_counts = _counts(run.get("attribution_status") for run in run_audits)
    ready_count = status_counts.get("ticker_specific_ready", 0)
    weak_count = sum(1 for run in run_audits if "weak_direct_price_ticker_documents" in run.get("issues", []))
    basket_count = sum(1 for run in run_audits if "selected_note_is_basket_or_sector" in run.get("issues", []))
    if run_count == 0:
        status = "no_runs"
        next_action = "provide_selected_research_replay_batch"
    elif ready_count == run_count:
        status = "ticker_attribution_ready"
        next_action = "rerun_replay_readiness_gate"
    elif weak_count:
        status = "blocked_weak_ticker_evidence"
        next_action = "backfill_direct_ticker_evidence"
    else:
        status = "blocked_basket_attribution"
        next_action = "improve_ticker_specific_note_selection"
    return {
        "attribution_status": status,
        "next_action": next_action,
        "run_count": run_count,
        "ticker_ready_run_count": ready_count,
        "basket_note_run_count": basket_count,
        "weak_direct_evidence_run_count": weak_count,
        "status_counts": status_counts,
        "price_ticker_counts": _counts(run.get("price_ticker") for run in run_audits),
        "research_batch_summary": research_batch.get("summary", {}),
        "can_change_analyst_weights": False,
        "can_write_learning_memory": False,
    }


def _issue_counts(run_audits: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for run in run_audits:
        counts.update(run.get("issues", []))
    return dict(sorted(counts.items()))


def _tasks(summary: dict[str, Any], run_audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    weak_runs = [run for run in run_audits if "weak_direct_price_ticker_documents" in run.get("issues", [])]
    basket_runs = [run for run in run_audits if "selected_note_is_basket_or_sector" in run.get("issues", [])]
    if basket_runs:
        tasks.append(
            {
                "priority": "P0",
                "task_id": "improve_ticker_specific_note_selection",
                "description": "Create or select a ticker-focused research note before treating a directional thesis as ticker-specific.",
                "affected_runs": [run.get("as_of") for run in basket_runs],
                "affected_price_tickers": sorted({run.get("price_ticker") for run in basket_runs if run.get("price_ticker")}),
            }
        )
    if weak_runs:
        tasks.append(
            {
                "priority": "P1",
                "task_id": "backfill_direct_price_ticker_documents",
                "description": "Add enough dated direct documents for the selected price ticker before using early windows for calibration.",
                "affected_runs": [run.get("as_of") for run in weak_runs],
                "affected_price_tickers": sorted({run.get("price_ticker") for run in weak_runs if run.get("price_ticker")}),
            }
        )
    if summary.get("ticker_ready_run_count", 0) < summary.get("run_count", 0):
        tasks.append(
            {
                "priority": "P2",
                "task_id": "rerun_selected_replay_after_attribution_fix",
                "description": "Rerun selected-window research replay, directionality diagnostic, readiness, and backfill after attribution fixes.",
            }
        )
    return tasks


def _commands(research_batch: dict[str, Any], run_audits: list[dict[str, Any]]) -> dict[str, str | None]:
    ready_dates = [run["as_of"] for run in run_audits if run.get("attribution_status") == "ticker_specific_ready"]
    inputs = research_batch.get("inputs", {})
    if not ready_dates or not inputs.get("price_data_path"):
        return {"ticker_ready_replay_batch": None}
    command = (
        f"python run_agent_historical_research_replay_batch.py {inputs['price_data_path']} "
        f"--tickers {' '.join(inputs.get('tickers', []))} "
        f"--as-of {' '.join(ready_dates)} "
        f"--lookback-days {inputs.get('lookback_days', 180)} "
        f"--horizon-days {' '.join(str(item) for item in inputs.get('horizon_days', []))}"
    )
    if inputs.get("news_data_paths"):
        command += f" --news-data {' '.join(inputs.get('news_data_paths', []))}"
    if inputs.get("macro_data_paths"):
        command += f" --macro-data {' '.join(inputs.get('macro_data_paths', []))}"
    tags = list(dict.fromkeys([*inputs.get("tags", []), "ticker_attribution_ready"]))
    if tags:
        command += f" --tags {' '.join(tags)}"
    command += " --output-dir reports\\dean_os\\historical_research_replay_batch_ticker_attribution_ready"
    return {"ticker_ready_replay_batch": command}


def _recommendations(summary: dict[str, Any]) -> list[str]:
    status = summary.get("attribution_status")
    if status == "ticker_attribution_ready":
        return ["Ticker-specific attribution is ready for a readiness rerun, but calibration still needs human review."]
    if status == "blocked_weak_ticker_evidence":
        return [
            "Backfill direct documents for the selected price ticker before using early windows for analyst calibration.",
            "Do not treat a sector/basket thesis as a ticker-specific signal until attribution improves.",
        ]
    if status == "blocked_basket_attribution":
        return ["Improve ticker-specific note selection before changing analyst weights."]
    return ["Provide selected-window research replay outputs before auditing attribution."]


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _upper_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _counts(values: Any) -> dict[str, int]:
    counts: Counter[str] = Counter(str(value) for value in values if value)
    return dict(sorted(counts.items()))


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
