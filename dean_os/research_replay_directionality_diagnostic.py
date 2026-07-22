from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_RESEARCH_BATCH = "reports/dean_os/historical_research_replay_batch_evidence_window_selected/latest.json"

DIRECTIONAL_VALUES = {"bullish", "bearish", "undervalued", "overvalued"}
THIN_EVIDENCE_VALUES = {"weak", "partial", "missing", "none", ""}


class ResearchReplayDirectionalityDiagnostic:
    """Diagnoses why evidence-backed research replay remains neutral or mixed."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/research_replay_directionality_diagnostic"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        research_batch_path: str | Path = DEFAULT_RESEARCH_BATCH,
        readiness_report_path: str | Path | None = None,
        backfill_plan_path: str | Path | None = None,
        min_directional_ratio: float = 0.25,
        min_strong_documents: int = 20,
        save: bool = True,
    ) -> dict[str, Any]:
        research_batch = _load_json(research_batch_path)
        readiness = _load_optional_json(readiness_report_path)
        backfill = _load_optional_json(backfill_plan_path)
        runs = [_run_diagnostic(run, min_strong_documents=min_strong_documents) for run in research_batch.get("runs", [])]
        summary = _summary(
            research_batch=research_batch,
            runs=runs,
            min_directional_ratio=min_directional_ratio,
            readiness=readiness,
            backfill=backfill,
        )
        payload = {
            "run_id": _run_id("research_replay_directionality_diagnostic"),
            "created_at": utc_now_iso(),
            "mode": "research_replay_directionality_diagnostic",
            "inputs": {
                "research_batch_path": str(research_batch_path),
                "readiness_report_path": str(readiness_report_path) if readiness_report_path else None,
                "backfill_plan_path": str(backfill_plan_path) if backfill_plan_path else None,
                "min_directional_ratio": min_directional_ratio,
                "min_strong_documents": min_strong_documents,
            },
            "summary": summary,
            "batch_context": _batch_context(research_batch, readiness, backfill),
            "run_diagnostics": runs,
            "issue_counts": _issue_counts(runs),
            "diagnostic_tasks": _tasks(summary, runs),
            "commands": _commands(research_batch, runs),
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
        rendered_md = render_research_replay_directionality_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_research_replay_directionality_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Research Replay Directionality Diagnostic",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('diagnostic_status')}`",
        f"- Runs: {summary.get('run_count')}",
        f"- Directional runs: {summary.get('directional_run_count')}",
        f"- Inconclusive runs: {summary.get('inconclusive_run_count')}",
        f"- Strong inconclusive runs: {summary.get('strong_inconclusive_run_count')}",
        f"- Missing tickers: `{summary.get('missing_tickers')}`",
        "",
        "## Top Issues",
        "",
    ]
    issue_counts = payload.get("issue_counts", {})
    lines.extend(f"- `{key}`: {value}" for key, value in issue_counts.items()) if issue_counts else lines.append("- None.")
    lines.extend(["", "## Tasks", ""])
    tasks = payload.get("diagnostic_tasks", [])
    lines.extend(f"- `{item.get('priority')}` {item.get('task_id')}: {item.get('description')}" for item in tasks) if tasks else lines.append("- None.")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _run_diagnostic(run: dict[str, Any], min_strong_documents: int) -> dict[str, Any]:
    expected = _lower(run.get("research_expected_direction"))
    stance = _lower(run.get("research_stance"))
    evidence_quality = _lower(run.get("evidence_data_quality"))
    ticker_specificity = _lower(run.get("ticker_specificity"))
    price_ticker = str(run.get("price_ticker") or "").upper()
    evidence_tickers = _upper_list(run.get("evidence_tickers", []))
    missing_tickers = _upper_list(run.get("evidence_missing_tickers", []))
    document_count = int(run.get("evidence_document_count") or 0)
    directional = expected in DIRECTIONAL_VALUES or stance in DIRECTIONAL_VALUES
    inconclusive = (not directional) or _lower(run.get("research_price_agreement")) == "research_inconclusive"
    issues: list[str] = []
    if evidence_quality in THIN_EVIDENCE_VALUES or document_count < min_strong_documents:
        issues.append("thin_or_partial_evidence")
    if missing_tickers:
        issues.append("missing_ticker_evidence")
    if ticker_specificity != "ticker_specific":
        issues.append("basket_or_sector_specificity")
    if price_ticker and price_ticker not in evidence_tickers:
        issues.append("price_ticker_not_in_evidence")
    if expected == "neutral" and _lower(run.get("price_expected_direction")) in DIRECTIONAL_VALUES:
        issues.append("neutral_research_vs_directional_price_candidate")
    if evidence_quality == "strong" and inconclusive:
        issues.append("strong_evidence_still_inconclusive")
    if (run.get("research_confidence") or 0) and float(run.get("research_confidence") or 0) >= 0.75 and expected == "neutral":
        issues.append("high_confidence_neutral")
    return {
        "run_id": run.get("run_id"),
        "as_of": run.get("as_of"),
        "horizon_days": run.get("horizon_days"),
        "research_stance": run.get("research_stance"),
        "research_expected_direction": run.get("research_expected_direction"),
        "research_confidence": run.get("research_confidence"),
        "research_price_agreement": run.get("research_price_agreement"),
        "exam_verdict": run.get("exam_verdict"),
        "evidence_document_count": document_count,
        "evidence_data_quality": run.get("evidence_data_quality"),
        "evidence_tickers": evidence_tickers,
        "evidence_missing_tickers": missing_tickers,
        "ticker_specificity": run.get("ticker_specificity"),
        "price_action": run.get("price_action"),
        "price_ticker": price_ticker,
        "price_expected_direction": run.get("price_expected_direction"),
        "outcome_label": run.get("outcome_label"),
        "realized_return": run.get("realized_return"),
        "directional": directional,
        "inconclusive": inconclusive,
        "issues": issues,
        "primary_diagnosis": _primary_diagnosis(issues),
    }


def _summary(
    research_batch: dict[str, Any],
    runs: list[dict[str, Any]],
    min_directional_ratio: float,
    readiness: dict[str, Any],
    backfill: dict[str, Any],
) -> dict[str, Any]:
    run_count = len(runs)
    directional_count = sum(1 for run in runs if run["directional"])
    inconclusive_count = sum(1 for run in runs if run["inconclusive"])
    weak_count = sum(1 for run in runs if "thin_or_partial_evidence" in run["issues"])
    strong_inconclusive_count = sum(1 for run in runs if "strong_evidence_still_inconclusive" in run["issues"])
    missing_tickers = sorted({ticker for run in runs for ticker in run["evidence_missing_tickers"]})
    directional_ratio = directional_count / run_count if run_count else 0.0
    if not runs:
        status = "no_runs"
        next_action = "provide_research_replay_batch"
    elif directional_ratio >= min_directional_ratio and strong_inconclusive_count == 0 and weak_count == 0:
        status = "directionality_ready"
        next_action = "rerun_readiness_gate"
    elif strong_inconclusive_count > 0:
        status = "diagnose_base_analyst_rules"
        next_action = "inspect_neutral_fallback_and_ticker_attribution"
    elif weak_count > 0 or missing_tickers:
        status = "evidence_and_directionality_blocked"
        next_action = "backfill_evidence_then_rerun_selected_windows"
    else:
        status = "directionality_blocked"
        next_action = "inspect_research_directionality_rules"
    return {
        "diagnostic_status": status,
        "next_action": next_action,
        "run_count": run_count,
        "directional_run_count": directional_count,
        "directional_ratio": round(directional_ratio, 6),
        "inconclusive_run_count": inconclusive_count,
        "weak_or_partial_run_count": weak_count,
        "strong_inconclusive_run_count": strong_inconclusive_count,
        "missing_tickers": missing_tickers,
        "research_stance_counts": _counts(run["research_stance"] for run in runs),
        "expected_direction_counts": _counts(run["research_expected_direction"] for run in runs),
        "price_ticker_counts": _counts(run["price_ticker"] for run in runs),
        "outcome_counts": research_batch.get("summary", {}).get("outcome_counts", {}),
        "readiness_status": readiness.get("summary", {}).get("readiness_status"),
        "backfill_status": backfill.get("summary", {}).get("backfill_status"),
        "can_change_analyst_weights": False,
        "can_write_learning_memory": False,
    }


def _batch_context(research_batch: dict[str, Any], readiness: dict[str, Any], backfill: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_batch_summary": research_batch.get("summary", {}),
        "research_batch_inputs": research_batch.get("inputs", {}),
        "readiness_summary": readiness.get("summary", {}),
        "backfill_summary": backfill.get("summary", {}),
    }


def _issue_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for run in runs:
        counts.update(run.get("issues", []))
    return dict(sorted(counts.items()))


def _tasks(summary: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    issue_counts = _issue_counts(runs)
    if issue_counts.get("strong_evidence_still_inconclusive", 0) > 0 or issue_counts.get("high_confidence_neutral", 0) > 0:
        tasks.append(
            {
                "priority": "P0",
                "task_id": "inspect_base_analyst_directionality_rules",
                "description": "Review why strong/high-confidence evidence-backed runs still produce neutral expected direction.",
                "affected_runs": [run["as_of"] for run in runs if "strong_evidence_still_inconclusive" in run["issues"]],
            }
        )
    if issue_counts.get("basket_or_sector_specificity", 0) > 0 or issue_counts.get("price_ticker_not_in_evidence", 0) > 0:
        tasks.append(
            {
                "priority": "P0",
                "task_id": "improve_ticker_specific_attribution",
                "description": "Separate basket/sector evidence from ticker-specific catalysts before judging analyst skill.",
                "affected_price_tickers": sorted({run["price_ticker"] for run in runs if run["price_ticker"]}),
            }
        )
    if summary.get("missing_tickers"):
        tasks.append(
            {
                "priority": "P1",
                "task_id": "backfill_missing_ticker_evidence",
                "description": "Add dated evidence for missing tickers before rerunning early selected windows.",
                "missing_tickers": summary["missing_tickers"],
            }
        )
    if summary.get("directional_run_count", 0) == 0:
        tasks.append(
            {
                "priority": "P1",
                "task_id": "add_directional_catalyst_materials",
                "description": "Provide filings, transcripts, sector reports, or catalyst notes that can support a directional thesis.",
            }
        )
    tasks.append(
        {
            "priority": "P2",
            "task_id": "rerun_selected_research_replay_after_fix",
            "description": "Rerun the selected-window historical research replay, readiness gate, and backfill plan after diagnostics are addressed.",
        }
    )
    return tasks


def _commands(research_batch: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, str | None]:
    inputs = research_batch.get("inputs", {})
    price_path = inputs.get("price_data_path")
    tickers = inputs.get("tickers", [])
    horizon_days = inputs.get("horizon_days", [])
    lookback_days = inputs.get("lookback_days", 180)
    strong_dates = [
        run["as_of"]
        for run in runs
        if run.get("evidence_data_quality") == "strong" and not run.get("evidence_missing_tickers")
    ]
    if not price_path or not tickers or not horizon_days or not strong_dates:
        return {"strong_evidence_replay_batch": None}
    news_paths = inputs.get("news_data_paths", [])
    macro_paths = inputs.get("macro_data_paths", [])
    materials_paths = inputs.get("materials_paths", [])
    tags = list(dict.fromkeys([*inputs.get("tags", []), "directionality_diagnostic"]))
    command = (
        f"python run_agent_historical_research_replay_batch.py {price_path} "
        f"--tickers {' '.join(tickers)} "
        f"--as-of {' '.join(str(date) for date in strong_dates)} "
        f"--lookback-days {lookback_days} "
        f"--horizon-days {' '.join(str(item) for item in horizon_days)}"
    )
    if news_paths:
        command += f" --news-data {' '.join(news_paths)}"
    if macro_paths:
        command += f" --macro-data {' '.join(macro_paths)}"
    if materials_paths:
        command += f" --materials {' '.join(materials_paths)}"
    if tags:
        command += f" --tags {' '.join(tags)}"
    command += " --output-dir reports\\dean_os\\historical_research_replay_batch_strong_evidence_directionality_check"
    return {"strong_evidence_replay_batch": command}


def _recommendations(summary: dict[str, Any]) -> list[str]:
    status = summary.get("diagnostic_status")
    if status == "directionality_ready":
        return ["Directionality is present enough for a readiness rerun, but learning and analyst weights remain blocked until review."]
    if status == "diagnose_base_analyst_rules":
        return [
            "Do not add more specialist profiles yet; first inspect why strong evidence still produces neutral output.",
            "Focus on ticker-specific attribution and neutral fallback thresholds before calibration.",
        ]
    if status == "evidence_and_directionality_blocked":
        return [
            "Backfill missing ticker evidence, then rerun only selected evidence-backed windows.",
            "If strong-evidence windows still stay neutral after backfill, inspect base analyst directionality rules.",
        ]
    if status == "no_runs":
        return ["Provide a historical research replay batch before diagnosing directionality."]
    return ["Treat current research replay as diagnostic only; do not change analyst weights or learning memory."]


def _primary_diagnosis(issues: list[str]) -> str:
    for key in [
        "strong_evidence_still_inconclusive",
        "missing_ticker_evidence",
        "price_ticker_not_in_evidence",
        "neutral_research_vs_directional_price_candidate",
        "basket_or_sector_specificity",
        "thin_or_partial_evidence",
    ]:
        if key in issues:
            return key
    return "no_major_issue"


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    if not path:
        return {}
    try:
        return DeanPaths.load_json(path)
    except Exception:
        return {"missing": True, "path": str(path)}


def _counts(values: Any) -> dict[str, int]:
    counts: Counter[str] = Counter(str(value) for value in values if value is not None)
    return dict(sorted(counts.items()))


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _upper_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
