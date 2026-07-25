from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from dean_os.agent_lab import AgentLabRunner
from dean_os.analyst_core.analyst_evidence_pack import AnalystEvidencePackRunner, documents_from_evidence_pack
from dean_os.dean_paths import DeanPaths
from dean_os.historical_replay import HistoricalReplayRunner as PriceReplayRunner
from dean_os.market_data_api import parse_datetime
from dean_os.regime_context import normalize_context_tags
from dean_os.schemas import AgentLabRunReport, ResearchNote, utc_now_iso
from dean_os.utils import json_ready

BULLISH_RESEARCH_PATTERNS = {
    "ai_compute_cycle",
    "defense_rearmament",
    "energy_security",
    "policy_easing",
    "pricing_power",
    "supply_chain_reshoring",
    "value_margin_safety",
}

RISK_RESEARCH_PATTERNS = {
    "balance_sheet_stress",
    "capacity_pressure",
    "regulatory_risk",
}


class HistoricalReplayRunner:
    """Runs an old-data research exam without learning writes or pipeline execution.

    This wraps three independent, already-live subsystems (AnalystEvidencePackRunner,
    AgentLabRunner, and the price-only PriceReplayRunner from historical_replay.py)
    into a single "research exam" comparing a research thesis against a realized
    price outcome, plus an optional ticker-focused overlay.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/historical_research_replay"):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        price_data_path: str | Path,
        tickers: list[str],
        as_of: str,
        lookback_days: int = 180,
        horizon_days: int = 60,
        news_data_paths: list[str | Path] | None = None,
        macro_data_paths: list[str | Path] | None = None,
        materials_paths: list[str | Path] | None = None,
        tags: list[str] | None = None,
        benchmark_ticker: str = "SPY",
        close_col: str = "close",
        datetime_col: str = "datetime",
        neutral_band: float = 0.01,
        max_rows_per_table: int = 300,
        max_documents: int = 600,
        normalize_daily_bars: bool = False,
        focused_overlay_path: str | Path | None = None,
        apply_focused_overlay: bool = False,
    ) -> dict[str, Any]:
        as_of_dt = parse_datetime(as_of)
        lookback_start = as_of_dt - timedelta(days=lookback_days)
        normalized_tickers = [ticker.upper() for ticker in tickers if str(ticker).strip()]
        normalized_tags = normalize_context_tags(tags or [])
        news_paths = list(news_data_paths or [])
        macro_paths = list(macro_data_paths or [])
        run_id = "historical_research_replay_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_")
        run_dir = self.output_dir / "runs" / run_id

        evidence_pack = AnalystEvidencePackRunner(output_dir=run_dir / "evidence_pack").run(
            materials_paths=materials_paths or [],
            news_data_paths=news_paths,
            macro_data_paths=macro_paths,
            tickers=normalized_tickers,
            tags=[*normalized_tags, "historical_research_replay"],
            start_at=lookback_start.isoformat(),
            as_of=as_of_dt.isoformat(),
            max_rows_per_table=max_rows_per_table,
            max_documents=max_documents,
        )
        evidence_json = evidence_pack.get("saved_paths", {}).get("latest_json") or evidence_pack.get("saved_paths", {}).get("json")
        documents = documents_from_evidence_pack(evidence_json) if evidence_json else []
        for document in documents:
            document.metadata["point_in_time_replay"] = {
                "as_of": as_of_dt.isoformat(),
                "evidence_pack_json": str(evidence_json),
            }

        lab_dir = run_dir / "agent_lab"
        lab_report = await AgentLabRunner(
            corpus_path=lab_dir / "research_corpus.sqlite",
            learning_path=lab_dir / "learning.sqlite",
            output_dir=lab_dir,
            memory_path=lab_dir / "memory.sqlite",
            log_path=lab_dir / "events.jsonl",
        ).run(
            documents=documents,
            tickers=normalized_tickers,
            tags=[*normalized_tags, "historical_research_replay"],
            as_of=as_of_dt.isoformat(),
            create_learning_records=False,
            include_operations_proposals=False,
        )

        price_replay = await PriceReplayRunner(output_dir=run_dir / "price_replay").run(
            price_data_path=price_data_path,
            tickers=normalized_tickers,
            as_of=as_of_dt.isoformat(),
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            news_data_path=_first_path(news_paths),
            macro_data_path=_first_path(macro_paths),
            benchmark_ticker=benchmark_ticker,
            close_col=close_col,
            datetime_col=datetime_col,
            neutral_band=neutral_band,
            normalize_daily_bars=normalize_daily_bars,
        )
        original_research_exam = _research_exam(lab_report, price_replay)
        compact_price = _compact_price_replay(price_replay)
        focused_overlay = _focused_overlay_for_run(
            focused_overlay_path=focused_overlay_path,
            as_of=as_of_dt.isoformat(),
            price_ticker=str(compact_price.get("decision", {}).get("ticker") or ""),
            horizon_days=horizon_days,
        )
        research_exam = _research_exam_with_optional_focused_overlay(
            original_exam=original_research_exam,
            focused_overlay=focused_overlay,
            apply_focused_overlay=apply_focused_overlay,
        )

        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "historical_research_replay",
            "inputs": {
                "price_data_path": str(price_data_path),
                "news_data_paths": [str(path) for path in news_paths],
                "macro_data_paths": [str(path) for path in macro_paths],
                "materials_paths": [str(path) for path in materials_paths or []],
                "tickers": normalized_tickers,
                "as_of": as_of_dt.isoformat(),
                "lookback_start": lookback_start.isoformat(),
                "lookback_days": lookback_days,
                "horizon_days": horizon_days,
                "benchmark_ticker": benchmark_ticker.upper(),
                "max_rows_per_table": max_rows_per_table,
                "max_documents": max_documents,
                "normalize_daily_bars": normalize_daily_bars,
                "focused_overlay_path": str(focused_overlay_path) if focused_overlay_path else None,
                "apply_focused_overlay": apply_focused_overlay,
                "tags": normalized_tags,
            },
            "safety": {
                "future_cutoff_enforced": True,
                "learning_records_created": 0,
                "operation_proposals_created": 0,
                "pipeline_started": False,
                "broker_connected": False,
                "production_memory_written": False,
                "focused_overlay_applied": bool(research_exam.get("focused_overlay_applied")),
            },
            "evidence_pack": _compact_evidence_pack(evidence_pack),
            "agent_lab": _compact_agent_lab(lab_report),
            "price_replay": compact_price,
            "research_exam": research_exam,
            "research_exam_original": original_research_exam if focused_overlay_path else None,
            "focused_research_exam_overlay": _compact_focused_overlay(focused_overlay),
            "recommendations": _recommendations(evidence_pack, lab_report, price_replay),
        }
        self.save_report(payload)
        return payload

    def save_report(self, payload: dict[str, Any]) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = payload["run_id"]
        json_path = self.output_dir / f"{run_id}.json"
        md_path = self.output_dir / f"{run_id}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        paths = {"json": json_path, "markdown": md_path, "latest_json": latest_json, "latest_markdown": latest_md}
        payload["saved_paths"] = {key: str(value) for key, value in paths.items()}
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False)
        rendered_md = render_historical_research_replay_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return paths


# Alias kept for dean_os/__init__.py's lazy export (which was already registered
# under this name) and to match the "research exam" naming used elsewhere.
HistoricalResearchReplayRunner = HistoricalReplayRunner


def render_historical_research_replay_markdown(payload: dict[str, Any]) -> str:
    exam = payload.get("research_exam", {})
    evidence = payload.get("evidence_pack", {})
    lab = payload.get("agent_lab", {})
    price = payload.get("price_replay", {})
    decision = price.get("decision", {})
    evaluation = price.get("evaluation", {})
    lines = [
        "# DEAN-OS Historical Research Replay",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- As of: `{payload.get('inputs', {}).get('as_of')}`",
        f"- Lookback days: {payload.get('inputs', {}).get('lookback_days')}",
        f"- Horizon days: {payload.get('inputs', {}).get('horizon_days')}",
        f"- Exam verdict: `{exam.get('exam_verdict')}`",
        f"- Learning gate: `{exam.get('learning_gate', {}).get('status')}`",
        "",
        "## Research View",
        "",
        f"- Stance: `{exam.get('research_stance')}`",
        f"- Expected direction: `{exam.get('research_expected_direction')}`",
        f"- Ticker specificity: `{exam.get('ticker_specificity')}`",
        f"- Thesis: {exam.get('research_thesis')}",
        f"- Evidence documents: {evidence.get('coverage', {}).get('document_count')}",
        f"- Agent Lab notes: {lab.get('note_count')}",
        "",
        "## Price Replay",
        "",
        f"- Decision: `{decision.get('action')}`",
        f"- Ticker: `{decision.get('ticker')}`",
        f"- Price expected direction: `{decision.get('expected_direction')}`",
        f"- Evaluation: `{evaluation.get('status')}` / `{evaluation.get('outcome_label')}`",
        f"- Realized return: {evaluation.get('realized_return')}",
        f"- Agreement: `{exam.get('research_price_agreement')}`",
        "",
        "## Quality",
        "",
    ]
    overlay = payload.get("focused_research_exam_overlay")
    if overlay:
        lines.extend(
            [
                f"- Focused overlay status: `{overlay.get('overlay_status')}`",
                f"- Focused overlay applied: `{exam.get('focused_overlay_applied', False)}`",
            ]
        )
    warnings = price.get("quality_warnings", [])
    if warnings:
        lines.extend(f"- Warning: {warning}" for warning in warnings)
    else:
        lines.append("- Price quality warnings: none")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    lines.extend(["", "## Research Notes", ""])
    for note in lab.get("research_notes", []):
        patterns = ", ".join(note.get("patterns", [])[:8]) or "none"
        lines.append(
            f"- `{note.get('agent_name')}` {note.get('data_quality')} "
            f"conf={note.get('confidence')}: {note.get('thesis')} Patterns: {patterns}"
        )
    return "\n".join(lines).strip() + "\n"


def _compact_evidence_pack(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": payload.get("run_id"),
        "coverage": payload.get("coverage", {}),
        "analyst_inputs": payload.get("analyst_inputs", {}),
        "warnings": payload.get("warnings", []),
        "dropped": payload.get("dropped", []),
        "saved_paths": payload.get("saved_paths", {}),
    }


def _compact_agent_lab(report: AgentLabRunReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "corpus_path": report.corpus_path,
        "document_count": report.document_count,
        "chunk_count": report.chunk_count,
        "note_count": report.note_count,
        "learning_record_count": len(report.learning_records),
        "action_proposal_count": len(report.action_proposals),
        "summary": report.summary,
        "research_notes": [_compact_note(note) for note in report.research_notes],
    }


def _compact_note(note: ResearchNote) -> dict[str, Any]:
    return {
        "agent_name": note.agent_name,
        "topic": note.topic,
        "thesis": note.thesis,
        "patterns": note.patterns,
        "catalysts": note.catalysts,
        "tailwinds": note.tailwinds,
        "headwinds": note.headwinds,
        "tickers": note.tickers,
        "sectors": note.sectors,
        "confidence": note.confidence,
        "data_quality": note.data_quality,
        "risks": note.risks,
        "blind_spots": note.blind_spots,
        "citation_count": len(note.citations),
    }


def _compact_price_replay(payload: dict[str, Any]) -> dict[str, Any]:
    replay = payload.get("historical_replay", {})
    price_quality = replay.get("coverage", {}).get("price_quality", {})
    return {
        "run_id": payload.get("run_id"),
        "decision": payload.get("decision", {}),
        "evaluation": payload.get("evaluation", {}),
        "coverage": replay.get("coverage", {}),
        "rankings": replay.get("rankings", [])[:10],
        "quality_warnings": list(price_quality.get("warnings", [])),
        "saved_paths": payload.get("saved_paths", {}),
    }


def _research_exam(lab_report: AgentLabRunReport, price_replay: dict[str, Any]) -> dict[str, Any]:
    note = _select_research_note(lab_report)
    stance = _research_stance(note)
    research_direction = _research_direction(stance)
    price_direction = str(price_replay.get("decision", {}).get("expected_direction") or "neutral")
    warnings = _compact_price_replay(price_replay).get("quality_warnings", [])
    evaluation = price_replay.get("evaluation", {})
    agreement = _agreement(research_direction, price_direction)
    exam_verdict = _exam_verdict(
        research_direction=research_direction,
        price_direction=price_direction,
        agreement=agreement,
        outcome_label=evaluation.get("outcome_label"),
        quality_warnings=warnings,
    )
    return {
        "selected_note_agent": note.agent_name if note else None,
        "research_thesis": note.thesis if note else "No research thesis was produced.",
        "research_stance": stance,
        "research_expected_direction": research_direction,
        "research_confidence": note.confidence if note else 0.0,
        "research_data_quality": note.data_quality if note else "weak",
        "ticker_specificity": _ticker_specificity(note),
        "price_expected_direction": price_direction,
        "research_price_agreement": agreement,
        "exam_verdict": exam_verdict,
        "learning_gate": _learning_gate(evaluation, warnings),
    }


def _focused_overlay_for_run(
    focused_overlay_path: str | Path | None,
    as_of: str,
    price_ticker: str,
    horizon_days: int,
) -> dict[str, Any] | None:
    if not focused_overlay_path:
        return None
    try:
        payload = DeanPaths.load_json(focused_overlay_path)
    except Exception:
        return {
            "overlay_status": "blocked_missing_focused_overlay",
            "issues": ["focused_overlay_file_missing"],
            "focused_exam": _focused_overlay_blocked_exam("focused_overlay_file_missing"),
        }
    normalized_ticker = price_ticker.upper()
    for overlay in payload.get("run_overlays", []):
        if str(overlay.get("as_of")) != as_of:
            continue
        if str(overlay.get("price_ticker") or "").upper() != normalized_ticker:
            continue
        if int(overlay.get("horizon_days") or horizon_days) != int(horizon_days):
            continue
        return overlay
    return {
        "overlay_status": "blocked_missing_focused_overlay",
        "issues": ["focused_overlay_run_missing"],
        "focused_exam": _focused_overlay_blocked_exam("focused_overlay_run_missing"),
    }


def _focused_overlay_blocked_exam(reason: str) -> dict[str, Any]:
    return {
        "selected_note_agent": None,
        "selected_note_id": None,
        "research_thesis": "Focused replay overlay is missing for this run.",
        "research_stance": "insufficient_data",
        "research_expected_direction": "neutral",
        "research_confidence": 0.0,
        "research_data_quality": "weak",
        "ticker_specificity": "none",
        "price_expected_direction": "neutral",
        "research_price_agreement": "research_inconclusive",
        "exam_verdict": "focused_note_blocked",
        "learning_gate": {
            "status": "blocked_focused_note",
            "can_write_learning_memory": False,
            "reason": reason,
        },
    }


def _research_exam_with_optional_focused_overlay(
    original_exam: dict[str, Any],
    focused_overlay: dict[str, Any] | None,
    apply_focused_overlay: bool,
) -> dict[str, Any]:
    if not focused_overlay:
        return original_exam
    if not apply_focused_overlay:
        return {
            **original_exam,
            "focused_overlay_available": True,
            "focused_overlay_applied": False,
            "focused_overlay_status": focused_overlay.get("overlay_status"),
        }
    focused_exam = dict(focused_overlay.get("focused_exam") or _focused_overlay_blocked_exam("focused_exam_missing"))
    focused_exam.update(
        {
            "focused_overlay_available": True,
            "focused_overlay_applied": True,
            "focused_overlay_status": focused_overlay.get("overlay_status"),
            "original_exam_verdict": original_exam.get("exam_verdict"),
            "original_research_stance": original_exam.get("research_stance"),
            "original_ticker_specificity": original_exam.get("ticker_specificity"),
        }
    )
    return focused_exam


def _compact_focused_overlay(focused_overlay: dict[str, Any] | None) -> dict[str, Any] | None:
    if not focused_overlay:
        return None
    focused_exam = focused_overlay.get("focused_exam", {})
    return {
        "overlay_status": focused_overlay.get("overlay_status"),
        "issues": focused_overlay.get("issues", []),
        "focused_exam": {
            "research_stance": focused_exam.get("research_stance"),
            "research_expected_direction": focused_exam.get("research_expected_direction"),
            "ticker_specificity": focused_exam.get("ticker_specificity"),
            "research_price_agreement": focused_exam.get("research_price_agreement"),
            "exam_verdict": focused_exam.get("exam_verdict"),
            "learning_gate": focused_exam.get("learning_gate", {}),
        },
        "focused_note": focused_overlay.get("focused_note", {}),
        "comparison": focused_overlay.get("comparison", {}),
    }


def _select_research_note(report: AgentLabRunReport) -> ResearchNote | None:
    for preferred in ("evidence_synthesis", "specialist_research", "financial_nlp"):
        for note in reversed(report.research_notes):
            if note.agent_name == preferred:
                return note
    return report.research_notes[-1] if report.research_notes else None


def _research_stance(note: ResearchNote | None) -> str:
    if note is None or note.data_quality == "weak":
        return "insufficient_data"
    pattern_set = set(note.patterns)
    bullish = len(pattern_set.intersection(BULLISH_RESEARCH_PATTERNS))
    risk = len(pattern_set.intersection(RISK_RESEARCH_PATTERNS))
    if bullish > risk:
        return "constructive"
    if risk > bullish:
        return "risk_heavy"
    thesis = note.thesis.lower()
    if "constructive" in thesis and bullish >= risk:
        return "constructive"
    if "risk-aware" in thesis or "risk aware" in thesis:
        return "risk_heavy"
    if "mixed" in thesis or "neutral" in thesis:
        return "mixed"
    return "mixed"


def _research_direction(stance: str) -> str:
    if stance == "constructive":
        return "bullish"
    if stance == "risk_heavy":
        return "bearish"
    return "neutral"


def _ticker_specificity(note: ResearchNote | None) -> str:
    if note is None or not note.tickers:
        return "none"
    if len(note.tickers) == 1:
        return "single_ticker"
    return "basket_or_sector"


def _agreement(research_direction: str, price_direction: str) -> str:
    if research_direction == "neutral":
        return "research_inconclusive"
    if price_direction == "neutral":
        return "price_inconclusive"
    if research_direction == price_direction:
        return "aligned"
    return "conflict"


def _exam_verdict(
    research_direction: str,
    price_direction: str,
    agreement: str,
    outcome_label: Any,
    quality_warnings: list[str],
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


def _learning_gate(evaluation: dict[str, Any], quality_warnings: list[str]) -> dict[str, Any]:
    if quality_warnings:
        return {
            "status": "blocked",
            "can_write_learning_memory": False,
            "reason": "Price-quality warnings make this research replay diagnostic only.",
        }
    if evaluation.get("status") != "evaluated":
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


def _recommendations(
    evidence_pack: dict[str, Any],
    lab_report: AgentLabRunReport,
    price_replay: dict[str, Any],
) -> list[str]:
    recommendations = [
        "Treat this as an old-data agent exam, not paper trading and not live trading.",
        "Review citations and price-quality warnings before promoting any lesson to learning memory.",
    ]
    coverage = evidence_pack.get("coverage", {})
    if coverage.get("data_quality") != "strong":
        recommendations.append("Add broader news, macro, filings, or sector materials before trusting the research stance.")
    if coverage.get("missing_requested_tickers"):
        recommendations.append("Some requested tickers have no matched evidence; improve source routing or ticker extraction.")
    if not lab_report.research_notes:
        recommendations.append("Agent Lab produced no research notes; inspect the evidence pack text extraction.")
    warnings = _compact_price_replay(price_replay).get("quality_warnings", [])
    if warnings:
        recommendations.append("Fix or normalize price history before using the replay result as calibration data.")
    if price_replay.get("evaluation", {}).get("outcome_label") == "miss":
        recommendations.append("Preserve this miss for review: it is valuable calibration material after quality checks pass.")
    recommendations.append("Run this over multiple as_of windows before adjusting analyst weights or pipeline tuning bounds.")
    return recommendations


def _first_path(paths: list[str | Path]) -> str | Path | None:
    return paths[0] if paths else None
