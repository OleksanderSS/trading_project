from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dean_os.replays.historical_replay_batch import resolve_as_of_dates
from dean_os.historical_research_replay import HistoricalReplayRunner
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class HistoricalResearchReplayBatchRunner:
    """Runs multiple old-data research replay exams without learning writes."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/historical_research_replay_batch"):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        price_data_path: str | Path,
        tickers: list[str],
        as_of_dates: list[str] | None = None,
        start_as_of: str | None = None,
        end_as_of: str | None = None,
        step_days: int = 30,
        lookback_days: int = 180,
        horizon_days: int | list[int] = 60,
        news_data_paths: list[str | Path] | None = None,
        macro_data_paths: list[str | Path] | None = None,
        materials_paths: list[str | Path] | None = None,
        tags: list[str] | None = None,
        benchmark_ticker: str = "SPY",
        close_col: str = "close",
        datetime_col: str = "datetime",
        neutral_band: float = 0.01,
        max_runs: int = 20,
        normalize_daily_bars: bool = False,
        focused_overlay_path: str | Path | None = None,
        apply_focused_overlay: bool = False,
    ) -> dict[str, Any]:
        horizons = _normalize_horizons(horizon_days)
        dates = resolve_as_of_dates(
            price_data_path=price_data_path,
            as_of_dates=as_of_dates,
            start_as_of=start_as_of,
            end_as_of=end_as_of,
            step_days=step_days,
            lookback_days=lookback_days,
            horizons=horizons,
            max_runs=max_runs,
            close_col=close_col,
            datetime_col=datetime_col,
        )
        run_id = "historical_research_replay_batch_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_")
        run_dir = self.output_dir / "runs" / run_id
        compact_runs: list[dict[str, Any]] = []
        for as_of in dates:
            for horizon in horizons:
                replay = await HistoricalReplayRunner(output_dir=run_dir / f"h{horizon}").run(
                    price_data_path=price_data_path,
                    tickers=tickers,
                    as_of=as_of,
                    lookback_days=lookback_days,
                    horizon_days=horizon,
                    news_data_paths=news_data_paths,
                    macro_data_paths=macro_data_paths,
                    materials_paths=materials_paths,
                    tags=tags,
                    benchmark_ticker=benchmark_ticker,
                    close_col=close_col,
                    datetime_col=datetime_col,
                    neutral_band=neutral_band,
                    normalize_daily_bars=normalize_daily_bars,
                    focused_overlay_path=focused_overlay_path,
                    apply_focused_overlay=apply_focused_overlay,
                )
                compact_runs.append(_compact_run(replay))

        summary = _summary(compact_runs)
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "historical_research_replay_batch",
            "inputs": {
                "price_data_path": str(price_data_path),
                "tickers": [ticker.upper() for ticker in tickers if str(ticker).strip()],
                "as_of_dates": dates,
                "start_as_of": start_as_of,
                "end_as_of": end_as_of,
                "step_days": step_days,
                "lookback_days": lookback_days,
                "horizon_days": horizons,
                "news_data_paths": [str(path) for path in news_data_paths or []],
                "macro_data_paths": [str(path) for path in macro_data_paths or []],
                "materials_paths": [str(path) for path in materials_paths or []],
                "tags": tags or [],
                "benchmark_ticker": benchmark_ticker.upper(),
                "max_runs": max_runs,
                "normalize_daily_bars": normalize_daily_bars,
                "focused_overlay_path": str(focused_overlay_path) if focused_overlay_path else None,
                "apply_focused_overlay": apply_focused_overlay,
            },
            "summary": summary,
            "learning_gate": _learning_gate(summary),
            "runs": compact_runs,
            "recommendations": _recommendations(summary),
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
        rendered_md = render_historical_research_replay_batch_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return paths


def render_historical_research_replay_batch_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    gate = payload.get("learning_gate", {})
    lines = [
        "# DEAN-OS Historical Research Replay Batch",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Total runs: {summary.get('total_runs')}",
        f"- Evaluated runs: {summary.get('evaluated_runs')}",
        f"- Research inconclusive runs: {summary.get('research_inconclusive_runs')}",
        f"- Weak evidence runs: {summary.get('weak_evidence_runs')}",
        f"- Quality-blocked runs: {summary.get('quality_blocked_runs')}",
        f"- Hit rate: {summary.get('hit_rate')}",
        f"- Learning gate: `{gate.get('status')}`",
        "",
        "## Research Stance",
        "",
    ]
    for label, count in summary.get("research_stance_counts", {}).items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Outcomes", ""])
    for label, count in summary.get("outcome_counts", {}).items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## By Price Ticker", ""])
    for ticker, item in summary.get("by_price_ticker", {}).items():
        lines.append(
            f"- `{ticker}`: runs={item.get('runs')} evaluated={item.get('evaluated')} "
            f"hit_rate={item.get('hit_rate')} avg_return={item.get('average_realized_return')}"
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _compact_run(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs", {})
    exam = payload.get("research_exam", {})
    evidence = payload.get("evidence_pack", {}).get("coverage", {})
    price = payload.get("price_replay", {})
    decision = price.get("decision", {})
    evaluation = price.get("evaluation", {})
    warnings = price.get("quality_warnings", [])
    return {
        "run_id": payload.get("run_id"),
        "as_of": inputs.get("as_of"),
        "horizon_days": inputs.get("horizon_days"),
        "research_stance": exam.get("research_stance"),
        "research_expected_direction": exam.get("research_expected_direction"),
        "research_confidence": exam.get("research_confidence"),
        "research_data_quality": exam.get("research_data_quality"),
        "ticker_specificity": exam.get("ticker_specificity"),
        "research_price_agreement": exam.get("research_price_agreement"),
        "exam_verdict": exam.get("exam_verdict"),
        "learning_gate_status": exam.get("learning_gate", {}).get("status"),
        "evidence_document_count": evidence.get("document_count", 0),
        "evidence_data_quality": evidence.get("data_quality"),
        "evidence_missing_tickers": evidence.get("missing_requested_tickers", []),
        "evidence_tickers": evidence.get("tickers", []),
        "price_action": decision.get("action"),
        "price_ticker": decision.get("ticker"),
        "price_expected_direction": decision.get("expected_direction"),
        "price_confidence": decision.get("confidence"),
        "evaluation_status": evaluation.get("status"),
        "outcome_label": evaluation.get("outcome_label"),
        "realized_return": evaluation.get("realized_return"),
        "quality_status": "blocked" if warnings else "clear",
        "quality_warnings": warnings,
        "focused_overlay_status": exam.get("focused_overlay_status"),
        "focused_overlay_applied": bool(exam.get("focused_overlay_applied")),
        "original_research_stance": payload.get("research_exam_original", {}).get("research_stance")
        if payload.get("research_exam_original")
        else None,
        "original_exam_verdict": payload.get("research_exam_original", {}).get("exam_verdict")
        if payload.get("research_exam_original")
        else None,
        "saved_paths": payload.get("saved_paths", {}),
    }


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    evaluated = [run for run in runs if run.get("evaluation_status") == "evaluated"]
    clear_evaluated = [run for run in evaluated if run.get("quality_status") == "clear"]
    return {
        "total_runs": total,
        "evaluated_runs": len(evaluated),
        "clear_evaluated_runs": len(clear_evaluated),
        "quality_blocked_runs": sum(1 for run in runs if run.get("quality_status") == "blocked"),
        "weak_evidence_runs": sum(1 for run in runs if run.get("evidence_data_quality") != "strong"),
        "research_inconclusive_runs": sum(1 for run in runs if run.get("research_expected_direction") == "neutral"),
        "research_stance_counts": _counts(runs, "research_stance"),
        "evidence_quality_counts": _counts(runs, "evidence_data_quality"),
        "outcome_counts": _counts(evaluated, "outcome_label"),
        "exam_verdict_counts": _counts(runs, "exam_verdict"),
        "hit_rate": _hit_rate(evaluated),
        "clear_hit_rate": _hit_rate(clear_evaluated),
        "average_realized_return": _average_return(evaluated),
        "clear_average_realized_return": _average_return(clear_evaluated),
        "by_research_stance": _group_summary(runs, "research_stance"),
        "by_price_ticker": _group_summary(runs, "price_ticker"),
        "quality_warnings": _warning_summary(runs),
    }


def _learning_gate(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("quality_blocked_runs", 0):
        return {
            "status": "blocked",
            "can_write_learning_memory": False,
            "reason": "At least one research replay has price-quality warnings.",
        }
    if summary.get("weak_evidence_runs", 0):
        return {
            "status": "blocked_weak_evidence",
            "can_write_learning_memory": False,
            "reason": "At least one research replay has weak or partial evidence coverage.",
        }
    if int(summary.get("clear_evaluated_runs", 0)) < 5:
        return {
            "status": "insufficient_sample",
            "can_write_learning_memory": False,
            "reason": "Clean evaluated replay sample is too small for calibration.",
        }
    return {
        "status": "review_required",
        "can_write_learning_memory": False,
        "reason": "Promotion still requires human review and a learning apply ceremony.",
    }


def _recommendations(summary: dict[str, Any]) -> list[str]:
    recommendations = [
        "Use this as an analyst calibration exam, not paper trading or live trading.",
        "Do not adjust analyst weights until timestamp, price quality, and evidence coverage gates are clean.",
    ]
    if summary.get("quality_blocked_runs", 0):
        recommendations.append("Fix price-quality warnings before treating hit/miss rates as calibration data.")
    if summary.get("weak_evidence_runs", 0):
        recommendations.append("Backfill source coverage or narrow the ticker universe before trusting research stance statistics.")
    if summary.get("research_inconclusive_runs", 0):
        recommendations.append("Track how often research remains neutral; that may be correct behavior when evidence is thin.")
    if summary.get("hit_rate") is not None:
        recommendations.append("Compare by research stance and price ticker before changing profile defaults.")
    return recommendations


def _normalize_horizons(value: int | list[int]) -> list[int]:
    raw = value if isinstance(value, list) else [value]
    horizons = sorted({int(item) for item in raw if int(item) > 0})
    if not horizons:
        raise ValueError("At least one positive horizon day is required.")
    return horizons


def _counts(runs: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(run.get(key) or "unknown") for run in runs).items()))


def _group_summary(runs: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get(key) or "unknown")].append(run)
    return {group_key: _mini_summary(items) for group_key, items in sorted(grouped.items())}


def _mini_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [run for run in runs if run.get("evaluation_status") == "evaluated"]
    clear_evaluated = [run for run in evaluated if run.get("quality_status") == "clear"]
    return {
        "runs": len(runs),
        "evaluated": len(evaluated),
        "quality_blocked": sum(1 for run in runs if run.get("quality_status") == "blocked"),
        "weak_evidence": sum(1 for run in runs if run.get("evidence_data_quality") != "strong"),
        "hit_rate": _hit_rate(evaluated),
        "clear_hit_rate": _hit_rate(clear_evaluated),
        "average_realized_return": _average_return(evaluated),
    }


def _hit_rate(runs: list[dict[str, Any]]) -> float | None:
    if not runs:
        return None
    hits = sum(1 for run in runs if run.get("outcome_label") == "hit")
    return round(hits / len(runs), 6)


def _average_return(runs: list[dict[str, Any]]) -> float | None:
    values = [float(run["realized_return"]) for run in runs if run.get("realized_return") is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _warning_summary(runs: list[dict[str, Any]]) -> dict[str, int]:
    warnings = Counter()
    for run in runs:
        for warning in run.get("quality_warnings", []):
            warnings[str(warning)] += 1
    return dict(warnings.most_common())
