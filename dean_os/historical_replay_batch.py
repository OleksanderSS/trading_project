from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.historical_replay import HistoricalReplayRunner
from dean_os.outcome_evaluation import _parse_datetime, _prepare_market_frame, _read_market_frame
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class HistoricalReplayBatchRunner:
    """Runs many historical replay slices without learning writes or pipeline execution."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/historical_replay_batch"):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        price_data_path: str | Path,
        tickers: list[str],
        as_of_dates: list[str] | None = None,
        start_as_of: str | None = None,
        end_as_of: str | None = None,
        step_days: int = 14,
        lookback_days: int = 180,
        horizon_days: int | list[int] = 60,
        news_data_path: str | Path | None = None,
        macro_data_path: str | Path | None = None,
        benchmark_ticker: str = "SPY",
        close_col: str = "close",
        datetime_col: str = "datetime",
        neutral_band: float = 0.01,
        max_runs: int = 50,
        stop_on_quality_warning: bool = False,
    ) -> dict[str, Any]:
        horizons = _normalize_horizons(horizon_days)
        dates = _resolve_as_of_dates(
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
        run_id = "historical_replay_batch_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_")
        replay_dir = self.output_dir / "runs" / run_id
        compact_runs: list[dict[str, Any]] = []
        for as_of in dates:
            for horizon in horizons:
                replay = await HistoricalReplayRunner(output_dir=replay_dir / f"h{horizon}").run(
                    price_data_path=price_data_path,
                    tickers=tickers,
                    as_of=as_of,
                    lookback_days=lookback_days,
                    horizon_days=horizon,
                    news_data_path=news_data_path,
                    macro_data_path=macro_data_path,
                    benchmark_ticker=benchmark_ticker,
                    close_col=close_col,
                    datetime_col=datetime_col,
                    neutral_band=neutral_band,
                    normalize_daily_bars=False,
                )
                summary = _compact_replay(replay, horizon)
                compact_runs.append(summary)
                if stop_on_quality_warning and summary["quality_warnings"]:
                    break
            if stop_on_quality_warning and compact_runs and compact_runs[-1]["quality_warnings"]:
                break

        aggregate = _aggregate_runs(compact_runs)
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "historical_replay_batch",
            "inputs": {
                "price_data_path": str(price_data_path),
                "tickers": [ticker.upper() for ticker in tickers if str(ticker).strip()],
                "as_of_dates": dates,
                "start_as_of": start_as_of,
                "end_as_of": end_as_of,
                "step_days": step_days,
                "lookback_days": lookback_days,
                "horizon_days": horizons,
                "news_data_path": str(news_data_path) if news_data_path else None,
                "macro_data_path": str(macro_data_path) if macro_data_path else None,
                "benchmark_ticker": benchmark_ticker.upper(),
                "max_runs": max_runs,
                "stop_on_quality_warning": stop_on_quality_warning,
            },
            "summary": aggregate,
            "learning_gate": _batch_learning_gate(aggregate),
            "runs": compact_runs,
            "recommendations": _recommendations(aggregate),
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
        rendered_md = render_historical_replay_batch_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return paths


def render_historical_replay_batch_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    gate = payload.get("learning_gate", {})
    lines = [
        "# DEAN-OS Historical Replay Batch",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Total replay runs: {summary.get('total_runs')}",
        f"- Evaluated runs: {summary.get('evaluated_runs')}",
        f"- Quality-blocked runs: {summary.get('quality_blocked_runs')}",
        f"- Hit rate: {summary.get('hit_rate')}",
        f"- Learning gate: `{gate.get('status')}`",
        "",
        "## Outcomes",
        "",
    ]
    for label, count in summary.get("outcome_counts", {}).items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## By Ticker", ""])
    for ticker, item in summary.get("by_ticker", {}).items():
        lines.append(
            f"- `{ticker}`: runs={item.get('runs')} evaluated={item.get('evaluated')} "
            f"hit_rate={item.get('hit_rate')} quality_blocked={item.get('quality_blocked')}"
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _resolve_as_of_dates(
    price_data_path: str | Path,
    as_of_dates: list[str] | None,
    start_as_of: str | None,
    end_as_of: str | None,
    step_days: int,
    lookback_days: int,
    horizons: list[int],
    max_runs: int,
    close_col: str,
    datetime_col: str,
) -> list[str]:
    if as_of_dates:
        return [_parse_datetime(value).isoformat() for value in as_of_dates[:max_runs]]
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(f"pandas is required for historical replay batch date discovery: {exc}") from exc

    path = Path(price_data_path)
    frame = _prepare_market_frame(
        pd=pd,
        frame=_read_market_frame(pd, path),
        close_col=close_col,
        datetime_col=datetime_col,
    )
    if frame.empty:
        raise ValueError("Cannot auto-generate as_of dates from an empty price frame.")
    min_date = frame["_dean_datetime"].min().to_pydatetime()
    max_date = frame["_dean_datetime"].max().to_pydatetime()
    start = _parse_datetime(start_as_of) if start_as_of else min_date + timedelta(days=lookback_days)
    end = _parse_datetime(end_as_of) if end_as_of else max_date - timedelta(days=max(horizons))
    if start > end:
        raise ValueError(
            "No valid replay date range. Move start_as_of earlier, end_as_of later, reduce lookback_days, or reduce horizon_days."
        )
    step = max(int(step_days), 1)
    dates: list[str] = []
    current = start
    while current <= end and len(dates) < max_runs:
        dates.append(current.astimezone(frame["_dean_datetime"].dt.tz).isoformat())
        current += timedelta(days=step)
    return dates


def _normalize_horizons(value: int | list[int]) -> list[int]:
    raw = value if isinstance(value, list) else [value]
    horizons = sorted({int(item) for item in raw if int(item) > 0})
    if not horizons:
        raise ValueError("At least one positive horizon day is required.")
    return horizons


def _compact_replay(payload: dict[str, Any], horizon_days: int) -> dict[str, Any]:
    decision = payload.get("decision", {})
    evaluation = payload.get("evaluation", {})
    replay = payload.get("historical_replay", {})
    price_quality = replay.get("coverage", {}).get("price_quality", {})
    warnings = list(price_quality.get("warnings", []))
    return {
        "run_id": payload.get("run_id"),
        "as_of": payload.get("inputs", {}).get("as_of"),
        "horizon_days": horizon_days,
        "action": decision.get("action"),
        "ticker": decision.get("ticker"),
        "expected_direction": decision.get("expected_direction"),
        "confidence": decision.get("confidence"),
        "evaluation_status": evaluation.get("status"),
        "outcome_label": evaluation.get("outcome_label"),
        "realized_return": evaluation.get("realized_return"),
        "quality_status": "blocked" if warnings else "clear",
        "quality_warnings": warnings,
        "top_rankings": replay.get("rankings", [])[:5],
        "regime_context": replay.get("regime_context", {}),
        "saved_paths": payload.get("saved_paths", {}),
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    evaluated = [run for run in runs if run.get("evaluation_status") == "evaluated"]
    clear_evaluated = [run for run in evaluated if run.get("quality_status") == "clear"]
    outcome_counts = Counter(str(run.get("outcome_label") or "unknown") for run in evaluated)
    quality_blocked = [run for run in runs if run.get("quality_status") == "blocked"]
    by_ticker = _group_summary(runs, "ticker")
    by_horizon = _group_summary(runs, "horizon_days")
    return {
        "total_runs": total,
        "evaluated_runs": len(evaluated),
        "clear_evaluated_runs": len(clear_evaluated),
        "quality_blocked_runs": len(quality_blocked),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "hit_rate": _hit_rate(evaluated),
        "clear_hit_rate": _hit_rate(clear_evaluated),
        "average_realized_return": _average_return(evaluated),
        "clear_average_realized_return": _average_return(clear_evaluated),
        "by_ticker": by_ticker,
        "by_horizon": by_horizon,
        "quality_warnings": _warning_summary(runs),
    }


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


def _batch_learning_gate(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("quality_blocked_runs", 0):
        return {
            "status": "blocked",
            "can_write_learning_memory": False,
            "reason": "At least one replay slice has price-quality warnings; batch outcomes remain diagnostic.",
        }
    if int(summary.get("clear_evaluated_runs", 0)) < 5:
        return {
            "status": "insufficient_sample",
            "can_write_learning_memory": False,
            "reason": "Clean replay sample is too small for learning-memory promotion.",
        }
    return {
        "status": "review_required",
        "can_write_learning_memory": False,
        "reason": "Batch data is clean, but promotion still requires a human-reviewed learning bridge.",
    }


def _recommendations(summary: dict[str, Any]) -> list[str]:
    recommendations = [
        "Use batch replay as an evidence exam, not as paper trading or live trading.",
        "Do not write batch outcomes to learning memory until quality gates are clean and review approves promotion.",
    ]
    if summary.get("quality_blocked_runs", 0):
        recommendations.append("Investigate repeated price-quality warnings before expanding the replay date grid.")
    if summary.get("clear_evaluated_runs", 0) < 5:
        recommendations.append("Increase the number of clean as_of slices before judging agent calibration.")
    if summary.get("hit_rate") is not None:
        recommendations.append("Compare hit rate and average return by ticker/horizon before changing agent weights.")
    return recommendations
