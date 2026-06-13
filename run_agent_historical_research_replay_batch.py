from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from dean_os.historical_research_replay_batch import HistoricalResearchReplayBatchRunner
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple historical research replay exams without learning writes or pipeline execution.",
    )
    parser.add_argument("price_data_path", help="Historical price CSV/parquet file.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--as-of", nargs="*", default=None, dest="as_of_dates", help="Explicit as_of dates.")
    parser.add_argument("--start-as-of", default=None)
    parser.add_argument("--end-as-of", default=None)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--horizon-days", nargs="+", type=int, default=[60])
    parser.add_argument("--news-data", nargs="*", default=None)
    parser.add_argument("--macro-data", nargs="*", default=None)
    parser.add_argument("--materials", nargs="*", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--neutral-band", type=float, default=0.01)
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--normalize-daily-bars", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/historical_research_replay_batch")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    return await HistoricalResearchReplayBatchRunner(output_dir=args.output_dir).run(
        price_data_path=args.price_data_path,
        tickers=args.tickers,
        as_of_dates=args.as_of_dates,
        start_as_of=args.start_as_of,
        end_as_of=args.end_as_of,
        step_days=args.step_days,
        lookback_days=args.lookback_days,
        horizon_days=args.horizon_days,
        news_data_paths=args.news_data,
        macro_data_paths=args.macro_data,
        materials_paths=args.materials,
        tags=args.tags,
        benchmark_ticker=args.benchmark_ticker,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        neutral_band=args.neutral_band,
        max_runs=args.max_runs,
        normalize_daily_bars=args.normalize_daily_bars,
    )


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    gate = payload.get("learning_gate", {})
    saved = payload.get("saved_paths", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Total runs: {summary.get('total_runs')} | evaluated={summary.get('evaluated_runs')}")
    print(f"Research inconclusive: {summary.get('research_inconclusive_runs')}")
    print(f"Weak evidence: {summary.get('weak_evidence_runs')} | quality-blocked={summary.get('quality_blocked_runs')}")
    print(f"Hit rate: {summary.get('hit_rate')} | avg return={summary.get('average_realized_return')}")
    print(f"Learning gate: {gate.get('status')} | can_write_learning_memory={gate.get('can_write_learning_memory')}")
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")
        print(f"Report Markdown: {saved.get('latest_markdown') or saved.get('markdown')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(main_async(args))
    if args.print_json:
        print(json.dumps(json_ready(payload), indent=2, ensure_ascii=False))
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
