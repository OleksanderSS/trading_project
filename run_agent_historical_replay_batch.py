from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from dean_os.historical_replay_batch import HistoricalReplayBatchRunner
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple safe DEAN-OS historical replay slices without learning writes or pipeline execution.",
    )
    parser.add_argument("price_data_path", help="Normalized or raw historical price CSV/parquet file.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--as-of", nargs="*", default=None, dest="as_of_dates", help="Explicit as_of dates.")
    parser.add_argument("--start-as-of", default=None)
    parser.add_argument("--end-as-of", default=None)
    parser.add_argument("--step-days", type=int, default=14)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--horizon-days", nargs="+", type=int, default=[60])
    parser.add_argument("--news-data", default=None)
    parser.add_argument("--macro-data", default=None)
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--neutral-band", type=float, default=0.01)
    parser.add_argument("--max-runs", type=int, default=50, help="Maximum generated as_of dates.")
    parser.add_argument("--stop-on-quality-warning", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/historical_replay_batch")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    runner = HistoricalReplayBatchRunner(output_dir=args.output_dir)
    return await runner.run(
        price_data_path=args.price_data_path,
        tickers=args.tickers,
        as_of_dates=args.as_of_dates,
        start_as_of=args.start_as_of,
        end_as_of=args.end_as_of,
        step_days=args.step_days,
        lookback_days=args.lookback_days,
        horizon_days=args.horizon_days,
        news_data_path=args.news_data,
        macro_data_path=args.macro_data,
        benchmark_ticker=args.benchmark_ticker,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        neutral_band=args.neutral_band,
        max_runs=args.max_runs,
        stop_on_quality_warning=args.stop_on_quality_warning,
    )


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    gate = payload.get("learning_gate", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Total runs: {summary.get('total_runs')} | evaluated={summary.get('evaluated_runs')}")
    print(f"Quality-blocked runs: {summary.get('quality_blocked_runs')}")
    print(f"Hit rate: {summary.get('hit_rate')} | clear hit rate={summary.get('clear_hit_rate')}")
    print(f"Average return: {summary.get('average_realized_return')}")
    print(f"Learning gate: {gate.get('status')} | can_write_learning_memory={gate.get('can_write_learning_memory')}")
    warnings = summary.get("quality_warnings", {})
    if warnings:
        print("Top quality warnings:")
        for warning, count in list(warnings.items())[:5]:
            print(f"- {count}x {warning}")
    saved = payload.get("saved_paths", {})
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
