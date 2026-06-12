from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from dean_os.historical_replay import HistoricalReplayRunner
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a safe old-data DEAN-OS replay without paper trades, live broker access, or heavy pipeline execution.",
    )
    parser.add_argument("price_data_path", help="Historical price CSV/parquet file.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers visible to the replay analyst.")
    parser.add_argument("--as-of", required=True, help="Cutoff timestamp; thesis sees only data at or before this time.")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--horizon-days", type=int, default=60)
    parser.add_argument("--news-data", default=None)
    parser.add_argument("--macro-data", default=None)
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--neutral-band", type=float, default=0.01)
    parser.add_argument("--max-news-items", type=int, default=80)
    parser.add_argument("--normalize-daily-bars", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/historical_replay")
    parser.add_argument("--print-json", action="store_true", help="Print full JSON payload.")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    runner = HistoricalReplayRunner(output_dir=args.output_dir)
    return await runner.run(
        price_data_path=args.price_data_path,
        tickers=args.tickers,
        as_of=args.as_of,
        lookback_days=args.lookback_days,
        horizon_days=args.horizon_days,
        news_data_path=args.news_data,
        macro_data_path=args.macro_data,
        benchmark_ticker=args.benchmark_ticker,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        neutral_band=args.neutral_band,
        max_news_items=args.max_news_items,
        normalize_daily_bars=args.normalize_daily_bars,
    )


def print_summary(payload: dict[str, Any]) -> None:
    decision = payload.get("decision", {})
    evaluation = payload.get("evaluation", {})
    replay = payload.get("historical_replay", {})
    price_quality = replay.get("coverage", {}).get("price_quality", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Decision: {decision.get('action')} | ticker={decision.get('ticker')} | confidence={decision.get('confidence')}")
    print(f"Evaluation: {evaluation.get('status')} | outcome={evaluation.get('outcome_label')} | return={evaluation.get('realized_return')}")
    warnings = price_quality.get("warnings", [])
    print(f"Price warnings: {len(warnings)}")
    for warning in warnings[:5]:
        print(f"- {warning}")
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
