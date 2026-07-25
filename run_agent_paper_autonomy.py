from __future__ import annotations

import argparse
import asyncio

from dean_os.cli_helpers import print_json
from dean_os.paper_autonomy import PaperAutonomyRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run supervised paper-autonomy diagnostics without broker access.")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--latest-processed-prices", default="1d")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--max-age-hours", type=float, default=72.0)
    parser.add_argument("--paper-store", default="data/dean_os/paper_trades.sqlite")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--position-size-pct", type=float, default=0.05)
    parser.add_argument("--include-watchlist", action="store_true")
    parser.add_argument("--review-snapshot", default=None)
    parser.add_argument("--max-drawdown-limit", type=float, default=0.10)
    parser.add_argument("--output-dir", default="reports/dean_os/paper_autonomy")
    parser.add_argument("--event-log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--decision-log-path", default="logs/dean_os/decisions.jsonl")
    parser.add_argument("--experience-diary", default="logs/experience_diary.csv")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict:
    runner = PaperAutonomyRunner(
        output_dir=args.output_dir,
        event_log_path=args.event_log_path,
        decision_log_path=args.decision_log_path,
        experience_diary_path=args.experience_diary,
    )
    return await runner.run(
        tickers=[ticker.upper() for ticker in args.tickers or []],
        timeframe=args.timeframe,
        market_data_path=args.market_data_path,
        latest_processed_prices=args.latest_processed_prices,
        as_of=args.as_of,
        max_age_hours=args.max_age_hours,
        paper_store_path=args.paper_store,
        initial_cash=args.initial_cash,
        position_size_pct=args.position_size_pct,
        include_watchlist=args.include_watchlist,
        review_snapshot_path=args.review_snapshot,
        max_drawdown_limit=args.max_drawdown_limit,
    )


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(main_async(args))
    if args.print_json:
        print_json(payload)
        return
    decision = payload.get("decision", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Status: {decision.get('status')} | allow_new_paper_decision={decision.get('allow_new_paper_decision')}")
    print(f"Reason: {decision.get('reason')}")
    print(f"Report JSON: {payload.get('saved_paths', {}).get('json')}")


if __name__ == "__main__":
    main()

