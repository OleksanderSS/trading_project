from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dean_os.agents.paper_portfolio import PaperPortfolioAgent
from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.schemas import MarketContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate logged paper decisions as a paper-only portfolio.")
    parser.add_argument("--store", default="data/dean_os/paper_trades.sqlite")
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--latest-processed-prices", default="1d")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--position-size-pct", type=float, default=0.05)
    parser.add_argument("--include-watchlist", action="store_true")
    parser.add_argument("--watchlist-position-size-pct", type=float, default=0.0)
    parser.add_argument("--confidence-weighting", action="store_true")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--statuses", nargs="*", default=["pending", "evaluated"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-drawdown-limit", type=float, default=0.10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/paper_portfolio")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    tickers = [ticker.upper() for ticker in args.tickers or []]
    context = MarketContext(tickers=tickers)
    report = await PaperPortfolioAgent(
        name="paper_portfolio",
        config={
            "store_path": args.store,
            "market_data_path": args.market_data_path,
            "latest_processed_prices": args.latest_processed_prices,
            "tickers": tickers,
            "as_of": args.as_of,
            "initial_cash": args.initial_cash,
            "position_size_pct": args.position_size_pct,
            "include_watchlist": args.include_watchlist,
            "watchlist_position_size_pct": args.watchlist_position_size_pct,
            "confidence_weighting": args.confidence_weighting,
            "slippage_bps": args.slippage_bps,
            "commission_bps": args.commission_bps,
            "close_col": args.close_col,
            "datetime_col": args.datetime_col,
            "statuses": args.statuses,
            "limit": args.limit,
            "max_drawdown_limit": args.max_drawdown_limit,
        },
    ).run(context)
    payload = {
        "run_id": run_id("paper_portfolio"),
        "mode": "paper_portfolio_agent",
        "inputs": vars(args),
        "report": report.model_dump(mode="json"),
        "paper_portfolio": context.metadata.get("paper_portfolio", {}),
    }
    return save_latest_json(args.output, args.output_dir, payload)


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(main_async(args))
    if args.print_json:
        print_json(payload)
        return
    summary = payload.get("paper_portfolio", {}).get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Verdict: {payload.get('report', {}).get('verdict')} | positions={summary.get('position_count')} | total_return={summary.get('total_return')}")
    print(f"Max drawdown: {summary.get('max_drawdown')} | skipped={summary.get('skipped_count')}")
    print(f"Report JSON: {payload.get('saved_paths', {}).get('latest_json') or payload.get('saved_paths', {}).get('json')}")


if __name__ == "__main__":
    main()

