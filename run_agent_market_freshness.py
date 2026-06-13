from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dean_os.agents.market_data_freshness import MarketDataFreshnessAgent
from dean_os.agents.operations import OperationsProposalAgent
from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.schemas import MarketContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local market data freshness without running the trading pipeline.")
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--latest-processed-prices", default="1d")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--max-age-hours", type=float, default=72.0)
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--include-operation-proposal", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/market_freshness")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    tickers = [ticker.upper() for ticker in args.tickers or [] if str(ticker).strip()]
    context = MarketContext(tickers=tickers, timeframes=[args.latest_processed_prices])
    report = await MarketDataFreshnessAgent(
        name="market_data_freshness",
        config={
            "market_data_path": args.market_data_path,
            "latest_processed_prices": args.latest_processed_prices,
            "tickers": tickers,
            "as_of": args.as_of,
            "max_age_hours": args.max_age_hours,
            "close_col": args.close_col,
            "datetime_col": args.datetime_col,
        },
    ).run(context)
    proposal_report = None
    if args.include_operation_proposal:
        proposal_report = await OperationsProposalAgent(name="operations_proposal", config={"proposal_only": True}).run(context)
    payload = {
        "run_id": run_id("market_freshness"),
        "mode": "market_data_freshness_agent",
        "inputs": vars(args),
        "report": report.model_dump(mode="json"),
        "proposal_report": proposal_report.model_dump(mode="json") if proposal_report else None,
        "data_freshness": context.metadata.get("data_freshness", {}),
        "action_proposals": [proposal.model_dump(mode="json") for proposal in context.action_proposals],
    }
    return save_latest_json(args.output, args.output_dir, payload)


def print_summary(payload: dict[str, Any]) -> None:
    report = payload["report"]
    prices = payload.get("data_freshness", {}).get("market_prices", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Verdict: {report.get('verdict')} | status={prices.get('status')} | age_hours={prices.get('age_hours')}")
    print(f"Missing tickers: {prices.get('missing_tickers', [])}")
    print(f"Action proposals: {len(payload.get('action_proposals', []))}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(main_async(args))
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()

