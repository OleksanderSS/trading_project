from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dean_os.agents.regime import RegimeAgent
from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.schemas import MarketContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RegimeAgent as a soft pipeline report without trading execution.")
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--latest-processed-prices", default="1d")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--engine", choices=["fallback", "project"], default="fallback")
    parser.add_argument("--manual-regime", default=None)
    parser.add_argument("--manual-tags", nargs="*", default=None)
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/regime")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    context = MarketContext(tickers=[args.ticker.upper()] if args.ticker else [], timeframes=[args.latest_processed_prices])
    report = await RegimeAgent(
        name="regime",
        config={
            "market_data_path": args.market_data_path,
            "latest_processed_prices": args.latest_processed_prices,
            "ticker": args.ticker,
            "engine": args.engine,
            "manual_regime": args.manual_regime,
            "manual_tags": args.manual_tags or [],
            "close_col": args.close_col,
            "volume_col": args.volume_col,
        },
    ).run(context)
    payload = {
        "run_id": run_id("regime"),
        "mode": "regime_agent",
        "inputs": vars(args),
        "report": report.model_dump(mode="json"),
        "regime_context": context.metadata.get("regime_context", {}),
        "regime_tags": context.metadata.get("regime_tags", []),
    }
    return save_latest_json(args.output, args.output_dir, payload)


def print_summary(payload: dict[str, Any]) -> None:
    report = payload["report"]
    regime = payload.get("regime_context", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Verdict: {report.get('verdict')} | signal={report.get('signal_strength')}")
    print(f"Regime: {regime.get('regime')} | confidence={regime.get('confidence')}")
    print(f"Tags: {', '.join(regime.get('context_tags', [])) or 'none'}")
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

