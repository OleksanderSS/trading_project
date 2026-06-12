from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dean_os.replay_price_normalizer import ReplayPriceNormalizer
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a normalized daily OHLCV artifact for safe DEAN-OS historical replay.",
    )
    parser.add_argument("price_data_path", help="Raw cached price CSV/parquet file.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker allow-list.")
    parser.add_argument("--output-dir", default="reports/dean_os/replay_price_normalizer")
    parser.add_argument("--artifact-dir", default="data/dean_os/replay_prices")
    parser.add_argument("--artifact-path", default=None, help="Optional explicit .csv or .parquet artifact path.")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--compare-replay", action="store_true", help="Run raw vs normalized historical replay comparison.")
    parser.add_argument("--as-of", default=None, help="Required with --compare-replay.")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--horizon-days", type=int, default=60)
    parser.add_argument("--news-data", default=None)
    parser.add_argument("--macro-data", default=None)
    parser.add_argument("--neutral-band", type=float, default=0.01)
    parser.add_argument("--print-json", action="store_true", help="Print full JSON payload.")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    runner = ReplayPriceNormalizer(output_dir=args.output_dir, artifact_dir=args.artifact_dir)
    return await runner.run(
        price_data_path=args.price_data_path,
        tickers=args.tickers,
        output_path=args.artifact_path,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        compare_replay=args.compare_replay,
        as_of=args.as_of,
        lookback_days=args.lookback_days,
        horizon_days=args.horizon_days,
        news_data_path=args.news_data,
        macro_data_path=args.macro_data,
        benchmark_ticker=args.benchmark_ticker,
        neutral_band=args.neutral_band,
    )


def print_summary(payload: dict[str, Any]) -> None:
    artifact = payload.get("artifact", {})
    gate = payload.get("learning_gate", {})
    quality = payload.get("quality", {})
    comparison = payload.get("replay_comparison", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Artifact: {artifact.get('path')}")
    print(f"Rows: {artifact.get('row_count')} | Tickers: {artifact.get('ticker_count')}")
    print(f"Learning gate: {gate.get('status')} | can_write_learning_memory={gate.get('can_write_learning_memory')}")
    warnings = quality.get("warnings", [])
    print(f"Price warnings: {len(warnings)}")
    for warning in warnings[:5]:
        print(f"- {warning}")
    if comparison:
        print(f"Replay comparison: {comparison.get('status')}")
        print(
            "Same ticker/action/outcome: "
            f"{comparison.get('same_decision_ticker')}/"
            f"{comparison.get('same_action')}/"
            f"{comparison.get('same_outcome_label')}"
        )
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
