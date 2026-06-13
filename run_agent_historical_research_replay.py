from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from dean_os.historical_research_replay import HistoricalResearchReplayRunner
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a safe historical research replay: evidence pack + Agent Lab + price outcome, "
            "without learning writes, broker access, or pipeline execution."
        ),
    )
    parser.add_argument("price_data_path", help="Historical price CSV/parquet file.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers visible to the replay exam.")
    parser.add_argument("--as-of", required=True, help="Cutoff timestamp; agents see only data at or before this time.")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--horizon-days", type=int, default=60)
    parser.add_argument("--news-data", nargs="*", default=None)
    parser.add_argument("--macro-data", nargs="*", default=None)
    parser.add_argument("--materials", nargs="*", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--neutral-band", type=float, default=0.01)
    parser.add_argument("--max-rows-per-table", type=int, default=300)
    parser.add_argument("--max-documents", type=int, default=600)
    parser.add_argument("--normalize-daily-bars", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/historical_research_replay")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    runner = HistoricalResearchReplayRunner(output_dir=args.output_dir)
    return await runner.run(
        price_data_path=args.price_data_path,
        tickers=args.tickers,
        as_of=args.as_of,
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
        max_rows_per_table=args.max_rows_per_table,
        max_documents=args.max_documents,
        normalize_daily_bars=args.normalize_daily_bars,
    )


def print_summary(payload: dict[str, Any]) -> None:
    exam = payload.get("research_exam", {})
    price = payload.get("price_replay", {})
    decision = price.get("decision", {})
    evaluation = price.get("evaluation", {})
    evidence = payload.get("evidence_pack", {}).get("coverage", {})
    saved = payload.get("saved_paths", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(
        "Research: "
        f"stance={exam.get('research_stance')} "
        f"direction={exam.get('research_expected_direction')} "
        f"specificity={exam.get('ticker_specificity')}"
    )
    print(
        "Price replay: "
        f"decision={decision.get('action')} "
        f"ticker={decision.get('ticker')} "
        f"outcome={evaluation.get('outcome_label')} "
        f"return={evaluation.get('realized_return')}"
    )
    print(f"Exam verdict: {exam.get('exam_verdict')} | agreement={exam.get('research_price_agreement')}")
    print(f"Evidence documents: {evidence.get('document_count')} | quality={evidence.get('data_quality')}")
    warnings = price.get("quality_warnings", [])
    print(f"Price warnings: {len(warnings)}")
    for warning in warnings[:5]:
        print(f"- {warning}")
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
