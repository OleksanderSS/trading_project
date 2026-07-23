from __future__ import annotations

import argparse
import asyncio

from dean_os.cli_helpers import print_json
from dean_os.pipeline_control.pipeline_control_bounded_evidence_batch import (
    PipelineControlBoundedEvidenceBatch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a predeclared set of offline bounded evidence contexts."
    )
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--ticker", action="append", dest="tickers", required=True)
    parser.add_argument("--frozen-context", action="append", dest="frozen_contexts")
    parser.add_argument("--macro-source-path", default=None)
    parser.add_argument("--rows-per-context", type=int, default=480)
    parser.add_argument("--max-features", type=int, default=40)
    parser.add_argument("--gap-size", type=int, default=5)
    parser.add_argument("--min-rows", type=int, default=180)
    parser.add_argument("--transaction-cost", type=float, default=0.0025, dest="transaction_cost_per_turn")
    parser.add_argument("--no-real-metric-review", action="store_false", dest="run_real_metric_review")
    parser.add_argument("--max-contexts", type=int, default=8)
    parser.add_argument("--input-is-enriched", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/pipeline_control_bounded_evidence_batch_current",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


async def run_batch(args: argparse.Namespace) -> None:
    payload = await PipelineControlBoundedEvidenceBatch(args.output_dir).run(
        coverage_json=args.coverage_json,
        tickers=args.tickers,
        macro_source_path=args.macro_source_path,
        rows_per_context=args.rows_per_context,
        max_features=args.max_features,
        gap_size=args.gap_size,
        min_rows=args.min_rows,
        transaction_cost_per_turn=args.transaction_cost_per_turn,
        run_real_metric_review=args.run_real_metric_review,
        frozen_contexts=args.frozen_contexts,
        max_contexts=args.max_contexts,
        input_is_enriched=args.input_is_enriched,
    )
    if args.print_json:
        print_json(payload)
        return
        
    summary = payload.get("summary", {})
    print(f"Batch status: {summary.get('batch_status')}")
    print(f"Configured context count: {summary.get('configured_context_count')}")
    print(f"Completed contexts: {summary.get('completed_context_count')}")
    print(f"Failed contexts: {summary.get('failed_context_count')}")
    
    saved = payload.get("saved_paths", {})
    if saved:
        print(f"Report JSON: {saved.get('latest_json')}")
        print(f"Report Markdown: {saved.get('latest_markdown')}")


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
