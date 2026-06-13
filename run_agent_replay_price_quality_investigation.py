from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.replay_price_quality_investigation import ReplayPriceQualityInvestigationPlan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only investigation plan for replay price-quality blockers.",
    )
    parser.add_argument("--report-json", action="append", default=None, help="Replay/normalizer/batch report JSON.")
    parser.add_argument("--artifact-only", action="store_true", help="Skip default report JSONs and inspect only price artifacts.")
    parser.add_argument("--price-data", action="append", default=None, help="Additional price CSV/parquet artifacts to inspect.")
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--large-step-threshold", type=float, default=0.15)
    parser.add_argument("--output-dir", default="reports/dean_os/replay_price_quality_investigation")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Investigation status: {summary.get('investigation_status')}")
    print(f"Reports loaded: {summary.get('reports_loaded')}")
    print(f"Price artifacts inspected: {summary.get('price_artifacts_inspected')}")
    print(f"Warning records: {summary.get('warning_record_count')}")
    print(f"Extreme benchmark warnings: {summary.get('extreme_benchmark_warning_count')}")
    print(f"Window diagnostics: {summary.get('window_diagnostic_count')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")
        print(f"Report Markdown: {payload['saved_paths'].get('latest_markdown') or payload['saved_paths'].get('markdown')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = ReplayPriceQualityInvestigationPlan(output_dir=args.output_dir).build(
        report_paths=[] if args.artifact_only else args.report_json,
        price_data_paths=args.price_data,
        benchmark_ticker=args.benchmark_ticker,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        large_step_threshold=args.large_step_threshold,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
