from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.outcome_price_coverage_plan import OutcomePriceCoveragePlan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only price coverage plan for analyst outcome evaluation.")
    parser.add_argument("--readiness-json", default="reports/dean_os/outcome_readiness_gate/latest.json")
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--latest-processed-prices", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--close-col", default=None)
    parser.add_argument("--datetime-col", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/outcome_price_coverage_plan")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Plan status: {summary.get('plan_status')}")
    print(
        "Coverage: "
        f"pending={summary.get('pending_record_count')} | "
        f"tickers={summary.get('ticker_count')} | "
        f"market_latest={summary.get('market_latest_timestamp')}"
    )
    print(f"Needs price after creation: {summary.get('tickers_need_price_after_creation')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = OutcomePriceCoveragePlan(output_dir=args.output_dir).build(
        readiness_path=args.readiness_json,
        market_data_path=args.market_data_path,
        latest_processed_prices=args.latest_processed_prices,
        tickers=args.tickers or [],
        close_col=args.close_col,
        datetime_col=args.datetime_col,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
