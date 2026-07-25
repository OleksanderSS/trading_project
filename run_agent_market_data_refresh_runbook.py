from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.market_data_refresh_runbook import MarketDataRefreshRunbook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only market-data refresh runbook for outcome blockers.")
    parser.add_argument("--coverage-plan-json", default="reports/dean_os/outcome_price_coverage_plan/latest.json")
    parser.add_argument("--collector-inventory-json", default="reports/dean_os/collector_inventory/latest.json")
    parser.add_argument("--price-glob", action="append", default=None)
    parser.add_argument("--max-price-artifacts", type=int, default=25)
    parser.add_argument("--refreshed-price-placeholder", default="PATH_TO_REFRESHED_PRICE_FILE")
    parser.add_argument("--output-dir", default="reports/dean_os/market_data_refresh_runbook")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Runbook status: {summary.get('runbook_status')}")
    print(f"Required tickers: {', '.join(summary.get('required_tickers', [])) or 'none'}")
    print(f"Primary price feed: {summary.get('primary_price_feed')}")
    print(f"Tasks: {summary.get('task_count')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = MarketDataRefreshRunbook(output_dir=args.output_dir).build(
        coverage_plan_path=args.coverage_plan_json,
        collector_inventory_path=args.collector_inventory_json,
        price_globs=args.price_glob,
        max_price_artifacts=args.max_price_artifacts,
        refreshed_price_placeholder=args.refreshed_price_placeholder,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
