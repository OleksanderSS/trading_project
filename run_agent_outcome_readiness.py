from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.outcome_readiness_gate import OutcomeReadinessGate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether pending analyst learning records are ready for outcome evaluation.")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--memory-store", default="data/dean_os/recommendation_memory.sqlite")
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--latest-processed-prices", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--neutral-band", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--agent-names", nargs="*", default=None)
    parser.add_argument("--include-non-analyst-records", action="store_true")
    parser.add_argument("--historical-diagnostic", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/outcome_readiness_gate")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Readiness: {summary.get('readiness_status')}")
    print(
        "Records: "
        f"pending={summary.get('pending_record_count')} | "
        f"evaluable={summary.get('evaluable_count')} | statuses={summary.get('status_counts')}"
    )
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = OutcomeReadinessGate(output_dir=args.output_dir).build(
        learning_path=args.learning_store,
        memory_path=args.memory_store,
        market_data_path=args.market_data_path,
        latest_processed_prices=args.latest_processed_prices,
        tickers=args.tickers or [],
        as_of=args.as_of,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        neutral_band=args.neutral_band,
        limit=args.limit,
        profile=args.profile,
        agent_names=args.agent_names or [],
        include_non_analyst_records=args.include_non_analyst_records,
        historical_diagnostic=args.historical_diagnostic,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
