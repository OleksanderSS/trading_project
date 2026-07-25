from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.outcome_evaluation import OutcomeEvaluationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pending learning records against local prices; dry-run by default.")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--latest-processed-prices", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--allow-early", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--neutral-band", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/outcome_evaluation")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = OutcomeEvaluationRunner(args.learning_store).evaluate(
        market_data_path=args.market_data_path,
        latest_processed_prices=args.latest_processed_prices,
        tickers=[ticker.upper() for ticker in args.tickers or []],
        as_of=args.as_of,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        allow_early=args.allow_early,
        apply_updates=args.apply,
        neutral_band=args.neutral_band,
        limit=args.limit,
    )
    payload = {"run_id": run_id("outcome_evaluation"), "inputs": vars(args), **result}
    payload = save_latest_json(args.output, args.output_dir, payload)
    if args.print_json:
        print_json(payload)
        return
    print(f"Pending checked: {payload.get('pending_record_count')} | evaluable={payload.get('evaluable_count')} | updated={payload.get('updated_count')}")
    print(f"Status counts: {payload.get('status_counts')}")
    for item in payload.get("recommendations", []):
        print(f"- {item}")
    print(f"Report JSON: {payload.get('saved_paths', {}).get('latest_json') or payload.get('saved_paths', {}).get('json')}")


if __name__ == "__main__":
    main()

