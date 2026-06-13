from __future__ import annotations

import argparse

from dean_os.analyst_outcome_evaluation_loop import AnalystOutcomeEvaluationLoop
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate reviewed analyst learning records against local prices; dry-run by default.",
    )
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--memory-store", default="data/dean_os/recommendation_memory.sqlite")
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--latest-processed-prices", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-early", action="store_true")
    parser.add_argument("--historical-diagnostic", action="store_true")
    parser.add_argument("--allow-diagnostic-apply", action="store_true")
    parser.add_argument("--neutral-band", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--agent-names", nargs="*", default=None)
    parser.add_argument("--include-non-analyst-records", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/analyst_outcome_evaluation")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    gate = payload.get("evaluation_gate", {})
    result = payload.get("outcome_evaluation", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Status: {gate.get('status')} | apply={gate.get('apply_requested')}")
    print(
        "Outcomes: "
        f"checked={result.get('pending_record_count')} | "
        f"evaluable={result.get('evaluable_count')} | updated={result.get('updated_count')}"
    )
    print(f"Status counts: {result.get('status_counts')}")
    for profile, summary in payload.get("profile_outcomes", {}).items():
        print(
            f"- {profile}: records={summary.get('record_count')} "
            f"completed={summary.get('completed_count')} pending={summary.get('pending_count')} "
            f"hit_rate={summary.get('hit_rate')}"
        )
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = AnalystOutcomeEvaluationLoop(output_dir=args.output_dir).run(
        learning_path=args.learning_store,
        memory_path=args.memory_store,
        market_data_path=args.market_data_path,
        latest_processed_prices=args.latest_processed_prices,
        tickers=args.tickers or [],
        as_of=args.as_of,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        apply=args.apply,
        allow_early=args.allow_early,
        historical_diagnostic=args.historical_diagnostic,
        allow_diagnostic_apply=args.allow_diagnostic_apply,
        neutral_band=args.neutral_band,
        limit=args.limit,
        profile=args.profile,
        agent_names=args.agent_names or [],
        include_non_analyst_records=args.include_non_analyst_records,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
