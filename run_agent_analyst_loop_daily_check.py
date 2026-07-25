from __future__ import annotations

import argparse

from dean_os.analyst_core.analyst_loop_daily_check import AnalystLoopDailyCheck
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only daily operator check for the analyst learning loop.",
    )
    parser.add_argument("--evidence-pack-json", default=None)
    parser.add_argument("--analyst-profiles-json", default=None)
    parser.add_argument("--profile-scorecard-json", default=None)
    parser.add_argument("--learning-bridge-json", default=None)
    parser.add_argument("--review-approved-learning-json", default=None)
    parser.add_argument("--outcome-evaluation-json", default=None)
    parser.add_argument("--calibration-gate-json", default=None)
    parser.add_argument("--calibration-proposals-json", default=None)
    parser.add_argument("--calibration-review-json", default=None)
    parser.add_argument("--manual-backlog-json", default=None)
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--latest-processed-prices", default="1d")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--max-age-hours", type=float, default=72.0)
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--event-log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--event-limit", type=int, default=10)
    parser.add_argument("--output-dir", default="reports/dean_os/analyst_loop_daily_check")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    checks = payload.get("checks", {})
    market = checks.get("market_freshness", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Decision: {summary.get('decision')}")
    print(f"Current: {summary.get('current_stage')} | status={summary.get('current_status')}")
    print(f"Market: {market.get('status')} | age_hours={market.get('age_hours')}")
    print(f"Blockers: {summary.get('blocker_count')} | warnings={summary.get('warning_count')}")
    print(f"Next command: {summary.get('next_command')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    stage_paths = {
        "evidence_pack": args.evidence_pack_json,
        "analyst_profiles": args.analyst_profiles_json,
        "profile_scorecard": args.profile_scorecard_json,
        "learning_bridge": args.learning_bridge_json,
        "review_approved_learning": args.review_approved_learning_json,
        "outcome_evaluation": args.outcome_evaluation_json,
        "calibration_gate": args.calibration_gate_json,
        "calibration_proposals": args.calibration_proposals_json,
        "calibration_review": args.calibration_review_json,
        "manual_backlog": args.manual_backlog_json,
    }
    payload = AnalystLoopDailyCheck(output_dir=args.output_dir).build(
        stage_paths=stage_paths,
        market_data_path=args.market_data_path,
        latest_processed_prices=args.latest_processed_prices,
        tickers=args.tickers,
        as_of=args.as_of,
        max_age_hours=args.max_age_hours,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        event_log_path=args.event_log_path,
        event_limit=args.event_limit,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
