from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.review_action_apply_ceremony import ReviewActionApplyCeremony


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record exactly one validated review action from a dry-run artifact.")
    parser.add_argument("--dry-run-json", default="reports/dean_os/review_action_dry_run/latest.json")
    parser.add_argument("--review-actions-store", default="data/dean_os/review_actions.sqlite")
    parser.add_argument("--operations-store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--apply-review-action", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/review_action_apply_ceremony")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    validation = payload.get("validation", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Source: {summary.get('source_type')}:{summary.get('source_id')}")
    print(f"Action: {summary.get('action_type')}")
    print(f"Apply status: {summary.get('apply_status')}")
    print(f"Review action write performed: {summary.get('review_action_write_performed')}")
    if summary.get("recorded_action_id"):
        print(f"Recorded action: {summary.get('recorded_action_id')}")
    print(f"Reasons: {validation.get('reasons')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = ReviewActionApplyCeremony(output_dir=args.output_dir).apply(
        dry_run_path=args.dry_run_json,
        review_actions_path=args.review_actions_store,
        operations_path=args.operations_store,
        event_log_path=args.log_path,
        apply_review_action=args.apply_review_action,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
