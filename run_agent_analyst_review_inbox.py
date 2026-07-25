from __future__ import annotations

import argparse

from dean_os.analyst_core.analyst_review_inbox import AnalystReviewInbox
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only analyst report review inbox.")
    parser.add_argument("--learning-bridge-json", default="reports/dean_os/analyst_learning_bridge/latest.json")
    parser.add_argument("--profile-run-json", default="reports/dean_os/analyst_profiles/latest.json")
    parser.add_argument("--review-actions-store", default="data/dean_os/review_actions.sqlite")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--operations-store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--output-dir", default="reports/dean_os/analyst_review_inbox")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Status: {summary.get('status')}")
    print(f"Sources: {summary.get('source_count')}")
    print(f"Ready for manual review: {summary.get('ready_for_manual_review_count')}")
    print(f"Needs more data: {summary.get('needs_more_data_candidate_count')}")
    print(f"Not reviewable yet: {summary.get('not_reviewable_yet_count')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = AnalystReviewInbox(output_dir=args.output_dir).build(
        learning_bridge_path=args.learning_bridge_json,
        profile_run_path=args.profile_run_json,
        review_actions_path=args.review_actions_store,
        learning_path=args.learning_store,
        operations_path=args.operations_store,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
