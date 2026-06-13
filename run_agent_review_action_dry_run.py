from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.review_action_dry_run import ReviewActionDryRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview a review action from a decision packet without writing it.")
    parser.add_argument("--packet-json", default="reports/dean_os/review_decision_packet/latest.json")
    parser.add_argument("--intent", choices=["mark_reviewed", "needs_more_data"], default="needs_more_data")
    parser.add_argument("--reviewer", default="human")
    parser.add_argument("--review-notes", default="")
    parser.add_argument(
        "--data-request",
        default="Add stronger citations or missing source coverage before learning promotion.",
    )
    parser.add_argument("--acknowledge-warnings", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/review_action_dry_run")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    validation = payload.get("validation", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Source: {summary.get('source_id')} | profile={summary.get('profile')}")
    print(f"Intent: {summary.get('intent')}")
    print(f"Dry-run status: {summary.get('dry_run_status')}")
    print(f"Can record: {summary.get('can_record_review_action')}")
    print(f"Reasons: {validation.get('reasons')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = ReviewActionDryRun(output_dir=args.output_dir).build(
        packet_path=args.packet_json,
        intent=args.intent,
        reviewer=args.reviewer,
        review_notes=args.review_notes,
        data_request=args.data_request,
        acknowledge_warnings=args.acknowledge_warnings,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
