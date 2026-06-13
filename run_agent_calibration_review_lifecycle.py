from __future__ import annotations

import argparse

from dean_os.calibration_review_lifecycle import CalibrationReviewLifecycle
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review calibration operation proposals without writing config or changing weights.",
    )
    parser.add_argument("--operations-store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--proposal-ids", nargs="*", default=None)
    parser.add_argument("--dry-run-proposals", action="store_true")
    parser.add_argument("--approve", nargs="*", default=None, help="Explicit proposal IDs to mark approved in OperationQueue.")
    parser.add_argument("--reject", nargs="*", default=None, help="Explicit proposal IDs to mark rejected in OperationQueue.")
    parser.add_argument("--include-non-calibration", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/calibration_review_lifecycle")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    gate = payload.get("lifecycle_gate", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(
        f"Status: {gate.get('status')} | proposals={gate.get('proposal_count')} | "
        f"dry_runs={gate.get('dry_run_count')} | approved_waiting={gate.get('approved_waiting_manual_implementation_count')}"
    )
    for proposal in payload.get("calibration_proposals", []):
        print(f"- {proposal.get('proposal_id')} | {proposal.get('status')} | {proposal.get('target')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = CalibrationReviewLifecycle(output_dir=args.output_dir).run(
        operations_path=args.operations_store,
        log_path=args.log_path,
        proposal_ids=args.proposal_ids or [],
        dry_run_proposals=args.dry_run_proposals,
        approve_ids=args.approve or [],
        reject_ids=args.reject or [],
        include_non_calibration=args.include_non_calibration,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
