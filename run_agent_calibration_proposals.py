from __future__ import annotations

import argparse

from dean_os.calibration_proposal_agent import CalibrationProposalAgent
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create proposal-only calibration review items from an analyst calibration gate report.",
    )
    parser.add_argument("calibration_gate_json")
    parser.add_argument("--operations-store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--include-caution", action="store_true")
    parser.add_argument("--enqueue", action="store_true", help="Write proposed review items to OperationQueue.")
    parser.add_argument("--output-dir", default="reports/dean_os/calibration_proposals")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    gate = payload.get("proposal_gate", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Status: {gate.get('status')} | proposals={gate.get('proposal_count')} | enqueued={gate.get('enqueued_count')}")
    for proposal in payload.get("proposals", []):
        print(f"- {proposal.get('proposal_id')} | {proposal.get('target')} | {proposal.get('status')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = CalibrationProposalAgent(output_dir=args.output_dir).run(
        calibration_gate_path=args.calibration_gate_json,
        operations_path=args.operations_store,
        log_path=args.log_path,
        include_caution=args.include_caution,
        enqueue=args.enqueue,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
