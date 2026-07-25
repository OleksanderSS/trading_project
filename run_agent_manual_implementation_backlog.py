from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.manual_implementation_backlog import ManualImplementationBacklog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report approved calibration proposals waiting for manual implementation.",
    )
    parser.add_argument("--operations-store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--include-proposed", action="store_true")
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--include-non-calibration", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/manual_implementation_backlog")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    gate = payload.get("backlog_gate", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(
        f"Status: {gate.get('status')} | tasks={gate.get('task_count')} | "
        f"approved={gate.get('approved_task_count')} | config_write={gate.get('config_write_performed')}"
    )
    for task in payload.get("tasks", []):
        print(f"- {task.get('task_id')} | {task.get('proposal_status')} | {task.get('target')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = ManualImplementationBacklog(output_dir=args.output_dir).build(
        operations_path=args.operations_store,
        include_proposed=args.include_proposed,
        include_rejected=args.include_rejected,
        include_non_calibration=args.include_non_calibration,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
