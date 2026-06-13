from __future__ import annotations

import argparse

from dean_os.analyst_learning_apply_ceremony import AnalystLearningApplyCeremony
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply pending analyst learning records from a validated bridge dry-run.")
    parser.add_argument("--bridge-dry-run-json", default="reports/dean_os/analyst_learning_bridge/latest.json")
    parser.add_argument("--learning-store", default=None)
    parser.add_argument("--review-actions-store", default=None)
    parser.add_argument("--operations-store", default=None)
    parser.add_argument("--apply-learning", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/analyst_learning_apply_ceremony")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    validation = payload.get("validation", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Apply status: {summary.get('apply_status')}")
    print(f"Learning write performed: {summary.get('learning_write_performed')}")
    print(f"Promoted: {summary.get('promoted_count')} / promotable={summary.get('promotable_count')}")
    print(f"Reasons: {validation.get('reasons')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = AnalystLearningApplyCeremony(output_dir=args.output_dir).apply(
        bridge_dry_run_path=args.bridge_dry_run_json,
        learning_path=args.learning_store,
        review_actions_path=args.review_actions_store,
        operations_path=args.operations_store,
        apply_learning=args.apply_learning,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
