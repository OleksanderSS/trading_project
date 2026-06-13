from __future__ import annotations

import argparse

from dean_os.analyst_learning_promotion_bridge import AnalystLearningPromotionBridge
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote reviewed analyst notes into learning records; dry-run by default.",
    )
    parser.add_argument("--profile-run-json", default=None, help="AnalystProfileOrchestrator JSON, usually reports/dean_os/analyst_profiles/latest.json.")
    parser.add_argument("--agent-lab-report-json", default=None, help="Direct Agent Lab report JSON.")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--review-actions-store", default="data/dean_os/review_actions.sqlite")
    parser.add_argument("--operations-store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--allow-unreviewed", action="store_true", help="Diagnostics only; do not use for durable promotion policy.")
    parser.add_argument("--allow-weak-notes", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--default-horizon-days", type=int, default=365)
    parser.add_argument("--apply", action="store_true", help="Write promotable learning records.")
    parser.add_argument("--output-dir", default="reports/dean_os/analyst_learning_bridge")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    gate = payload.get("promotion_gate", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Status: {gate.get('status')} | apply={payload.get('inputs', {}).get('apply')}")
    print(
        "Candidates: "
        f"{gate.get('candidate_count')} | promotable={gate.get('promotable_count')} | "
        f"blocked={gate.get('blocked_count')} | promoted={gate.get('promoted_count')}"
    )
    for source in payload.get("sources", []):
        print(
            f"- {source.get('source_id')} profile={source.get('profile')} "
            f"reviewed={source.get('review', {}).get('reviewed')} promotable={source.get('promotable_count')}"
        )
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = AnalystLearningPromotionBridge(output_dir=args.output_dir).run(
        profile_run_path=args.profile_run_json,
        agent_lab_report_path=args.agent_lab_report_json,
        learning_path=args.learning_store,
        review_actions_path=args.review_actions_store,
        operations_path=args.operations_store,
        require_review=not args.allow_unreviewed,
        apply=args.apply,
        allow_weak_notes=args.allow_weak_notes,
        allow_duplicates=args.allow_duplicates,
        default_horizon_days=args.default_horizon_days,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()

