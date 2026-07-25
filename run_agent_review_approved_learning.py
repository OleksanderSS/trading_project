from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.review_approved_learning_loop import ReviewApprovedLearningLoop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the explicit review-approved loop for analyst learning promotion.",
    )
    parser.add_argument("--profile-run-json", default=None, help="AnalystProfileOrchestrator JSON.")
    parser.add_argument("--agent-lab-report-json", default=None, help="Direct Agent Lab report JSON.")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--review-actions-store", default="data/dean_os/review_actions.sqlite")
    parser.add_argument("--operations-store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--memory-store", default="data/dean_os/recommendation_memory.sqlite")
    parser.add_argument("--reviewer", default="human")
    parser.add_argument("--review-notes", default="")
    parser.add_argument("--mark-reviewed", action="store_true", help="Record mark_reviewed for discovered Agent Lab reports.")
    parser.add_argument("--needs-more-data", default=None, help="Record an open data request instead of promotion approval.")
    parser.add_argument("--apply", action="store_true", help="Apply learning promotion after review gates pass.")
    parser.add_argument("--allow-weak-notes", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--default-horizon-days", type=int, default=365)
    parser.add_argument("--no-context-summary", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/review_approved_learning")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    gate = payload.get("loop_gate", {})
    bridge_gate = payload.get("final_bridge", {}).get("promotion_gate", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Status: {gate.get('status')} | apply={gate.get('apply_requested')}")
    print(
        "Bridge: "
        f"{bridge_gate.get('status')} | candidates={bridge_gate.get('candidate_count')} | "
        f"promotable={bridge_gate.get('promotable_count')} | promoted={bridge_gate.get('promoted_count')}"
    )
    print(f"Review actions: {gate.get('review_action_count')}")
    for source in payload.get("final_bridge", {}).get("sources", []):
        print(
            f"- {source.get('source_id')} profile={source.get('profile')} "
            f"reviewed={source.get('review', {}).get('reviewed')} promoted={source.get('promoted_count')}"
        )
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = ReviewApprovedLearningLoop(output_dir=args.output_dir).run(
        profile_run_path=args.profile_run_json,
        agent_lab_report_path=args.agent_lab_report_json,
        learning_path=args.learning_store,
        review_actions_path=args.review_actions_store,
        operations_path=args.operations_store,
        memory_path=args.memory_store,
        reviewer=args.reviewer,
        review_notes=args.review_notes,
        mark_reviewed=args.mark_reviewed,
        needs_more_data_request=args.needs_more_data,
        apply=args.apply,
        allow_weak_notes=args.allow_weak_notes,
        allow_duplicates=args.allow_duplicates,
        default_horizon_days=args.default_horizon_days,
        include_context_summary=not args.no_context_summary,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
