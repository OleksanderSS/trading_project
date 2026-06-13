from __future__ import annotations

import argparse

from dean_os.agent_learning_loop_runbook import AgentLearningLoopRunbook
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only operator runbook for the safe analyst learning loop.",
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
    parser.add_argument("--output-dir", default="reports/dean_os/agent_learning_loop_runbook")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    position = payload.get("loop_position", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Current: {summary.get('current_stage')} | status={summary.get('current_status')}")
    print(f"Stop reason: {position.get('stop_reason')}")
    print(f"Next command: {position.get('next_command')}")
    print(f"Artifacts: {summary.get('available_artifact_count')}/{summary.get('stage_count')}")
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
    payload = AgentLearningLoopRunbook(output_dir=args.output_dir).build(stage_paths=stage_paths)
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
