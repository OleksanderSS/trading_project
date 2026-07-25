from __future__ import annotations

import argparse

from dean_os.analyst_core.analyst_calibration_gate import AnalystCalibrationGate
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build proposal-only analyst calibration guidance from scorecards and outcomes.",
    )
    parser.add_argument("--profile-scorecard-json", default=None)
    parser.add_argument("--profile-runs-dir", default="reports/dean_os/analyst_profiles")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--memory-store", default="data/dean_os/recommendation_memory.sqlite")
    parser.add_argument("--min-profile-runs", type=int, default=3)
    parser.add_argument("--min-completed-outcomes", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, default=0.55)
    parser.add_argument("--max-miss-rate", type=float, default=0.4)
    parser.add_argument("--allow-scorecard-candidate", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/analyst_calibration_gate")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Profiles: {summary.get('profile_count')} | statuses={summary.get('status_counts')}")
    print(f"Ready for review: {summary.get('ready_for_review_profiles')}")
    for profile, card in payload.get("profiles", {}).items():
        print(
            f"- {profile}: {card.get('calibration_status')} | "
            f"completed_outcomes={card.get('outcomes', {}).get('completed_count')} | "
            f"hit_rate={card.get('outcomes', {}).get('hit_rate')} | "
            f"delta={card.get('suggested_weight_delta')}"
        )
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = AnalystCalibrationGate(output_dir=args.output_dir).run(
        profile_scorecard_path=args.profile_scorecard_json,
        profile_runs_dir=args.profile_runs_dir,
        learning_path=args.learning_store,
        memory_path=args.memory_store,
        min_profile_runs=args.min_profile_runs,
        min_completed_outcomes=args.min_completed_outcomes,
        min_hit_rate=args.min_hit_rate,
        max_miss_rate=args.max_miss_rate,
        require_scorecard_ready=not args.allow_scorecard_candidate,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
