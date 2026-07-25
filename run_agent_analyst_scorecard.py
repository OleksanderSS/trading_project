from __future__ import annotations

import argparse

from dean_os.analyst_core.analyst_profile_scorecard import AnalystProfileScorecard
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build activation scorecards for analyst profiles from saved profile runs.")
    parser.add_argument("--profile-runs-dir", default="reports/dean_os/analyst_profiles")
    parser.add_argument("--min-completed-runs", type=int, default=3)
    parser.add_argument("--min-avg-confidence", type=float, default=0.55)
    parser.add_argument("--min-avg-citations", type=float, default=1.0)
    parser.add_argument("--output-dir", default="reports/dean_os/analyst_profile_scorecard")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = AnalystProfileScorecard(output_dir=args.output_dir).build(
        profile_runs_dir=args.profile_runs_dir,
        min_completed_runs=args.min_completed_runs,
        min_avg_confidence=args.min_avg_confidence,
        min_avg_citations=args.min_avg_citations,
    )
    if args.print_json:
        print_json(payload)
        return
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Profile runs: {payload.get('summary', {}).get('orchestrator_run_count')}")
    print(f"Ready profiles: {payload.get('summary', {}).get('activation_ready_profiles')}")
    print(f"Candidate profiles: {payload.get('summary', {}).get('keep_candidate_profiles')}")
    print(f"Blocked profiles: {payload.get('summary', {}).get('blocked_profiles')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


if __name__ == "__main__":
    main()

