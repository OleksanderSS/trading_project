from __future__ import annotations

import argparse
import asyncio

from dean_os.analyst_core.analyst_profile_orchestrator import AnalystProfileOrchestrator
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the managed analyst profile flow from an evidence pack.",
    )
    parser.add_argument("evidence_pack_json", help="Evidence pack JSON from run_agent_analyst_evidence_pack.py.")
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--sectors", nargs="*", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--allow-candidate-profiles", action="store_true")
    parser.add_argument("--create-learning-records", action="store_true")
    parser.add_argument("--include-operation-proposals", action="store_true")
    parser.add_argument("--no-review-snapshot", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/analyst_profiles")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--learning-store", default=None)
    parser.add_argument("--operations-store", default=None)
    parser.add_argument("--review-actions-store", default=None)
    parser.add_argument("--memory-store", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict:
    orchestrator = AnalystProfileOrchestrator(
        output_dir=args.output_dir,
        corpus_path=args.corpus,
        learning_path=args.learning_store,
        operations_path=args.operations_store,
        review_actions_path=args.review_actions_store,
        memory_path=args.memory_store,
        log_path=args.log_path,
    )
    return await orchestrator.run(
        evidence_pack_path=args.evidence_pack_json,
        profiles=args.profiles,
        tickers=args.tickers,
        sectors=args.sectors,
        tags=args.tags,
        allow_candidate_profiles=args.allow_candidate_profiles,
        create_learning_records=args.create_learning_records,
        include_operation_proposals=args.include_operation_proposals,
        build_review_snapshot=not args.no_review_snapshot,
    )


def print_summary(payload: dict) -> None:
    plan = payload.get("profile_plan", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Profiles to run: {', '.join(plan.get('profiles_to_run', [])) or 'none'}")
    print(f"Skipped profiles: {len(plan.get('skipped_profiles', []))}")
    for skipped in plan.get("skipped_profiles", []):
        print(f"- skipped {skipped.get('profile')}: {skipped.get('reason')}")
    for run in payload.get("profile_runs", []):
        print(f"- {run.get('profile')}: {run.get('status')} via {run.get('runner')}")
    if payload.get("review_snapshot"):
        print("Review snapshot: created")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(main_async(args))
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()

