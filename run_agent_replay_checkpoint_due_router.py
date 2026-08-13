from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.replays.replay_checkpoint_due_router import ReplayCheckpointDueRouter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReplayCheckpointDueRouter (replay_checkpoint_due_router).")
    parser.add_argument("--output-dir", default="reports/dean_os/replay_checkpoint_due_router_current")
    parser.add_argument("--registration-json", required=True)
    parser.add_argument("--review-gate-json", required=True)
    parser.add_argument("--as-of", default=None,
                        help="ISO-8601 timestamp; defaults to now in UTC.")
    parser.add_argument("--verified-price-paths", action="append", dest="verified_price_paths", default=None,
                        help="Repeatable path to a saved artifact.")
    parser.add_argument("--pipeline-paths", action="append", dest="pipeline_paths", default=None,
                        help="Repeatable path to a saved artifact.")
    parser.add_argument("--outcome-json-paths", action="append", dest="outcome_json_paths", default=None,
                        help="Repeatable path to a saved artifact.")
    parser.add_argument("--due-soon-days", type=int, default=3)
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = ReplayCheckpointDueRouter(
        output_dir=args.output_dir,
    )
    payload = builder.build(
        registration_json=args.registration_json,
        review_gate_json=args.review_gate_json,
        as_of=args.as_of or utc_now_iso(),
        verified_price_paths=args.verified_price_paths,
        pipeline_paths=args.pipeline_paths,
        outcome_json_paths=args.outcome_json_paths,
        due_soon_days=args.due_soon_days,
        save=args.save,
    )
    if args.print_json:
        print_json(payload)
        return
    print(f"Run ID: {payload.get('run_id')}")
    for key, value in (payload.get("summary") or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"{key}: {value}")
    saved = payload.get("saved_paths") or {}
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")


if __name__ == "__main__":
    main()
