from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.replays.historical_replay_outcome_review import HistoricalReplayOutcomeReview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HistoricalReplayOutcomeReview (historical_replay_outcome_review).")
    parser.add_argument("--review-gate-json", default=None)
    parser.add_argument("--registration-json", default=None)
    parser.add_argument("--price-paths", action="append", dest="price_paths", default=None)
    parser.add_argument("--pipeline-paths", action="append", dest="pipeline_paths", default=None)
    parser.add_argument("--task-ids", action="append", dest="task_ids", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/historical_replay_outcome_review_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = HistoricalReplayOutcomeReview(output_dir=args.output_dir).build(
        review_gate_json=args.review_gate_json,
        registration_json=args.registration_json,
        price_paths=args.price_paths,
        pipeline_paths=args.pipeline_paths,
        task_ids=args.task_ids,
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
