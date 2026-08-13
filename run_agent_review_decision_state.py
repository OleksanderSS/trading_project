from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.review_decision_state import ReviewDecisionStateBuilder

# Build parameters not exposed on this CLI (non-scalar types); call the
# builder directly if you need them:
#   previous_state: ReviewDecisionState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReviewDecisionStateBuilder (review_decision_state).")
    parser.add_argument("--evidence-plan-path", default=None)
    parser.add_argument("--voi-review-path", default=None)
    parser.add_argument("--actor", default='dean_os_policy')
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/review_decision_state_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = ReviewDecisionStateBuilder(output_dir=args.output_dir).build(
        evidence_plan_path=args.evidence_plan_path,
        voi_review_path=args.voi_review_path,
        actor=args.actor,
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
