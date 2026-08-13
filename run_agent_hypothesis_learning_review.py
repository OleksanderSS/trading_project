from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.research_corpus.hypothesis_learning_review import HypothesisLearningReview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HypothesisLearningReview (hypothesis_learning_review).")
    parser.add_argument("--output-dir", default="reports/dean_os/hypothesis_learning_review_current")
    parser.add_argument("--packet-json", required=True,
                        help="Path to a saved JSON artifact.")
    parser.add_argument("--review-gate-json", required=True,
                        help="Path to a saved JSON artifact.")
    parser.add_argument("--outcome-json", default=None,
                        help="Path to a saved JSON artifact.")
    parser.add_argument("--journal-path", default='data/dean_os/system_journal.jsonl')
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = HypothesisLearningReview(
        output_dir=args.output_dir,
    )
    payload = builder.build(
        packet_json=args.packet_json,
        review_gate_json=args.review_gate_json,
        outcome_json=args.outcome_json,
        journal_path=args.journal_path,
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
