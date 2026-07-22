from __future__ import annotations

import argparse
import json
from pathlib import Path

from dean_os.current_cycle_journal import CurrentCycleJournalBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import one verified analysis cycle into SystemJournal."
    )
    parser.add_argument(
        "--cycle-json",
        default="reports/dean_os/full_system_review_cycle_current/latest.json",
    )
    parser.add_argument(
        "--world-model-json",
        default="reports/dean_os/world_model_event_learning_cycle_current/latest.json",
    )
    parser.add_argument(
        "--review-gate-json",
        default="reports/dean_os/world_model_replay_review_gate_cycle_current/latest.json",
    )
    parser.add_argument(
        "--closure-json",
        default="reports/dean_os/full_system_cycle_closure_current/latest.json",
    )
    parser.add_argument(
        "--learning-review-json",
        default="reports/dean_os/hypothesis_learning_review_current/latest.json",
    )
    parser.add_argument("--reasoning-snapshot-json")
    parser.add_argument(
        "--journal-path", default="data/dean_os/system_journal.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/current_cycle_journal_current",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--exclude-full-news", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reasoning_path = (
        Path(args.reasoning_snapshot_json)
        if args.reasoning_snapshot_json
        else None
    )
    payload = CurrentCycleJournalBuilder(args.output_dir).build(
        cycle_json=args.cycle_json,
        world_model_json=args.world_model_json,
        review_gate_json=args.review_gate_json,
        closure_json=args.closure_json,
        learning_review_json=args.learning_review_json,
        reasoning_snapshot_json=reasoning_path,
        journal_path=args.journal_path,
        apply=args.apply,
        include_all_news=not args.exclude_full_news,
        save=not args.no_save,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
