from __future__ import annotations

import argparse
import json

from dean_os.analyst_core_reasoning_snapshot import (
    AnalystCoreReasoningSnapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a review-only analyst reasoning snapshot."
    )
    parser.add_argument(
        "--runtime-json",
        default=(
            "reports/dean_os/"
            "semiconductor_analyst_runtime_current/latest.json"
        ),
    )
    parser.add_argument("--hypothesis-journal-path")
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/analyst_core_reasoning_snapshot_current",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = AnalystCoreReasoningSnapshot(args.output_dir).build(
        runtime_json=args.runtime_json,
        hypothesis_journal_path=args.hypothesis_journal_path,
        save=not args.no_save,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
