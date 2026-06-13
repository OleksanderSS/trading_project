from __future__ import annotations

import argparse
import json
from pathlib import Path

from dean_os.review import AgentReviewBuilder
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a DEAN-OS human-review snapshot from lab, learning, queue, memory, and logs.")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--reports-dir", default="reports/dean_os/agent_lab")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--operations-store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--review-actions-store", default="data/dean_os/review_actions.sqlite")
    parser.add_argument("--memory-store", default="data/dean_os/recommendation_memory.sqlite")
    parser.add_argument("--log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--output-dir", default="reports/dean_os/review")
    parser.add_argument("--event-limit", type=int, default=10)
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = AgentReviewBuilder(
        reports_dir=args.reports_dir,
        learning_path=args.learning_store,
        operations_path=args.operations_store,
        review_actions_path=args.review_actions_store,
        memory_path=args.memory_store,
        log_path=args.log_path,
        output_dir=args.output_dir,
    )
    snapshot = builder.build(report_path=args.report_path, event_limit=args.event_limit)
    json_path, md_path = builder.save(snapshot)
    latest_json = Path(args.output_dir) / "latest.json"
    latest_md = Path(args.output_dir) / "latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    snapshot["saved_paths"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }
    if args.print_json:
        print(json.dumps(json_ready(snapshot), indent=2, ensure_ascii=False))
        return
    print(f"Report available: {snapshot['report'].get('available')}")
    print(f"Learning records: {snapshot['learning'].get('total_record_count')}")
    print(f"Operation proposals: {snapshot['operations'].get('proposal_count')}")
    print("Next actions:")
    for action in snapshot.get("next_actions", []):
        print(f"- {action}")
    print(f"Review JSON: {latest_json}")


if __name__ == "__main__":
    main()

