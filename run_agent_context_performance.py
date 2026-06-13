from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.context_performance import AgentPerformanceByContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize agent performance by theme/regime context.")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--memory-store", default="data/dean_os/recommendation_memory.sqlite")
    parser.add_argument("--agent-name", default=None)
    parser.add_argument("--context-tag", default=None)
    parser.add_argument("--min-completed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/context_performance")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = AgentPerformanceByContext(args.learning_store, args.memory_store).build_summary(
        agent_name=args.agent_name,
        context_tag=args.context_tag,
        min_completed=args.min_completed,
        limit=args.limit,
    )
    payload = {"run_id": run_id("context_performance"), "inputs": vars(args), **summary}
    payload = save_latest_json(args.output, args.output_dir, payload)
    if args.print_json:
        print_json(payload)
        return
    print(f"Completed outcomes: {payload.get('overall', {}).get('completed_count')}")
    print(f"Weak contexts: {len(payload.get('weak_contexts', []))} | strengths={len(payload.get('strengths', []))}")
    for item in payload.get("recommendations", []):
        print(f"- {item}")
    print(f"Report JSON: {payload.get('saved_paths', {}).get('latest_json') or payload.get('saved_paths', {}).get('json')}")


if __name__ == "__main__":
    main()

