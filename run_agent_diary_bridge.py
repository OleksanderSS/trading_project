from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dean_os.agents.diary_bridge import DiaryBridgeAgent
from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.operation_queue import OperationQueue
from dean_os.schemas import MarketContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect whether DEAN paper outcomes can safely bridge into pipeline diary review.")
    parser.add_argument("--experience-diary", default="logs/experience_diary.csv")
    parser.add_argument("--paper-store", default="data/dean_os/paper_trades.sqlite")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/diary_bridge")
    parser.add_argument("--print-json", action="store_true")
    # Where the proposal goes so a human can act on it.
    #
    # The agent deliberately does not write to the pipeline diary -- it says so
    # itself, and that boundary is the same one every other agent here honours.
    # But its proposal was being written to a JSON file and nowhere else, so
    # the last link of the loop was a report nobody was required to read: paper
    # trades accumulated outcomes, the agent noticed them, and the notice
    # evaporated with the process.
    #
    # `agent_lab` already queues proposals when it is given a store
    # (`if self.operation_queue_path and context.action_proposals`), and
    # DiaryBridgeAgent is not in the registry, so it never runs through
    # agent_lab and never got one. Same destination, reachable from here.
    #
    # Gemini's §15.1 called for the agent to INSERT results into the diary
    # directly. That would remove the boundary rather than close the gap; the
    # queue is where a proposal waits for a person, which is what it is for.
    parser.add_argument(
        "--operations-store",
        default="data/dean_os/operation_queue.sqlite",
        help="queue the proposal here for review; pass an empty string to skip",
    )
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    context = MarketContext()
    report = await DiaryBridgeAgent(
        name="diary_bridge",
        config={
            "experience_diary_path": args.experience_diary,
            "paper_store_path": args.paper_store,
        },
    ).run(context)
    payload = {
        "run_id": run_id("diary_bridge"),
        "mode": "diary_bridge_agent",
        "inputs": vars(args),
        "report": report.model_dump(mode="json"),
        "diary_bridge": context.metadata.get("diary_bridge", {}),
        "action_proposals": [proposal.model_dump(mode="json") for proposal in context.action_proposals],
    }

    queued: list[str] = []
    if args.operations_store and context.action_proposals:
        queued = OperationQueue(args.operations_store).add_many(context.action_proposals)
    payload["queued_proposal_ids"] = queued
    payload["operations_store"] = args.operations_store or None

    return save_latest_json(args.output, args.output_dir, payload)


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(main_async(args))
    if args.print_json:
        print_json(payload)
        return
    bridge = payload.get("diary_bridge", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Status: {bridge.get('status')} | verdict={payload.get('report', {}).get('verdict')}")
    print(f"Bridge candidates: {bridge.get('paper_records', {}).get('bridge_candidate_count')}")
    print(f"Action proposals: {len(payload.get('action_proposals', []))}"
          f" | queued for review: {len(payload.get('queued_proposal_ids', []))}")
    print(f"Report JSON: {payload.get('saved_paths', {}).get('latest_json') or payload.get('saved_paths', {}).get('json')}")


if __name__ == "__main__":
    main()

