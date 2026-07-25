from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dean_os.agents.diary_bridge import DiaryBridgeAgent
from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.schemas import MarketContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect whether DEAN paper outcomes can safely bridge into pipeline diary review.")
    parser.add_argument("--experience-diary", default="logs/experience_diary.csv")
    parser.add_argument("--paper-store", default="data/dean_os/paper_trades.sqlite")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/diary_bridge")
    parser.add_argument("--print-json", action="store_true")
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
    print(f"Action proposals: {len(payload.get('action_proposals', []))}")
    print(f"Report JSON: {payload.get('saved_paths', {}).get('latest_json') or payload.get('saved_paths', {}).get('json')}")


if __name__ == "__main__":
    main()

