from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dean_os.agents.source_routing import SourceRoutingAgent
from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.schemas import MarketContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route local materials and collector inventory to specialist/pipeline intake paths.")
    parser.add_argument("materials_path", nargs="?", default=None)
    parser.add_argument("--collector-inventory", dest="collector_inventory", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/source_routing")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    context = MarketContext()
    report = await SourceRoutingAgent(
        name="source_routing",
        config={
            "materials_path": args.materials_path,
            "collector_inventory_path": args.collector_inventory,
        },
    ).run(context)
    payload = {
        "run_id": run_id("source_routing"),
        "mode": "source_routing_agent",
        "inputs": vars(args),
        "report": report.model_dump(mode="json"),
        "source_routing": context.metadata.get("source_routing", {}),
    }
    return save_latest_json(args.output, args.output_dir, payload)


def print_summary(payload: dict[str, Any]) -> None:
    report = payload["report"]
    summary = payload.get("source_routing", {}).get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Verdict: {report.get('verdict')} | routable={summary.get('routable_source_count')} | warnings={summary.get('warning_count')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(main_async(args))
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()

