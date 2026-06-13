from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dean_os.agents.collector_inventory import CollectorInventoryAgent
from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.schemas import MarketContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Map collector configs/classes without importing collectors or calling networks.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config-path", default="src/config/collectors.yaml")
    parser.add_argument("--collectors-dir", default="src/data/collectors")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/collector_inventory")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    context = MarketContext()
    report = await CollectorInventoryAgent(
        name="collector_inventory",
        config={
            "project_root": args.project_root,
            "config_path": args.config_path,
            "collectors_dir": args.collectors_dir,
        },
    ).run(context)
    payload = {
        "run_id": run_id("collector_inventory"),
        "mode": "collector_inventory_agent",
        "inputs": vars(args),
        "report": report.model_dump(mode="json"),
        "collector_inventory": context.metadata.get("collector_inventory", {}),
    }
    return save_latest_json(args.output, args.output_dir, payload)


def print_summary(payload: dict[str, Any]) -> None:
    report = payload["report"]
    summary = payload.get("collector_inventory", {}).get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Verdict: {report.get('verdict')} | configured={summary.get('configured_count')} | enabled={summary.get('enabled_count')}")
    print(f"Enabled missing classes: {summary.get('enabled_missing_classes', [])}")
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

