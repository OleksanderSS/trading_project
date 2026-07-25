from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dean_os.agents.model_performance import ModelPerformanceAgent
from dean_os.agents.operations import OperationsProposalAgent
from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.schemas import MarketContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect model/backtest metrics without training, tuning, or pipeline execution.")
    parser.add_argument("performance_path", nargs="?", default=None)
    parser.add_argument("--min-validation-score", type=float, default=0.55)
    parser.add_argument("--min-sharpe", type=float, default=0.0)
    parser.add_argument("--max-drawdown", type=float, default=0.25)
    parser.add_argument("--min-sample-count", type=int, default=50)
    parser.add_argument("--max-age-hours", type=float, default=24 * 30)
    parser.add_argument("--include-operation-proposal", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/model_performance")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    context = MarketContext()
    report = await ModelPerformanceAgent(
        name="model_performance",
        config={
            "performance_path": args.performance_path,
            "min_validation_score": args.min_validation_score,
            "min_sharpe": args.min_sharpe,
            "max_drawdown": args.max_drawdown,
            "min_sample_count": args.min_sample_count,
            "max_age_hours": args.max_age_hours,
        },
    ).run(context)
    proposal_report = None
    if args.include_operation_proposal:
        proposal_report = await OperationsProposalAgent(name="operations_proposal", config={"proposal_only": True}).run(context)
    payload = {
        "run_id": run_id("model_performance"),
        "mode": "model_performance_agent",
        "inputs": vars(args),
        "report": report.model_dump(mode="json"),
        "proposal_report": proposal_report.model_dump(mode="json") if proposal_report else None,
        "model_performance": context.metadata.get("model_performance", {}),
        "action_proposals": [proposal.model_dump(mode="json") for proposal in context.action_proposals],
    }
    return save_latest_json(args.output, args.output_dir, payload)


def print_summary(payload: dict[str, Any]) -> None:
    report = payload["report"]
    metrics = payload.get("model_performance", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Verdict: {report.get('verdict')} | score={metrics.get('performance_score')}")
    print(f"Threshold failures: {metrics.get('threshold_failures', [])}")
    print(f"Action proposals: {len(payload.get('action_proposals', []))}")
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

