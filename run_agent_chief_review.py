from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dean_os.agents.chief_review import ChiefReviewAgent
from dean_os.cli_helpers import load_json, print_json, run_id, save_latest_json
from dean_os.schemas import MarketContext, PipelineActionProposal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize saved DEAN-OS state into one supervised-autonomy review.")
    parser.add_argument("--review-snapshot", default=None)
    parser.add_argument("--model-performance-json", default=None)
    parser.add_argument("--regime-context-json", default=None)
    parser.add_argument("--tuning-json", default=None)
    parser.add_argument("--context-performance-json", default=None)
    parser.add_argument("--autonomy-mode", default="paper_supervised")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/chief_review")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    proposals = []
    if args.review_snapshot:
        metadata["review_snapshot"] = load_json(args.review_snapshot)
    if args.model_performance_json:
        payload = load_json(args.model_performance_json)
        metadata["model_performance"] = payload.get("model_performance", payload.get("metrics_snapshot", payload))
        proposals.extend(payload.get("action_proposals", []))
    if args.regime_context_json:
        payload = load_json(args.regime_context_json)
        metadata["regime_context"] = payload.get("regime_context", payload)
    if args.tuning_json:
        payload = load_json(args.tuning_json)
        metadata["tuning"] = payload.get("tuning", payload)
        proposals.extend(payload.get("action_proposals", []))
    if args.context_performance_json:
        metadata["context_performance"] = load_json(args.context_performance_json)

    context = MarketContext(metadata=metadata)
    for proposal in proposals:
        if isinstance(proposal, dict):
            context.action_proposals.append(PipelineActionProposal(**proposal))

    report = await ChiefReviewAgent(name="chief_review", config={"autonomy_mode": args.autonomy_mode}).run(context)
    payload = {
        "run_id": run_id("chief_review"),
        "mode": "chief_review_agent",
        "inputs": vars(args),
        "report": report.model_dump(mode="json"),
        "chief_review": context.metadata.get("chief_review", {}),
        "action_proposals": [proposal.model_dump(mode="json") for proposal in context.action_proposals],
    }
    return save_latest_json(args.output, args.output_dir, payload)


def print_summary(payload: dict[str, Any]) -> None:
    review = payload.get("chief_review", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Decision: {review.get('decision')} | verdict={review.get('verdict')}")
    print(f"Autonomy: {review.get('autonomy_recommendation')}")
    print("Next actions:")
    for action in review.get("next_actions", []):
        print(f"- {action}")
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

