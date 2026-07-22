from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.full_system_orchestrator import create_full_agent_system
from dean_os.schemas import MarketContext


def _load_json(path: str | None) -> Any:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


async def _run(args: argparse.Namespace) -> int:
    pipeline_result = _load_json(args.pipeline_stage03_json)
    evidence = _load_json(args.evidence_json) or []
    if not isinstance(evidence, list):
        raise ValueError("--evidence-json must contain a JSON list")

    context = MarketContext(
        as_of=args.as_of,
        tickers=args.tickers,
        timeframes=args.timeframes,
        metadata={"knowledge_cutoff": args.knowledge_cutoff or args.as_of},
    )
    system = create_full_agent_system(
        project_root=args.project_root,
        domain_id=args.domain_id,
        soft_mode=args.soft_mode,
        persistence_enabled=not args.no_persistence,
        reports_root=args.reports_root,
        briefing_output_dir=args.briefing_output_dir,
    )
    result = await system.run(
        context,
        pipeline_stage03_result=pipeline_result,
        evidence_payloads=evidence,
    )
    payload = result.model_dump(mode="json")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete DEAN-OS agent-system skeleton against existing pipeline stages 0-3 outputs."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--domain-id", default="semiconductor_ai_infrastructure")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--knowledge-cutoff")
    parser.add_argument("--pipeline-stage03-json")
    parser.add_argument("--evidence-json")
    parser.add_argument("--tickers", nargs="*", default=[])
    parser.add_argument("--timeframes", nargs="*", default=["1d"])
    parser.add_argument("--reports-root")
    parser.add_argument("--briefing-output-dir")
    parser.add_argument("--output")
    parser.add_argument("--soft-mode", action="store_true", default=False)
    parser.add_argument("--no-persistence", action="store_true")
    return parser


def main() -> int:
    return asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
