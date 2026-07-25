from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dean_os.agent_lab import AgentLabRunner
from dean_os.analyst_core.analyst_evidence_pack import documents_from_evidence_pack
from dean_os.cli_helpers import load_json, print_json
from dean_os.sample_materials import agent_lab_sample_documents
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated DEAN-OS Agent Lab without starting the trading pipeline.")
    parser.add_argument("materials_path", nargs="?", default=None)
    parser.add_argument("--sample", action="store_true", help="Use deterministic sample documents.")
    parser.add_argument("--evidence-pack-json", default=None, help="Analyst evidence pack JSON from run_agent_analyst_evidence_pack.py.")
    parser.add_argument("--corpus", default="data/dean_os/research_corpus.sqlite")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--operations-store", default=None)
    parser.add_argument("--memory-store", default="data/dean_os/recommendation_memory.sqlite")
    parser.add_argument("--log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--output-dir", default="reports/dean_os/agent_lab")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--sectors", nargs="*", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--regime-tags", nargs="*", default=None)
    parser.add_argument("--regime-context-json", default=None)
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--no-financial-nlp", action="store_true")
    parser.add_argument("--no-synthesis", action="store_true")
    parser.add_argument("--no-learning-records", action="store_true")
    parser.add_argument("--no-operation-proposals", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main_async(args: argparse.Namespace):
    tickers = [ticker.upper() for ticker in args.tickers or [] if str(ticker).strip()]
    sectors = [sector for sector in args.sectors or [] if str(sector).strip()]
    tags = [tag for tag in args.tags or [] if str(tag).strip()]
    documents = []
    if args.sample:
        documents.extend(agent_lab_sample_documents(tickers=tickers, sectors=sectors, tags=tags))
    if args.evidence_pack_json:
        documents.extend(documents_from_evidence_pack(args.evidence_pack_json))
    regime_context = load_json(args.regime_context_json) if args.regime_context_json else None

    runner = AgentLabRunner(
        corpus_path=args.corpus,
        learning_path=args.learning_store,
        output_dir=args.output_dir,
        operation_queue_path=args.operations_store,
        memory_path=args.memory_store,
        log_path=args.log_path,
        chunk_size=args.chunk_size,
    )
    return await runner.run(
        materials_path=args.materials_path,
        documents=documents or None,
        tickers=tickers,
        sectors=sectors,
        tags=tags,
        regime_tags=args.regime_tags or [],
        regime_context=regime_context,
        source_type=args.source_type,
        include_financial_nlp=not args.no_financial_nlp,
        include_synthesis=not args.no_synthesis,
        create_learning_records=not args.no_learning_records,
        include_operations_proposals=not args.no_operation_proposals,
    )


def print_summary(report, output_dir: str) -> None:
    print(f"Run ID: {report.run_id}")
    print(f"Documents: {report.document_count} | notes={report.note_count} | proposals={len(report.action_proposals)}")
    print(f"Learning records: {len(report.learning_records)}")
    print(f"Latest thesis: {report.summary.get('latest_thesis') or 'none'}")
    print(f"Report JSON: {Path(output_dir) / f'{report.run_id}.json'}")


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(main_async(args))
    if args.print_json:
        print_json(json_ready(report))
        return
    print_summary(report, args.output_dir)


if __name__ == "__main__":
    main()
