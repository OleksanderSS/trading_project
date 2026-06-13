from __future__ import annotations

import argparse

from dean_os.analyst_evidence_pack import AnalystEvidencePackRunner
from dean_os.cli_helpers import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local-only evidence pack for analyst agents from materials, cached news, and macro data.",
    )
    parser.add_argument("--materials", nargs="*", default=None, help="Local files/directories with research materials.")
    parser.add_argument("--news-data", nargs="*", default=None, help="Cached news CSV/parquet/json files.")
    parser.add_argument("--macro-data", nargs="*", default=None, help="Cached macro CSV/parquet/json files.")
    parser.add_argument("--source-routing-json", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--sectors", nargs="*", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--end-at", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--max-rows-per-table", type=int, default=200)
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--max-text-chars", type=int, default=6000)
    parser.add_argument("--no-routed-materials", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/analyst_evidence_pack")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    coverage = payload.get("coverage", {})
    analyst = payload.get("analyst_inputs", {}).get("base_analyst", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Quality: {coverage.get('data_quality')} | documents={coverage.get('document_count')}")
    print(f"Source types: {coverage.get('by_source_type')}")
    print(f"Tickers: {', '.join(coverage.get('tickers', [])) or 'none'}")
    print(f"Base analyst ready: {analyst.get('ready')}")
    print(f"Agent Lab: {analyst.get('agent_lab_command_preview')}")
    if payload.get("saved_paths"):
        print(f"Evidence pack JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = AnalystEvidencePackRunner(output_dir=args.output_dir).run(
        materials_paths=args.materials or [],
        news_data_paths=args.news_data or [],
        macro_data_paths=args.macro_data or [],
        source_routing_path=args.source_routing_json,
        tickers=args.tickers or [],
        sectors=args.sectors or [],
        tags=args.tags or [],
        start_at=args.start_at,
        end_at=args.end_at,
        as_of=args.as_of,
        max_rows_per_table=args.max_rows_per_table,
        max_documents=args.max_documents,
        max_text_chars=args.max_text_chars,
        include_routed_materials=not args.no_routed_materials,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()

