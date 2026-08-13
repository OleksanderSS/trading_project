from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.industry_operational_source_coverage import IndustryOperationalSourceCoverageBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IndustryOperationalSourceCoverageBuilder (industry_operational_source_coverage).")
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--research-sqlite-path", default=None)
    parser.add_argument("--knowledge-pack-path", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/industry_operational_source_coverage_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = IndustryOperationalSourceCoverageBuilder(output_dir=args.output_dir).build(
        duckdb_path=args.duckdb_path,
        research_sqlite_path=args.research_sqlite_path,
        knowledge_pack_path=args.knowledge_pack_path,
        save=args.save,
    )
    if args.print_json:
        print_json(payload)
        return
    print(f"Run ID: {payload.get('run_id')}")
    for key, value in (payload.get("summary") or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"{key}: {value}")
    saved = payload.get("saved_paths") or {}
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")


if __name__ == "__main__":
    main()
