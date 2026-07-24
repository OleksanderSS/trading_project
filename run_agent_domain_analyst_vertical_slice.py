from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_vertical_slice_run import DomainAnalystVerticalSliceRun, render_domain_analyst_vertical_slice_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the domain analyst vertical slice.")
    parser.add_argument("--domain-id", default=None)
    parser.add_argument("--evidence-pack-json", default=None)
    parser.add_argument("--source-gate-json", default=None)
    parser.add_argument("--pipeline-context-json", default=None)
    parser.add_argument("--materials", nargs="+", default=None)
    parser.add_argument("--news-data", nargs="+", default=None)
    parser.add_argument("--macro-data", nargs="+", default=None)
    parser.add_argument("--source-routing-path", default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--sectors", nargs="+", default=None)
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--sector-keywords", nargs="+", default=None)
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--end-at", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_vertical_slice_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "domain_id": args.domain_id,
        "evidence_pack_json": args.evidence_pack_json,
        "source_gate_json": args.source_gate_json,
        "pipeline_context_json": args.pipeline_context_json,
        "materials_paths": args.materials,
        "news_data_paths": args.news_data,
        "macro_data_paths": args.macro_data,
        "source_routing_path": args.source_routing_path,
        "tickers": args.tickers,
        "sectors": args.sectors,
        "tags": args.tags,
        "sector_keywords": args.sector_keywords,
        "start_at": args.start_at,
        "end_at": args.end_at,
        "as_of": args.as_of,
        "horizon_days": args.horizon_days,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = DomainAnalystVerticalSliceRun(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_domain_analyst_vertical_slice_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
