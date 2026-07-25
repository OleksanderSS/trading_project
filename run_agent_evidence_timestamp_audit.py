from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.evidence_timestamp_audit import EvidenceTimestampAudit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit cached evidence source timestamps before historical research replay.",
    )
    parser.add_argument("--source-data", nargs="*", default=None, help="Generic CSV/parquet/json source tables.")
    parser.add_argument("--news-data", nargs="*", default=None, help="Cached news CSV/parquet/json tables.")
    parser.add_argument("--macro-data", nargs="*", default=None, help="Cached macro CSV/parquet/json tables.")
    parser.add_argument("--evidence-pack-json", default=None, help="Optional AnalystEvidencePack JSON to compare.")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--min-parse-rate", type=float, default=0.75)
    parser.add_argument("--collapse-share-threshold", type=float, default=0.95)
    parser.add_argument("--collapse-min-rows", type=int, default=10)
    parser.add_argument("--output-dir", default="reports/dean_os/evidence_timestamp_audit")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Audit status: {summary.get('audit_status')}")
    print(
        "Sources: "
        f"ready={summary.get('ready_count')} "
        f"suspicious={summary.get('suspicious_count')} "
        f"blocked={summary.get('blocked_count')}"
    )
    print(f"Evidence pack status: {summary.get('evidence_pack_status')}")
    print(f"Can run historical research replay: {summary.get('can_run_historical_research_replay')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")
        print(f"Report Markdown: {payload['saved_paths'].get('latest_markdown') or payload['saved_paths'].get('markdown')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = EvidenceTimestampAudit(output_dir=args.output_dir).run(
        source_paths=args.source_data,
        news_data_paths=args.news_data,
        macro_data_paths=args.macro_data,
        evidence_pack_path=args.evidence_pack_json,
        as_of=args.as_of,
        start_at=args.start_at,
        min_parse_rate=args.min_parse_rate,
        collapse_share_threshold=args.collapse_share_threshold,
        collapse_min_rows=args.collapse_min_rows,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
