from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.evidence_gap_resolution_plan import EvidenceGapResolutionPlan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only plan for resolving analyst evidence gaps.")
    parser.add_argument("--review-action-json", default="reports/dean_os/review_action_apply_ceremony/latest.json")
    parser.add_argument("--decision-packet-json", default="reports/dean_os/review_decision_packet/latest.json")
    parser.add_argument("--evidence-pack-json", default=None)
    parser.add_argument("--source-routing-json", default=None)
    parser.add_argument("--min-documents-per-missing-ticker", type=int, default=2)
    parser.add_argument("--min-date-span-days", type=int, default=30)
    parser.add_argument("--suggested-max-rows-per-table", type=int, default=200)
    parser.add_argument("--output-dir", default="reports/dean_os/evidence_gap_resolution_plan")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Source: {summary.get('source_type')}:{summary.get('source_id')}")
    print(f"Plan status: {summary.get('plan_status')}")
    print(f"Missing tickers: {', '.join(summary.get('missing_tickers', [])) or 'none'}")
    print(f"Tasks: {summary.get('task_count')}")
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = EvidenceGapResolutionPlan(output_dir=args.output_dir).build(
        review_action_path=args.review_action_json,
        decision_packet_path=args.decision_packet_json,
        evidence_pack_path=args.evidence_pack_json,
        source_routing_path=args.source_routing_json,
        min_documents_per_missing_ticker=args.min_documents_per_missing_ticker,
        min_date_span_days=args.min_date_span_days,
        suggested_max_rows_per_table=args.suggested_max_rows_per_table,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
