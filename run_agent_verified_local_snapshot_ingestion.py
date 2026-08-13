from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.verified_local_snapshot_ingestion import VerifiedLocalSnapshotIngestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VerifiedLocalSnapshotIngestion (verified_local_snapshot_ingestion).")
    parser.add_argument("--source-router-json", default=None)
    parser.add_argument("--candidate-path", default=None)
    parser.add_argument("--registration-json", default=None)
    parser.add_argument("--review-gate-json", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--pipeline-paths", action="append", dest="pipeline_paths", default=None)
    parser.add_argument("--prior-outcome-json-paths", action="append", dest="prior_outcome_json_paths", default=None)
    parser.add_argument("--packet-json", default=None)
    parser.add_argument("--journal-path", default='data/dean_os/system_journal.jsonl')
    parser.add_argument("--apply-ingestion", dest="apply_ingestion", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/verified_local_snapshot_ingestion_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = VerifiedLocalSnapshotIngestion(output_dir=args.output_dir).build(
        source_router_json=args.source_router_json,
        candidate_path=args.candidate_path,
        registration_json=args.registration_json,
        review_gate_json=args.review_gate_json,
        as_of=args.as_of,
        pipeline_paths=args.pipeline_paths,
        prior_outcome_json_paths=args.prior_outcome_json_paths,
        packet_json=args.packet_json,
        journal_path=args.journal_path,
        apply_ingestion=args.apply_ingestion,
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
