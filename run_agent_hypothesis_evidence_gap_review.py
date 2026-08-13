from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.research_corpus.hypothesis_evidence_gap_review import HypothesisEvidenceGapReview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HypothesisEvidenceGapReview (hypothesis_evidence_gap_review).")
    parser.add_argument("--output-dir", default="reports/dean_os/hypothesis_evidence_gap_review_current")
    parser.add_argument("--analyst-review-path", required=True)
    parser.add_argument("--fundamental-artifact-path", required=True)
    parser.add_argument("--ratio-artifact-path", default=None)
    parser.add_argument("--primary-snapshot-path", default=None)
    parser.add_argument("--operational-metrics-path", default=None)
    parser.add_argument("--filing-order-evidence-path", default=None)
    parser.add_argument("--as-of", default=None,
                        help="ISO-8601 timestamp; defaults to now in UTC.")
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = HypothesisEvidenceGapReview(
        output_dir=args.output_dir,
    )
    payload = builder.build(
        analyst_review_path=args.analyst_review_path,
        fundamental_artifact_path=args.fundamental_artifact_path,
        ratio_artifact_path=args.ratio_artifact_path,
        primary_snapshot_path=args.primary_snapshot_path,
        operational_metrics_path=args.operational_metrics_path,
        filing_order_evidence_path=args.filing_order_evidence_path,
        as_of=args.as_of or utc_now_iso(),
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
