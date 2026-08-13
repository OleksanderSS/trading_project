from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.filing_order_evidence import FilingOrderEvidenceBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FilingOrderEvidenceBuilder (filing_order_evidence).")
    parser.add_argument("--output-dir", default="reports/dean_os/filing_order_evidence_current")
    parser.add_argument("--companyfacts-paths", action="append", dest="companyfacts_paths", required=True,
                        help="Repeatable path to a saved artifact.")
    parser.add_argument("--as-of", default=None,
                        help="ISO-8601 timestamp; defaults to now in UTC.")
    parser.add_argument("--max-age-days", type=int, default=730)
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = FilingOrderEvidenceBuilder(
        output_dir=args.output_dir,
    )
    payload = builder.build(
        companyfacts_paths=args.companyfacts_paths,
        as_of=args.as_of or utc_now_iso(),
        max_age_days=args.max_age_days,
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
