from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.industry_operational_metrics import IndustryOperationalMetricsBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IndustryOperationalMetricsBuilder (industry_operational_metrics).")
    parser.add_argument("--output-dir", default="reports/dean_os/industry_operational_metrics_current")
    parser.add_argument("--records", action="append", dest="records", required=True,
                        help="Repeatable path to a saved artifact.")
    parser.add_argument("--as-of", default=None,
                        help="ISO-8601 timestamp; defaults to now in UTC.")
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--input-reference", default='in_memory_operator_packet')
    parser.add_argument("--input-sha256", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = IndustryOperationalMetricsBuilder(
        output_dir=args.output_dir,
    )
    payload = builder.build(
        records=args.records,
        as_of=args.as_of or utc_now_iso(),
        domain_id=args.domain_id,
        input_reference=args.input_reference,
        input_sha256=args.input_sha256,
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
