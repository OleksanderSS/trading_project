from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.verified_market_source_router import VerifiedMarketSourceRouter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VerifiedMarketSourceRouter (verified_market_source_router).")
    parser.add_argument("--lifecycle-json", default=None)
    parser.add_argument("--registration-json", default=None)
    parser.add_argument("--review-gate-json", default=None)
    parser.add_argument("--source-policy-json", default='dean_os/config/replay_verified_market_sources.template.json')
    parser.add_argument("--previous-refresh-json-paths", action="append", dest="previous_refresh_json_paths", default=None)
    parser.add_argument("--local-snapshot-paths", action="append", dest="local_snapshot_paths", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/verified_market_source_router_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = VerifiedMarketSourceRouter(output_dir=args.output_dir).build(
        lifecycle_json=args.lifecycle_json,
        registration_json=args.registration_json,
        review_gate_json=args.review_gate_json,
        source_policy_json=args.source_policy_json,
        previous_refresh_json_paths=args.previous_refresh_json_paths,
        local_snapshot_paths=args.local_snapshot_paths,
        as_of=args.as_of,
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
