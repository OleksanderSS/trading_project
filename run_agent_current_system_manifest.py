from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.current_system_manifest import CurrentSystemManifestBuilder

# Build parameters not exposed on this CLI (non-scalar types); call the
# builder directly if you need them:
#   operating_profile: SystemOperatingProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CurrentSystemManifestBuilder (current_system_manifest).")
    parser.add_argument("--output-dir", default="reports/dean_os/current_system_manifest_current")
    parser.add_argument("--topology-path", default='dean_os/config/system_topology.yaml')
    parser.add_argument("--authorization-ledger-path", default='data/dean_os/accumulation_authorization_ledger.jsonl')
    parser.add_argument("--as-of", default=None,
                        help="ISO-8601 timestamp; defaults to now in UTC.")
    parser.add_argument("--domain-id", default='semiconductor_ai_infrastructure')
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = CurrentSystemManifestBuilder(
        output_dir=args.output_dir,
        topology_path=args.topology_path,
        authorization_ledger_path=args.authorization_ledger_path,
    )
    payload = builder.build(
        as_of=args.as_of or utc_now_iso(),
        domain_id=args.domain_id,
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
