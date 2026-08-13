from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.packets.source_extraction_fixture_packet import SourceExtractionFixturePacket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SourceExtractionFixturePacket (source_extraction_fixture_packet).")
    parser.add_argument("--contract-json", default='reports/dean_os/source_extraction_review_packet_current/latest.json')
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--no-prefer-timestamped", dest="prefer_timestamped", action="store_false")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/source_extraction_fixture_packet_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = SourceExtractionFixturePacket(output_dir=args.output_dir).build(
        contract_json=args.contract_json,
        max_items=args.max_items,
        prefer_timestamped=args.prefer_timestamped,
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
