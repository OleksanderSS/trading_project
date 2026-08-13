from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.packets.sector_to_ticker_review_packet import SectorToTickerReviewPacket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SectorToTickerReviewPacket (sector_to_ticker_review_packet).")
    parser.add_argument("--bridge-path", default='reports/dean_os/sector_thesis_to_ticker_basket_current/latest.json')
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/sector_to_ticker_review_packet_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = SectorToTickerReviewPacket(output_dir=args.output_dir).build(
        bridge_path=args.bridge_path,
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
