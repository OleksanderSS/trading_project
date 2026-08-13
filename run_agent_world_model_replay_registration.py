from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.world_model.world_model_replay_registration import WorldModelReplayRegistrationBridge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WorldModelReplayRegistrationBridge (world_model_replay_registration).")
    parser.add_argument("--output-dir", default="reports/dean_os/world_model_replay_registration_current")
    parser.add_argument("--gate-json", required=True,
                        help="Path to a saved JSON artifact.")
    parser.add_argument("--source-packet-json", default=None,
                        help="Path to a saved JSON artifact.")
    parser.add_argument("--tracker-db-path", default='data/dean_os/outcome_tracker.sqlite')
    parser.add_argument("--apply", dest="apply", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = WorldModelReplayRegistrationBridge(
        output_dir=args.output_dir,
    )
    payload = builder.build(
        gate_json=args.gate_json,
        source_packet_json=args.source_packet_json,
        tracker_db_path=args.tracker_db_path,
        apply=args.apply,
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
