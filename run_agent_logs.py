from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.event_log import EventLog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect DEAN-OS structured event logs.")
    parser.add_argument("--log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--print-json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary")
    tail = sub.add_parser("tail")
    tail.add_argument("--limit", type=int, default=10)
    tail.add_argument("--event-type", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log = EventLog(args.log_path)
    payload = log.summary() if args.command == "summary" else {"events": log.read(limit=args.limit, event_type=args.event_type)}
    if args.print_json:
        print_json(payload)
        return
    if args.command == "summary":
        print(f"Events: {payload.get('event_count')} | latest={payload.get('latest_event', {}).get('event_type') if payload.get('latest_event') else None}")
        print(f"By source: {payload.get('source_counts')}")
    else:
        for event in payload["events"]:
            print(f"- {event.get('timestamp')} | {event.get('event_type')} | {event.get('source')} | run={event.get('run_id')}")


if __name__ == "__main__":
    main()

