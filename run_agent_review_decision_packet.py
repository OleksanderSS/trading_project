from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.review_decision_packet import ReviewDecisionPacket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only review decision packet for an analyst inbox source.")
    parser.add_argument("--inbox-json", default="reports/dean_os/analyst_review_inbox/latest.json")
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--max-notes", type=int, default=6)
    parser.add_argument("--max-citations-per-note", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=500)
    parser.add_argument("--output-dir", default="reports/dean_os/review_decision_packet")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Source: {summary.get('source_id')} | profile={summary.get('profile')}")
    print(f"Packet status: {summary.get('packet_status')}")
    print(f"Recommended action: {summary.get('recommended_review_action')}")
    print(
        "Checks: "
        f"pass={guidance.get('pass_count')} "
        f"warn={guidance.get('warning_count')} "
        f"fail={guidance.get('fail_count')}"
    )
    if payload.get("saved_paths"):
        print(f"Report JSON: {payload['saved_paths'].get('latest_json') or payload['saved_paths'].get('json')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = ReviewDecisionPacket(output_dir=args.output_dir).build(
        inbox_path=args.inbox_json,
        source_id=args.source_id,
        max_notes=args.max_notes,
        max_citations_per_note=args.max_citations_per_note,
        max_text_chars=args.max_text_chars,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
