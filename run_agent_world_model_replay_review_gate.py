from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.world_model.world_model_replay_review_gate import WorldModelReplayReviewGate

# Build parameters not exposed on this CLI (non-scalar types); call the
# builder directly if you need them:
#   hypothesis_dispositions: dict[str, Any] | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WorldModelReplayReviewGate (world_model_replay_review_gate).")
    parser.add_argument("--output-dir", default="reports/dean_os/world_model_replay_review_gate_current")
    parser.add_argument("--packet-json", required=True,
                        help="Path to a saved JSON artifact.")
    parser.add_argument("--approve", dest="approve", action="store_true")
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--review-notes", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = WorldModelReplayReviewGate(
        output_dir=args.output_dir,
    )
    payload = builder.build(
        packet_json=args.packet_json,
        approve=args.approve,
        reviewer=args.reviewer,
        review_notes=args.review_notes,
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
