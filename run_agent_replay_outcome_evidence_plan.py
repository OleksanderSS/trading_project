from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.replays.replay_outcome_evidence_plan import ReplayOutcomeEvidencePlanBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReplayOutcomeEvidencePlanBuilder (replay_outcome_evidence_plan).")
    parser.add_argument("--packet-path", default=None)
    parser.add_argument("--routing-path", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/replay_outcome_evidence_plan_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = ReplayOutcomeEvidencePlanBuilder(output_dir=args.output_dir).build(
        packet_path=args.packet_path,
        routing_path=args.routing_path,
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
