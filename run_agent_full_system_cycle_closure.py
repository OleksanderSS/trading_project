from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.full_system_cycle_closure import FullSystemCycleClosureBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FullSystemCycleClosureBuilder (full_system_cycle_closure).")
    parser.add_argument("--cycle-path", default=None)
    parser.add_argument("--world-model-path", default=None)
    parser.add_argument("--prior-checkpoint-monitor-path", default=None)
    parser.add_argument("--replay-review-gate-path", default=None)
    parser.add_argument("--replay-registration-path", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/full_system_cycle_closure_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = FullSystemCycleClosureBuilder(output_dir=args.output_dir).build(
        cycle_path=args.cycle_path,
        world_model_path=args.world_model_path,
        prior_checkpoint_monitor_path=args.prior_checkpoint_monitor_path,
        replay_review_gate_path=args.replay_review_gate_path,
        replay_registration_path=args.replay_registration_path,
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
