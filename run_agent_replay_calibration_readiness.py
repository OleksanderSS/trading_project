from __future__ import annotations

import argparse
import json
from typing import Any

from dean_os.replay_calibration_readiness_gate import ReplayCalibrationReadinessGate
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether repaired replay evidence is ready for manual analyst calibration review.",
    )
    parser.add_argument("--repair-report-json", default="reports/dean_os/replay_price_artifact_repair_current/latest.json")
    parser.add_argument(
        "--price-quality-json",
        default="reports/dean_os/replay_price_quality_investigation_repaired_artifact_only_v2/latest.json",
    )
    parser.add_argument("--replay-batch-json", default="reports/dean_os/historical_replay_batch_repaired_202603_202604/latest.json")
    parser.add_argument(
        "--research-batch-json",
        default="reports/dean_os/historical_research_replay_batch_repaired_202603_202604/latest.json",
    )
    parser.add_argument("--min-clean-replay-runs", type=int, default=10)
    parser.add_argument("--min-clean-research-runs", type=int, default=10)
    parser.add_argument("--max-quality-blocked-runs", type=int, default=0)
    parser.add_argument("--max-price-warning-records", type=int, default=0)
    parser.add_argument("--max-weak-evidence-runs", type=int, default=0)
    parser.add_argument("--min-directional-research-ratio", type=float, default=0.25)
    parser.add_argument("--output-dir", default="reports/dean_os/replay_calibration_readiness_gate")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main_payload(args: argparse.Namespace) -> dict[str, Any]:
    return ReplayCalibrationReadinessGate(output_dir=args.output_dir).build(
        repair_report_path=args.repair_report_json,
        price_quality_report_path=args.price_quality_json,
        replay_batch_path=args.replay_batch_json,
        research_batch_path=args.research_batch_json,
        min_clean_replay_runs=args.min_clean_replay_runs,
        min_clean_research_runs=args.min_clean_research_runs,
        max_quality_blocked_runs=args.max_quality_blocked_runs,
        max_price_warning_records=args.max_price_warning_records,
        max_weak_evidence_runs=args.max_weak_evidence_runs,
        min_directional_research_ratio=args.min_directional_research_ratio,
    )


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    gate = payload.get("gate", {})
    saved = payload.get("saved_paths", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Readiness status: {summary.get('readiness_status')}")
    print(f"Next action: {summary.get('next_action')}")
    print(f"Blockers: {summary.get('blocker_count')} | cautions={summary.get('caution_count')}")
    print(f"Can create calibration review packet: {summary.get('can_create_calibration_review_packet')}")
    for blocker in gate.get("blockers", [])[:5]:
        print(f"- blocker {blocker.get('check')}: {blocker.get('reason')}")
    for caution in gate.get("cautions", [])[:5]:
        print(f"- caution {caution.get('check')}: {caution.get('reason')}")
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")
        print(f"Report Markdown: {saved.get('latest_markdown') or saved.get('markdown')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = main_payload(args)
    if args.print_json:
        print(json.dumps(json_ready(payload), indent=2, ensure_ascii=False))
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
