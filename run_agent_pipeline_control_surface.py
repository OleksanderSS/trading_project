from __future__ import annotations

import argparse
import json

from dean_os.pipeline_control_surface import PipelineControlSurface
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a bounded control surface for proposal-only pipeline tuning.",
    )
    parser.add_argument("--model-performance", default=None, help="JSON/CSV model or backtest performance artifact.")
    parser.add_argument("--replay-batch", default=None, help="Historical replay batch JSON artifact.")
    parser.add_argument("--feature-report", default=None, help="JSON/CSV feature importance/stability artifact.")
    parser.add_argument("--data-quality", default=None, help="JSON/CSV leakage/data-quality artifact.")
    parser.add_argument("--constraints", default=None, help="Optional JSON constraints override.")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_surface")
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    surface = payload.get("surface", {})
    gate = payload.get("proposal_gate", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Surface status: {surface.get('status')} | feasible={surface.get('feasible')}")
    print(f"Proposal gate: {gate.get('status')} | can_propose_tuning={gate.get('can_propose_tuning')}")
    print("Axes:")
    for axis in surface.get("axes", []):
        print(f"- {axis.get('name')}: {axis.get('status')} score={axis.get('score')}")
        for reason in axis.get("reasons", [])[:2]:
            print(f"  - {reason}")
    saved = payload.get("saved_paths", {})
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")
        print(f"Report Markdown: {saved.get('latest_markdown') or saved.get('markdown')}")


def main() -> None:
    args = build_parser().parse_args()
    payload = PipelineControlSurface(output_dir=args.output_dir).run(
        model_performance_path=args.model_performance,
        replay_batch_path=args.replay_batch,
        feature_report_path=args.feature_report,
        data_quality_path=args.data_quality,
        constraints_path=args.constraints,
    )
    if args.print_json:
        print(json.dumps(json_ready(payload), indent=2, ensure_ascii=False))
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
