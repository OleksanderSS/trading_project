from __future__ import annotations

import argparse
import json
from typing import Any

from dean_os.historical_evidence_backfill_plan import HistoricalEvidenceBackfillPlan
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only plan for backfilling weak historical research replay evidence.",
    )
    parser.add_argument("--readiness-json", default="reports/dean_os/replay_calibration_readiness_gate_after_step14_research/latest.json")
    parser.add_argument(
        "--research-batch-json",
        default="reports/dean_os/historical_research_replay_batch_repaired_expanded_step14/latest.json",
    )
    parser.add_argument("--news-data", nargs="*", default=None)
    parser.add_argument("--macro-data", nargs="*", default=None)
    parser.add_argument("--materials", nargs="*", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--min-documents-per-run", type=int, default=5)
    parser.add_argument("--output-dir", default="reports/dean_os/historical_evidence_backfill_plan")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main_payload(args: argparse.Namespace) -> dict[str, Any]:
    return HistoricalEvidenceBackfillPlan(output_dir=args.output_dir).build(
        readiness_report_path=args.readiness_json,
        research_batch_path=args.research_batch_json,
        news_data_paths=args.news_data,
        macro_data_paths=args.macro_data,
        materials_paths=args.materials,
        tickers=args.tickers,
        lookback_days=args.lookback_days,
        min_documents_per_run=args.min_documents_per_run,
    )


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    saved = payload.get("saved_paths", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Backfill status: {summary.get('backfill_status')}")
    print(f"Weak runs: {summary.get('weak_run_count')} | missing tickers={summary.get('missing_tickers')}")
    print(f"Tasks: {summary.get('task_count')}")
    for task in payload.get("backfill_tasks", [])[:6]:
        print(f"- {task.get('priority')} {task.get('task_id')}: {task.get('description')}")
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
