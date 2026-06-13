from __future__ import annotations

import argparse
import json
from typing import Any

from dean_os.replay_price_artifact_repair import ReplayPriceArtifactRepairPlan
from dean_os.utils import json_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive candidate repair for mixed replay price artifacts.",
    )
    parser.add_argument("price_data_path", help="Raw cached or normalized price CSV/parquet file.")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker allow-list.")
    parser.add_argument("--output-dir", default="reports/dean_os/replay_price_artifact_repair")
    parser.add_argument("--artifact-dir", default="data/dean_os/replay_prices")
    parser.add_argument("--artifact-path", default=None, help="Optional explicit .csv or .parquet artifact path.")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--anomaly-threshold", type=float, default=0.30)
    parser.add_argument("--anchor-bridge-threshold", type=float, default=0.15)
    parser.add_argument("--write-artifact", action="store_true", help="Write the candidate repaired artifact.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main_payload(args: argparse.Namespace) -> dict[str, Any]:
    return ReplayPriceArtifactRepairPlan(output_dir=args.output_dir, artifact_dir=args.artifact_dir).build(
        price_data_path=args.price_data_path,
        tickers=args.tickers,
        output_path=args.artifact_path,
        close_col=args.close_col,
        datetime_col=args.datetime_col,
        benchmark_ticker=args.benchmark_ticker,
        anomaly_threshold=args.anomaly_threshold,
        anchor_bridge_threshold=args.anchor_bridge_threshold,
        write_artifact=args.write_artifact,
    )


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    quarantine = payload.get("quarantine", {})
    gate = payload.get("learning_gate", {})
    artifact = payload.get("artifact", {})
    saved = payload.get("saved_paths", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Repair status: {summary.get('repair_status')}")
    print(f"Artifact: {artifact.get('path')}")
    print(f"Input rows: {summary.get('input_rows')} | candidate rows={summary.get('candidate_rows')}")
    print(f"Quarantined rows: {quarantine.get('row_count')} | affected ticker/dates={quarantine.get('date_count')}")
    print(f"Learning gate: {gate.get('status')} | can_write_learning_memory={gate.get('can_write_learning_memory')}")
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
