from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.pipeline_control.pipeline_control_saved_price_repair import (
    PipelineControlSavedPriceRepair,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build non-destructive clean and resampled price candidates from coverage."
    )
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--required-model-rows", type=int, default=180)
    parser.add_argument("--min-daily-source-bars", type=int, default=24)
    parser.add_argument(
        "--domain-id",
        help=(
            "Require a recursively verified domain sector-market coverage "
            "bridge instead of generic pipeline coverage."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/pipeline_control_saved_price_repair_current",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = PipelineControlSavedPriceRepair(args.output_dir).build(
        coverage_json=args.coverage_json,
        required_model_rows=args.required_model_rows,
        min_daily_source_bars=args.min_daily_source_bars,
        domain_id=args.domain_id,
    )
    if args.print_json:
        print_json(payload)
        return
    summary = payload.get("summary", {})
    print(f"Status: {summary.get('repair_status')}")
    print(f"Clean 15m rows: {summary.get('clean_15m_row_count')}")
    print(f"Resampled 60m rows: {summary.get('resampled_60m_row_count')}")
    print(f"Resampled 1d rows: {summary.get('resampled_1d_row_count')}")
    print(f"Ready timeframes: {summary.get('timeframes_ready_for_required_rows')}")
    print(f"Still short: {summary.get('timeframes_still_short')}")
    print(f"Can trade: {summary.get('can_trade')}")
    saved = payload.get("saved_paths", {})
    if saved:
        print(f"Report JSON: {saved.get('latest_json')}")
        print(f"Report Markdown: {saved.get('latest_markdown')}")


if __name__ == "__main__":
    main()
