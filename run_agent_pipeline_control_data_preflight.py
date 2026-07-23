from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.pipeline_control.pipeline_control_data_preflight import (
    PipelineControlDataPreflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run saved-data coverage and non-destructive repair as one offline command."
    )
    parser.add_argument("--assets-yaml", default="src/config/assets.yaml")
    parser.add_argument("--price-path", action="append", dest="price_paths")
    parser.add_argument("--macro-path", action="append", dest="macro_paths")
    parser.add_argument("--required-model-rows", type=int, default=180)
    parser.add_argument("--min-daily-source-bars", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/pipeline_control_data_preflight_current",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = PipelineControlDataPreflight(args.output_dir).build(
        assets_yaml=args.assets_yaml,
        price_paths=args.price_paths,
        macro_paths=args.macro_paths,
        required_model_rows=args.required_model_rows,
        min_daily_source_bars=args.min_daily_source_bars,
    )
    if args.print_json:
        print_json(payload)
        return
    
    summary = payload.get("summary", {})
    print(f"Preflight status: {summary.get('preflight_status')}")
    print(f"Configured assets: {summary.get('configured_asset_count')}")
    print(f"Eligible contexts: {summary.get('eligible_context_count')}")
    print(f"Timeframes ready: {summary.get('timeframes_ready_for_required_rows', [])}")
    print(f"Timeframes short: {summary.get('timeframes_still_short', [])}")
    print(f"Can trade: {summary.get('can_trade')}")
    
    saved = payload.get("saved_paths", {})
    if saved:
        print(f"Report JSON: {saved.get('latest_json')}")
        print(f"Report Markdown: {saved.get('latest_markdown')}")


if __name__ == "__main__":
    main()
