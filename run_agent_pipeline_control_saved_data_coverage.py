from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.pipeline_control.pipeline_control_saved_data_coverage import (
    PipelineControlSavedDataCoverage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory saved asset, timeframe, and macro coverage without training."
    )
    parser.add_argument("--assets-yaml", default="src/config/assets.yaml")
    parser.add_argument("--price-path", action="append", dest="price_paths")
    parser.add_argument("--macro-path", action="append", dest="macro_paths")
    parser.add_argument("--min-rows", type=int, default=180)
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--max-abs-return", type=float, default=0.25)
    parser.add_argument("--min-cadence-ratio", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/pipeline_control_saved_data_coverage_current",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = PipelineControlSavedDataCoverage(args.output_dir).build(
        assets_yaml=args.assets_yaml,
        price_paths=args.price_paths,
        macro_paths=args.macro_paths,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        max_abs_return=args.max_abs_return,
        min_cadence_ratio=args.min_cadence_ratio,
    )
    if args.print_json:
        print_json(payload)
        return
    summary = payload.get("summary", {})
    print(f"Status: {summary.get('coverage_status')}")
    print(f"Configured assets: {summary.get('configured_asset_count')}")
    print(f"Assets with price data: {summary.get('configured_assets_with_price_data')}")
    print(f"Eligible contexts: {summary.get('eligible_context_count')}")
    print(f"Eligible 15m contexts: {summary.get('eligible_15m_context_count')}")
    print(f"Recommended macro source: {summary.get('recommended_macro_source')}")
    print(f"Can trade: {summary.get('can_trade')}")
    saved = payload.get("saved_paths", {})
    if saved:
        print(f"Report JSON: {saved.get('latest_json')}")
        print(f"Report Markdown: {saved.get('latest_markdown')}")


if __name__ == "__main__":
    main()
