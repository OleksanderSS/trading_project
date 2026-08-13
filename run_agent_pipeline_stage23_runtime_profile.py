from __future__ import annotations

import argparse
import asyncio

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.pipeline_stage23_runtime_profile import PipelineStage23RuntimeProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PipelineStage23RuntimeProfile (pipeline_stage23_runtime_profile).")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_stage23_runtime_profile_current")
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--tickers", action="append", dest="tickers", required=True)
    parser.add_argument("--timeframes", action="append", dest="timeframes", required=True)
    parser.add_argument("--max-rows-per-ticker", type=int, default=80)
    parser.add_argument("--include-stage2", dest="include_stage2", action="store_true")
    parser.add_argument("--include-stage3", dest="include_stage3", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    builder = PipelineStage23RuntimeProfile(
        output_dir=args.output_dir,
    )
    payload = await builder.build(
        source_path=args.source_path,
        tickers=args.tickers,
        timeframes=args.timeframes,
        max_rows_per_ticker=args.max_rows_per_ticker,
        include_stage2=args.include_stage2,
        include_stage3=args.include_stage3,
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
    asyncio.run(main())
