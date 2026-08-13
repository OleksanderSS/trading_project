from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.pipeline_control.pipeline_control_historical_price_recovery import PipelineControlHistoricalPriceRecovery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PipelineControlHistoricalPriceRecovery (pipeline_control_historical_price_recovery).")
    parser.add_argument("--historical-15m-path", default=None)
    parser.add_argument("--current-15m-path", default=None)
    parser.add_argument("--historical-1d-path", default=None)
    parser.add_argument("--required-development-rows", type=int, default=180)
    parser.add_argument("--minimum-past-evaluation-rows", type=int, default=60)
    parser.add_argument("--min-daily-source-bars", type=int, default=24)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_historical_price_recovery_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = PipelineControlHistoricalPriceRecovery(output_dir=args.output_dir).build(
        historical_15m_path=args.historical_15m_path,
        current_15m_path=args.current_15m_path,
        historical_1d_path=args.historical_1d_path,
        required_development_rows=args.required_development_rows,
        minimum_past_evaluation_rows=args.minimum_past_evaluation_rows,
        min_daily_source_bars=args.min_daily_source_bars,
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
